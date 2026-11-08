"""Ingestion pipeline: PDF -> semantic chunks -> hybrid Qdrant collection."""

import hashlib
from typing import Callable, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
import re

import config

ProgressFn = Optional[Callable[[str, float], None]]  # (message, 0.0-1.0)

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------
_client = QdrantClient(url=config.QDRANT_URL)
_embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
_sparse = FastEmbedSparse(model_name=config.SPARSE_MODEL)


def _doc_filter(doc_id: str, chunk_type: Optional[str] = None) -> models.Filter:
    must = [
        models.FieldCondition(
            key="metadata.doc_id", match=models.MatchValue(value=doc_id)
        )
    ]
    if chunk_type:
        must.append(
            models.FieldCondition(
                key="metadata.type", match=models.MatchValue(value=chunk_type)
            )
        )
    return models.Filter(must=must)


# ---------------------------------------------------------------------------
# Collection / vector store
# ---------------------------------------------------------------------------
def ensure_collection() -> None:
    """Create the hybrid (dense + BM25 sparse) collection if it doesn't exist."""
    if _client.collection_exists(config.COLLECTION_NAME):
        return
    _client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config={
            config.DENSE_VECTOR_NAME: models.VectorParams(
                size=config.EMBED_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF  # required for BM25-style scoring
            )
        },
    )
    # Payload indexes make filtering (delete / overview lookup) fast.
    for field in ("metadata.doc_id", "metadata.type"):
        _client.create_payload_index(
            config.COLLECTION_NAME, field, models.PayloadSchemaType.KEYWORD
        )


def get_vector_store() -> QdrantVectorStore:
    ensure_collection()
    return QdrantVectorStore(
        client=_client,
        collection_name=config.COLLECTION_NAME,
        embedding=_embeddings,
        sparse_embedding=_sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def file_hash(path: str) -> str:
    """Content hash used as the document ID (re-uploads dedupe automatically)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_pdf(path: str, doc_id: str, filename: str) -> list[Document]:
    pages = PyPDFLoader(path).load()
    valid_pages = []
    
    for p in pages:
        text = p.page_content.strip()
        if not text:
            continue
            
        # 1. Truncate at the References/Bibliography section
        # We check if it's past the halfway point to avoid cutting off at the Table of Contents
        ref_match = re.search(r'\n(References|Bibliography)\s*\n', text, re.IGNORECASE)
        if ref_match and int(p.metadata.get("page", 0)) > len(pages) / 2:
            # Keep only the text before the "References" header
            text = text[:ref_match.start()].strip()
            if text:
                p.page_content = text
                p.metadata = {
                    "doc_id": doc_id,
                    "source": filename,
                    "page": int(p.metadata.get("page", 0)) + 1,
                }
                valid_pages.append(p)
            break  # Stop processing any subsequent pages completely
            
        # 2. Clean up header/footer noise (e.g., stray page numbers)
        text = re.sub(r'^\d+\n', '', text)
        text = re.sub(r'\n\d+$', '', text)
        
        p.page_content = text
        p.metadata = {
            "doc_id": doc_id,
            "source": filename,
            "page": int(p.metadata.get("page", 0)) + 1,
        }
        valid_pages.append(p)

    return valid_pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def semantic_chunk(pages: list[Document], progress: ProgressFn = None) -> list[Document]:
    """Semantic split per page (keeps page metadata accurate)."""
    chunker = SemanticChunker(
        _embeddings,
        breakpoint_threshold_type=config.BREAKPOINT_TYPE,
        breakpoint_threshold_amount=config.BREAKPOINT_AMOUNT,
    )
    out: list[Document] = []
    for i, page in enumerate(pages):
        try:
            out.extend(chunker.split_documents([page]))
        except Exception:
            # Very short pages can break the percentile calc; keep as one chunk.
            out.append(page)
        if progress:
            progress(f"Chunking page {i + 1}/{len(pages)}", (i + 1) / len(pages))
    return out


def normalize_chunks(chunks: list[Document]) -> list[Document]:
    """Merge tiny chunks into their predecessor; force-split giant ones."""
    merged: list[Document] = []
    for c in chunks:
        text = c.page_content.strip()
        if not text:
            continue
        if (
            merged
            and len(text) < config.MIN_CHUNK_CHARS
            and merged[-1].metadata["doc_id"] == c.metadata["doc_id"]
        ):
            merged[-1].page_content += " " + text
        else:
            merged.append(Document(page_content=text, metadata=dict(c.metadata)))

    # A leading tiny chunk has nothing before it; fold it forward.
    if len(merged) > 1 and len(merged[0].page_content) < config.MIN_CHUNK_CHARS:
        merged[1].page_content = merged[0].page_content + " " + merged[1].page_content
        merged.pop(0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_CHARS, chunk_overlap=config.CHUNK_OVERLAP
    )
    final: list[Document] = []
    for c in merged:
        if len(c.page_content) > config.MAX_CHUNK_CHARS:
            final.extend(splitter.split_documents([c]))
        else:
            final.append(c)

    for i, c in enumerate(final):
        c.metadata["type"] = "chunk"
        c.metadata["chunk_index"] = i
    return final


# ---------------------------------------------------------------------------
# Document-level summary (fixes the "Keyhole Problem")
# ---------------------------------------------------------------------------
def _sample_text(pages: list[Document], limit: int) -> str:
    """Take text from the start, middle, and end so the summary isn't biased."""
    full = "\n\n".join(p.page_content for p in pages)
    if len(full) <= limit:
        return full
    third = limit // 3
    mid = len(full) // 2
    return "\n...\n".join(
        [full[:third], full[mid - third // 2 : mid + third // 2], full[-third:]]
    )


def summarize_document(pages: list[Document], filename: str) -> str:
    llm = ChatOllama(
        model=config.LLM_MODEL, temperature=0, num_ctx=config.LLM_NUM_CTX
    )
    sample = _sample_text(pages, config.SUMMARY_SAMPLE_CHARS)
    prompt = (
        f"Below are excerpts from a document named '{filename}'.\n"
        "Write a concise overview (4-6 sentences): what the document is, who it "
        "is for, and its main topics. Use only the text provided.\n\n"
        f"{sample}"
    )
    return llm.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def document_exists(doc_id: str) -> bool:
    ensure_collection()
    n = _client.count(
        config.COLLECTION_NAME, count_filter=_doc_filter(doc_id), exact=True
    ).count
    return n > 0


def ingest_pdf(path: str, filename: str, progress: ProgressFn = None) -> dict:
    """Full pipeline. Returns {'doc_id', 'chunks', 'skipped'}."""
    doc_id = file_hash(path)
    if document_exists(doc_id):
        return {"doc_id": doc_id, "chunks": 0, "skipped": True}

    if progress:
        progress("Loading PDF", 0.0)
    pages = load_pdf(path, doc_id, filename)
    if not pages:
        raise ValueError(
            "No extractable text found. This may be a scanned PDF (needs OCR)."
        )

    chunks = normalize_chunks(semantic_chunk(pages, progress))

    if progress:
        progress("Generating document summary", 0.0)
    summary = summarize_document(pages, filename)
    summary_doc = Document(
        page_content=summary,
        metadata={
            "doc_id": doc_id,
            "source": filename,
            "page": 0,
            "type": "summary",
        },
    )

    if progress:
        progress("Embedding and storing", 0.0)
    store = get_vector_store()
    store.add_documents(chunks + [summary_doc])
    if progress:
        progress("Done", 1.0)

    return {"doc_id": doc_id, "chunks": len(chunks), "skipped": False}


def delete_document(doc_id: str) -> None:
    ensure_collection()
    _client.delete(
        config.COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
    )


def list_documents() -> list[dict]:
    """Returns [{'doc_id', 'source'}] using the summary chunks as an index."""
    ensure_collection()
    points, _ = _client.scroll(
        config.COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type", match=models.MatchValue(value="summary")
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    return [
        {
            "doc_id": p.payload["metadata"]["doc_id"],
            "source": p.payload["metadata"]["source"],
        }
        for p in points
    ]


def get_summary(doc_id: str) -> Optional[str]:
    ensure_collection()
    points, _ = _client.scroll(
        config.COLLECTION_NAME,
        scroll_filter=_doc_filter(doc_id, "summary"),
        limit=1,
        with_payload=True,
    )
    return points[0].payload["page_content"] if points else None