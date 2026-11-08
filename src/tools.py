"""Retrieval + re-ranking, exposed as LangChain tools for the agent."""

from functools import lru_cache
from typing import Optional, Sequence

import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.tools import tool
from qdrant_client import models

import config
import document_processor as dp

# Which documents the agent may search. None = all documents.
# (Module-level state is fine for a single-user local app.)
_active_doc_ids: Optional[Sequence[str]] = None


def set_active_documents(doc_ids: Optional[Sequence[str]]) -> None:
    global _active_doc_ids
    _active_doc_ids = list(doc_ids) if doc_ids else None


# ---------------------------------------------------------------------------
# Re-ranker (loaded once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_reranker() -> HuggingFaceCrossEncoder:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceCrossEncoder(
        model_name=config.RERANK_MODEL, model_kwargs={"device": device}
    )


def _search_filter() -> models.Filter:
    """Only real chunks (never summaries), optionally scoped to active docs."""
    must = [
        models.FieldCondition(
            key="metadata.type", match=models.MatchValue(value="chunk")
        )
    ]
    if _active_doc_ids:
        must.append(
            models.FieldCondition(
                key="metadata.doc_id", match=models.MatchAny(any=list(_active_doc_ids))
            )
        )
    return models.Filter(must=must)


# ---------------------------------------------------------------------------
# Plain retrieval function (testable without the LLM)
# ---------------------------------------------------------------------------
def retrieve(
    query: str, k: Optional[int] = None, top_n: Optional[int] = None
) -> list[tuple[Document, float]]:
    """Hybrid search (top-k) -> cross-encoder re-rank -> top-n with scores."""
    k = k or config.RETRIEVE_K
    top_n = top_n or config.RERANK_TOP_N

    store = dp.get_vector_store()
    candidates = store.similarity_search(query, k=k, filter=_search_filter())
    if not candidates:
        return []

    scores = _get_reranker().score([(query, d.page_content) for d in candidates])
    ranked = sorted(
        ((d, float(s)) for d, s in zip(candidates, scores)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if config.RERANK_MIN_SCORE is not None:
        ranked = [(d, s) for d, s in ranked if s >= config.RERANK_MIN_SCORE]
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Agent tools. The docstrings ARE the instructions the 3B model sees.
# ---------------------------------------------------------------------------
@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents for specific facts, details, terms,
    names, numbers, definitions, or explanations. Use this for any specific
    question about the content of the documents. The query should be a short,
    self-contained search phrase containing the key terms, not the user's whole
    message. Do NOT use for greetings or small talk, and do NOT use for broad
    questions like 'what is this document about'."""
    results = retrieve(query)
    if not results:
        return "No relevant passages found in the documents."

    blocks = []
    for doc, _score in results:
        m = doc.metadata
        blocks.append(f"[Source: {m.get('source', '?')}, page {m.get('page', '?')}]\n"
                      f"{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def get_document_overview() -> str:
    """Get a high-level summary of the uploaded document(s). Use this for
    broad questions such as 'what is this document about?', 'summarize this',
    or 'what topics does it cover?'. Takes no input."""
    docs = dp.list_documents()
    if _active_doc_ids:
        docs = [d for d in docs if d["doc_id"] in _active_doc_ids]
    if not docs:
        return "No documents have been uploaded yet."

    parts = []
    for d in docs:
        summary = dp.get_summary(d["doc_id"]) or "(no summary available)"
        parts.append(f"Document: {d['source']}\n{summary}")
    return "\n\n".join(parts)


TOOLS = [search_documents, get_document_overview]