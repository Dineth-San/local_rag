"""Step 2 test run.

Usage (from the project root, same folder as tools.py):
    uv run python test_step2.py
"""

import statistics
import time
from collections import defaultdict

from langchain_qdrant import QdrantVectorStore, RetrievalMode

import config
import document_processor as dp
import tools

# ---------------------------------------------------------------------------
# EDIT THIS SECTION.
# Each entry: (category, question, [needles])
# A "needle" is a short exact string (a number, name, or phrase) that MUST
# appear in the chunk that answers the question. Matching is case-insensitive.
# If ANY needle appears in a retrieved chunk, that chunk counts as a hit.
# Only use questions whose answer lives in ONE place in the PDF.
# The examples below are for the transformer survey paper (test.pdf).
# ---------------------------------------------------------------------------
EVAL_SET = [
    ("keyword",  "What WER did the Speech-Transformer achieve on eval92?", ["10.9"]),
    ("keyword",  "Who proposed the Audio Spectrogram Transformer?", ["Gong"]),
    ("keyword",  "What accuracy was reached on Speech commands v2?", ["98.11"]),
    ("keyword",  "What does TRACKE estimate?", ["keywords"]),
    ("semantic", "When was the transformer model first introduced?", ["June 2017"]),
    ("semantic", "How did the authors collect data for the review?", ["data extraction"]),
]

# Questions the document CANNOT answer (used to calibrate RERANK_MIN_SCORE).
UNANSWERABLE = [
    "What is the recipe for chocolate chip cookies?",
    "Who won the 2018 FIFA World Cup?",
    "How do I change a flat tire?",
]
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def norm(s: str) -> str:
    """Lowercase and collapse all whitespace (PDF text has stray line breaks)."""
    return " ".join(s.lower().split())


def first_hit(docs, needles):
    """Index of the first doc containing any needle, else None."""
    needles = [norm(n) for n in needles]
    for i, d in enumerate(docs):
        text = norm(d.page_content)
        if any(n in text for n in needles):
            return i
    return None


# ---------------------------------------------------------------------------
# A. Collection sanity
# ---------------------------------------------------------------------------
def section_a() -> None:
    header("A. COLLECTION SANITY")
    points, _ = dp._client.scroll(
        config.COLLECTION_NAME, limit=10000, with_payload=True
    )
    by_doc = defaultdict(list)
    for p in points:
        m = p.payload.get("metadata", {})
        key = (m.get("doc_id", "?"), m.get("source", "?"))
        by_doc[key].append((m, p.payload.get("page_content", "")))

    print(f"Total points: {len(points)}   Documents: {len(by_doc)}")
    for (doc_id, source), items in by_doc.items():
        chunks = [(m, t) for m, t in items if m.get("type") == "chunk"]
        summaries = [1 for m, _ in items if m.get("type") == "summary"]
        other = len(items) - len(chunks) - len(summaries)
        print(f"\n  {source}  [{doc_id}]")
        print(f"    chunks={len(chunks)}  summaries={len(summaries)}  untyped/legacy={other}")
        if chunks:
            sizes = [len(t) for _, t in chunks]
            pages = [m["page"] for m, _ in chunks if m.get("page") is not None]
            print(f"    chunk chars min/avg/max = "
                  f"{min(sizes)}/{int(statistics.mean(sizes))}/{max(sizes)}")
            if pages:
                print(f"    pages {min(pages)} to {max(pages)}")
        if len(summaries) != 1:
            print("    WARNING: expected exactly 1 summary for this document")
        if other:
            print("    WARNING: legacy points found (old Phase 1 data in this collection?)")
    if len(by_doc) > 1:
        print("\n  NOTE: more than one document is in the collection. Searches will "
              "cover all of them unless you call tools.set_active_documents([...]).")


# ---------------------------------------------------------------------------
# B. Tool checks
# ---------------------------------------------------------------------------
def section_b() -> None:
    header("B. TOOL CHECKS")

    overview = tools.get_document_overview.invoke({})
    print("get_document_overview (first 300 chars):")
    print("  " + overview[:300].replace("\n", "\n  "))
    print("  ->", "PASS" if overview.strip() and "No documents" not in overview else "FAIL")

    out = tools.search_documents.invoke({"query": "transformer"})
    print("\nsearch_documents output starts with a [Source: ...] label:",
          "PASS" if out.startswith("[Source:") else "FAIL")

    results = tools.retrieve("what is this document about")
    leaked = [d for d, _ in results if d.metadata.get("type") != "chunk"]
    print("Summary chunks never appear in search results:",
          "PASS" if not leaked else "FAIL (summary leaked into search)")


# ---------------------------------------------------------------------------
# C. Retrieval eval
# ---------------------------------------------------------------------------
def dense_only_store() -> QdrantVectorStore:
    dp.ensure_collection()
    return QdrantVectorStore(
        client=dp._client,
        collection_name=config.COLLECTION_NAME,
        embedding=dp._embeddings,
        retrieval_mode=RetrievalMode.DENSE,
        vector_name=config.DENSE_VECTOR_NAME,
    )


def section_c() -> list[float]:
    header("C. RETRIEVAL EVAL  (needle must be in the top "
           f"{config.RERANK_TOP_N} after re-ranking)")
    if config.RERANK_MIN_SCORE is not None:
        print("NOTE: RERANK_MIN_SCORE is set; low-scoring chunks are dropped "
              "and may show up as misses.\n")

    tools.retrieve("warm up the model")  # first call loads the cross-encoder
    dense = dense_only_store()
    k = config.RETRIEVE_K

    stats = defaultdict(lambda: [0, 0])  # category -> [passes, total]
    hybrid_only_wins = 0
    passed_top1_scores: list[float] = []

    for category, question, needles in EVAL_SET:
        t0 = time.time()
        ranked = tools.retrieve(question, k=k, top_n=k)  # all k, re-ranked
        elapsed = time.time() - t0

        docs = [d for d, _ in ranked]
        rank = first_hit(docs, needles)  # 0-based position after re-ranking

        dense_docs = dense.similarity_search(question, k=k, filter=tools._search_filter())
        dense_rank = first_hit(dense_docs, needles)

        if rank is None:
            verdict = "RETRIEVAL MISS"
        elif rank < config.RERANK_TOP_N:
            verdict = "PASS"
        else:
            verdict = "RERANKER MISS"

        stats[category][1] += 1
        if verdict == "PASS":
            stats[category][0] += 1
            if ranked:
                passed_top1_scores.append(ranked[0][1])
        if rank is not None and dense_rank is None:
            hybrid_only_wins += 1

        rank_txt = "-" if rank is None else str(rank + 1)
        dense_txt = "miss" if dense_rank is None else str(dense_rank + 1)
        top1 = f"{ranked[0][1]:.2f}" if ranked else "n/a"
        print(f"[{verdict:<14}] {category:<8} final rank {rank_txt:>2} | "
              f"dense-only rank {dense_txt:>4} | top score {top1:>6} | "
              f"{elapsed:.2f}s | {question}")

    print("\nSummary by category:")
    for cat, (ok, total) in stats.items():
        print(f"  {cat:<9} {ok}/{total} passed")
    print(f"  Questions where hybrid found the answer but dense-only did not: "
          f"{hybrid_only_wins}")
    print("\nHow to read verdicts:")
    print("  PASS            answer chunk is in the top results sent to the LLM")
    print("  RERANKER MISS   retrieved in the top 10 but ranked below the cut-off")
    print("                  -> try RERANK_TOP_N = 4 or 5, or inspect that chunk")
    print("  RETRIEVAL MISS  answer chunk never made the top 10")
    print("                  -> chunking or search problem; also check the needle")
    print("                     really appears in a single chunk (it may be split)")
    return passed_top1_scores


# ---------------------------------------------------------------------------
# D. Score calibration
# ---------------------------------------------------------------------------
def section_d(answerable_scores: list[float]) -> None:
    header("D. SCORE CALIBRATION (for RERANK_MIN_SCORE)")
    unans = []
    for q in UNANSWERABLE:
        ranked = tools.retrieve(q)
        top1 = ranked[0][1] if ranked else float("-inf")
        unans.append(top1)
        print(f"  unanswerable top score {top1:7.2f} | {q}")

    if not answerable_scores or not unans:
        print("\nNot enough data to suggest a threshold.")
        return
    low_ok, high_bad = min(answerable_scores), max(unans)
    print(f"\n  Lowest top score among PASSED answerable questions: {low_ok:.2f}")
    print(f"  Highest top score among unanswerable questions:      {high_bad:.2f}")
    if low_ok > high_bad:
        print(f"  -> A threshold between them separates the two, "
              f"e.g. RERANK_MIN_SCORE = {(low_ok + high_bad) / 2:.1f}")
    else:
        print("  -> The ranges overlap; no clean threshold yet. "
              "Leave RERANK_MIN_SCORE = None and add more questions.")


if __name__ == "__main__":
    section_a()
    section_b()
    scores = section_c()
    section_d(scores)