"""Step 3 test run: does the agent route correctly, answer correctly, and remember?

Run it the same way you ran test_step2.py (same folder as agent.py).
Make sure Ollama is running and Qdrant is up before you start.
"""

import sqlite3
import time
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

import agent
import config
import tools

# ---------------------------------------------------------------------------
# EDIT ME if your PDF is not the transformer survey.
# Each turn: (question, expected_tool, [answer must contain ANY of these])
#   expected_tool: "none"     -> NO tool may be called
#                  "overview" -> get_document_overview must be called
#                  "search"   -> search_documents must be called
#                  "any"      -> don't check tool use
# An empty list means "don't check answer text".
# Turns inside one scenario share a conversation (same thread).
# ---------------------------------------------------------------------------
NOT_FOUND = [
    "couldn't find", "could not find", "not mention", "no mention",
    "doesn't mention", "does not mention", "not discuss", "doesn't discuss",
    "does not discuss", "no information",
]

SCENARIOS = [
    ("Small talk: greeting", [("Hi there!", "none", [])]),
    ("Small talk: thanks", [("Thanks, that's really helpful!", "none", [])]),
    ("Broad question", [("What is this document about?", "overview", ["transformer"])]),
    ("Specific fact 1", [("What WER did the Speech-Transformer achieve on eval92?", "search", ["10.9"])]),
    ("Specific fact 2", [("Who proposed the Audio Spectrogram Transformer?", "search", ["gong"])]),
    ("Unanswerable but on-topic", [("Does the paper discuss quantum computing hardware?", "search", NOT_FOUND)]),
    ("Follow-up uses history", [
        ("Who proposed the Audio Spectrogram Transformer?", "search", ["gong"]),
        ("What year was it proposed?", "any", ["2021"]),
    ]),
]

# On-topic questions the document cannot answer (for RERANK_MIN_SCORE).
ON_TOPIC_UNANSWERABLE = [
    "Does the paper discuss quantum computing hardware?",
    "What are the home addresses of the authors?",
    "How much did the GPUs used in the experiments cost?",
]
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_turn(res: dict, expect: str, answer_any: list[str]) -> tuple[bool, str]:
    names = {c["name"] for c in res["tool_calls"]}
    reasons = []
    if expect == "none" and names:
        reasons.append(f"called tools {sorted(names)} but should not have")
    elif expect == "overview" and "get_document_overview" not in names:
        reasons.append("did not call get_document_overview")
    elif expect == "search" and "search_documents" not in names:
        reasons.append("did not call search_documents")
    if answer_any and not any(a in res["answer"].lower() for a in answer_any):
        reasons.append(f"answer lacks any of {answer_any[:3]}...")
    if not res["answer"]:
        reasons.append("empty answer")
    return (not reasons, "; ".join(reasons))


def run_scenarios() -> tuple[int, int, str]:
    header("A. AGENT SCENARIOS")
    passed = total = 0
    followup_thread = ""
    for name, turns in SCENARIOS:
        thread_id = f"test-{uuid.uuid4()}"
        if name.startswith("Follow-up"):
            followup_thread = thread_id
        print(f"\n--- {name} ---")
        for question, expect, answer_any in turns:
            t0 = time.time()
            try:
                res = agent.ask(question, thread_id)
            except Exception as e:  # show the error instead of crashing the run
                print(f"[ERROR] {question}\n        {type(e).__name__}: {e}")
                total += 1
                continue
            elapsed = time.time() - t0
            ok, why = check_turn(res, expect, answer_any)
            total += 1
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {question}  ({elapsed:.1f}s)")
            if not ok:
                print(f"        why: {why}")
            if res["tool_calls"]:
                for c in res["tool_calls"]:
                    print(f"        tool call: {c['name']}({c['args']})")
            else:
                print("        tool call: (none)")
            print(f"        sources: {res['sources'] or '(none)'}")
            print(f"        answer: {res['answer'][:350]!r}")
    print(f"\nScenario turns passed: {passed}/{total}")
    return passed, total, followup_thread


def check_persistence(thread_id: str) -> None:
    header("B. MEMORY PERSISTS ACROSS A FRESH AGENT")
    if not thread_id:
        print("[SKIP] no follow-up thread was created")
        return
    conn = sqlite3.connect(config.MEMORY_DB_PATH, check_same_thread=False)
    fresh = agent.build_agent(SqliteSaver(conn))  # brand-new graph + connection
    state = fresh.get_state({"configurable": {"thread_id": thread_id}})
    n = len(state.values.get("messages", []))
    print(f"[{'PASS' if n >= 4 else 'FAIL'}] fresh agent loaded {n} stored messages "
          f"for the follow-up thread (expected 4 or more)")


def calibrate_threshold() -> None:
    header("C. SCORES FOR ON-TOPIC UNANSWERABLE QUESTIONS")
    for q in ON_TOPIC_UNANSWERABLE:
        ranked = tools.retrieve(q)
        top1 = ranked[0][1] if ranked else float("-inf")
        print(f"  top score {top1:7.2f} | {q}")
    print("\nFor comparison, your answerable questions in test_step2 scored "
          "between about 2.1 and 7.5 at rank 1.")
    print("Send me these numbers; they decide whether a RERANK_MIN_SCORE is safe.")


if __name__ == "__main__":
    p, t, followup = run_scenarios()
    check_persistence(followup)
    calibrate_threshold()