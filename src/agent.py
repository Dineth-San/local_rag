""" LangGraph Agent: Llama 3.2 + 2 retrieval tools + SQLite conversation memory"""

import re 
import sqlite3
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from tools import TOOLS

SYSTEM_PROMPT = """You are an assistant that answers questions about the user's uploaded documents. You have two tools.
 
RULES:
1. Greetings, thanks, or casual chat: reply briefly and DO NOT call any tool.
2. Broad questions (what is the document about, summarize it, main topics): call get_document_overview.
3. Any specific question about the content (facts, numbers, names, definitions, how or why): call search_documents with a short keyword query.
4. Answer ONLY from the tool results. Never use outside knowledge.
5. After each fact, cite its source like this: (source, page N).
6. If the tool results do not contain the answer, reply exactly: "I couldn't find that in the document." Do not guess.
7. If the user's message depends on earlier conversation (for example "tell me more" or "what year was it?"), rewrite it as a full, self-contained query before searching."""
 
_OMITTED = "[Earlier search results omitted to save space.]"
_FORCE_ANSWER = (
    "You have searched enough. Answer now using only what the tool results above "
    "contain. If they do not contain the answer, say: I couldn't find that in the document."
)
_SOURCE_RE = re.compile(r"\[Source: (.+?), page (\d+)\]")


# Context Management (a 3B model with an 8k window needs this)

def _human_indices(messages) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]

def _last_turns(messages, n: int) -> list:
    """Keeps only the last n user turns (cut at a humanmessage so tool pairs stay intact)"""
    idx = _human_indices(messages)
    if len(idx) <= n:
        return list(messages)
    return list(messages[idx[-n]:])

def _compact_old_tool_results(messages) -> list:
    """Replace tool output from PREVIOUS turrns with a stub; keep this turn's intact"""
    idx = _human_indices(messages)
    if not idx:
        return list(messages)
    last = idx[-1]
    out = []
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage) and i < last:
            out.append(ToolMessage(content=_OMITTED, tool_call_id=m.tool_call_id, name=m.name))
        else:
            out.append(m)
    return out


def _tool_calls_this_turn(messages) -> int:
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            count += 1

    return count


## GRAPH ----------------------------

def build_agent(checkpointer=None):
    llm = ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.AGENT_TEMPERATURE,
        num_ctx=config.LLM_NUM_CTX
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        history = _compact_old_tool_results(
            _last_turns(state["messages"], config.HISTORY_TURNS)
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

        if _tool_calls_this_turn(messages) >= config.MAX_TOOL_CALLS:
            response = llm.invoke(messages + [SystemMessage(content=_FORCE_ANSWER)])

        else:
            response = llm_with_tools.invoke(messages)

        return {"messages": [response]}


    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


_graph = None

def get_agent():
    """Lazily build one shared agent with persistent SQLite memory"""
    global _graph
    if _graph is None:
        conn = sqlite3.connect(config.MEMORY_DB_PATH, check_same_thread=False)
        _graph = build_agent(SqliteSaver(conn))

    return _graph


# Public API ----------------------------------

def summarize_turn(messages) -> dict:
    """Pull the answer, tool calls, and cited sources out of the latest turn"""
    idx = _human_indices(messages)
    turn = messages[idx[-1] + 1:] if idx else list(messages)

    tool_calls: list[dict] = []
    sources: list[tuple[str, int]] = []

    for m in turn:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_calls += [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            for src, page in _SOURCE_RE.findall(str(m.content)):
                if (src, int(page)) not in sources:
                    sources.append({src, int(page)})


    answer = ""
    if turn and isinstance(turn[-1], AIMessage):
        answer = str(turn[-1].content).strip()

    return {"answer": answer, "tool_calls": tool_calls, "sources": sources}


def ask(question: str, thread_id: str) -> dict:
    """Run one user turn. Returns {'answer', 'tool_calls', 'sources'}"""

    result = get_agent().invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}}
    )

    return summarize_turn(result["messages"])