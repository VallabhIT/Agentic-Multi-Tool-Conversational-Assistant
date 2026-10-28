import json
from typing import Literal

from langgraph.graph import END, StateGraph

from app.config import get_llm
from app.prompts import ANSWER_PROMPT, JUDGE_PROMPT, REWRITE_PROMPT
from app.state import RAGState
from app.tools import retrieve_docs, rewrite_query

def judge_docs(query: str, docs: list[str]) -> dict:
    llm = get_llm()
    prompt = JUDGE_PROMPT.format(query=query, docs="\n---\n".join(docs))
    result = llm.invoke(prompt)
    text = str(result.content).strip()

    try:
        parsed = json.loads(text)
        return {
            "relevant": bool(parsed.get("relevant", False)),
            "score": float(parsed.get("score", 0.0)),
            "reason": parsed.get("reason", "")
        }
    except Exception:
        return {
            "relevant": False,
            "score": 0.0,
            "reason": "Parse error in judge output"
        }

def generate_answer(query: str, docs: list[str]) -> str:
    llm = get_llm()
    context = "\n---\n".join(f"[Doc {i+1}] {doc}" for i, doc in enumerate(docs))
    prompt = ANSWER_PROMPT.format(query=query, context=context)
    result = llm.invoke(prompt)
    return str(result.content).strip()

def retrieve_node(state: RAGState):
    docs = retrieve_docs(state["user_query"], k=5)
    return {"retrieved_docs": docs, "status": "retrieved"}

def judge_node(state: RAGState):
    result = judge_docs(state["user_query"], state["retrieved_docs"])
    return {
        "judge_score": result["score"],
        "relevant": result["relevant"],
        "status": "judged"
    }

def rewrite_node(state: RAGState):
    new_query = rewrite_query(state["user_query"])
    return {"rewritten_query": new_query, "status": "rewritten"}

def generate_node(state: RAGState):
    answer = generate_answer(state["user_query"], state["retrieved_docs"])
    return {"final_answer": answer, "status": "answered"}

def route_after_retrieve(state: RAGState):
    if not state.get("retrieved_docs"):
        return "rewrite"
    return "judge"

def route_after_judge(state: RAGState):
    if state.get("relevant") and state.get("judge_score", 0) >= 0.5:
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite"

def build_rag_subgraph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "rewrite": "rewrite",
            "judge": "judge"
        }
    )
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
