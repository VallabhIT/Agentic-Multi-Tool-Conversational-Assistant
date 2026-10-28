from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.config import get_llm
from app.prompts import ROUTER_PROMPT
from app.rag_subgraph import build_rag_subgraph
from app.state import AgentState
from app.tools import duckduckgo_search, is_dangerous_sql, run_python, run_sql, detect_dangerous_python

def router_node(state: AgentState):
    query = state["user_query"]
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(query=query)
    resp = llm.invoke(prompt)
    text = str(resp.content).strip()
    route = "direct"
    try:
        import json
        route = json.loads(text).get("route", "direct")
    except Exception:
        route = "direct"
    return {"route": route, "status": "routed"}

def rag_node(state: AgentState):
    rag = build_rag_subgraph()
    rag_state = {
        "user_query": state["user_query"],
        "retrieved_docs": [],
        "judge_score": 0.0,
        "relevant": False,
        "retry_count": 0,
        "rewritten_query": "",
        "final_answer": "",
        "status": "idle"
    }

    rag_result = rag.invoke(rag_state)
    final_answer = rag_result.get("final_answer", "No answer generated.")
    return {
        "final_answer": final_answer,
        "status": "rag_complete"
    }

def web_fallback_node(state: AgentState):
    query = state["user_query"]
    results = duckduckgo_search(query, max_results=5)
    return {
        "web_results": results,
        "final_answer": "I used web search fallback because internal docs were insufficient.\n\n" + "\n\n".join(results[:3]),
        "status": "web_complete"
    }

def sql_guardrail_node(state: AgentState):
    query = state["user_query"]
    if is_dangerous_sql(query):
        return {
            "approval_required": True,
            "approval_message": f"Approve this SQL before execution:\n\n{query}",
            "status": "awaiting_approval",
            "approved": None,
        }
    try:
        rows = run_sql(query)
        return {
            "final_answer": f"SQL result: {rows}",
            "status": "sql_complete"
        }
    except Exception as exc:
        return {
            "final_answer": f"SQL execution failed: {str(exc)}",
            "status": "sql_error"
        }

def python_guardrail_node(state: AgentState):
    code = state["python_code"] or state["user_query"]
    if detect_dangerous_python(code):
        return {
            "approval_required": True,
            "approval_message": f"Approve this Python code before execution:\n\n{code}",
            "status": "awaiting_approval",
            "approved": None,
        }
    try:
        result = run_python(code)
        return {
            "final_answer": f"Python execution result:\n{result}",
            "status": "python_complete"
        }
    except Exception as exc:
        return {
            "final_answer": f"Python execution failed: {str(exc)}",
            "status": "python_error"
        }

def finalize_node(state: AgentState):
    if state.get("final_answer"):
        return {"final_answer": state["final_answer"], "status": "finalized"}
    return {
        "final_answer": "I could not determine the best answer path.",
        "status": "finalized"
    }

def route_after_router(state: AgentState):
    return state["route"]

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("sql_guardrail", sql_guardrail_node)
    workflow.add_node("python_guardrail", python_guardrail_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag",
            "sql": "sql_guardrail",
            "python": "python_guardrail",
            "web": "web_fallback",
            "direct": "finalize",
        }
    )

    workflow.add_edge("rag", "finalize")
    workflow.add_edge("sql_guardrail", "finalize")
    workflow.add_edge("python_guardrail", "finalize")
    workflow.add_edge("web_fallback", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
