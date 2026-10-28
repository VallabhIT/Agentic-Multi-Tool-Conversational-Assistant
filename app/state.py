from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    route: str
    retrieved_docs: list[str]
    web_results: list[str]
    final_answer: str
    sql_query: str
    python_code: str
    approval_required: bool
    approval_message: str
    approved: Optional[bool]
    status: str
    judge_score: float
    retry_count: int
    error: Optional[str]

class RAGState(TypedDict):
    user_query: str
    retrieved_docs: list[str]
    judge_score: float
    relevant: bool
    retry_count: int
    rewritten_query: str
    final_answer: str
    status: str
