import ast
import json
import re
import sqlite3
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from app.config import DATA_DIR, VECTORSTORE_PATH

DANGEROUS_SQL = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bALTER\b",
    r"\bINSERT\b",
    r"\bCREATE\s+TABLE\b",
    r"\bTRUNCATE\b",
    r"\bEXEC\b",
]

def load_demo_documents() -> List[Document]:
    file_path = DATA_DIR / "demo_docs.json"
    if not file_path.exists():
        sample = [
            {
                "id": "doc1",
                "title": "Product Overview",
                "content": "Our product is an AI assistant platform for enterprises. It supports document retrieval, SQL analysis, Python execution, and web search. The system is built for secure internal knowledge access and reporting."
            },
            {
                "id": "doc2",
                "title": "Company Policy",
                "content": "Employees should check internal documentation before making product claims. For sensitive data tasks, approval is required before executing SQL or Python code that touches production systems."
            },
            {
                "id": "doc3",
                "title": "Data Warehouse Guide",
                "content": "The analytics database contains customers, orders, products, and support tickets. Use read-only queries for data exploration. Avoid destructive statements like DROP, DELETE, ALTER, or UPDATE without approval."
            },
            {
                "id": "doc4",
                "title": "AI Security Rules",
                "content": "The platform blocks dangerous operations such as os.system, subprocess, eval, exec, and direct file writes outside the sandbox. Always ask for approval before running high-risk scripts."
            },
            {
                "id": "doc5",
                "title": "Knowledge Base FAQ",
                "content": "The system can answer questions using private documents, search the web when necessary, and route to SQL or Python when the task requires computation or data access."
            },
        ]
        file_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    docs = []
    for item in data:
        docs.append(
            Document(
                page_content=item["content"],
                metadata={"title": item["title"], "id": item["id"]}
            )
        )
    return docs

def init_vectorstore(force_rebuild: bool = False):
    persist_dir = str(VECTORSTORE_PATH)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if force_rebuild or not Path(persist_dir).exists():
        docs = load_demo_documents()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        return vectorstore

    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def retrieve_docs(query: str, k: int = 5):
    vectorstore = init_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def rewrite_query(query: str) -> str:
    from app.config import get_llm
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
    Rewrite the following query to improve retrieval quality for a knowledge base.
    Keep it concise, precise, and searchable.
    Query: {query}
    """)
    result = llm.invoke(prompt.format(query=query))
    return str(result.content).strip()

def duckduckgo_search(query: str, max_results: int = 5):
    search = DuckDuckGoSearchRun()
    raw = search.run(query)
    if isinstance(raw, str):
        return [raw]
    return raw[:max_results]

def is_dangerous_sql(sql: str) -> bool:
    sql_norm = sql.strip()
    if not sql_norm:
        return True
    for pattern in DANGEROUS_SQL:
        if re.search(pattern, sql_norm, flags=re.IGNORECASE):
            return True
    return False

def detect_dangerous_python(code: str) -> bool:
    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError:
        return True

    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in {"system", "popen", "check_output", "exec", "eval"}:
                    return True
            if isinstance(func, ast.Name) and func.id in {"exec", "eval", "open"}:
                return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False

def run_sql(query: str):
    db_path = DATA_DIR / "demo.db"
    if not db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, spend REAL)")
        cur.execute("INSERT INTO customers (name, city, spend) VALUES ('Alice', 'New York', 2500)")
        cur.execute("INSERT INTO customers (name, city, spend) VALUES ('Bob', 'Boston', 1800)")
        cur.execute("INSERT INTO customers (name, city, spend) VALUES ('Charlie', 'Chicago', 3200)")
        conn.commit()
        conn.close()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute(query).fetchall()
    conn.close()
    return rows

def run_python(code: str):
    local_vars = {"__builtins__": __builtins__}
    exec(code, local_vars, local_vars)
    return local_vars
