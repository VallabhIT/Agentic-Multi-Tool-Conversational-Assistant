ROUTER_PROMPT = """
You are a routing classifier for an AI assistant.

Classify the user request into one of:
- rag
- sql
- python
- web
- direct

Rules:
- Use "rag" for internal knowledge / documentation / policy / company info / FAQs / product details.
- Use "sql" for database lookup, reports, counts, filtering records, analytics, table questions.
- Use "python" for computation, analysis, transformation, coding, script execution, numbers, CSV processing.
- Use "web" for current events, breaking news, latest facts, live info, or when internal docs are likely insufficient.
- Use "direct" for greetings, small talk, general conversations.

Return only JSON:
{"route":"rag"} or {"route":"sql"} etc.
User query: {query}
"""

JUDGE_PROMPT = """
You are a relevance judge for a RAG pipeline.

Given a user question and retrieved documents, decide if the documents are relevant enough to answer the question.

Return JSON only:
{"relevant": true|false, "score": 0.0-1.0, "reason": "brief explanation"}

User question:
{query}

Retrieved docs:
{docs}
"""

REWRITE_PROMPT = """
You are helping improve a retrieval query.
Rewrite the user question to be more precise and likely to find the right documents in a vector search.

Return only the rewritten query, no extra text.

Original question:
{query}
"""

ANSWER_PROMPT = """
You are a grounded answer generator.

Use the supplied context to answer the user question. If the context is missing or weak, say so honestly and use web search fallback or mention uncertainty.

Cite documents by using:
[Doc 1], [Doc 2], etc.

Return a clear answer with concise reasoning.

Question:
{query}

Context:
{context}
"""
