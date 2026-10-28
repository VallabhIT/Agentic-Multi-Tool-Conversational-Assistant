# Agentic LangGraph RAG Project

This project upgrades a simple tool-calling assistant into a LangGraph workflow with:
- router node for intent classification
- RAG subgraph with retrieval, judge, retry, and answer generation
- fallback to web search when internal docs are insufficient
- human approval gate for dangerous SQL and Python execution
- LLM-as-judge evaluation
- selective tool routing for production-style orchestration

## Setup

1. Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies
   pip install -r requirements.txt

3. Set environment variables
   cp .env.example .env
   Edit .env with your API key if using OpenAI.

4. Run
   python -m app.main

## Example queries

- What does the product do?
- Show me the company policy
- Which customers spent more than 2000?
- Run Python to compute the result of 2 + 2
- What is the latest info on OpenAI?

## Notes

- For SQL or Python operations, the app uses a human approval gate.
- For retrieval failures, it retries and then falls back to web search.
- This is designed as a strong interview-ready project rather than a toy demo.
