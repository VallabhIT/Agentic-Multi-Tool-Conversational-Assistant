import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", str(DATA_DIR / "chroma")))

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.1)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        return ChatOllama(model=model, temperature=0.1)

    raise ValueError(f"Unsupported LLM provider: {provider}")
