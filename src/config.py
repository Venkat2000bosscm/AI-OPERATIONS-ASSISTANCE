from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
SUPPORTED_GROQ_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
}


def resolve_groq_model(raw_value: str | None) -> str:
    model = (raw_value or "").strip()
    if not model:
        return DEFAULT_GROQ_MODEL
    if model == "llama-3.1-8b-instant":
        return DEFAULT_GROQ_MODEL
    if model in SUPPORTED_GROQ_MODELS:
        return model
    return DEFAULT_GROQ_MODEL


class Settings(BaseModel):
    # Groq / LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    groq_model: str = resolve_groq_model(os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL"))
    groq_max_retries: int = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    groq_retry_base_delay: float = float(os.getenv("GROQ_RETRY_BASE_DELAY", "0.75"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # Backward-compatible aliases for older code paths
    @property
    def openai_api_key(self) -> str:
        return self.groq_api_key

    @property
    def openai_model(self) -> str:
        return self.groq_model

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "cloudops-sentinel-self-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "incident-runbooks")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Internet search
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Self-RAG controls
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_support_retries: int = int(os.getenv("MAX_SUPPORT_RETRIES", "2"))
    max_retrieval_rewrites: int = int(os.getenv("MAX_RETRIEVAL_REWRITES", "2"))
    max_web_rewrites: int = int(os.getenv("MAX_WEB_REWRITES", "2"))
    database_path: str = os.getenv("DATABASE_PATH", "data/audit.db")

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p



@lru_cache
def get_settings() -> Settings:
    return Settings()