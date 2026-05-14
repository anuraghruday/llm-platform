from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Google Gemini
    google_api_key: str
    gemini_model: str = "gemini-2.5-pro"
    gemini_model_fast: str = "gemini-2.5-flash"

    # Paths
    chroma_db_path: str = "./chroma_db"
    mlflow_tracking_uri: str = "./mlruns"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG
    retrieval_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # API
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # vLLM
    vllm_url: str = "http://localhost:8001"
    vllm_model: str = "finetuned"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Router
    router_confidence_threshold: float = 0.8

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
