from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Nexo Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8081

    # Google Gen AI — set USE_VERTEX_AI=true to use Vertex AI instead of Gemini API
    USE_VERTEX_AI: bool = False
    GEMINI_API_KEY: Optional[str] = None        # Required when USE_VERTEX_AI=false
    VERTEX_PROJECT: Optional[str] = None        # Required when USE_VERTEX_AI=true
    VERTEX_LOCATION: str = "us-central1"        # Required when USE_VERTEX_AI=true
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_RETRIES: int = 3                    # Retry attempts on 429 quota errors

    # MongoDB
    MONGODB_URL: str = "mongodb+srv://deepang_db_user:bNhnLJyiyA4K4wOf@deepan.7g36kzf.mongodb.net/"
    MONGODB_DB: str = "nexo"
    # Encryption
    ENCRYPTION_KEY: str = "change-me-to-a-strong-secret-key"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "domain_docs"
    QDRANT_API_KEY: Optional[str] = None

    # Embeddings
    # Gemini (GEMINI_API_KEY set): model=models/gemini-embedding-001, dim=3072
    # Local  (no API key):         model=all-MiniLM-L6-v2,            dim=384
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIM: int = 3072              # set 384 when using local fallback only

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600

    # Web Search (Tavily)
    TAVILY_API_KEY: Optional[str] = None
    SERPAPI_KEY: Optional[str] = None

    # Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3  # Lowered from 0.65 for testing
    RERANK_TOP_N: int = 3

    # Ingestion
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 75
    MAX_PAGES_PER_DOMAIN: int = 100
    
    # File Upload Limits
    MAX_FILE_SIZE_MB: int = 10                 # Maximum file size in MB
    MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB in bytes

    # WebSocket
    WS_TIMEOUT_SECONDS: int = 120               # Auto-close idle connections after 2 min

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"          # silently ignore unknown env vars (e.g. OPENAI_API_KEY)

@lru_cache()
def get_settings() -> Settings:
    return Settings()
