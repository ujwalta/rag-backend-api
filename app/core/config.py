"""
Configuration settings for the RAG backend application.
Loads from environment variables with sensible defaults.
"""
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "RAG Backend API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./rag_database.db",
        description="Database connection URL"
    )
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CHAT_TTL: int = 3600  # 1 hour
    
    # Vector Database
    VECTOR_DB_TYPE: Literal["pinecone", "qdrant", "weaviate", "milvus"] = "qdrant"
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "rag-documents"
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_API_KEY: Optional[str] = None
    
    # Weaviate
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    WEAVIATE_CLASS_NAME: str = "Document"
    
    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "documents"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    BATCH_SIZE: int = 32
    
    # LLM (using local model or API)
    LLM_MODEL: str = "gpt-3.5-turbo"  # Can be changed to local model
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    OPENAI_API_KEY: Optional[str] = None
    
    # Document Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt"}
    
    # RAG
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()