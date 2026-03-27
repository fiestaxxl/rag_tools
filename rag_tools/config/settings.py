"""
Configuration settings for the RAG Tools module.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseModel):
    """Qdrant vector database configuration."""
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False


class PostgresSettings(BaseModel):
    """PostgreSQL configuration."""
    host: str = "localhost"
    port: int = 5432
    user: str = "rag_tools"
    password: str = "rag_tools_password"
    database: str = "rag_tools"
    min_connections: int = 2
    max_connections: int = 20


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda" for GPU
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True

class APIEmbeddingSettings(BaseModel):
    """API Embedding model configuration."""
    url: str = "http://localhost:1234"
    api_key: Optional[str] = None
    timeout: int = 30
    normalize_embeddings: bool = False


class RerankerSettings(BaseModel):
    """Cross-encoder reranker configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32
    top_k: int = 20  # Number of candidates to rerank

class APIRerankerSettings(BaseModel):
    """API Cross-encoder reranker configuration."""
    url: str = "http://localhost:1234"
    api_key: Optional[str] = None
    timeout: int = 30
    batch_size: int = 32
    top_k: int = 20  # Number of candidates to rerank

class RAGSettings(BaseModel):
    """RAG pipeline configuration."""
    default_top_k: int = 10
    rerank_top_k: int = 5
    min_relevance_score: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 50


class Settings(BaseSettings):
    """Main application settings."""
    # Database settings
    qdrant: QdrantSettings = QdrantSettings()
    postgres: PostgresSettings = PostgresSettings()

    # Model settings
    embedding: EmbeddingSettings = EmbeddingSettings()
    api_embedding: APIEmbeddingSettings = APIEmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    api_reranker: APIRerankerSettings = APIRerankerSettings()
    rag: RAGSettings = RAGSettings()

    # Storage paths
    work_dir: Path = Path("/tmp/rag_tools")

    # Collection names
    tools_collection: str = "mcp_tools"
    tools_chunks_collection: str = "mcp_tools_chunks"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )

    # class Config:
    #     env_file = "/app/.env"
    #     env_nested_delimiter = "__"
    #     extra = "ignore"


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
