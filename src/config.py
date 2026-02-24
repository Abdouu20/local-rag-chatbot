"""
Configuration Management for Local RAG Chatbot
Loads settings from .env file
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from .env file"""
    
    # Model Configuration
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    
    # China Mirror
    hf_endpoint: str = "https://hf-mirror.com"
    
    # Embedding Model
    embedding_model: str = "BAAI/bge-m3"
    
    # Vector Database
    chroma_persist_dir: str = "./data/vector_store"
    chroma_collection: str = "rag_documents"
    
    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 100
    top_k_retrieval: int = 5
    
    # Security
    max_input_length: int = 2000
    rate_limit_per_minute: int = 10
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_project_root() -> Path:
    """Get project root directory"""
    return PROJECT_ROOT