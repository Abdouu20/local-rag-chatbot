"""
Embedding Generation Module
Creates vector embeddings for text chunks
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using local models"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of embedding model to use
        """
        self.model_name = model_name
        
        # Set China mirror for HuggingFace
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        
        logger.info(f"Initializing embedding model: {model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Use CPU for embeddings
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info("Embedding model loaded successfully")
    
    def get_embeddings(self) -> Embeddings:
        """Get the embeddings object for use with LangChain"""
        return self.embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)