"""
Vector Store Module
Manages ChromaDB for storing and retrieving embeddings
"""

from typing import List, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pathlib import Path
import logging
import os
os.environ["CHROMA_DB_TELEMETRY"] = "false"

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store"""
    
    def __init__(self, persist_directory: str, collection_name: str, embeddings: Embeddings):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
            embeddings: Embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = embeddings
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB: {persist_directory}/{collection_name}")
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info("ChromaDB initialized successfully")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        ids = self.vectorstore.add_documents(documents)
        
        logger.info(f"Added {len(ids)} documents")
        
        return ids
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Searching for: {query[:50]}... (k={k})")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Found {len(results)} relevant documents")
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def delete_collection(self):
        """Delete the entire collection"""
        logger.warning("Deleting entire collection")
        self.vectorstore.delete_collection()
    
    def get_document_count(self) -> int:
        """Get total number of documents in store"""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0
    
    def clear_store(self):
        """Clear all documents from store"""
        logger.info("Clearing vector store")
        # Create new empty store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )