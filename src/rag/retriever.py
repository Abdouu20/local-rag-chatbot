"""
Retriever Module
Combines vector store with retrieval logic
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from src.rag.vector_store import VectorStoreManager
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieve relevant documents for RAG"""
    
    def __init__(self, vector_store: VectorStoreManager, top_k: int = 5):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store manager
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, k=self.top_k)
    
    def retrieve_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents with relevance scores
        
        Args:
            query: User query
            
        Returns:
            List of dicts with document and score
        """
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return formatted_results
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)