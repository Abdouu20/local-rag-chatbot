"""
Prompt Templates Module
Defines RAG prompt templates with security measures
"""

from typing import List
from langchain_core.documents import Document


class RAGPrompts:
    """Prompt templates for RAG system"""
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the provided context below
2. If the answer is not in the context, say "I don't have enough information to answer that question"
3. Always cite your sources using [Source X] format
4. Do not make up information or hallucinate
5. Keep answers concise and relevant

Context from documents:
{context}

User Question: {question}

Your Answer:"""

    def __init__(self):
        pass
    
    def create_rag_prompt(self, context: str, question: str) -> str:
        """
        Create a RAG prompt with context and question
        
        Args:
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        return self.SYSTEM_PROMPT.format(context=context, question=question)
    
    def format_sources(self, documents: List[Document]) -> str:
        """
        Format document sources for citation
        
        Args:
            documents: List of source documents
            
        Returns:
            Formatted sources string
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            sources.append(f"[Source {i}]: {source}")
        
        return "\n".join(sources)