"""
Text Chunking Module
Splits documents into optimal chunks for RAG
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Split documents into chunks for RAG"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"TextChunker initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split raw text into chunks
        
        Args:
            text: Raw text to chunk
            metadata: Optional metadata for all chunks
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking text of length {len(text)}")
        
        chunks = self.text_splitter.create_documents([text], metadatas=[metadata or {}])
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks