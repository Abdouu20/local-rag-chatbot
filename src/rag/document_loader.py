"""
Document Loading Module
Supports PDF, TXT, MD, DOCX files
"""

from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various file formats"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.docx': Docx2txtLoader,
    }
    
    def __init__(self):
        self.supported_extensions = list(self.SUPPORTED_EXTENSIONS.keys())
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and return documents
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.supported_extensions}"
            )
        
        logger.info(f"Loading document: {file_path}")
        
        try:
            loader = self.SUPPORTED_EXTENSIONS[extension](str(path))
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/chunks from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported files from a directory
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of all Document objects
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"*{ext}"):
                try:
                    docs = self.load_file(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions"""
        return self.supported_extensions.copy()