"""
Main RAG Pipeline
Orchestrates all components
"""

from src.config import settings, VECTOR_STORE_DIR
from src.rag.document_loader import DocumentLoader
from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import RAGRetriever
from src.llm.model import LocalLLM
from src.llm.prompts import RAGPrompts
from src.security.validator import InputValidator
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG Pipeline"""
    
    def __init__(self):
        """Initialize all RAG components"""
        logger.info("Initializing RAG Pipeline...")
        
        # Document processing
        self.document_loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Embeddings and vector store
        self.embedding_generator = EmbeddingGenerator(settings.embedding_model)
        self.vector_store = VectorStoreManager(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=settings.chroma_collection,
            embeddings=self.embedding_generator.get_embeddings()
        )
        
        # Retrieval
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            top_k=settings.top_k_retrieval
        )
        
        # LLM
        self.llm = LocalLLM(settings.llm_model)
        
        # Prompts
        self.prompts = RAGPrompts()
        
        # Security
        self.validator = InputValidator()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_document(self, file_path: str) -> int:
        """
        Ingest a document into the vector store
        
        Args:
            file_path: Path to document file
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Ingesting document: {file_path}")
        
        # Load document
        documents = self.document_loader.load_file(file_path)
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        # Add to vector store
        ids = self.vector_store.add_documents(chunks)
        
        logger.info(f"Ingested {len(ids)} chunks from {file_path}")
        
        return len(ids)
    
    def ingest_directory(self, directory_path: str) -> int:
        """
        Ingest all documents from a directory
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Total number of chunks added
        """
        logger.info(f"Ingesting directory: {directory_path}")
        
        documents = self.document_loader.load_directory(directory_path)
        chunks = self.chunker.chunk_documents(documents)
        ids = self.vector_store.add_documents(chunks)
        
        logger.info(f"Ingested {len(ids)} chunks from {directory_path}")
        
        return len(ids)
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Validate input
        is_valid, error = self.validator.validate_input(question)
        if not is_valid:
            return {
                "answer": f"Error: {error}",
                "sources": [],
                "success": False
            }
        
        # Sanitize input
        question = self.validator.sanitize_input(question)
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(question)
        
        if not documents:
            return {
                "answer": "No relevant documents found. Please add documents first.",
                "sources": [],
                "success": True
            }
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Create prompt
        prompt = self.prompts.create_rag_prompt(context, question)
        
        # Generate response
        answer = self.llm.generate(prompt)
        
        # Format sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:200] + "..."
            }
            for doc in documents
        ]
        
        logger.info("Query completed successfully")
        
        return {
            "answer": answer,
            "sources": sources,
            "success": True
        }
    
    def query_stream(self, question: str):
        """
        Query with streaming response
        
        Args:
            question: User question
            
        Yields:
            Chunks of the response
        """
        # Validate input
        is_valid, error = self.validator.validate_input(question)
        if not is_valid:
            yield f"Error: {error}"
            return
        
        question = self.validator.sanitize_input(question)
        
        # Retrieve documents
        documents = self.retriever.retrieve(question)
        
        if not documents:
            yield "No relevant documents found. Please add documents first."
            return
        
        context = self.retriever.format_context(documents)
        prompt = self.prompts.create_rag_prompt(context, question)
        
        for chunk in self.llm.generate_stream(prompt):
            yield chunk
    
    def clear_store(self):
        """Clear the vector store"""
        self.vector_store.clear_store()
        logger.info("Vector store cleared")
    
    def get_document_count(self) -> int:
        """Get number of documents in store"""
        return self.vector_store.get_document_count()


# Convenience function for testing
def create_pipeline() -> RAGPipeline:
    """Create and return a RAG pipeline instance"""
    return RAGPipeline()


if __name__ == "__main__":
    # Test the pipeline
    print("Testing RAG Pipeline...")
    pipeline = create_pipeline()
    print(f"Document count: {pipeline.get_document_count()}")
    print("Pipeline ready!")