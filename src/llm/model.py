"""
LLM Module
Manages local LLM inference via Ollama
"""

from langchain_ollama import OllamaLLM
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class LocalLLM:
    """Manage local LLM via Ollama"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize LLM
        
        Args:
            model_name: Ollama model name (default from settings)
        """
        self.model_name = model_name or settings.llm_model
        self.base_url = settings.ollama_base_url
        
        logger.info(f"Initializing LLM: {self.model_name} @ {self.base_url}")
        
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.7,
            num_ctx=4096,
        )
        
        logger.info("LLM initialized successfully")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        logger.info(f"Generating response for prompt length: {len(prompt)}")
        
        response = self.llm.invoke(prompt)
        
        logger.info(f"Generated response length: {len(response)}")
        
        return response
    
    def generate_stream(self, prompt: str):
        """
        Generate streaming response from LLM
        
        Args:
            prompt: Input prompt
            
        Yields:
            Chunks of generated text
        """
        for chunk in self.llm.stream(prompt):
            yield chunk