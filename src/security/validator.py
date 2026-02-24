"""
Security Module
Input validation and prompt injection prevention
"""

import re
from typing import Tuple, Optional
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate and sanitize user inputs"""
    
    # Patterns that might indicate prompt injection
    SUSPICIOUS_PATTERNS = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are now",
        r"bypass",
        r"jailbreak",
        r"dan mode",
        r"developer mode",
    ]
    
    def __init__(self):
        self.max_length = settings.max_input_length
    
    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check length
        if len(text) > self.max_length:
            return False, f"Input too long. Maximum {self.max_length} characters."
        
        if len(text.strip()) == 0:
            return False, "Input cannot be empty."
        
        # Check for suspicious patterns
        text_lower = text.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in text_lower:
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return False, "Invalid input detected."
        
        return True, None
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input
        
        Args:
            text: User input
            
        Returns:
            Sanitized text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove potentially dangerous characters
        text = text.replace("<|", "").replace("|>", "")
        
        return text.strip()