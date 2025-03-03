"""
Token counter for Llm
"""
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TokenCounter:
    """
    Counts tokens for language models
    """
    
    def __init__(self):
        logger.info("TokenCounter initialized")
        # Simple ratio estimates for tokens-to-characters
        self.token_ratios = {
            "english": 0.25,  # ~4 characters per token for English
            "code": 0.3,     # ~3.33 characters per token for code
        }
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count the number of tokens in a text
        
        Args:
            text: The text to count tokens for
            model: The model to use for tokenization
            
        Returns:
            Estimated token count
        """
        logger.info(f"Counting tokens for text of length {len(text)}")
        
        # Dummy implementation using character ratio
        # In a real implementation, this would use the model's tokenizer
        
        # Detect if text is mostly code
        code_patterns = ['{', '}', 'def ', 'class ', 'function', '=>', '()', '[]']
        code_matches = sum(1 for pattern in code_patterns if pattern in text)
        
        content_type = "code" if code_matches >= 2 else "english"
        ratio = self.token_ratios[content_type]
        
        # Estimate token count based on character count
        estimated_tokens = int(len(text) * ratio)
        
        logger.info(f"Estimated {estimated_tokens} tokens (content type: {content_type})")
        return estimated_tokens
        
    def is_within_limit(self, text: str, model: str = "gpt-3.5-turbo", max_tokens: int = None) -> bool:
        """
        Check if the text is within the token limit for the model
        
        Args:
            text: The text to check
            model: The model to check against
            max_tokens: Maximum token limit (overrides model default)
            
        Returns:
            True if within limit, False otherwise
        """
        token_count = self.count_tokens(text, model)
        
        # Default model limits
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "claude-3-sonnet": 100000,
            "llama-2-70b": 4096
        }
        
        limit = max_tokens if max_tokens is not None else model_limits.get(model, 4096)
        
        return token_count <= limit