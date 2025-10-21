"""
Rate limiter for LLM
"""
import logging
import time
from typing import Dict, Any
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limits API calls to language models
    """
    
    def __init__(self):
        logger.info("RateLimiter initialized")
        # Default rate limits per minute for different services
        self.rate_limits = {
            "openai": {
                "gpt-3.5-turbo": 3500,  # tokens per minute
                "gpt-4": 10000,         # tokens per minute
                "default": 3000         # default rate limit
            },
            "anthropic": {
                "claude-3-sonnet": 5000,
                "default": 4000
            },
            "meta": {
                "default": 2000
            },
            "default": 1000
        }
        
        # Request history for tracking usage
        self.request_history = {
            "openai": deque(),
            "anthropic": deque(),
            "meta": deque(),
            "default": deque()
        }
    
    def wait_if_needed(self, provider: str, model: str, token_count: int) -> float:
        """
        Wait if needed to respect rate limits
        
        Args:
            provider: The API provider (openai, anthropic, etc.)
            model: The model being used
            token_count: Number of tokens in the request
            
        Returns:
            Wait time in seconds
        """
        logger.info(f"Checking rate limits for {provider}/{model} with {token_count} tokens")
        
        if provider not in self.rate_limits:
            provider = "default"
            
        provider_limits = self.rate_limits[provider]
        rate_limit = provider_limits.get(model, provider_limits.get("default", 1000))
        
        # Clean up old requests (older than 1 minute)
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        history = self.request_history[provider]
        while history and history[0]["timestamp"] < one_minute_ago:
            history.popleft()
            
        # Calculate current token usage in the past minute
        current_usage = sum(req["token_count"] for req in history)
        
        # Check if adding this request would exceed the rate limit
        if current_usage + token_count > rate_limit:
            # Calculate wait time (add a 10% buffer)
            wait_time = 60 * (1.1 * (current_usage + token_count) / rate_limit - 1)
            wait_time = max(0, wait_time)
            
            if wait_time > 0:
                logger.warning(f"Rate limit approached for {provider}/{model}. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                
            return wait_time
            
        # Record this request
        history.append({
            "timestamp": now,
            "token_count": token_count,
            "model": model
        })
        
        return 0.0
        
    def update_rate_limit(self, provider: str, model: str, limit: int) -> None:
        """
        Update the rate limit for a provider/model
        
        Args:
            provider: The API provider
            model: The model
            limit: New rate limit (tokens per minute)
        """
        if provider not in self.rate_limits:
            self.rate_limits[provider] = {"default": 1000}
            self.request_history[provider] = deque()
            
        self.rate_limits[provider][model] = limit
        logger.info(f"Updated rate limit for {provider}/{model} to {limit} tokens per minute")