"""
Caching service for LLMService
"""
import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CachingService:
    """
    Caches responses from language models to reduce API calls
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the caching service
        
        Args:
            ttl_seconds: Time to live for cache entries in seconds
        """
        logger.info("CachingService initialized")
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a cached response
        
        Args:
            key: Dictionary containing the request parameters
            
        Returns:
            Cached response or None if not found
        """
        # Create a deterministic hash from the key dictionary
        hash_key = self._hash_dict(key)
        
        if hash_key in self.cache:
            entry = self.cache[hash_key]
            
            # Check if the entry has expired
            now = datetime.now()
            if now - entry["timestamp"] < timedelta(seconds=self.ttl_seconds):
                logger.info(f"Cache hit for key: {hash_key[:8]}...")
                return entry["response"]
            else:
                # Expired entry
                logger.info(f"Cache entry expired for key: {hash_key[:8]}...")
                del self.cache[hash_key]
                
        logger.info(f"Cache miss for key: {hash_key[:8]}...")
        return None
        
    def set(self, key: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Set a cache entry
        
        Args:
            key: Dictionary containing the request parameters
            response: Response to cache
        """
        hash_key = self._hash_dict(key)
        
        self.cache[hash_key] = {
            "response": response,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Cached response for key: {hash_key[:8]}...")
        
    def invalidate(self, key: Dict[str, Any]) -> bool:
        """
        Invalidate a cache entry
        
        Args:
            key: Dictionary containing the request parameters
            
        Returns:
            True if an entry was invalidated, False otherwise
        """
        hash_key = self._hash_dict(key)
        
        if hash_key in self.cache:
            del self.cache[hash_key]
            logger.info(f"Invalidated cache entry for key: {hash_key[:8]}...")
            return True
            
        return False
        
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache = {}
        logger.info("Cache cleared")
        
    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """Create a deterministic hash from a dictionary"""
        # Sort the dictionary to ensure deterministic hashing
        serialized = json.dumps(d, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()