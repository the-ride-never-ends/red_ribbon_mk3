"""
Cache manager for Top10DocumentRetrieval
"""
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching of search results
    """
    
    def __init__(self, cache_ttl_seconds: int = 3600):
        """
        Initialize cache manager
        
        Args:
            cache_ttl_seconds: Time-to-live for cache entries in seconds
        """
        logger.info(f"CacheManager initialized with TTL: {cache_ttl_seconds} seconds")
        self.cache: dict[str, Any] = {}
        self.cache_timestamps: dict[str, float] = {}
        self.cache_ttl_seconds = cache_ttl_seconds
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Check if key exists in cache
        if key not in self.cache:
            return None
            
        # Check if cache entry has expired
        timestamp = self.cache_timestamps.get(key, 0)
        current_time = time.time()
        
        if current_time - timestamp > self.cache_ttl_seconds:
            # Remove expired entry
            del self.cache[key]
            del self.cache_timestamps[key]
            return None
            
        logger.info(f"Cache hit for key: {key}")
        return self.cache[key]
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            None
        """
        logger.info(f"Caching results for key: {key}")
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()
        
    def clear(self) -> None:
        """
        Clear all cache entries
        
        Returns:
            None
        """
        logger.info("Clearing cache")
        self.cache = {}
        self.cache_timestamps = {}
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "keys": list(self.cache.keys()),
            "ttl_seconds": self.cache_ttl_seconds
        }