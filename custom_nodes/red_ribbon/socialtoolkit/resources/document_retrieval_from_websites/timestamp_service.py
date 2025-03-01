"""
Timestamp service for DocumentRetrievalFromWebsites
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TimestampService:
    """
    Service for generating consistent timestamps
    """
    
    def __init__(self):
        logger.info("TimestampService initialized")
    
    def now(self) -> str:
        """
        Get current timestamp in ISO format
        
        Returns:
            ISO formatted timestamp string
        """
        return datetime.now().isoformat()
        
    def formatted(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Get formatted timestamp
        
        Args:
            format_str: Format string for the timestamp
            
        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime(format_str)