"""
Context manager for PromptDecisionTree
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context variables during tree traversal
    """
    
    def __init__(self):
        logger.info("ContextManager initialized")
        self.context = {}
    
    def update(self, new_values: Dict[str, Any]) -> None:
        """
        Update context with new values
        
        Args:
            new_values: Dictionary of new context values to add
        """
        logger.info(f"Updating context with {len(new_values)} new values")
        self.context.update(new_values)
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a specific context value
        
        Args:
            key: Context variable key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        return self.context.get(key, default)
        
    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete context dictionary
        
        Returns:
            Complete context dictionary
        """
        return self.context.copy()
        
    def reset(self) -> None:
        """Reset the context to empty"""
        self.context = {}