"""
Response formatter for PromptDecisionTree
"""
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Formats responses from the decision tree
    """
    
    def __init__(self):
        logger.info("ResponseFormatter initialized")
    
    def format(self, node_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format a response based on the current node and context
        
        Args:
            node_data: Current node data
            context: Current context variables
            
        Returns:
            Formatted response as a dictionary
        """
        logger.info(f"Formatting response for node {node_data.get('id', 'unknown')}")
        
        # Dummy implementation
        response_template = node_data.get("response_template", "")
        formatted_response = response_template
        
        # Simple variable replacement in template
        if context and response_template:
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                formatted_response = formatted_response.replace(placeholder, str(value))
        
        return {
            "text": formatted_response,
            "node_id": node_data.get("id"),
            "is_terminal": node_data.get("is_terminal", False),
            "metadata": {
                "response_type": node_data.get("response_type", "text"),
                "suggested_actions": node_data.get("suggested_actions", [])
            }
        }