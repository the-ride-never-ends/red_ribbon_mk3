"""
Node evaluator for PromptDecisionTree
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NodeEvaluator:
    """
    Evaluates decision tree nodes against user input
    """
    
    def __init__(self):
        logger.info("NodeEvaluator initialized")
    
    def evaluate(self, node_data: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        Evaluate a node condition against user input to determine next node
        
        Args:
            node_data: Current node data including conditions
            user_input: User input text to evaluate against
            
        Returns:
            Dictionary with evaluation results including next node ID
        """
        logger.info(f"Evaluating node {node_data.get('id', 'unknown')} against user input")
        
        # Dummy implementation
        # In a real implementation, this might use natural language understanding
        # or pattern matching to determine which path to take
        
        conditions = node_data.get("conditions", [])
        default_next = node_data.get("default_next")
        
        # Simple keyword matching for the dummy implementation
        for condition in conditions:
            if "keywords" in condition:
                keywords = condition.get("keywords", [])
                for keyword in keywords:
                    if keyword.lower() in user_input.lower():
                        logger.info(f"Matched condition for keyword '{keyword}', proceeding to node {condition.get('next_node')}")
                        return {
                            "matched": True,
                            "matched_condition": condition,
                            "next_node": condition.get("next_node"),
                            "confidence": 0.8
                        }
        
        # Default path if no conditions matched
        logger.info(f"No conditions matched, proceeding to default node {default_next}")
        return {
            "matched": False,
            "next_node": default_next,
            "confidence": 0.5
        }