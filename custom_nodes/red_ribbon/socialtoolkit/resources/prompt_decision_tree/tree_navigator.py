"""
Tree navigator for PromptDecisionTree
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TreeNavigator:
    """
    Navigates through the decision tree structure
    """
    
    def __init__(self):
        logger.info("TreeNavigator initialized")
        self.visited_nodes = []
    
    def get_node(self, tree_data: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific node from the tree by ID
        
        Args:
            tree_data: Complete tree data structure
            node_id: ID of the node to retrieve
            
        Returns:
            Node data dictionary or None if not found
        """
        logger.info(f"Getting node {node_id}")
        
        nodes = tree_data.get("nodes", {})
        node = nodes.get(node_id)
        
        if node:
            self.visited_nodes.append(node_id)
            return node
        
        logger.warning(f"Node {node_id} not found in tree")
        return None
        
    def get_path_history(self) -> list:
        """
        Get the history of visited nodes
        
        Returns:
            List of visited node IDs
        """
        return self.visited_nodes
        
    def reset(self) -> None:
        """Reset the navigation history"""
        self.visited_nodes = []