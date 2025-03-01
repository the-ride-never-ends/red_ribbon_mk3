"""
Tree loader for PromptDecisionTree
"""
import logging
import json
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TreeLoader:
    """
    Loads decision tree data from files or databases
    """
    
    def __init__(self):
        logger.info("TreeLoader initialized")
    
    def load_from_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a decision tree from a JSON file
        
        Args:
            filepath: Path to the JSON file containing tree definition
            
        Returns:
            Tree data dictionary or None if loading failed
        """
        logger.info(f"Loading tree from file: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"Tree file not found: {filepath}")
            return None
            
        try:
            with open(filepath, 'r') as f:
                tree_data = json.load(f)
                logger.info(f"Successfully loaded tree with {len(tree_data.get('nodes', {}))} nodes")
                return tree_data
        except Exception as e:
            logger.error(f"Error loading tree file: {str(e)}")
            return None
            
    def load_from_string(self, json_string: str) -> Optional[Dict[str, Any]]:
        """
        Load a decision tree from a JSON string
        
        Args:
            json_string: JSON string containing tree definition
            
        Returns:
            Tree data dictionary or None if loading failed
        """
        logger.info("Loading tree from JSON string")
        
        try:
            tree_data = json.loads(json_string)
            logger.info(f"Successfully loaded tree with {len(tree_data.get('nodes', {}))} nodes")
            return tree_data
        except Exception as e:
            logger.error(f"Error loading tree from JSON string: {str(e)}")
            return None