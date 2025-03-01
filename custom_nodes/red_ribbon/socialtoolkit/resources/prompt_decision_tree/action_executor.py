"""
Action executor for PromptDecisionTree
"""
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    Executes actions defined in decision tree nodes
    """
    
    def __init__(self):
        logger.info("ActionExecutor initialized")
        self.registered_actions = {}
    
    def register_action(self, action_name: str, action_function: Callable) -> None:
        """
        Register an action function
        
        Args:
            action_name: Name of the action
            action_function: Function to execute for this action
        """
        logger.info(f"Registering action: {action_name}")
        self.registered_actions[action_name] = action_function
    
    def execute(self, node: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute an action defined in a node
        
        Args:
            node: Node containing action definition
            context: Context data for action execution
            
        Returns:
            Result of the action or None if execution failed
        """
        action_name = node.get("action", "")
        
        if not action_name:
            logger.error("Action node missing action name")
            return None
            
        action_function = self.registered_actions.get(action_name)
        
        if not action_function:
            logger.error(f"Unknown action: {action_name}")
            return None
            
        try:
            logger.info(f"Executing action: {action_name}")
            result = action_function(context, node.get("action_params", {}))
            return result
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {str(e)}")
            return None