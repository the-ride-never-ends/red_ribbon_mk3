"""
Prompt generator for Llm
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PromptGenerator:
    """
    Generates prompts for language models
    """
    
    def __init__(self):
        logger.info("PromptGenerator initialized")
        self.templates = {
            "default": "Answer the following question: {question}",
            "summarize": "Summarize the following text: {text}",
            "chat": "User: {user_message}\nAssistant:",
            "extraction": "Extract the following information from the text: {fields}\n\nText: {text}"
        }
    
    def generate(self, template_name: str, variables: Dict[str, Any]) -> str:
        """
        Generate a prompt using a template and variables
        
        Args:
            template_name: Name of the template to use
            variables: Variables to fill in the template
            
        Returns:
            Generated prompt
        """
        logger.info(f"Generating prompt with template: {template_name}")
        
        if template_name not in self.templates:
            logger.warning(f"Unknown template: {template_name}, falling back to default")
            template_name = "default"
            
        template = self.templates[template_name]
        
        # Apply variables to template
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template: {e}")
            prompt = f"Error in prompt generation: missing variable {e}"
            
        return prompt
        
    def add_template(self, name: str, template: str) -> None:
        """
        Add a new template
        
        Args:
            name: Name of the template
            template: Template string with placeholders
        """
        self.templates[name] = template
        logger.info(f"Added new template: {name}")
        
    def list_templates(self) -> List[str]:
        """
        List available templates
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())