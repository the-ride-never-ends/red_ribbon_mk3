from logging import Logger
from typing import Any


#


class LLMService:

    def __init__(self, resources: dict[str, Any], configs):
        self.resources = resources
        self.configs = configs
        self.logger: Logger = resources.get("logger")

        self.llm_model = resources.get("llm_model")
        self.llm_tokenizer = resources.get("llm_tokenizer")
        self.llm_vector_maker = resources.get("llm_vector_maker")
        self.llm_vector_storage = resources.get("llm_vector_storage")

        self.logger.info("LLMService initialized with services")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self, command_context: str, *args, **kwargs):
        """
        
        
        """
        pass


