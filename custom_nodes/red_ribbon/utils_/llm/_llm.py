from logging import Logger
from typing import Any, Callable
import importlib.util
import os
import sys
from typing import Callable
import logging



from ..configs._configs import Configs
from ._caching_service import CachingService
from ._model_provider import ModelProvider
from ._prompt_generator import PromptGenerator
from ._response_parser import ResponseParser
from ._token_counter import TokenCounter
from ..logger._logger import make_logger


# Augmentoolkit LLM tools
from ._generation_step import GenerationStep
from ._engine_wrapper import EngineWrapper
from ._pipeline_step import PipelineStep


class LLMError(RuntimeError):
    """Custom exception for LLM errors."""

    def __init__(self, *args):
        super().__init__(*args)


class LLM:

    _JSON_INSTRUCTIONS = "Return your answer in JSON format"

    def __init__(self, 
                 resources: dict[str, Any] = None, 
                 configs: Configs = None
                 ) -> None:
        self.resources = resources
        self.configs = configs

        self._logger: logging.Logger = resources['logger']

        self.llm_model = resources["llm_model"]
        self.llm_tokenizer = resources["llm_tokenizer"]
        self.llm_vector_maker = resources["llm_vector_maker"]
        self.llm_vector_storage = resources["llm_vector_storage"]

        self.client = resources["client"]
        self.usage_tracker = resources["usage_tracker"]

        self._logger.info("LLM initialized")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    @classmethod
    def enter(cls, *args, **kwargs) -> 'LLM':
        instance = cls(*args, **kwargs)
        return instance

    def execute(self, func_name: str, *args, **kwargs) -> Any:
        """
        """
        func: Callable = None
        match func_name:
            case "caching_service":
                func = self.caching_service
            case "model_provider":
                func = self.model_provider
            case "prompt_generator":
                return self.prompt_generator
            case "response_parser":
                return self.response_parser
            case "token_counter":
                return self.token_counter
            case _:
                self._logger.error(f"Unknown function '{func_name}'")
                raise ValueError(f"Unknown function '{func_name}'")

        # Run the function with error handling
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._logger.error(f"Error executing {func_name}: {e}")
            raise RuntimeError(f"Failed to execute {func_name}") from e

    def caching_service(self, *args, **kwargs):
        """
        """
        pass

    def model_provider(self, *args, **kwargs):
        """
        """
        pass

    def prompt_generator(self, *args, **kwargs):
        """
        """
        pass

    def response_parser(self, *args, **kwargs):
        """
        """
        pass

    def token_counter(self, *args, **kwargs):
        """
        """
        pass

    def _add_json_instructions_if_needed(self, system_message):
        """Add JSON instruction to system message if needed."""
        if self._JSON_INSTRUCTIONS.casefold() not in system_message.casefold():
            self._logger.debug(
                "JSON instructions not found in system message. Adding..."
            )
            system_message = f"{system_message} {self._JSON_INSTRUCTIONS}."
            self._logger.debug("New system message:\n%s", system_message)
        return system_message