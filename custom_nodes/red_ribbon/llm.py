from logging import Logger
from typing import Any, Callable
import importlib.util
import os
import sys
from typing import Callable
import logging


import dependency_injector
import dependency_injector.resources


from .configs import Configs
from .utils.llm.caching_service import CachingService
from .utils.llm.model_provider import ModelProvider
from .utils.llm.prompt_generator import PromptGenerator
from .utils.llm.response_parser import ResponseParser
from .utils.llm.token_counter import TokenCounter
from .logger import get_logger


# Augmentoolkit LLM tools
from .utils.llm.generation_step import GenerationStep
from .utils.llm.enginer_wrapper import EngineWrapper
from .utils.llm.pipeline_step import PipelineStep


dependency_injector.resources.Resource

class Llm:

    _JSON_INSTRUCTIONS = "Return your answer in JSON format"

    def __init__(self, resources: dict[str, Any], configs: Configs):
        self.resources = resources
        self.configs = configs

        self.logger = get_logger(self.__class__.__name__)

        self.llm_model = resources["llm_model"]
        self.llm_tokenizer = resources["llm_tokenizer"]
        self.llm_vector_maker = resources["llm_vector_maker"]
        self.llm_vector_storage = resources["llm_vector_storage"]

        self.client = resources["client"]
        self.usage_tracker = resources["usage_tracker"]

        self.logger.info("Llm initialized")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    @classmethod
    def enter(cls, *args, **kwargs) -> 'Llm':
        instance = cls(*args, **kwargs)
        return instance

    def execute(self, func: str, *args, **kwargs) -> 'Llm':
        """
        """
        match func:
            case "caching_service":
                return self.caching_service(*args, **kwargs)
            case "model_provider":
                return self.model_provider(*args, **kwargs)
            case "prompt_generator":
                return self.prompt_generator(*args, **kwargs)
            case "response_parser":
                return self.response_parser(*args, **kwargs)
            case "token_counter":
                return self.token_counter(*args, **kwargs)
            case _:
                self.logger.error(f"Unknown function {func}")
                raise ValueError(f"Unknown function {func}")


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
            self.logger.debug(
                "JSON instructions not found in system message. Adding..."
            )
            system_message = f"{system_message} {self._JSON_INSTRUCTIONS}."
            self.logger.debug("New system message:\n%s", system_message)
        return system_message