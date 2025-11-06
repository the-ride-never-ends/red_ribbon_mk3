import logging


from ._llm import LLM
from ._caching_service import CachingService
from ._model_provider import ModelProvider
from ._prompt_generator import PromptGenerator
from ._response_parser import ResponseParser
from ._token_counter import TokenCounter
from ._generation_step import GenerationStep
from ._engine_wrapper import EngineWrapper
from ._pipeline_step import PipelineStep


from .clients.openai import OpenAiApi
from ..configs._configs import configs


class LLMInitializationError(Exception):
    """Custom exception for LLM initialization errors."""

    def __init__(self, *args):
        super().__init__(*args)


def make_llm(client_name: str = "openai") -> LLM:
    """
    Factory function to create an LLM instance.
    
    Args:
        client_name: Name of the LLM service to use
    
    Returns:
        An instance of the LLM class
        
    Raises:
        TypeError: If client_name is not a string
        ValueError: If client_name is not supported
        LLMInitializationError: If LLM resources or LLM initialization fails
    """
    if not isinstance(client_name, str):
        raise TypeError("client_name must be a string, got {type(client_name).__name__}")

    # Choose the appropriate API based on the service name
    Client = None
    match client_name:
        case "openai":
            Client = OpenAiApi
        case _:
            raise ValueError(f"Unsupported client name: {client_name}")

    try:
        resources = {
            "llm_model": ModelProvider(),
            "llm_tokenizer": TokenCounter(),
            "llm_vector_maker": PromptGenerator(),
            "llm_vector_storage": ResponseParser(),
            "client": Client(configs),
            "usage_tracker": CachingService(),
            "logger": logging.getLogger(__name__)
        }
    except Exception as e:
        raise LLMInitializationError(f"Failed to create resources for LLM: {e}") from e

    try:
        return LLM(resources=resources, configs=configs)
    except Exception as e:
        raise LLMInitializationError(f"Failed to initialize LLM: {e}") from e