import logging


from ._llm import LLM
from ._caching_service import CachingService
from ._model_provider import ModelProvider
from ._prompt_generator import PromptGenerator
from ._response_parser import ResponseParser
from ._token_counter import TokenCounter
from ._generation_step import GenerationStep
from ._enginer_wrapper import EngineWrapper
from ._pipeline_step import PipelineStep


from .clients.openai import OpenAiApi
from ...configs import configs


def make_llm(client_name: str = "openai") -> LLM:
    """
    Factory function to create an LLM instance.
    
    Args:
        client_name: Name of the LLM service to use
        resources: Resources required for the LLM
        configs: Configuration settings for the LLM
    
    Returns:
        An instance of the LLM class
    """
    # Choose the appropriate API based on the service name
    match client_name:
        case "openai":
            Client = OpenAiApi

    resources = {
        "llm_model": ModelProvider(),
        "llm_tokenizer": TokenCounter(),
        "llm_vector_maker": PromptGenerator(),
        "llm_vector_storage": ResponseParser(),
        "client": Client(configs),
        "usage_tracker": CachingService(),
        "logger": logging.getLogger(__name__)
    }

    return LLM(resources=resources, configs=configs)