
from typing import Any


from ._llm_client import OpenAiClient
from ._embeddings_manager import EmbeddingsManager
from ._interface import LLMInterface


from ..configs import Configs
from ..logger import logger
from custom_nodes.red_ribbon._custom_errors import InitializationError, ResourceError, ConfigurationError


# Define a type alias for LLM
LLM = LLMInterface


def _validate_configs(configs: Configs) -> None:
    try:
        configs.model_validate()
    except Exception as e:
        raise ConfigurationError(f"Invalid configurations: {e}") from e


def _make_embeddings_manager(
        configs: Configs = Configs(), 
        resources: dict[str, Any] = {}
    ) -> EmbeddingsManager:
    _validate_configs(configs)
    try:
        _resources = {
            "logger": resources.get("logger" , logger),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize EmbeddingsManager resources: {e}") from e

    try:
        return EmbeddingsManager(configs=configs, resources=_resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize EmbeddingsManager: {e}") from e


def _make_openai_client(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> OpenAiClient:
    """Factory function to create OpenAiClient instance"""
    _validate_configs(configs)
    api_key = configs.OPENAI_API_KEY.get_secret_value()

    try:
        from openai import AsyncOpenAI, OpenAI
    except ImportError as e:
        raise ImportError("openai package is required to use OpenAiClient. Please install it via 'pip install openai'.") from e

    try:
        _resources = {
            "logger": resources.get("logger" , logger),
            "async_client": resources.get("async_client", AsyncOpenAI(api_key=api_key)),
            "client": resources.get("client", OpenAI(api_key=api_key)),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize OpenAiClient resources: {e}") from e

    try:
        return OpenAiClient(configs=configs, resources=_resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize OpenAiClient: {e}") from e


def make_llm(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> LLM:
    """Factory function to create LLMInterface instance"""
    _validate_configs(configs)
    try:
        llm_resources = {
            "logger": resources.get("logger" , logger),
            "embeddings_manager": resources.get("embeddings_manager", _make_embeddings_manager()),
            "openai_client": resources.get("openai_client", _make_openai_client()),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize LLM resources for LLMInterface: {e}") from e

    try:
        return LLMInterface(resources=llm_resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize LLMInterface: {e}") from e
