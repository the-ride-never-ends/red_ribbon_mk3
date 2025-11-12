
from typing import Any, Callable, Optional, Type, Union


from ._llm_client import OpenAiClient
from ._async_interface import AsyncLLMInterface
from ._embeddings_manager import EmbeddingsManager
from ._interface import LLMInterface


from ..configs import Configs
from ..logger import logger
from custom_nodes.red_ribbon._custom_errors import InitializationError, ResourceError, ConfigurationError


from openai import OpenAIError, AsyncOpenAI, OpenAI


LLM = Union[LLMInterface, AsyncLLMInterface]


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
        return EmbeddingsManager(configs=configs, resources=resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize EmbeddingsManager: {e}") from e


def _make_openai_client(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> OpenAiClient:
    _validate_configs(configs)
    try:
        return OpenAiClient(configs=configs, resources=resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize OpenAiClient: {e}") from e


def _make_async_openai_client(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> OpenAiClient:
    """Factory function to create OpenAiClient instance"""
    _validate_configs(configs)
    try:
        _resources = {
            "client": resources.get("client", AsyncOpenAI(api_key=configs.OPENAI_API_KEY.get_secret_value())),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize OpenAiClient resources: {e}") from e

    try:
        return OpenAiClient(configs=configs, resources=resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize OpenAiClient: {e}") from e


def make_llm(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> LLM:
    """Factory function to create LLMInterface instance"""
    _validate_configs(configs)
    try:
        llm_resources = {
            "logger": resources.get("logger" , logger),
            "embeddings_manager": resources.get("embeddings_manager", _make_embeddings_manager()),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize LLM resources for LLMInterface: {e}") from e

    try:
        return LLMInterface(resources=llm_resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize LLMInterface: {e}") from e


def make_async_llm(configs: Configs = Configs(), resources: dict[str, Any] = {}) -> LLM:
    _validate_configs(configs)

    try:
        llm_resources = {
            "logger": resources.get("logger" , logger),
            "embeddings_manager": resources.get("embeddings_manager", _make_embeddings_manager()),
            "llm": resources.get("llm", _make_async_openai_client()),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize LLM resources for AsyncLLMInterface: {e}") from e

    try:
        return AsyncLLMInterface(resources=llm_resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize AsyncLLMInterface: {e}") from e


