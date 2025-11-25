"""
LLM integration module for American Law database with OpenAI API integration.
Provides RAG components and embeddings functionality.
"""
from ._async_interface import AsyncLLMInterface
from ._embeddings_manager import EmbeddingsManager
from .dependencies.async_openai_client import AsyncOpenAIClient
from .factory import make_llm, LLM

__all__ = [
    "LLM",
    "AsyncLLMInterface",
    "EmbeddingsManager",
    "AsyncOpenAIClient",
    "make_llm",
]
