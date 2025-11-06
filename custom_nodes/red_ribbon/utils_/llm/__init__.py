"""
Resource implementations for LLM
"""


from ._llm import LLM
from .llm_factory import make_llm


__all__ = [
    "LLM",
    "make_llm",
]
