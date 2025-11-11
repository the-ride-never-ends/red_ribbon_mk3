"""
Custom Types for Red Ribbon Nodes

These are register with ComfyUI to be used as input/output types for custom nodes.
"""
from dataclasses import dataclass
from typing import TypeVar, Optional, Union, Any, Generic


try:
    from networkx import DiGraph # type: ignore # NOTE We do this so that we can register the nx.DiGraph type in ComfyUI
except ImportError as e:
    raise ImportError(f"Critical local import 'networkx' not found. Please ensure all Red Ribbon files are present.\n{e}") from e


from .custom_easy_nodes import register_type
from .custom_easy_nodes.easy_nodes import AnythingVerifier
from .utils_.configs import Configs
from .utils_ import DatabaseAPI
from .utils_ import LLM


# Force the AnyType to always be equal in not equal comparisons.
# Source: https://github.com/rgthree/rgthree-comfy/blob/main/py/display_any.py#L15
class AnyType:
    """Special type that accepts anything and always compares as equal."""
    def __ne__(self, other: Any) -> bool:
        return False

# Type aliases for custom node types
Excel = Union[str, list[str], dict[str, str]]

Vectors = Union[list[float], list[list[float]]]
Metadata = Union[str, dict, DiGraph, tuple, list[str], list[dict], list[tuple], list[DiGraph]]
Prompts = Union[str, dict, DiGraph]

Urls = Union[str, list[str]]
Answers = Union[str, list[str]]

# Compound type aliases
Documents = Union[str, list[str]]
Data = Union[DatabaseAPI, Excel]
LlmApi = Union[str, LLM, dict]

@dataclass
class Laws:
    documents: Documents | None = None
    metadata: Metadata | None = None
    vectors: Vectors | None = None
    prompts: Prompts | None = None
    answers: Answers | None = None

# Register the types with ComfyUI
def register_custom_types():
    types = {
        "Database": DatabaseAPI, "LLM": LLM, "Configs": Configs, 
        "Prompts": Prompts, "DiGraph": DiGraph, "dict": dict,
        "Vectors": Vectors, "Documents": Documents, "Urls": Urls,
        "Metadata": Metadata, "AnyType": AnyType, "Answers": Answers,
        "Excel": Excel, "Data": Data, "Laws": Laws, "LlmApi": LlmApi,
    }
    for type_name, type_class in types.items():
        try:
            type_class.__qualname__ 
        except AttributeError:  # If the class doesn't have a __qualname__ attribute, monkeypatch one in.
            # This came up when testing TypeVar aliases.
            type_class.__qualname__ = type_name
            #print(f"Added __qualname__ to type {type_class.__name__}")

        register_type(type_class, type_name, verifier=AnythingVerifier())
