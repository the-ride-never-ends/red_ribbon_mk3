import re
from typing import Any, Optional


import numpy as np
import torch
from torch import nn, Tensor
from torch.functional import F


class CustomFunctionNode:
    """
    Node that applies a custom mathematical function to a tensor input.
    
    Inputs:
      - x: Input tensor (B, T, C)
      - custom_func: String representation of a custom function
      - number_of_embeddings: Embedding dimension (for reference)
    
    Outputs:
      - output: Result after applying custom MLP
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "custom_func": ("STRING", {
                    "multiline": True, 
                    "default": '''def mlp(x, number_of_embeddings: int):\n    # Default MLP implementation\n    h = F.gelu(torch.nn.Linear(number_of_embeddings, 4 * number_of_embeddings)(x))\n    return torch.nn.Linear(4 * number_of_embeddings, number_of_embeddings)(h)'''
                }),
                "kwargs": ("STRING", {
                    "multiline": True, 
                    "default": "{'number_of_embeddings': 512}"
                }),
            },
            "optional": {
                "residual": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_custom_function"
    CATEGORY = "transformer/custom"
    
    def __init__(self):
        self.fc1 = None
        self.fc2 = None

    @staticmethod
    def _check_for_potentially_dangerous_operations_in(string: str) -> None:
        """Check the custom function string for potentially dangerous operations."""
        dangerous_patterns = [
            r'\beval\b', r'\bexec\b', r'\b__import__\b', r'\bopen\b', 
            r'\bfile\b', r'\bos\b', r'\bsys\b', r'\bsubprocess\b'
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, string):
                raise ValueError(f"custom_func string contains potentially dangerous operation: {pattern}")

    @staticmethod
    def _check_function_signature_for_(arg: str, custom_func: str) -> None:
        """Check if the custom function accepts the required argument."""
        func_signature = re.search(r'def\s+\w+\s*\((.*?)\)', custom_func)
        if func_signature is None:
            raise ValueError("Could not find function signature in custom_func string.")
        if arg not in func_signature.group(1):
            raise ValueError(f"custom_func must accept '{arg}' as an argument.")

    @staticmethod
    def _get_function_name_from(custom_func: str) -> str:
        """Find the functions's name in the provided string."""
        match = re.search(r'def\s+(\w+)\s*\(', custom_func)
        if match is None:
            raise ValueError("Cannot find function name in custom_func string.")
        return match.group(1)

    @staticmethod
    def _turn_kwargs_str_into_dict(kwargs: str) -> dict:
        """Attempt to parse the kwargs string as a dictionary."""
        try:
            kwargs_dict = eval(kwargs)
            match kwargs_dict:
                case dict():
                    return kwargs_dict
                case _:
                    raise ValueError(f"kwargs must be a dict, got {type(kwargs_dict).__name__}.")
        except Exception as e:
            raise ValueError(f"Error parsing kwargs_dict: {e}") from e

    @staticmethod
    def _validate_tensor(x: Any) -> None:
        """Validate that the input tensor is 3D."""
        match x:
            case Tensor():
                if x.dim() != 3:
                    raise ValueError(f"Input tensor x must be 3D (B, T, C), got {x.dim()}D.")
            case _:
                raise ValueError(f"Input x must be a Tensor, got {type(x).__name___}.")

    def apply_custom_function(self, 
                              x: Tensor, 
                              residual: Optional[Tensor],
                              custom_func: str, 
                              kwargs: str | dict = "",
                              save: bool = False
                            ) -> tuple[Tensor]:
        kwargs_dict = {}

        self._check_for_potentially_dangerous_operations_in(custom_func)

        # NOTE Doing this also allows us to check if the function exists.
        function_name = self._get_function_name_from(custom_func)

        self._check_function_signature_for_('x', custom_func)

        match kwargs:
            case str():
                if kwargs.strip() != "":
                    self._check_for_potentially_dangerous_operations_in(kwargs)
                    kwargs_dict = self._turn_kwargs_str_into_dict(kwargs)
            case dict():
                kwargs_dict = kwargs if kwargs else {}
            case _:
                raise ValueError(f"kwargs must be a dict or a string, got {type(kwargs).__name__}.")

        # Validate input tensor shape
        self._validate_tensor(x)

        if residual is not None:
            self._validate_tensor(residual)
            kwargs_dict["residual"] = residual

        # Compile the custom function
        try:
            compiled_mlp_fn = compile(custom_func, "<string>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid syntax in custom_func: {e}")

        # Setup execution environment
        local_env = {
            "x": x,
            # Common PyTorch modules
            "torch": torch,
            "Tensor": Tensor,
            "F": F,
            "nn": nn,
            # Numpy
            "np": np,
        }
        local_env.update(kwargs_dict)

        try:
            # Execute custom function
            exec(compiled_mlp_fn, globals(), local_env)
            tup = (x, kwargs_dict.values()) if kwargs_dict else (x,)
            result = local_env[function_name](*tup)
            if not isinstance(result, Tensor):
                raise ValueError(f"Custom function must return a Tensor, got {type(result).__name__}.")

            return (result,)
        except Exception as e:
            print(f"Error executing custom function: {e}")
            raise RuntimeError(f"Error in custom function: {e}") from e
