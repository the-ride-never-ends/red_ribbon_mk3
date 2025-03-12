import math


import torch
from torch.functional import F


class TensorTransformNode:
    """
    General-purpose node for applying custom transformations to tensors.
    
    Inputs:
      - input_tensor: The input tensor to transform
      - transform_fn: String representation of a custom transformation function
    
    Outputs:
      - output: Transformed tensor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "transform_fn": ("STRING", {
                    "multiline": True, 
                    "default": "def transform(x):\n    # Identity transform by default\n    return x"
                }),
            },
            "optional": {
                "aux_tensor": ("TENSOR", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "transform_tensor"
    CATEGORY = "transformer/custom"
    
    def transform_tensor(self, input_tensor, transform_fn, aux_tensor=None):
        # Setup execution environment
        local_env = {
            "x": input_tensor,
            "aux": aux_tensor,
            "torch": torch,
            "F": F,
            "math": math,
        }
        
        try:
            # Execute custom transform function
            exec(transform_fn, globals(), local_env)
            result = local_env["transform"](input_tensor)
            
            return (result,)
        except Exception as e:
            print(f"Error in custom transform function: {str(e)}")
            return (torch.zeros_like(input_tensor).fill_(-999),)

