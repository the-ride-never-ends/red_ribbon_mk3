from torch import Tensor


def add_residual(x: Tensor, residual: Tensor):
    # Simple addition operation
    output = x + residual
    return output


class ResidualAddNode:
    """
    Node that adds a residual connection.
    
    Inputs:
      - x: Original input tensor
      - residual: Tensor to be added as residual
    
    Outputs:
      - output: Result after adding residual
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "residual": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "add_residual"
    CATEGORY = "transformer/residual"
    
    def add_residual(self, x: Tensor, residual: Tensor):
        # Simple addition operation
        output = x + residual
        
        return (output,)