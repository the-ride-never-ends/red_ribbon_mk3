from torch import nn, Tensor


class LayerNormNode:
    """
    Node that applies layer normalization.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - normalized_shape: Size of normalized shape (usually embedding dimension)
      - eps: Small constant for numerical stability
    
    Outputs:
      - normalized: Normalized tensor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "normalized_shape": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "eps": ("FLOAT", {"default": 1e-5, "min": 1e-10, "max": 1e-2, "step": 1e-6}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("normalized",)
    FUNCTION = "normalize"
    CATEGORY = "transformer/norm"
    
    def __init__(self):
        self.ln = None  # Will be initialized at runtime
    
    def normalize(self, x: Tensor, normalized_shape: int, eps: float) -> tuple[Tensor]:
        # Get input shape
        B, T, C = x.shape
        
        # Initialize layer norm if needed
        if self.ln is None or self.ln.normalized_shape[0] != normalized_shape:
            self.ln = nn.LayerNorm(normalized_shape, eps=eps)
        elif self.ln.eps != eps:
            self.ln = nn.LayerNorm(normalized_shape, eps=eps)
        
        # Apply normalization
        normalized = self.ln(x)
        
        return (normalized,)