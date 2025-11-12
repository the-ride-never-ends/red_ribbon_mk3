import math


import torch
from torch import nn, Tensor
import torch.nn.functional as F

# from custom_nodes.red_ribbon.logger import make_logger
# logger = make_logger(__name__)
import logging
logger = logging.getLogger(__name__)


class QKVProjectionNode:
    """
    Node that projects input embeddings into query, key, and value vectors.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - number_of_attn_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension
    
    Outputs:
      - q: Query tensor of shape (B, nh, T, hs)
      - k: Key tensor of shape (B, nh, T, hs)
      - v: Value tensor of shape (B, nh, T, hs)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "number_of_attn_heads": ("INT", {"default": 8, "min": 1, "max": 100}),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("q", "k", "v")
    FUNCTION = "project"
    CATEGORY = "transformer/attention"
    
    def __init__(self):
        self.c_attn: nn.Linear | None = None  # Will be initialized at runtime
    
    def project(self, x: Tensor, number_of_attn_heads: int, number_of_embeddings: int
                ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, C = x.size()
        
        # Initialize projection if needed
        if self.c_attn is None or self.c_attn.in_features != C:
            self.c_attn = nn.Linear(C, 3 * C)
            
        # Split into query, key, value
        qkv: Tensor = self.c_attn(x)
        q, k, v = qkv.split(number_of_embeddings, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        return (q, k, v,)


# # TODO Make a config file for all the constants.
# THIS_FILE = os.path.realpath(__file__)
# THIS_DIR = os.path.dirname(THIS_FILE)
# CUSTOM_NODES_DIR = os.path.dirname(THIS_DIR)
# COMFYUI_DIR = os.path.dirname(CUSTOM_NODES_DIR)
# LLM_OUTPUTS_DIR = os.path.join(COMFYUI_DIR, "llm_outputs")


# def qkv_projection(
#     x: Tensor, 
#     number_of_attn_heads: int = NumberInput(default=8, min=1, max=1128, step=1), 
#     number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16), 
#     ) -> tuple[Tensor, Tensor, Tensor]:
#     """
#     Node that projects input embeddings into query, key, and value vectors for multi-head attention.

#     Inputs:
#       - x: Input tensor of shape (B, T, C)
#       - number_of_attn_heads: Number of attention heads
#       - number_of_embeddings: Embedding dimension
    
#     Outputs:
#       - q: Query tensor of shape (B, nh, T, hs)
#       - k: Key tensor of shape (B, nh, T, hs)
#       - v: Value tensor of shape (B, nh, T, hs)
#     """
#     c_attn: nn.Linear = None  # Will be initialized at runtime
#     B, T, C = x.size()
    
#     # Initialize projection if needed
#     if c_attn is None or c_attn.in_features != C:
#         c_attn = nn.Linear(C, 3 * C)

#     # Split into query, key, value
#     qkv: Tensor = c_attn(x)
#     q, k, v = qkv.split(number_of_embeddings, dim=2)

#     # Reshape for multi-head attention
#     k = k.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
#     q = q.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
#     v = v.view(B, T, number_of_attn_heads, C // number_of_attn_heads).transpose(1, 2)  # (B, nh, T, hs)
    
#     return q, k, v



class OutputProjectionNode:
    """
    Node that applies final projection and dropout to attention output.
    
    Inputs:
      - y: Attention output tensor of shape (B, T, C)
      - number_of_embeddings: Embedding dimension
      - dropout_rate: Residual dropout rate
    
    Outputs:
      - output: Final attention output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "y": ("TENSOR",),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("attention_output",)
    FUNCTION = "project_output"
    CATEGORY = "transformer/attention"
    
    def __init__(self):
        self.c_proj = None  # Output projection layer
        self.resid_dropout = nn.Dropout(0.1)  # Default, will be updated
    
    def project_output(self, y: Tensor, number_of_embeddings: int, dropout_rate: float) -> tuple[Tensor]:
        # Initialize or update projection layer if needed
        if self.c_proj is None or self.c_proj.out_features != number_of_embeddings:
            self.c_proj = nn.Linear(number_of_embeddings, number_of_embeddings)
        
        # Update dropout if needed
        if self.resid_dropout.p != dropout_rate:
            self.resid_dropout = nn.Dropout(dropout_rate)
        
        # Apply output projection and dropout
        output = self.resid_dropout(self.c_proj(y))
        
        return (output,)


class CalculateCausalAttentionMatrixNode:
    """
    Node that calculates the attention matrix with causal masking.
    
    Inputs:
      - q: Query tensor of shape (B, nh, T, hs)
      - k: Key tensor of shape (B, nh, T, hs)
      - block_size: Maximum sequence length
      - dropout_rate: Attention dropout rate
    
    Outputs:
      - attention: Attention matrix of shape (B, nh, T, T)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "q": ("TENSOR",),
                "k": ("TENSOR",),
                "block_size": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("attention_matrix",)
    FUNCTION = "calculate_attention"
    CATEGORY = "transformer/attention"
    
    def __init__(self):
        self.bias = None  # Causal mask buffer
        self.attn_dropout = nn.Dropout(0.1)  # Default, will be updated
    
    def calculate_attention(self, q: Tensor, k: Tensor, block_size: int, dropout_rate: float) -> tuple[Tensor]:
        # Update dropout if needed
        if self.attn_dropout.p != dropout_rate:
            self.attn_dropout = nn.Dropout(dropout_rate)
        
        # Create or retrieve causal mask
        if self.bias is None or self.bias.size(-1) < block_size:
            self.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        
        # Get current sequence length
        T = q.size(2)
        
        # Calculate attention scores
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)
        
        return (attention,)


class ApplyAttentionNode:
    """
    Node that applies attention scores to values.
    
    Inputs:
      - attention: Attention matrix of shape (B, nh, T, T)
      - v: Value tensor of shape (B, nh, T, hs)
      - number_of_embeddings: Total embedding dimension
    
    Outputs:
      - y: Output tensor after attention is applied (B, T, C)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention": ("TENSOR",),
                "v": ("TENSOR",),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("attended_output",)
    FUNCTION = "apply_attention"
    CATEGORY = "transformer/attention"
    
    def apply_attention(self, attention: Tensor, v: Tensor, number_of_embeddings: int) -> tuple[Tensor]:
        # Apply attention to values
        y = attention @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Reshape back to original dimensions
        B, nh, T, hs = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, number_of_embeddings)  # (B, T, C)
        
        return (y,)
    

class CustomAttentionFunctionNode:
    """
    Node that applies a custom mathematical function to the attention mechanism.
    
    Inputs:
      - q: Query tensor (B, nh, T, hs)
      - k: Key tensor (B, nh, T, hs)
      - v: Value tensor (B, nh, T, hs)
      - attention_fn: String representation of a custom function to compute attention scores
      - post_attention_fn: Optional function to apply after attention scores are computed
    
    Outputs:
      - output: Result after applying custom attention
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "q": ("TENSOR",),
                "k": ("TENSOR",),
                "v": ("TENSOR",),
                "attention_fn": ("STRING", {
                    "multiline": True, 
                    "default": "def attention(q, k, v):\n    # Default attention calculation\n    attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))\n    # Apply causal mask\n    mask = torch.tril(torch.ones(attention.size(-2), attention.size(-1))).to(attention.device)\n    attention = attention.masked_fill(mask == 0, float('-inf'))\n    attention = F.softmax(attention, dim=-1)\n    return attention @ v"
                }),
            },
            "optional": {
                "post_attention_fn": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_custom_attention"
    CATEGORY = "transformer/custom"
    
    def apply_custom_attention(self, q, k, v, attention_fn, post_attention_fn=""):
        # Setup execution environment with necessary imports
        local_env = {
            "q": q,
            "k": k,
            "v": v,
            "torch": torch,
            "F": F,
            "math": math,
        }
        
        try:
            # Execute custom attention function
            exec(attention_fn, globals(), local_env)
            result = local_env["attention"](q, k, v)
            
            # Apply post-processing if provided
            if post_attention_fn.strip():
                local_env["result"] = result
                exec(post_attention_fn, globals(), local_env)
                result = local_env["processed_result"]
            
            return (result,)
        except Exception as e:
            # Return error tensor
            print(f"Error in custom attention function: {str(e)}")
            # Return a copy of the input with an error flag
            return (torch.zeros_like(q[:, :, :, 0:1]).fill_(-999),)


class CausalSelfAttentionNode:
    """
    Combined node that implements full causal self-attention.
    Internally uses the modular components but provides a simplified interface.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - number_of_attn_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension 
      - block_size: Maximum sequence length
      - attn_dropout_rate: Attention dropout rate
      - residual_dropout_rate: Residual dropout rate
    
    Outputs:
      - output: Final attention output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "number_of_attn_heads": ("INT", {"default": 8, "min": 1, "max": 100}),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "block_size": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "attn_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
                "residual_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("attention_output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/attention"
    
    def __init__(self):
        self.qkv_node = QKVProjectionNode()
        self.att_matrix_node = CalculateCausalAttentionMatrixNode()
        self.apply_att_node = ApplyAttentionNode()
        self.output_node = OutputProjectionNode()
    
        # AKA Forward
    def calculate_attention(self, 
                            x, 
                            number_of_attn_heads, 
                            number_of_embeddings, 
                            block_size, 
                            attn_dropout_rate, 
                            residual_dropout_rate
                        ):
        # Apply each component in sequence
        q, k, v = self.qkv_node.project(x, number_of_attn_heads, number_of_embeddings)
        attention, = self.att_matrix_node.calculate_attention(q, k, block_size, attn_dropout_rate)
        y, = self.apply_att_node.apply_attention(attention, v, number_of_embeddings)
        output, = self.output_node.project_output(y, number_of_embeddings, residual_dropout_rate)

        return (output,)

    def forward(self, x, number_of_attn_heads, number_of_embeddings, block_size, attn_dropout_rate, residual_dropout_rate):
        # Alias for calculate_attention
        args = (x, number_of_attn_heads, number_of_embeddings, block_size, attn_dropout_rate, residual_dropout_rate)
        return self.calculate_attention(*args)

