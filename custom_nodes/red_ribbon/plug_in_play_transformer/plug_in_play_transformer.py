"""
Plug-in-Play Transformer - Build and train LLMs without knowing how to code!
"""

class TransformerAPI:
    """API for accessing Plug-in-Play Transformer functionality from ComfyUI"""
    
    def __init__(self, resources=None, configs=None):
        self.configs = configs
        self.resources = resources

# Main function that can be called when using this as a script
def main():
    print("Plug-in-Play Transformer module loaded successfully")
    print("Available tools:")
    print("- PiPTransformerNode: Node for ComfyUI integration")
    print("- TransformerAPI: API for programmatic access")

if __name__ == "__main__":
    main()


import math
import random


import torch
from torch import Tensor, nn
from torch.nn import Dropout

from easy_nodes.easy_nodes import register_type, AnythingVerifier

# Piece-wise Attention
from .attention import (
    OutputProjectionNode, 
    ApplyAttentionNode, 
    QKVProjectionNode, 
    CalculateCausalAttentionMatrixNode
)
# Piece-wise MLP
from .mlp import (
    ActivationFunctionNode,
    MLP as OriginalMLP,
    MLPExpansionNode,
    MLPContractionNode,
    DropoutNode,
    MLPNode,
    CustomMLPFunctionNode,
)

from .layer_normalization import LayerNormNode

from .architecture_explorer import (
    PositionalEncodingExplorerNode,
    ModelArchitectureExplorerNode,
)

from easy_nodes.easy_nodes import AnythingVerifier
from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
    register_type,
)
from torch import nn, Tensor


import logging
# from custom_nodes.socialtoolkit.logger.logger import Logger
logger = logging.getLogger(__name__)


register_these_classes = [
    Tensor,
    Dropout
]

# Register the custom types with custom JS bindings.
for this_class in register_these_classes:
    with_its_class_name_in_all_caps: str = this_class.__qualname__.upper()
    register_type(this_class, with_its_class_name_in_all_caps, verifier=AnythingVerifier())

# @ComfyNode("Plug-in-Play-Transformer", 
#            color="#d30e0e", 
#            bg_color="#ff0000",
#            display_name="Apply Attention",
#            return_names=["attended_tensor"])
# def apply_attention(
#     attention: Tensor, 
#     v: Tensor, 
#     number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16), 
#     ) -> Tensor:
#     """
#     Node that applies attention scores to values.
    
#     Inputs:
#       - att: Attention matrix of shape (B, nh, T, T)
#       - v: Value tensor of shape (B, nh, T, hs)
#       - number_of_embeddings: Total embedding dimension
    
#     Outputs:
#       - y: Output tensor after attention is applied (B, T, C)
#     """
#     # Apply attention to values
#     y = attention @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    
#     # Reshape back to original dimensions
#     B, nh, T, hs = y.shape
#     y = y.transpose(1, 2).contiguous().view(B, T, number_of_embeddings)  # (B, T, C)

#     return y


# @ComfyNode("Plug-in-Play-Transformer", 
#            color="#d30e0e", 
#            bg_color="#ff0000",
#            display_name="QKV Projection",
#            return_names=["q","k","v"])
# def qkv_projection(
#     x: Tensor, 
#     number_of_attention_heads: int = NumberInput(default=8, min=1, max=1128, step=1), 
#     number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16), 
#     ) -> tuple[Tensor, Tensor, Tensor]:
#     """
#     Node that projects input embeddings into query, key, and value vectors for multi-head attention.

#     Inputs:
#       - x: Input tensor of shape (B, T, C)
#       - number_of_attention_heads: Number of attention heads
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
#     k = k.view(B, T, number_of_attention_heads, C // number_of_attention_heads).transpose(1, 2)  # (B, nh, T, hs)
#     q = q.view(B, T, number_of_attention_heads, C // number_of_attention_heads).transpose(1, 2)  # (B, nh, T, hs)
#     v = v.view(B, T, number_of_attention_heads, C // number_of_attention_heads).transpose(1, 2)  # (B, nh, T, hs)
    
#     return q, k, v


# @ComfyNode("Plug-in-Play-Transformer", 
#            color="#d30e0e", 
#            bg_color="#ff0000",
#            display_name="Calculate Causal Attention Matrix",
#            return_names=["attention"])
# def calculate_attention(
#     q: Tensor, 
#     k: Tensor, 
#     block_size: int = NumberInput(default=1024, min=1, max=8192), 
#     dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
#     ) -> Tensor:

#     bias = None  # Causal mask buffer
#     attn_dropout = nn.Dropout(0.1)  # Default, will be updated

#     # Update dropout if needed
#     if attn_dropout.p != dropout_rate:
#         attn_dropout = nn.Dropout(dropout_rate)
    
#     # Create or retrieve causal mask
#     if bias is None or bias.size(-1) < block_size:
#         bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
    
#     # Get current sequence length
#     T = q.size(2)

#     # Calculate attention scores
#     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#     att = att.masked_fill(bias[:, :, :T, :T] == 0, float('-inf'))
#     att = nn.functional.softmax(att, dim=-1)
#     att = attn_dropout(att)
    
#     return att


# @ComfyNode("Plug-in-Play-Transformer", 
#            color="#d30e0e", 
#            bg_color="#ff0000",
#            display_name="Project Output",
#            return_names=["output"])
# def project_output(
#     attended_tensor: Tensor, # y
#     number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16),
#     dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
#     ) -> Tensor:
#     """
#     Applies the final projection and dropout to attention output.
    
#     Inputs:
#       - y: Attention output tensor of shape (B, T, C)
#       - number_of_embeddings: Embedding dimension
#       - dropout_rate: Residual dropout rate
    
#     Outputs:
#       - output: Final attention output
#     """

#     c_proj: nn.Linear = None  # Output projection layer
#     resid_dropout = nn.Dropout(0.1)  # Default, will be updated
    
#     # Initialize or update projection layer if needed
#     if c_proj is None or c_proj.out_features != number_of_embeddings:
#         c_proj = nn.Linear(number_of_embeddings, number_of_embeddings)
    
#     # Update dropout if needed
#     if resid_dropout.p != dropout_rate:
#         resid_dropout = nn.Dropout(dropout_rate)
    
#     # Apply output projection and dropout
#     output = resid_dropout(c_proj(attended_tensor))
    
#     return output


@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="Causal Self Attention",
           return_names=["output"])
def forward(
        x: Tensor, 
        number_of_attention_heads: int = NumberInput(default=8, min=1, max=1128, step=1),
        number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16),
        block_size: int = NumberInput(default=1024, min=1, max=8192),
        attention_dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05),
        resid_dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
    ) -> Tensor:

    qkv_node = QKVProjectionNode()
    att_matrix_node = CalculateCausalAttentionMatrixNode()
    apply_att_node = ApplyAttentionNode()
    output_node = OutputProjectionNode()

    # Apply each component in sequence
    q, k, v = qkv_node.project(x, number_of_attention_heads, number_of_embeddings)
    att, = att_matrix_node.calculate_attention(q, k, block_size, attention_dropout_rate)
    y, = apply_att_node.apply_attention(att, v, number_of_embeddings)
    output, = output_node.project_output(y, number_of_embeddings, resid_dropout_rate)
    
    return output



def my_is_changed_func():
    return random()

# Wrapper functions
##################################
#### ATTENTION CLASS WRAPPERS ####
##################################


@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="Project Output",
           return_names=["output"])
def project_output(
    attended_tensor: Tensor, # y
    number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16),
    dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
    ) -> Tensor:
    """
    Node that applies final projection and dropout to attention output.
    
    Inputs:
      - y: Attention output tensor of shape (B, T, C)
      - number_of_embeddings: Embedding dimension
      - dropout_rate: Residual dropout rate
    
    Outputs:
      - output: Final attention output
    """
    node = OutputProjectionNode()
    output: tuple[Tensor] = node.project_output(attended_tensor, number_of_embeddings, dropout_rate)
    return output[0]


@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="Calculate Causal Attention Matrix",
           return_names=["attention"])
def calculate_attention(
    q: Tensor, 
    k: Tensor, 
    block_size: int = NumberInput(default=1024, min=1, max=8192), 
    dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
    ) -> Tensor:

    node = CalculateCausalAttentionMatrixNode()
    attention = node.calculate_attention(q, k, block_size, dropout_rate)
    return attention[0]



@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="QKV Projection",
           return_names=["q","k","v"])
def qkv_projection(
    x: Tensor, 
    number_of_attention_heads: int = NumberInput(default=8, min=1, max=1128, step=1), 
    number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16), 
    ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Node that projects input embeddings into query, key, and value vectors for multi-head attention.

    Inputs:
      - x: Input tensor of shape (B, T, C)
      - number_of_attention_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension
    
    Outputs:
      - q: Query tensor of shape (B, nh, T, hs)
      - k: Key tensor of shape (B, nh, T, hs)
      - v: Value tensor of shape (B, nh, T, hs)
    """
    node = ApplyAttentionNode()
    q, k, v = node.apply_attention(x, number_of_attention_heads, number_of_embeddings)
    return q, k, v


@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="Apply Attention",
           return_names=["attended_tensor"])
def apply_attention(
    attention: Tensor, 
    v: Tensor, 
    number_of_embeddings: int = NumberInput(default=512, min=16, max=8192, step=16), 
    ) -> Tensor:
    """
    Node that applies attention scores to values.
    
    Inputs:
      - att: Attention matrix of shape (B, nh, T, T)
      - v: Value tensor of shape (B, nh, T, hs)
      - number_of_embeddings: Total embedding dimension
    
    Outputs:
      - y: Output tensor after attention is applied (B, T, C)
    """

    node = ApplyAttentionNode()
    attended_tensor = node.apply_attention(attention, v, number_of_embeddings)
    return attended_tensor[0]


@ComfyNode("Plug-in-Play-Transformer", 
           color="#d30e0e", 
           bg_color="#ff0000",
           display_name="Layer Normalization",
           return_names=["normalized_tensor"])
def layer_norm(
    x: Tensor,
    normalized_shape: int = NumberInput(default=512, min=16, max=8192, step=16),
    eps: float = NumberInput(default=1e-5, min=1e-10, max=1e-2, step=1e-6),
    ) -> Tensor:
    """
    Node that applies layer normalization.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - normalized_shape: Size of normalized shape (usually embedding dimension)
      - eps: Small constant for numerical stability
    
    Outputs:
      - normalized: Normalized tensor
    """
    node = LayerNormNode()
    output = node.normalize(x, normalized_shape, eps)
    return output[0]


@ComfyNode("Plug-in-Play-Transformer", 
        color="#d30e0e", 
        bg_color="#ff0000",
        display_name="MLP",
        return_names=["mlp_output"])
def mlp_node(
    x: Tensor,
    expansion_factor: float = NumberInput(default=4.0, min=1.0, max=8.0, step=0.5),
    activation_function: str = Choice(["gelu", "relu", "silu", "swish"]),
    dropout_rate: float = NumberInput(default=0.1, min=0.0, max=0.9, step=0.05)
    ) -> Tensor:
    """
    Node that applies MLP transformation.

    Inputs:
    - x: Input tensor of shape (B, T, C)
    - expansion_factor: Factor by which to expand the embedding dimension
    - activation_function: Type of activation to apply
    - dropout_rate: Dropout probability

    Outputs:
    - mlp_output: Final MLP output
    """
    node = MLPNode()
    output = node.forward(x, expansion_factor, activation_function, dropout_rate)
    return output[0]


@ComfyNode("Plug-in-Play-Transformer", 
        color="#d30e0e", 
        bg_color="#ff0000",
        display_name="Generate Architecture",
        return_names=["model_config"])
def generate_architecture(
                        embedding_dim: int, #= NumberInput(default=512, min=16, max=4096, step=1),
                        num_heads: int = NumberInput(default=8, min=1, max=128, step=1), 
                        num_layers: int = NumberInput(default=12, min=1, max=1000000, step=1),
                        attention_type: str = Choice(["standard", "flash", "linear", "local", "sparse"]),
                        mlp_type:  str = Choice(["standard", "gated", "swiglu", "geglu"]),
                        normalization_type: str = Choice(["layernorm", "rmsnorm", "scalednorm"]),
                        positional_encoding: str = Choice(["sinusoidal", "learned", "rotary", "alibi"])
                        ) -> str:
    """Node that allows exploring different transformer model architectures.
    
    Inputs:
      - embedding_dim: Dimension of the embeddings
      - num_heads: Number of attention heads
      - num_layers: Number of transformer layers
      - attention_type: Type of attention mechanism
      - mlp_type: Type of MLP
      - normalization_type: Type of normalization
      - positional_encoding: Type of positional encoding
    
    Outputs:
      - model_config: Complete model configuration
    """
    explorer = ModelArchitectureExplorerNode()
    arch = explorer.generate_architecture(embedding_dim, num_heads, num_layers, attention_type, mlp_type, normalization_type, positional_encoding)
    return arch[0]
