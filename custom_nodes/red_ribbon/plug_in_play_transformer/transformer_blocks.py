
from pathlib import Path

import torch
from torch import nn
import torch.utils.checkpoint

from .layer_normalization import LayerNormNode
from .attention import CausalSelfAttentionNode
from .add_residual import ResidualAddNode
from .mlp import MLPNode

class TransformerBlockNode:
    """
    Combined node that implements a full transformer block.
    Internally uses all the modular components but provides a simplified interface.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - number_of_attention_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension
      - block_size: Maximum sequence length
      - attention_dropout_rate: Attention dropout rate
      - residual_dropout_rate: Residual dropout rate
      - mlp_expansion_factor: MLP expansion factor
      - activation_type: MLP activation type
    
    Outputs:
      - output: Final transformer block output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "number_of_attention_heads": ("INT", {"default": 8, "min": 1, "max": 100}),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "block_size": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "attention_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
                "residual_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
                "mlp_expansion_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "activation_type": (["gelu", "relu", "silu", "swish"], {"default": "gelu"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("block_output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/block"
    
    def __init__(self):
        # Initialize all sub-components
        self.ln1 = LayerNormNode()
        self.attention = CausalSelfAttentionNode()
        self.res_add1 = ResidualAddNode()
        
        self.ln2 = LayerNormNode()
        self.mlp = MLPNode()
        self.res_add2 = ResidualAddNode()
    
    def forward(self, 
                x, 
                number_of_attention_heads, 
                number_of_embeddings, 
                block_size, 
                attention_dropout_rate, 
                residual_dropout_rate, 
                mlp_expansion_factor, 
                activation_type
            ):
        # Get shape
        B, T, C = x.shape
        
        # First sub-layer: Attention
        ln1_out, = self.ln1.normalize(x, C, 1e-5)
        attn_out, = self.attention.forward(ln1_out, 
                                        number_of_attention_heads, 
                                        number_of_embeddings, 
                                        block_size, 
                                        attention_dropout_rate, 
                                        residual_dropout_rate
                                        )
        res1_out, = self.res_add1.add_residual(x, attn_out)
        
        # Second sub-layer: MLP
        ln2_out, = self.ln2.normalize(res1_out, C, 1e-5)
        mlp_out, = self.mlp.forward(ln2_out, mlp_expansion_factor, 
                                   activation_type, residual_dropout_rate)
        res2_out, = self.res_add2.add_residual(res1_out, mlp_out)
        
        return (res2_out,)


class StackedTransformerBlocksNode:
    """
    Node that applies multiple identical transformer blocks in sequence.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - n_layers: Number of identical transformer blocks to apply
      - number_of_attention_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension
      - block_size: Maximum sequence length
      - attention_dropout_rate: Attention dropout rate
      - residual_dropout_rate: Residual dropout rate
      - mlp_expansion_factor: MLP expansion factor
      - activation_type: MLP activation type
    
    Outputs:
      - output: Final output after all transformer blocks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "n_layers": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "number_of_attention_heads": ("INT", {"default": 8, "min": 1, "max": 100}),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "block_size": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "attention_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
                "residual_dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
                "mlp_expansion_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "activation_type": (["gelu", "relu", "silu", "swish"], {"default": "gelu"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/stack"
    
    def __init__(self):
        self.transformer_block = TransformerBlockNode()
    
    def forward(self, x, n_layers, number_of_attention_heads, number_of_embeddings, block_size, attention_dropout_rate, 
                residual_dropout_rate, mlp_expansion_factor, activation_type):
        # Apply transformer blocks sequentially
        for idx in range(n_layers):
            x, = self.transformer_block.forward(
                x, 
                number_of_attention_heads, 
                number_of_embeddings, 
                block_size, 
                attention_dropout_rate, 
                residual_dropout_rate, 
                mlp_expansion_factor, 
                activation_type
            )
            
        return (x,)
    


class ConfigBasedTransformerNode:
    """
    Node that builds and applies a transformer based on a configuration.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - config_json: JSON string specifying the transformer architecture
    
    Outputs:
      - output: Transformer output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "config_json": ("STRING", {
                    "multiline": True, 
                    "default": """{
                        "n_layers": 12,
                        "number_of_attention_heads": 8,
                        "number_of_embeddings": 512,
                        "block_size": 1024,
                        "attn_pdrop": 0.1,
                        "resid_pdrop": 0.1,
                        "mlp_expansion": 4.0,
                        "activation": "gelu"
                    }"""
                }),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/config"
    
    def __init__(self):
        self.stacked_transformer = None
    
    def forward(self, x, config_json: Path):
        import json
        
        # Parse configuration
        config: dict = json.loads(config_json.read_text())
        
        # Initialize transformer
        if self.stacked_transformer is None:
            self.stacked_transformer = StackedTransformerBlocksNode()
        
        # Apply transformer blocks
        output, = self.stacked_transformer.forward(
            x, 
            config.get("n_layers", 12),
            config.get("number_of_attention_heads", 8),
            config.get("number_of_embeddings", 512),
            config.get("block_size", 1024),
            config.get("attn_pdrop", 0.1),
            config.get("resid_pdrop", 0.1),
            config.get("mlp_expansion", 4.0),
            config.get("activation", "gelu")
        )
        
        return (output,)
    

class CheckpointedTransformerNode:
    """
    Node that applies multiple transformer blocks with gradient checkpointing
    for memory efficiency.
    
    Inputs:
      - x: Input tensor of shape (B, T, C) 
      - n_layers: Number of transformer blocks
      - number_of_attention_heads: Number of attention heads
      - number_of_embeddings: Embedding dimension
      - use_checkpointing: Whether to use gradient checkpointing (saves memory)
    
    Outputs:
      - output: Transformer output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "n_layers": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "number_of_attention_heads": ("INT", {"default": 8, "min": 1, "max": 100}),
                "number_of_embeddings": ("INT", {"default": 512, "min": 16, "max": 8192}),
                "use_checkpointing": ("BOOLEAN", {"default": True}),
                # ... other parameters # TODO Fix this
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/optimized"
    
    def __init__(self):
        self.blocks = None
    
    def forward(self, x, n_layers, number_of_attention_heads, number_of_embeddings, use_checkpointing, **kwargs):
        # Initialize blocks if needed
        if self.blocks is None or len(self.blocks) != n_layers:
            self.blocks = nn.ModuleList([
                TransformerBlockNode() for _ in range(n_layers)
            ])
        
        # Apply blocks with checkpointing if enabled
        for idx, block in enumerate(self.blocks):
            if use_checkpointing and idx < n_layers - 1:  # Don't checkpoint last layer
                x = torch.utils.checkpoint.checkpoint(
                    block.forward, x, number_of_attention_heads, number_of_embeddings, **kwargs
                )
            else:
                x, = block.forward(x, number_of_attention_heads, number_of_embeddings, **kwargs)
        
        return (x,)
    

class TransformerConfigLoaderNode:
    """
    Node that loads transformer configurations from external files
    and builds the corresponding architecture.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - config_path: Path to configuration file
    
    Outputs:
      - output: Transformer output
    """
    _MODEL_CONFIGS = {
        "gpt2-small": {"n_layers": 12, "number_of_attention_heads": 12, "number_of_embeddings": 768},
        "openai-gpt": {"n_layers": 12, "number_of_attention_heads": 12, "number_of_embeddings": 768},  # 117M params
        "gpt2-medium": {"n_layers": 24, "number_of_attention_heads": 16, "number_of_embeddings": 1024},
        "gpt2-large": {"n_layers": 36, "number_of_attention_heads": 20, "number_of_embeddings": 1280},
        "gpt2-xl": {"n_layers": 48, "number_of_attention_heads": 25, "number_of_embeddings": 1600},  # 1558M params
        "gpt-mini": {"n_layers": 6, "number_of_attention_heads": 6, "number_of_embeddings": 192},
        "gopher-44m": {"n_layers": 8, "number_of_attention_heads": 16, "number_of_embeddings": 512},
        "gpt-micro": {"n_layers": 4, "number_of_attention_heads": 4, "number_of_embeddings": 128},
        "gpt-nano": {"n_layers": 3, "number_of_attention_heads": 3, "number_of_embeddings": 48},
        "custom": {},  # Placeholder for custom configurations
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "config_name": ([key for key in cls._MODEL_CONFIGS.keys()], {"default": "gpt2-small"}),
                "custom_config_path": ("STRING", {"default": "", "required": False}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/config"

    def __init__(self):
        self.configs = self._MODEL_CONFIGS
        self.stacked_transformer = StackedTransformerBlocksNode()
    
    def forward(self, x, config_name, custom_config_path=""):
        # Load configuration
        if config_name == "custom" and custom_config_path:
            import json
            with open(custom_config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self.configs.get(config_name, self.configs["gpt2-small"])
        
        args = (
            x, 
            config.get("n_layers", 12),
            config.get("number_of_attention_heads", 12),
            config.get("number_of_embeddings", 768),
            config.get("block_size", 1024),  # Default block size
            config.get("attn_pdrop", 0.1),   # Default attention dropout
            config.get("resid_pdrop", 0.1),  # Default residual dropout
            config.get("mlp_expansion", 4.0),# Default MLP expansion factor
            config.get("activation", "gelu") # Default activation type
        )

        # Apply transformer
        output, = self.stacked_transformer.forward(*args)

        return (output,)