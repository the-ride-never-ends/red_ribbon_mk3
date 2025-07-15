import re


import torch
from torch import nn, Tensor
from torch.functional import F
import numpy as np
from typing import Any, Optional

class MLP(nn.Module):
    """
    A simple multi-layer perceptron module in a transformer block.
    Consists of: Linear → GELU → Linear → Dropout
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.number_of_embeddings, 4 * config.number_of_embeddings)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.number_of_embeddings, config.number_of_embeddings)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLPExpansionNode:
    """
    Node that applies the first linear expansion in an MLP.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - expansion_factor: Factor by which to expand the embedding dimension
    
    Outputs:
      - expanded: Expanded tensor after linear projection
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "expansion_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("expanded",)
    FUNCTION = "expand"
    CATEGORY = "transformer/mlp"
    
    def __init__(self):
        self.fc = None  # Will be initialized at runtime
    
    def expand(self, x: Tensor, expansion_factor: float):
        B, T, C = x.size()
        
        # Initialize projection if needed
        if self.fc is None or self.fc.in_features != C:
            self.fc = nn.Linear(C, int(C * expansion_factor))
        elif self.fc.out_features != int(C * expansion_factor):
            self.fc = nn.Linear(C, int(C * expansion_factor))
            
        # Apply expansion
        expanded = self.fc(x)
        
        return (expanded,)


class ActivationFunctionNode:
    """
    Node that applies a non-linear activation function.
    
    Inputs:
      - x: Input tensor
      - activation_type: Type of activation to apply
    
    Outputs:
      - activated: Tensor after activation function
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "activation_type": (["gelu", "relu", "silu", "swish"], {"default": "gelu"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("activated",)
    FUNCTION = "activate"
    CATEGORY = "transformer/mlp"
    
    def __init__(self):
        self.activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU()  # Swish is the same as SiLU
        }
    
    def activate(self, x, activation_type):
        # Apply selected activation function
        activated = self.activations[activation_type](x)
        
        return (activated,)


class MLPContractionNode:
    """
    Node that applies the second linear projection in an MLP 
    to project back to the original dimension.
    
    Inputs:
      - x: Input expanded tensor
      - target_dim: Target dimension to project to
    
    Outputs:
      - contracted: Tensor after projection to original dimension
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "target_dim": ("INT", {"default": 512, "min": 16, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("contracted",)
    FUNCTION = "contract"
    CATEGORY = "transformer/mlp"
    
    def __init__(self):
        self.fc = None  # Will be initialized at runtime
    
    def contract(self, x, target_dim):
        B, T, C = x.size()
        
        # Initialize projection if needed
        if self.fc is None or self.fc.out_features != target_dim or self.fc.in_features != C:
            self.fc = nn.Linear(C, target_dim)
            
        # Apply contraction
        contracted = self.fc(x)
        
        return (contracted,)


class DropoutNode:
    """
    Node that applies dropout to a tensor.
    
    Inputs:
      - x: Input tensor
      - dropout_rate: Dropout probability
    
    Outputs:
      - output: Tensor after dropout
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_dropout"
    CATEGORY = "transformer/mlp"
    
    def __init__(self):
        self.dropout = nn.Dropout(0.1)  # Default, will be updated
    
    def apply_dropout(self, x: Tensor, dropout_rate: float) -> tuple[Tensor]:
        # Update dropout if needed
        if self.dropout.p != dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Apply dropout
        output = self.dropout(x)
        
        return (output,)


class MLPNode:
    """
    Combined node that implements the full MLP module.
    Internally uses the modular components but provides a simplified interface.
    
    Inputs:
      - x: Input tensor of shape (B, T, C)
      - expansion_factor: Factor by which to expand the embedding dimension
      - activation_type: Type of activation to apply
      - dropout_rate: Dropout probability
    
    Outputs:
      - output: Final MLP output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "expansion_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "activation_type": (["gelu", "relu", "silu", "swish"], {"default": "gelu"}),
                "dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("mlp_output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/mlp"
    
    def __init__(self):
        self.expansion_node = MLPExpansionNode()
        self.activation_node = ActivationFunctionNode()
        self.contraction_node = MLPContractionNode()
        self.dropout_node = DropoutNode()
    
    def forward(self, x: Tensor, expansion_factor: float, activation_type: str, dropout_rate: float) -> tuple[Tensor]:
        # Get input dimensions to use for contraction target
        B, T, C = x.shape
        
        # Apply each component in sequence
        expanded, = self.expansion_node.expand(x, expansion_factor)
        activated, = self.activation_node.activate(expanded, activation_type)
        contracted, = self.contraction_node.contract(activated, C)
        output, = self.dropout_node.apply_dropout(contracted, dropout_rate)
        
        return (output,)




