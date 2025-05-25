
import torch
from .transformer_blocks import StackedTransformerBlocksNode


class TransformerModuleRegistry:
    """Helper class to manage transformer modules"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TransformerModuleRegistry()
        return cls._instance
    
    def __init__(self):
        self.modules = {}
    
    def register_module(self, name, module):
        """Register a module by name"""
        self.modules[name] = module
        return module
    
    def get_module(self, name):
        """Get a module by name"""
        return self.modules.get(name)


class TransformerModelNode:
    """
    High-level transformer model node for practical use.
    
    Inputs:
      - x: Input tensor
      - model_size: Predefined configuration (small, medium, large)
      - num_layers: Optional override for number of layers
    
    Outputs:
      - output: Model output
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "model_size": (["tiny", "small", "medium", "large", "xl"], {"default": "small"}),
            },
            "optional": {
                "num_layers": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "custom_config": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "transformer/models"
    
    def __init__(self):
        self.configs = {
            "tiny": {"n_layers": 4, "n_head": 4, "n_embd": 256},
            "small": {"n_layers": 12, "n_head": 12, "n_embd": 768},
            "medium": {"n_layers": 24, "n_head": 16, "n_embd": 1024},
            "large": {"n_layers": 36, "n_head": 20, "n_embd": 1280},
            "xl": {"n_layers": 48, "n_head": 25, "n_embd": 1600},
        }
        self.stacked_blocks = StackedTransformerBlocksNode()
    
    def forward(self, x, model_size, num_layers=0, custom_config=""):
        # Determine configuration
        config = self.configs.get(model_size, self.configs["small"])
        
        # Override number of layers if specified
        if num_layers > 0:
            config["n_layers"] = num_layers
            
        # Override with custom config if provided
        if custom_config:
            import json
            try:
                custom = json.loads(custom_config)
                for k, v in custom.items():
                    config[k] = v
            except:
                print("Error parsing custom config, using default")
        
        # Apply transformer
        output, = self.stacked_blocks.forward(
            x,
            config.get("n_layers", 12),
            config.get("n_head", 12),
            config.get("n_embd", 768),
            config.get("block_size", 1024),
            config.get("attn_pdrop", 0.1),
            config.get("resid_pdrop", 0.1),
            config.get("mlp_expansion", 4.0),
            config.get("activation", "gelu")
        )
        
        return (output,)


class AttentionVisualizationNode: # TODO
    """
    Node that visualizes attention patterns from a specified layer.
    
    Inputs:
      - model: Transformer model reference
      - layer_index: Which layer to visualize (0-indexed)
      - head_index: Which attention head to visualize (-1 for all heads)
    
    Outputs:
      - visualization: Attention pattern visualization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "default_model"}),
                "layer_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "head_index": ("INT", {"default": -1, "min": -1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "transformer/visualization"
    
    def visualize(self, model_name, layer_index, head_index):
        # Get model from registry
        registry = TransformerModuleRegistry.get_instance()
        model = registry.get_module(model_name)
        
        if model is None:
            # Return empty visualization
            return (torch.zeros(1, 3, 100, 100),)
        
        # Extract attention patterns # TODO
        # ... implementation depends on model structure
        
        # Create visualization # TODO
        # ... convert attention patterns to image
        attention_image = None
        return (attention_image,)
    


class LayerOutputNode:
    """
    Node that extracts the output of a specific layer from a transformer model.
    
    Inputs:
      - model: Transformer model reference
      - layer_index: Which layer's output to extract
    
    Outputs:
      - layer_output: Output tensor from the specified layer
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "default_model"}),
                "layer_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("layer_output",)
    FUNCTION = "extract"
    CATEGORY = "transformer/introspection"
    
    def extract(self, model_name, layer_index):
        # Get model from registry
        registry = TransformerModuleRegistry.get_instance()
        model = registry.get_module(model_name)
        
        if model is None or not hasattr(model, "layer_outputs"):
            # Return empty tensor
            return (torch.zeros(1, 1, 1),)
        
        # Extract layer output
        if layer_index < len(model.layer_outputs):
            return (model.layer_outputs[layer_index],)
        else:
            return (torch.zeros(1, 1, 1),)