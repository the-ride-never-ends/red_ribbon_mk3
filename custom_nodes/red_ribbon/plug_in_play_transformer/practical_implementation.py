import gc
import traceback
from io import BytesIO
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .transformer_blocks import StackedTransformerBlocksNode


class ImageTensor(torch.Tensor):
    pass


# Use non-interactive backend to prevent memory leaks
matplotlib.use('Agg')


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

# cmaps = {}

# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))


# def plot_color_gradients(category, cmap_list):
#     # Create figure and adjust figure height to number of colormaps
#     nrows = len(cmap_list)
#     figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
#     fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
#     fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
#                         left=0.2, right=0.99)
#     axs[0].set_title(f'{category} colormaps', fontsize=14)

#     for ax, name in zip(axs, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
#         ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
#                 transform=ax.transAxes)

#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axs:
#         ax.set_axis_off()

#     # Save colormap list for later.
#     cmaps[category] = cmap_list



class ChooseColorMap:
    """
    Node that selects a color map for visualizations.
    
    Inputs:
      - color_map: Name of the color map to use
    
    Outputs:
      - cmap: Color map function
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": ([], {"default": "diverging"}),
                "color_map": (["viridis", "plasma", "inferno", "magma", "cividis"], {"default": "viridis"}),
            }
        }
    
    RETURN_TYPES = ("COLOR_MAP",)
    RETURN_NAMES = ("cmap",)
    FUNCTION = "get_color_map"
    CATEGORY = "transformer/visualization"
    
    def get_color_map(self, color_map):
        import matplotlib.pyplot as plt
        return (plt.get_cmap(color_map),)




import torch
from torch import Tensor
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import gc
from typing import Tuple, Literal
from scipy import stats

# Use non-interactive backend to prevent memory leaks
matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

class VisualizeTensorNode:
    """
    Node that visualizes tensors as images.
    
    Inputs:
      - x: Input tensor to visualize
      - cmap: Colormap for visualization
      - save_as: Output format (jpeg, png, tiff, npy)
      - comparison_method: Method for reducing multi-dimensional tensors (average, max, min)
    
    Outputs:
      - visualization: Tensor visualization as an image
    """

    # Available colormaps
    COLOR_MAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                   'Blues', 'Reds', 'Greens', 'hot', 'cool', 'gray', 
                   'spring', 'summer', 'autumn', 'winter', 'bone',
                   'copper', 'YlOrRd', 'YlGnBu', 'RdBu', 'coolwarm']

    COMPARISON_METHODS = [
        # Basic statistics
        "average", "max", "min", "median", "sum",
        # Advanced statistics
        "std", "var", "range", "iqr", 
        "percentile_25", "percentile_50", "percentile_75", "percentile_95",
        # Norms
        "l1_norm", "l2_norm", "inf_norm", "frobenius",
        # Other statistical measures
        "rms", "energy", "entropy", "skewness", "kurtosis",
        # Means
        "geometric_mean", "harmonic_mean", "trimmed_mean_10",
        # Selection methods
        "first", "last", "middle", "random",
        # Absolute value methods
        "max_abs", "min_abs", "mean_abs"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),
                "cmap": (cls.COLOR_MAPS, {"default": "viridis"}),
                "save_as": (["jpeg", "png", "tiff", "npy"], {"default": "jpeg"}),
                "comparison_method": (cls.COMPARISON_METHODS, {"default": "average"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "transformer/visualization"

    def __init__(self):
        pass

    @property
    def _zero_tensor(self) -> Tensor:
        """Returns a zero tensor for empty visualizations."""
        # Create a small black image using PIL and convert to tensor
        img = Image.new('RGB', (100, 100), color='black')
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        return img_tensor.permute(2, 0, 1).unsqueeze(0)

    def visualize(self, 
                  x: Tensor, 
                  cmap: str, 
                  save_as: str, 
                  comparison_method: str
                  ) -> Tuple[ImageTensor]:
        """
        Visualize a tensor as an image.
        
        Args:
            x: Input tensor to visualize
            cmap: Colormap to use for visualization
            save_as: Output format for the visualization
            comparison_method: Method for reducing multi-dimensional tensors
            
        Returns:
            Tuple containing the visualization as an image tensor
        """
        output_image = self._zero_tensor  # Default to zero tensor if visualization fails
        buf = None

        try:
            # Handle different tensor shapes and reduce to 2D for visualization
            tensor_2d = self._prepare_tensor_for_visualization(x, comparison_method)

            match save_as:
                case "tiff":
                    print("Saving as TIFF format")
                    output_image = self._save_as_tiff(tensor_2d)
                case "npy":
                    print("Saving as raw numpy format")
                    output_image = self._save_as_npy(x)  # Use original tensor for raw
                case _:  # jpeg or png
                    print(f"Saving as {save_as.upper()} format")
                    output_image = self._save_as_image(tensor_2d, save_as, cmap)
                
        except Exception as e:
            print(f"Error during tensor visualization: {e}")
            # Return zero tensor on error
            output_image = self._zero_tensor
        finally:
            # Clean up resources
            if buf is not None and hasattr(buf, 'close'):
                buf.close()
            plt.close('all')
            gc.collect()

        print(f"Visualization complete: {output_image.shape} {output_image.dtype}")
        return (output_image,)  # type: ignore[return-value]

    def _prepare_tensor_for_visualization(self, x: Tensor, comparison_method: str) -> np.ndarray:
        """
        Prepare tensor for 2D visualization by reducing dimensions if necessary.
        
        Args:
            x: Input tensor
            comparison_method: Method for reduction
            
        Returns:
            2D numpy array ready for visualization
        """
        # Convert to numpy and remove batch dimension if present
        tensor_np = x.detach().cpu().numpy()
        
        # Handle different tensor shapes
        if len(tensor_np.shape) >= 4:  # [batch, channels, height, width] or more
            tensor_np = tensor_np[0]  # Take first batch
            
        if len(tensor_np.shape) > 2:
            # Reduce to 2D based on comparison method
            tensor_np = self._apply_reduction(tensor_np, comparison_method)
                    
        return tensor_np
    
    def _apply_reduction(self, tensor_np: np.ndarray, method: str) -> np.ndarray:
        """Apply the specified reduction method to reduce tensor dimensions."""
        # Keep reducing until we have 2D
        while len(tensor_np.shape) > 2:
            axis = 0  # Reduce along first dimension
            
            match method:
                # Basic statistics
                case "average":
                    tensor_np = np.mean(tensor_np, axis=axis)
                case "max":
                    tensor_np = np.max(tensor_np, axis=axis)
                case "min":
                    tensor_np = np.min(tensor_np, axis=axis)
                case "median":
                    tensor_np = np.median(tensor_np, axis=axis)
                case "sum":
                    tensor_np = np.sum(tensor_np, axis=axis)
                    
                # Advanced statistics
                case "std":
                    tensor_np = np.std(tensor_np, axis=axis)
                case "var":
                    tensor_np = np.var(tensor_np, axis=axis)
                case "range":
                    tensor_np = np.ptp(tensor_np, axis=axis)  # peak-to-peak (max - min)
                case "iqr":
                    q75 = np.percentile(tensor_np, 75, axis=axis)  # type: ignore[assignment]
                    q25 = np.percentile(tensor_np, 25, axis=axis)  # type: ignore[assignment]
                    tensor_np = q75 - q25
                    
                # Percentiles
                case method if method.startswith("percentile_"):
                    percentile = int(method.split("_")[1])
                    tensor_np = np.percentile(tensor_np, percentile, axis=axis)
                    
                # Norms
                case "l1_norm":
                    tensor_np = np.sum(np.abs(tensor_np), axis=axis)
                case "l2_norm":
                    tensor_np = np.sqrt(np.sum(tensor_np**2, axis=axis))
                case "inf_norm":
                    tensor_np = np.max(np.abs(tensor_np), axis=axis)
                case "frobenius":
                    tensor_np = np.sqrt(np.sum(tensor_np**2, axis=axis))
                    
                # Other measures
                case "rms":
                    tensor_np = np.sqrt(np.mean(tensor_np**2, axis=axis))
                case "energy":
                    tensor_np = np.sum(tensor_np**2, axis=axis)
                case "entropy":
                    # Normalize to probabilities
                    tensor_flat = tensor_np.reshape(tensor_np.shape[0], -1)
                    tensor_flat = tensor_flat - tensor_flat.min(axis=1, keepdims=True)
                    tensor_flat = tensor_flat / (tensor_flat.sum(axis=1, keepdims=True) + 1e-10)
                    entropy = -np.sum(tensor_flat * np.log(tensor_flat + 1e-10), axis=1)
                    tensor_np = entropy.reshape(tensor_np.shape[1:])
                case "skewness":
                    tensor_np = stats.skew(tensor_np, axis=axis)
                case "kurtosis":
                    tensor_np = stats.kurtosis(tensor_np, axis=axis)
                    
                # Means
                case "geometric_mean":
                    # Handle negative values by using absolute values
                    tensor_np = stats.gmean(np.abs(tensor_np) + 1e-10, axis=axis)  # type: ignore[assignment]
                case "harmonic_mean":
                    # Avoid division by zero
                    tensor_np = stats.hmean(np.abs(tensor_np) + 1e-10, axis=axis)  # type: ignore[assignment]
                case "trimmed_mean_10":
                    tensor_np = stats.trim_mean(tensor_np, 0.1, axis=axis)  # type: ignore[assignment]
                    
                # Selection methods
                case "first":
                    tensor_np = tensor_np[0]
                case "last":
                    tensor_np = tensor_np[-1]
                case "middle":
                    mid_idx = tensor_np.shape[axis] // 2
                    tensor_np = np.take(tensor_np, mid_idx, axis=axis)
                case "random":
                    idx = np.random.randint(0, tensor_np.shape[axis])
                    tensor_np = np.take(tensor_np, idx, axis=axis)
                    
                # Absolute value methods
                case "max_abs":
                    tensor_np = np.max(np.abs(tensor_np), axis=axis)
                case "min_abs":
                    tensor_np = np.min(np.abs(tensor_np), axis=axis)
                case "mean_abs":
                    tensor_np = np.mean(np.abs(tensor_np), axis=axis)
                    
                case _:
                    # Default to average if method not recognized
                    print(f"Unknown comparison method '{method}', using average")
                    tensor_np = np.mean(tensor_np, axis=axis)
                
        return tensor_np

    def _save_as_tiff(self, tensor_np: np.ndarray) -> Tensor:
        """Save tensor as TIFF format."""
        try:
            import rasterio
            
            # Normalize to 0-1 range for TIFF
            tensor_min = tensor_np.min()
            tensor_max = tensor_np.max()
            if tensor_max > tensor_min:
                tensor_np = (tensor_np - tensor_min) / (tensor_max - tensor_min)
            else:
                tensor_np = np.zeros_like(tensor_np)
            
            # Create in-memory TIFF
            buf = BytesIO()
            
            # Configure TIFF metadata
            height, width = tensor_np.shape
            
            with rasterio.open(
                buf, 'w', driver='GTiff',
                height=height, width=width,
                count=1, dtype='float32'
            ) as dst:
                dst.write(tensor_np.astype('float32'), 1)
                
            buf.seek(0)
            
            # Read back and convert to tensor
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Convert to RGB tensor format [batch, channels, height, width]
            img_tensor = torch.from_numpy(img_array).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            # Ensure 4D tensor with 3 channels
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            else:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
                
            buf.close()
            return img_tensor
            
        except ImportError:
            print("rasterio not installed. Using PIL for TIFF instead.")
            # Fall back to regular image saving with TIFF format
            return self._save_as_image(tensor_np, 'tiff', 'gray')
        except Exception as e:
            print(f"Error creating TIFF visualization: {e}")
            return self._zero_tensor

    def _save_as_npy(self, x: Tensor) -> Tensor:
        """Save raw tensor data and return a placeholder visualization."""
        try:
            # Save raw tensor to file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                tensor_np = x.detach().cpu().numpy()
                np.save(f.name, tensor_np)
                print(f"Raw tensor saved to: {f.name}")
                print(f"Shape: {tensor_np.shape}, dtype: {tensor_np.dtype}")

            # Create a simple visualization showing tensor info
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 
                    f"Raw Tensor\nShape: {x.shape}\nDtype: {x.dtype}\n"
                    f"Min: {x.min():.4f}\nMax: {x.max():.4f}",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            
            # Convert to tensor
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            img = Image.open(buf)
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            buf.close()
            plt.close(fig)
            return img_tensor
            
        except Exception as e:
            print(f"Error saving raw tensor: {e}")
            return self._zero_tensor

    def _save_as_image(self, tensor_np: np.ndarray, format: str, cmap: str) -> Tensor:
        """Save tensor as image format (JPEG/PNG/TIFF via PIL)."""
        try:
            print(f"Creating {format} visualization with colormap '{cmap}'")
            print(f"Input tensor shape: {tensor_np.shape}, dtype: {tensor_np.dtype}")
            print(f"Tensor min: {tensor_np.min():.4f}, max: {tensor_np.max():.4f}")
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            print("Created matplotlib figure")
            
            # Use the specified colormap
            im = ax.imshow(tensor_np, cmap=cmap, aspect='auto')
            print(f"Applied imshow with colormap: {cmap}")
            
            # Add colorbar and title
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Tensor Visualization\nShape: {tensor_np.shape}')
            print("Added colorbar and title")
            
            # Save to buffer in the requested format
            buf = BytesIO()
            plt.savefig(buf, format=format, dpi=200, bbox_inches='tight')
            buf.seek(0)
            print(f"Saved figure to buffer in {format} format")
            
            # Read back and convert to tensor
            img = Image.open(buf)
            img_array = np.array(img)
            print(f"Loaded image from buffer: shape {img_array.shape}, dtype {img_array.dtype}")
            
            # Convert to tensor format [batch, channels, height, width]
            if len(img_array.shape) == 3:
                print("Converting RGB image to tensor")
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
                print(f"RGB tensor shape: {img_tensor.shape}")
            else:
                # Handle grayscale
                print("Converting grayscale image to tensor")
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                img_tensor = img_tensor.unsqueeze(-1).unsqueeze(0)  # Add channel dimension
                print(f"Grayscale to RGB tensor shape: {img_tensor.shape}")
            
            print(f"Final tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
            
            buf.close()
            plt.close(fig)
            print("Cleaned up matplotlib figure and buffer")
            return img_tensor

        except Exception as e:
            print(f"Error creating {format} visualization: {e}")
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            return self._zero_tensor
