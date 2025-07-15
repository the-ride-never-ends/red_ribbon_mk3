import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path


print(sys.path)

# Add the top-level directory to the path to import the custom node
COMFY_DIR = (Path.home() / "red_ribbon_mk3" ).resolve() 
print(COMFY_DIR)
sys.path.append(COMFY_DIR)

import os
# Add the project root to the Python path so imports work
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
print(proj_root)
sys.path.append(proj_root)

import time

time.sleep(1)  # Ensure the path is set before importing the node








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

import traceback

class ImageTensor(torch.Tensor):
    pass

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
                    output_image = self._save_as_tiff(tensor_2d)
                case "npy":
                    output_image = self._save_as_npy(x)  # Use original tensor for raw
                case _:  # jpeg or png
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
        return (output_image,)

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
                    print(f"[DEBUG] Applying average reduction on axis {axis}. Original shape: {tensor_np.shape}")
                    tensor_np = np.mean(tensor_np, axis=axis)
                    print(f"[DEBUG] Applying average reduction, new shape: {tensor_np.shape}")
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
                    q75 = np.percentile(tensor_np, 75, axis=axis)
                    q25 = np.percentile(tensor_np, 25, axis=axis)
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
                    tensor_np = stats.gmean(np.abs(tensor_np) + 1e-10, axis=axis)
                case "harmonic_mean":
                    # Avoid division by zero
                    tensor_np = stats.hmean(np.abs(tensor_np) + 1e-10, axis=axis)
                case "trimmed_mean_10":
                    tensor_np = stats.trim_mean(tensor_np, 0.1, axis=axis)
                    
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
            print(f"[DEBUG] _save_as_image called with format: {format}, cmap: {cmap}")
            print(f"[DEBUG] Input tensor_np shape: {tensor_np.shape}, dtype: {tensor_np.dtype}")
            print(f"[DEBUG] Tensor_np min: {tensor_np.min()}, max: {tensor_np.max()}")
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            print(f"[DEBUG] Created matplotlib figure")
            
            # Use the specified colormap
            im = ax.imshow(tensor_np, cmap=cmap, aspect='auto')
            print(f"[DEBUG] Applied imshow with colormap: {cmap}")
            
            # Add colorbar and title
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Tensor Visualization\nShape: {tensor_np.shape}')
            print(f"[DEBUG] Added colorbar and title")
            
            # Save to buffer in the requested format
            buf = BytesIO()
            plt.savefig(buf, format=format, dpi=100, bbox_inches='tight')
            buf.seek(0)
            print(f"[DEBUG] Saved matplotlib figure to buffer")
            
            # Read back and convert to tensor
            img = Image.open(buf)
            print(f"[DEBUG] Opened PIL image, mode: {img.mode}, size: {img.size}")
            
            img_array = np.array(img)
            print(f"[DEBUG] Converted PIL to numpy, shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # Convert to tensor format [batch, channels, height, width]
            if len(img_array.shape) == 3:
                print(f"[DEBUG] Processing 3D image array (RGB/RGBA)")
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                print(f"[DEBUG] After normalization, shape: {img_tensor.shape}, min: {img_tensor.min()}, max: {img_tensor.max()}")
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                print(f"[DEBUG] After permute and unsqueeze, final shape: {img_tensor.shape}")
            else:
                # Handle grayscale
                print(f"[DEBUG] Processing grayscale image array")
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                print(f"[DEBUG] After normalization, shape: {img_tensor.shape}")
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                print(f"[DEBUG] After grayscale to RGB conversion, final shape: {img_tensor.shape}")
            
            buf.close()
            plt.close(fig)
            print(f"[DEBUG] Cleaned up resources, returning tensor with shape: {img_tensor.shape}")
            return img_tensor

        except Exception as e:
            print(f"[ERROR] Error creating {format} visualization: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return self._zero_tensor


class TestVisualizeTensorNodeShapeIssue:
    """Test cases to reproduce and verify the shape issue in VisualizeTensorNode."""
    
    @pytest.fixture
    def mock_node(self):
        """Set up test fixtures."""
        # Import the actual node class
        
        return VisualizeTensorNode()
        
    @pytest.mark.parametrize("tensor,expected_channels", [
        (torch.randn(1, 665), 3),         # 2D tensor with 665 features -> RGB image
        (torch.randn(665), 3),            # 1D tensor with 665 elements -> RGB image
        (torch.randn(1, 1, 665), 3),      # 3D tensor that might be problematic -> RGB image
        (torch.randn(1, 665, 1), 3),      # Another 3D configuration -> RGB image
        (torch.randn(1, 3, 224, 224), 3), # Standard image tensor -> RGB image
    ])
    def test_tensor_visualization_output_shape(self, mock_node, tensor, expected_channels):
        """Test that visualize always returns proper image tensor shape."""
        # Call the visualize method
        result = mock_node.visualize(
            x=tensor,
            cmap='viridis',
            save_as='png',
            comparison_method='average'
        )
        
        # Extract the output tensor
        output_tensor = result[0]
        
        # Verify the output shape
        assert len(output_tensor.shape) == 4, f"Expected 4D tensor, got {output_tensor.shape}"
        assert output_tensor.shape[0] == 1, f"Expected batch size 1, got {output_tensor.shape[0]}"
        assert output_tensor.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {output_tensor.shape[1]}"
        assert output_tensor.shape[2] > 0, f"Expected non-zero height, got {output_tensor.shape[2]}"
        assert output_tensor.shape[3] > 0, f"Expected non-zero width, got {output_tensor.shape[3]}"
        
        # Verify the output is in valid range
        assert output_tensor.min() >= 0.0, f"Expected min >= 0, got {output_tensor.min()}"
        assert output_tensor.max() <= 1.0, f"Expected max <= 1, got {output_tensor.max()}"
        
    def test_problematic_shape_handling(self, mock_node):
        """Specifically test the (1, 1, 665) shape that caused the error."""
        # Create the problematic tensor
        tensor = torch.randn(1, 1, 665)
        
        # Call visualize
        result = mock_node.visualize(
            x=tensor,
            cmap='viridis',
            save_as='png',
            comparison_method='average'
        )
        
        output_tensor = result[0]
        
        # The output should be a valid image tensor
        assert output_tensor.shape[0] == 1  # Batch
        assert output_tensor.shape[1] == 3  # RGB channels
        assert output_tensor.shape[2] > 0  # Height
        assert output_tensor.shape[3] > 0  # Width
        
        # Verify it can be converted to PIL Image without error
        # First, convert to numpy and rearrange dimensions
        img_np = output_tensor[0].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img_np = (img_np * 255).astype(np.uint8)
        
        # This should not raise an error
        img = Image.fromarray(img_np)
        assert img.mode == 'RGB'
        
    def test_prepare_tensor_for_visualization(self, mock_node):
        """Test the tensor preparation method directly."""
        # Test various tensor shapes
        test_cases = [
            (torch.randn(1, 665), 'average'),           # 2D
            (torch.randn(1, 1, 665), 'average'),        # 3D 
            (torch.randn(1, 3, 224, 224), 'average'),   # 4D standard image
            (torch.randn(2, 10, 20, 30), 'max'),        # 4D batch
        ]
        
        for tensor, method in test_cases:
            result = mock_node._prepare_tensor_for_visualization(tensor, method)
            
            # Result should always be 2D numpy array
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 2, f"Expected 2D array, got shape {result.shape}"
            
    def test_edge_case_tensor_shapes(self, mock_node):
        """Test various edge case tensor shapes that might cause issues."""
        edge_cases = [
            torch.randn(665, 1, 1),      # Tensor with single spatial dimensions
            torch.randn(1, 1, 1, 665),   # 4D tensor with 665 in last dimension
            torch.randn(1, 3, 665, 1),   # 4D tensor with 665 in height
            torch.zeros(0, 665),         # Empty tensor with 665 features
            torch.randn(1),              # Single element tensor
            torch.randn(1, 1, 1, 1),     # All dimensions are 1
        ]
        
        for tensor in edge_cases:
            try:
                result = mock_node.visualize(
                    x=tensor,
                    cmap='viridis',
                    save_as='png',
                    comparison_method='average'
                )
                
                output = result[0]
                
                # Should always produce valid image tensor
                assert output.shape[0] == 1  # Batch
                assert output.shape[1] == 3  # RGB
                assert output.shape[2] > 0   # Height
                assert output.shape[3] > 0   # Width
                
            except Exception as e:
                pytest.fail(f"Failed on tensor shape {tensor.shape}: {str(e)}")
                
    def test_zero_tensor_property(self, mock_node):
        """Test the _zero_tensor property returns valid image."""
        zero_tensor = mock_node._zero_tensor
        
        assert zero_tensor.shape == (1, 3, 100, 100)
        assert zero_tensor.min() >= 0.0
        assert zero_tensor.max() <= 1.0
        
    def test_save_as_image_output_shape(self, mock_node):
        """Test that _save_as_image produces correct tensor shape."""
        # Create a simple 2D array
        tensor_2d = np.random.randn(50, 50)
        
        # Call _save_as_image
        result = mock_node._save_as_image(tensor_2d, 'png', 'viridis')
        
        # Check shape
        assert result.shape[0] == 1  # Batch
        assert result.shape[1] == 3  # RGB channels
        assert result.shape[2] > 0   # Height
        assert result.shape[3] > 0   # Width
        
    def test_comparison_methods(self, mock_node):
        """Test all comparison methods work correctly."""
        tensor = torch.randn(1, 5, 10, 20)  # 4D tensor
        
        for method in mock_node.COMPARISON_METHODS:
            try:
                result = mock_node.visualize(
                    x=tensor,
                    cmap='viridis', 
                    save_as='png',
                    comparison_method=method
                )
                
                output = result[0]
                assert output.shape[1] == 3  # Should always be RGB
                
            except Exception as e:
                pytest.fail(f"Method '{method}' failed: {str(e)}")
                
    def test_empty_tensor_handling(self, mock_node):
        """Test handling of empty tensors."""
        empty_tensor = torch.zeros(0, 665)
        
        result = mock_node.visualize(
            x=empty_tensor,
            cmap='viridis',
            save_as='png', 
            comparison_method='average'
        )
        
        output = result[0]
        
        # Should return valid zero tensor
        assert output.shape == (1, 3, 100, 100)  # Default zero tensor shape
        
    @patch('matplotlib.pyplot.savefig')
    @patch('PIL.Image.open')
    def test_matplotlib_to_pil_conversion(self, mock_pil_open, mock_savefig, mock_node):
        """Test the conversion from matplotlib figure to PIL image."""
        # Create test tensor
        tensor = torch.randn(1, 665)
        
        # Mock PIL Image.open to return a proper RGB image
        mock_img = MagicMock()
        mock_img_array = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        mock_img.__array__ = lambda: mock_img_array
        mock_pil_open.return_value = mock_img
        
        # Call visualize
        result = mock_node.visualize(
            x=tensor,
            cmap='viridis',
            save_as='png',
            comparison_method='average'
        )
        
        output = result[0]
        
        # Verify correct shape
        assert output.shape == (1, 3, 400, 400)
        
    def test_integration_with_comfyui_save_images(self, mock_node):
        """Test that output works with ComfyUI's save_images function."""
        # Create problematic tensor
        tensor = torch.randn(1, 1, 665)
        
        # Get visualization
        result = mock_node.visualize(
            x=tensor,
            cmap='viridis',
            save_as='png',
            comparison_method='average'
        )
        
        output = result[0]
        
        # Simulate what save_images does
        # It expects tensor in shape [batch, height, width, channels]
        # So we need to permute from [batch, channels, height, width]
        for i in range(output.shape[0]):
            img_tensor = output[i]  # [channels, height, width]
            img_tensor = img_tensor.permute(1, 2, 0)  # [height, width, channels]
            
            # Convert to numpy and scale to 0-255
            img_np = img_tensor.numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            # This should not raise an error
            img = Image.fromarray(img_np)
            assert img.mode == 'RGB'
            assert img.size[0] > 0  # width
            assert img.size[1] > 0  # height