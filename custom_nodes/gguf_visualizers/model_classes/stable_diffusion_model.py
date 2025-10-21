import sys
from pathlib import Path
from typing import Iterable
from collections import OrderedDict


import numpy as np
import numpy.typing as npt

from .model_protocol import Model


class StableDiffusionModel(Model):
    """
    A class to interact with stable diffusion text-to-image models.

    Methods:
        __init__(filename: Path | str) -> None:
            Initialize the model with a file path.
        tensor_names() -> Iterable[str]:
            Get the model's tensor names.
        valid(key: str) -> tuple[bool, None | str]:
            Check if a tensor is valid.
        get_as_f32(key: str) -> npt.NDArray[np.float32]:
            Get a tensor as a float32 numpy array.
        get_type_name(key: str) -> str:
            Get the type name of a tensor.
    
    Attributes:
        model: The loaded model object.
        tensors: An OrderedDict containing the model's tensors.
    """

    def __init__(self, filename: Path | str) -> None:
        """Initialize the model with a file path.
        
        Args:
            filename (Path | str): The path to the model file.

        Attributes initialized:
            model: The loaded model object.
            tensors: An OrderedDict containing the model's tensors.
        """
        self.model = None
        self.tensors = None
        self.torch = None

        try:
            import torch
            self.torch = torch
        except ImportError:
            print("! Loading Stable Diffusion models requires the Torch Python module")
            sys.exit(1)
        
        try:
            self.model = self.torch.load(filename, map_location='cpu')
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Validate that model is a dict-like structure
        if not hasattr(self.model, 'keys'):
            print("Invalid checkpoint structure")
            sys.exit(1)
        
        # Create tensors OrderedDict with squeezed tensors
        self.tensors = OrderedDict()
        for key, tensor in self.model.items():
            self.tensors[key] = tensor.squeeze()
        
        # Check for SD components and warn if missing
        self._check_sd_components()
    
    def _check_sd_components(self):
        """Check for standard SD components and warn if missing."""
        # Only print warnings if we have any tensors to check
        if not self.tensors:
            return
            
        # Check for VAE components
        vae_found = any(key.startswith('first_stage_model.') for key in self.tensors.keys())
        if not vae_found:
            print("Warning: No VAE components found")
        
        # Check for UNet components
        unet_found = any(key.startswith('model.diffusion_model.') for key in self.tensors.keys())
        if not unet_found:
            print("Warning: No UNet components found")
        
        # Check for text encoder components
        text_encoder_found = any(key.startswith('cond_stage_model.') for key in self.tensors.keys())
        if not text_encoder_found:
            print("Warning: No text encoder components found")

    def tensor_names(self) -> Iterable[str]:
        """Get the models tensor names.
        
        Returns:
            Iterable[str]: An iterable containing the names of the tensors in the model.
        """
        return self.tensors.keys()

    def valid(self, key: str) -> tuple[bool, None | str]:
        """Check if a tensor is valid.
        
        Args:
            key (str): The name of the tensor to check.

        Returns:
            tuple[bool, None | str]: A tuple where the first element is a boolean indicating
                whether the tensor is valid, and the second element is an optional error message.
        """
        # Check if tensor exists
        if key not in self.tensors:
            return (False, "Tensor not found")
        
        tensor = self.tensors[key]
        
        # Check dtype
        valid_dtypes = ['float32', 'float16', 'bfloat16']
        if tensor.dtype.name not in valid_dtypes:
            return (False, "Unhandled type")
        
        # Check dimensions (SD models typically have tensors with various dimensions)
        # Allow up to 4D tensors for conv layers (common in UNet)
        if tensor.ndim > 4:
            return (False, "Unhandled dimensions")
        
        return (True, "OK")

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        """Get a tensor as a float32 numpy array.
        
        Args:
            key (str): The name of the tensor to retrieve.

        Returns:
            npt.NDArray[np.float32]: The tensor data as a numpy array of type float32.
            
        Raises:
            KeyError: If the tensor key is not found.
            ValueError: If the tensor cannot be converted to numpy array or to float32.
        """
        if key not in self.tensors:
            raise KeyError(f"Tensor '{key}' not found")
        
        tensor = self.tensors[key]
        
        try:
            numpy_array = tensor.numpy()
        except Exception as e:
            raise ValueError(f"Failed to convert tensor '{key}' to numpy array: {e}")
        
        # Convert to float32 if not already
        if numpy_array.dtype != np.float32:
            try:
                numpy_array = numpy_array.astype(np.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert tensor '{key}' to float32: {e}")
        
        return numpy_array

    def get_type_name(self, key: str) -> str:
        """Get the type name of a tensor.
        
        Args:
            key (str): The name of the tensor.

        Returns:
            str: The type name of a tensor (e.g. "torch.float32", "torch.float16", etc.).
        """
        if key not in self.tensors:
            raise KeyError(f"Tensor '{key}' not found")
        
        tensor = self.tensors[key]
        
        # Return the string representation of the tensor's dtype
        return str(tensor.dtype)













# class StableDiffusionModel(Model):
#     """
#     A class to interact with stable diffusion text-to-image models.

#     Methods:
#         __init__(filename: Path | str) -> None:
#             Initialize the model with a file path.
#         tensor_names() -> Iterable[str]:
#             Get the model's tensor names.
#         valid(key: str) -> tuple[bool, None | str]:
#             Check if a tensor is valid.
#         get_as_f32(key: str) -> npt.NDArray[np.float32]:
#             Get a tensor as a float32 numpy array.
#         get_type_name(key: str) -> str:
#             Get the type name of a tensor.
    
#     Attributes:
#         model: The loaded model object.
#         tensors: An OrderedDict containing the model's tensors.
#     """

#     def __init__(self, filename: Path | str) -> None:
#         """Initialize the model with a file path.
        
#         Args:
#             filename (Path | str): The path to the model file.

#         Attributes initialized:
#             model: The loaded model object.
#             tensors: An OrderedDict containing the model's tensors.
#         """
#         self.model = None
#         self.tensors = None

#     def tensor_names(self) -> Iterable[str]:
#         """Get the models tensor names.
        
#         Returns:
#             Iterable[str]: An iterable containing the names of the tensors in the model.
#         """
#         pass

#     def valid(self, key: str) -> tuple[bool, None | str]:
#         """Check if a tensor is valid.
        
#         Args:
#             key (str): The name of the tensor to check.

#         Returns:
#             tuple[bool, None | str]: A tuple where the first element is a boolean indicating
#                 whether the tensor is valid, and the second element is an optional error message.
#         """
#         pass

#     def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
#         """Get a tensor as a float32 numpy array.
        
#         Args:
#             key (str): The name of the tensor to retrieve.

#         Returns:
#             npt.NDArray[np.float32]: The tensor data as a numpy array of type float32.
#         """
#         pass

#     def get_type_name(self, key: str) -> str:
#         """Get the type name of a tensor.
        
#         Args:
#             key (str): The name of the tensor.

#         Returns:
#             str: The type name of the tensor (e.g. "unet", "attention", etc.).
#         """
#         pass


