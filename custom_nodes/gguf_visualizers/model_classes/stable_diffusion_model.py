from pathlib import Path
from typing import Iterable, Protocol

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

    def tensor_names(self) -> Iterable[str]:
        """Get the models tensor names.
        
        Returns:
            Iterable[str]: An iterable containing the names of the tensors in the model.
        """
        pass

    def valid(self, key: str) -> tuple[bool, None | str]:
        """Check if a tensor is valid.
        
        Args:
            key (str): The name of the tensor to check.

        Returns:
            tuple[bool, None | str]: A tuple where the first element is a boolean indicating
                whether the tensor is valid, and the second element is an optional error message.
        """
        pass

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        """Get a tensor as a float32 numpy array.
        
        Args:
            key (str): The name of the tensor to retrieve.

        Returns:
            npt.NDArray[np.float32]: The tensor data as a numpy array of type float32.
        """
        pass

    def get_type_name(self, key: str) -> str:
        """Get the type name of a tensor.
        
        Args:
            key (str): The name of the tensor.

        Returns:
            str: The type name of the tensor (e.g. "unet", "attention", etc.).
        """
        pass