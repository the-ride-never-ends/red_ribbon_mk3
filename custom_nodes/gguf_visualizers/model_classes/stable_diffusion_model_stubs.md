# Function and Class stubs from '/home/kylerose1946/red_ribbon_mk3/custom_nodes/gguf_visualizers/model_classes/stable_diffusion_model.py'

Files last updated: 1752562724.8953884

Stub file last updated: 2025-07-14 23:59:11

## StableDiffusionModel

```python
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
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## __init__

```python
def __init__(self, filename: Path | str) -> None:
    """
    Initialize the model with a file path.

Args:
    filename (Path | str): The path to the model file.

Attributes initialized:
    model: The loaded model object.
    tensors: An OrderedDict containing the model's tensors.
    """
```
* **Async:** False
* **Method:** True
* **Class:** StableDiffusionModel

## get_as_f32

```python
def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
    """
    Get a tensor as a float32 numpy array.

Args:
    key (str): The name of the tensor to retrieve.

Returns:
    npt.NDArray[np.float32]: The tensor data as a numpy array of type float32.
    """
```
* **Async:** False
* **Method:** True
* **Class:** StableDiffusionModel

## get_type_name

```python
def get_type_name(self, key: str) -> str:
    """
    Get the type name of a tensor.

Args:
    key (str): The name of the tensor.

Returns:
    str: The type name of the tensor (e.g. "unet", "attention", etc.).
    """
```
* **Async:** False
* **Method:** True
* **Class:** StableDiffusionModel

## tensor_names

```python
def tensor_names(self) -> Iterable[str]:
    """
    Get the models tensor names.

Returns:
    Iterable[str]: An iterable containing the names of the tensors in the model.
    """
```
* **Async:** False
* **Method:** True
* **Class:** StableDiffusionModel

## valid

```python
def valid(self, key: str) -> tuple[bool, None | str]:
    """
    Check if a tensor is valid.

Args:
    key (str): The name of the tensor to check.

Returns:
    tuple[bool, None | str]: A tuple where the first element is a boolean indicating
        whether the tensor is valid, and the second element is an optional error message.
    """
```
* **Async:** False
* **Method:** True
* **Class:** StableDiffusionModel
