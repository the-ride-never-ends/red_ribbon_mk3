import logging
from typing import TypeAlias


import numpy as np
import numpy.typing as npt


Float32Array: TypeAlias = npt.NDArray[np.float32]


logger = logging.getLogger(__name__)


def _get_min_max(x: Float32Array) -> tuple[float, float]:
    return x.min(), x.max()


def _get_mean(x: Float32Array) -> float:
    return np.mean(x, dtype=np.float64)


def _get_std_dev(x: Float32Array) -> float:
    return np.std(x, dtype=np.float64)


def _normalize_array(x: Float32Array, y: Float32Array) -> Float32Array:
    # Calculate numerically stable means and standard deviations.
    mean_x, std_dev_x = _get_mean(x), _get_std_dev(x)
    mean_y, std_dev_y = _get_mean(y), _get_std_dev(y)

    # Normalize arrays
    normalized_x = (x - mean_x) / std_dev_x
    normalized_y = (y - mean_y) / std_dev_y

    # Calculate the difference between normalized arrays
    diff_array = normalized_x - normalized_y

    # Check that the difference array is normalized correctly.
    min_val, max_val = _get_min_max(diff_array)

    if min_val == max_val:
        logger.warning(f"Uniform array detected, max = {max_val}, min = {min_val}")
        raise ValueError("Array is uniform.")

    return diff_array


def validate_array(
        diff_array: npt.NDArray[np.float32],
        tensor1: npt.NDArray[np.float32],
        tensor2: npt.NDArray[np.float32],
        scale_factor: float = 100.0
        ) -> npt.NDArray[np.float32]:
    """
    Check if the difference array is normalized correctly and attempt to normalize it if not.

    This function checks if the input difference array is properly normalized. If not, it attempts
    to normalize it by increasing precision and, if necessary, scaling the input tensors.

    Args:
        diff_array (npt.NDArray[np.float32]): The initial difference array to check.
        tensor1 (npt.NDArray[np.float32]): The first input tensor.
        tensor2 (npt.NDArray[np.float32]): The second input tensor.
        scale_factor (float): The factor by which to scale the input tensors if normalization fails. Defaults to 100.0.

    Returns:
        npt.NDArray[np.float32]: The normalized difference array.

    Raises:
        ValueError: If the array cannot be normalized even after increasing precision and scaling.

    Notes:
        - The function first checks if the difference array has uniform values.
        - If uniform, it increases precision to float64 and attempts normalization.
        - If still uniform, it scales the input tensors by a factor of 100 and tries again.
    """
    # Get min and max values of the tensor
    min_val, max_val = _get_min_max(diff_array)

    if min_val == max_val:
        # NOTE We use a generator here to avoid unnecessary computations if the first pair of tensors already normalizes correctly.
        for x, y in ((tensor1, tensor2), (tensor1 * scale_factor, tensor2 * scale_factor)):
            try:
                diff_array = _normalize_array(x, y)
            except ValueError as e:
                logger.warning(f"Normalization failed: {e}")
                continue
            else:
                logger.info("Array normalization successful.")
                break
        else:
            raise ValueError(f"Difference array failed to normalize, even after scaling by {scale_factor}")
    logger.info(f"Min/Max Values in diff_array: max = {max_val}, min = {min_val}")
    return diff_array
