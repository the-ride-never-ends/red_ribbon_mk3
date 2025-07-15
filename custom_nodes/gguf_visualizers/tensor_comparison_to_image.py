#!/usr/bin/env python3
"""Produce heatmaps of differences in tensor values for AI models (GGUF and PyTorch)"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import re
from textwrap import dedent
from typing import Any, Callable, Never, TypeAlias
from pathlib import Path


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import numpy.typing as npt
from PIL import Image


from .tensor_to_image import GGUFModel, TorchModel
from .visualizer_utils import (
    validate_array,
    have_the_same_file_extension,
    right_now,
    write_array_to_geotiff,
)


from .visualizer_configs import (
    get_config,
    FileSpecificConfigs,
)





# Define hard-coded constants
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = script_dir = os.path.dirname(script_dir)
MAIN_FOLDER = os.path.dirname(script_dir)
YEAR_IN_DAYS: int = 365
DEBUG_FILEPATH: str = os.path.join(MAIN_FOLDER, "debug_logs")
RANDOM_SEED: int = 420

# Define program-specific hard-coded constants
# Clip values to at max 7 standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7
# Number of standard deviations above the mean to be positively scaled.
CFG_SD_POSITIVE_THRESHOLD = 1
# Number of standard deviations below the mean to be negatively scaled.
CFG_SD_NEGATIVE_THRESHOLD = 1
# RGB scaling for pixels that meet the negative threshold.
CFG_NEG_SCALE = (1.2, 0.2, 1.2)
# RGB scaling for pixels that meet the positive threshold.
CFG_POS_SCALE = (0.2, 1.2, 1.2)
# RGB scaling for pixels between those ranges.
CFG_MID_SCALE = (0.1, 0.1, 0.1)
# CFG_MID_SCALE = (0.6, 0.6, 0.9) Original Values


import logging
logger = logging.getLogger(__name__)

Float32Array: TypeAlias = npt.NDArray[np.float32]

config: Callable = FileSpecificConfigs().config

def comfyui_node():
    pass


#@comfyui_node
def tensor_comparison_to_image_comfy_ui_node(
                                            model_file1: str, 
                                            model_file2: str, 
                                            tensor_name: str, 
                                            comparison_type: str, 
                                            color_mode: str, 
                                            output_name: str, 
                                            output_mode: str,
                                            ) -> None:
        """
        ComfyUI node for generating a heatmap of the difference between two tensors in two models.

        Args:
            model_file1 (str): Path to the first model file. Can be a GGUF or PyTorch model.
        """
        run = TensorComparisonToImage(
            model_file1=model_file1, 
            model_file2=model_file2, 
            tensor_name=tensor_name,
            comparison_type=comparison_type,
            color_mode=color_mode,
            output_name=output_name,
            output_mode=output_mode
            )
        run.tensor_comparison_to_image()

        return {"ui": {"images": [run.heatmap_img]}}




class TensorComparisonToImage:

    SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.geotiff', '.tif')


    # Define hard-coded constants
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent
    MAIN_FOLDER = PROJECT_ROOT.parent
    YEAR_IN_DAYS: int = 365
    DEBUG_FILEPATH: Path = MAIN_FOLDER / "debug_logs"
    RANDOM_SEED: int = 420
    INPUT_FOLDER: Path = MAIN_FOLDER / "input"
    OUTPUT_FOLDER: Path = MAIN_FOLDER / "output"

    # Define program-specific hard-coded constants
        # Clip values to at max 7 standard deviations from the mean.
    CFG_SD_CLIP_THRESHOLD = 7
        # Number of standard deviations above the mean to be positively scaled.
    CFG_SD_POSITIVE_THRESHOLD = 1
        # Number of standard deviations below the mean to be negatively scaled.
    CFG_SD_NEGATIVE_THRESHOLD = 1
        # RGB scaling for pixels that meet the negative threshold.
    CFG_NEG_SCALE = (1.2, 0.2, 1.2)
        # RGB scaling for pixels that meet the positive threshold.
    CFG_POS_SCALE = (0.2, 1.2, 1.2)
        # RGB scaling for pixels between those ranges.
    CFG_MID_SCALE = (0.1, 0.1, 0.1)
        # CFG_MID_SCALE = (0.6, 0.6, 0.9) Original Values

    FIRST_MODEL_PATH: str = config("MODEL_FILE_PATH1")
    SECOND_MODEL_PATH: str = config("MODEL_FILE_PATH2")
    TENSOR_NAME: str = config("TENSOR_NAME")
    COMPARISON_TYPE: str = config("COMPARISON_TYPE")
    COLOR_MODE: str = config("COLOR_MODE")
    OUTPUT_NAME: str = config("OUTPUT_NAME")

    def __init__(self, **kwargs):

        model_path1 = self.FIRST_MODEL_PATH or kwargs.pop("model_file1")
        model_path2 = self.SECOND_MODEL_PATH or kwargs.pop("model_file2")

        self.model_file1: str = str(self.INPUT_FOLDER / model_path1)
        self.model_file2: str = str(self.INPUT_FOLDER / model_path2)
    
        self.type_check_model_files()

        self.tensor_name: str = self.TENSOR_NAME or kwargs.pop("tensor_name")
        if self.tensor_name is None or self.tensor_name == "":
            raise ValueError("Tensor name cannot be empty.")

        self.comparison_type: str = self.COMPARISON_TYPE or kwargs.pop("comparison_type")
        self.color_mode: str = self.COLOR_MODE or kwargs.pop("color_mode")


        self.output_name = self.OUTPUT_NAME or kwargs.pop(
            "output_name", 
            f"diff_map_{os.path.basename(self.model_file1)}_and_{os.path.basename(self.model_file1)}_{self.comparison_type}_{right_now()}.png"
        )
        # find_this_file_under_this_directory_and_return_the_files_path
        self.output_path: Path = self.OUTPUT_FOLDER / self.output_name

        # If the image path does not end with a file-type, default to png
        if not self.output_path.suffix.lower() in self.SUPPORTED_IMAGE_TYPES:
            self.output_path = self.output_path.with_suffix('.png')

        self.x: Float32Array = None
        self.y: Float32Array = None
        self.central_tendency: float | Float32Array = None
        self.deviation: float | Float32Array = None

        # Load tensors from the models.
        self.x = self._extract_tensor_from_model(self.model_file1)
        self.y = self._extract_tensor_from_model(self.model_file2)

        # Check if the tensors have the same dimensions.
        if self.x.shape != self.y.shape:
            raise ValueError("Tensors must be of the same dimensions.")

        self.heatmap_img: Image = None

    def type_check_model_files(self) -> Never: 
        if self.model_file1 is None or self.model_file2 is None:
            msg = f"self.model_file1: {self.model_file1}\nself.model_file2: {self.model_file2}"
            logger.error(msg)
            raise ValueError(f"Both model_file1 and model_file2 must be provided.\n{msg}")
        if not os.path.exists(self.model_file1) or not os.path.exists(self.model_file2):
            msg = f"self.model_file1: {self.model_file1}\nself.model_file2: {self.model_file2}"
            logger.error(msg)
            raise FileNotFoundError(f"One or both model files not found under the given paths.\n{msg}")
        
        # Check if the models have the same ending prefix e.g. gguf, pth, etc.
        if not have_the_same_file_extension(self.model_file1, self.model_file2):
            raise ValueError(f"Model prefixes do not match\n{os.path.basename(self.model_file1)}\n{os.path.basename(self.model_file2)}")


    def _extract_tensor_from_model(
        self,
        model_file: str, 
        ) -> Float32Array:
        """
        Extracts a tensor from a given model file based on the tensor name.

        Args:
            model_file (str): Path to the model file. Can be a GGUF or PyTorch model.

        Returns:
            Float32Array: The extracted tensor as a NumPy array.

        Raises:
            ValueError: If the model type is unknown or if the tensor extraction fails.
            NotImplementedError: If a Stable Diffusion model is provided (currently unsupported).
        """
        # Initialize the model
        match model_file.split('.')[-1].lower():
            case "gguf":
                model = GGUFModel(model_file)
            case "pth":
                model = TorchModel(model_file)
            case "stable_diffusion":
                raise NotImplementedError("Stable Diffusion models are not yet supported.")
            case _:
                raise ValueError("Unknown Model Type")

        # Validate and retrieve the tensor
        tensor_is_valid, error_message = model.valid(self.tensor_name)
        if tensor_is_valid:
            return model.get_as_f32(self.tensor_name)
        else:
            raise ValueError(f"Error extracting tensor from {model_file}: {error_message}")

    def _calc_absolute(self) -> Float32Array:
        # Scale vectors by 1000 to prevent creation of a null array.
        scale_factor = 1000
        scaled_x = self.x * scale_factor
        scaled_y = self.y * scale_factor

        # Compare element-wise differences into a single array for visualization
        abs_diff_array = scaled_x - scaled_y

        logger.info(f"""
            * Direct comparison:
                max diff * 1000 = {abs_diff_array.max()}, 
                min diff * 1000 = {abs_diff_array.min()}
        """)
        return abs_diff_array / scale_factor


    @staticmethod
    def _calc_median_and_mad(tensor: Float32Array) -> tuple[float, float]:
        """
        Calculate the median and median absolute deviation (MAD) of a tensor.

        Args:
            tensor (Float32Array): Input tensor for which to calculate statistics.

        Returns:
            tuple[float, float]: A tuple containing:
                - median (float): The median value of the tensor calculated with float64 precision
                - mad (float): The median absolute deviation from the median
        """
        return np.median(tensor, dtype=np.float64), np.median(np.abs(tensor - np.median(tensor)))


    @staticmethod
    def _calc_mean_and_std_dev(tensor: Float32Array) -> tuple[float, float]:
        """
        Calculate the mean and standard deviation of a tensor.

        Args:
            tensor (Float32Array): Input tensor array for statistical calculation.

        Returns:
            tuple[float, float]: A tuple containing the mean and standard deviation
                                of the tensor, computed with float64 precision.
        """
        return np.mean(tensor, dtype=np.float64), np.std(tensor, dtype=np.float64)


    def _compare_tensors(self, comparison_type: str) -> Float32Array:
        """
        Compare two tensors using the specified comparison method.

        This method performs tensor comparison using one of three approaches:
        - Absolute difference: Direct subtraction of tensor values
        - Mean-based comparison: Normalizes tensors using mean and standard deviation
        - Median-based comparison: Normalizes tensors using median and median absolute deviation

        For statistical comparisons (mean/median), tensors are first normalized by their
        central tendency and deviation to make them directly comparable, then the difference
        is calculated between the normalized arrays.

        Args:
            comparison_type (str): The type of comparison to perform. Must be one of:
                - "absolute": Calculate direct difference between tensors
                - "mean": Normalize using mean/std_dev then calculate difference
                - "median": Normalize using median/MAD then calculate difference
        Returns:
            Float32Array: The difference array between the compared tensors.
                For absolute comparison, returns direct difference.
                For statistical comparisons, returns difference of normalized tensors.

        Raises:
            ValidationError: If the resulting difference array fails validation against input tensors

        Side Effects:
            - Sets self.central_tendency and self.deviation attributes for statistical comparisons
            - Logs diagnostic information about tensor statistics and comparison results

        """
        calc_func: Callable = None
        calc_mode: tuple[str] = None
        absolute: bool = False
        match comparison_type:
            case "absolute":
                absolute = True
            case "mean":
                calc_func = self._calc_mean_and_std_dev
                calc_mode = ("mean", "std_dev")
            case "median":
                calc_func = self._calc_median_and_mad
                calc_mode = ("median", "mad")

        if absolute:
            diff_array = self._calc_absolute()
        else:
            tensor_list = []
            for idx, tensor in enumerate([self.x, self.y], start=1):
                model_name = getattr(self, f"model_file{idx}")

                # Calculate numerically stable means and standard deviations
                central_tendency, deviation = calc_func(tensor)

                # Log tensor values for diagnostics
                logger.debug(f"""
                    * Tensor{idx} ({self.tensor_name} from {model_name}) stats:
                        {calc_mode[0]} = {central_tendency},
                        {calc_mode[1]} = {deviation}
                        max = {np.max(tensor)},
                        min = {np.min(tensor)}
                """,f=True)

                # Normalize arrays to make their means directly comparable.
                normalized = (tensor - central_tendency) / deviation
                tensor_list.append(normalized)

            # Calculate the difference between normalized arrays.
            diff_array = tensor_list[0] - tensor_list[1]

            # Get the central tendency and deviation for the difference array.
            self.central_tendency, self.deviation = calc_func(diff_array)

            logger.info(f"""
                * {calc_mode[0].capitalize()} comparison:
                {calc_mode[0]}_diff = {diff_array}
            """,f=True)
        return validate_array(diff_array, self.x, self.y)

    @staticmethod
    def _make_color_map_with_discrete_bins(diff_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Create a custom colormap with discrete bins for visualizing tensor differences.

        This method generates a discrete colormap by dividing the range of difference values
        into 10 equal bins and applying the 'coolwarm' colormap with boundary normalization.
        The resulting colormap provides clear visual distinction between different ranges
        of tensor difference values.

        Args:
            diff_array (npt.NDArray[np.uint8]): Array containing difference values between
                tensors. Values should be in the range [0, 255] representing the magnitude
                of differences at each position.

        Returns:
            npt.NDArray[np.uint8]: RGBA color array with the same shape as input array
                plus an additional dimension for color channels. Each pixel is mapped to
                a discrete color bin based on its difference value using the coolwarm
                colormap (blue for low differences, red for high differences).

        Note:
            The colormap uses 10 discrete bins with boundary normalization to ensure
            clear visual separation between different difference magnitudes. The coolwarm
            colormap provides intuitive color coding where cooler colors (blue) represent
            smaller differences and warmer colors (red) represent larger differences.
        """
        n_bins = 10  # Number of bins
        cmap = plt.cm.coolwarm
        bounds = np.linspace(diff_array.min(), diff_array.max(), n_bins + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap(norm(diff_array))

    def _covert_diff_result_to_heatmap(self, color_mode: str, diff_array: Float32Array) -> Image:

        logger.debug("Checking if array is 2D and numerical...")
        if len(diff_array.shape) != 2 or not np.issubdtype(diff_array.dtype, np.number):
            raise ValueError("Input diff_array must be a 2D numerical array")

        logger.info(f"""
            Input diff_array stats (in SD): 
                max = {diff_array.max()}, 
                min = {diff_array.min()}, 
                mean = {np.mean(diff_array, dtype=np.float64)}, 
                sd = {np.std(diff_array, dtype=np.float64)}
            """,f=True)

        logger.info(f"Applying colormap '{color_mode}'...")
        map_func: Callable = None
        match color_mode:
            case "grayscale":            map_func = plt.cm.gray
            case "false color jet":      map_func = plt.cm.jet
            case "false color viridis":  map_func = plt.cm.viridis
            case "false color plasma":   map_func = plt.cm.plasma
            case "false color inferno":  map_func = plt.cm.inferno
            case "false color magma":    map_func = plt.cm.magma
            case "false color cividis":  map_func = plt.cm.cividis
            case "false color twilight": map_func = plt.cm.twilight
            case "false color rainbow":  map_func = plt.cm.rainbow
            case "false color seismic":  map_func = plt.cm.seismic
            case "binned coolwarm":      map_func = self._make_color_map_with_discrete_bins
            case "tensor_to_image style":
                if self.central_tendency and self.deviation:
                    normalized_diff_array = self.normalize_tensor_by_central_tendency_and_deviation(diff_array, self.central_tendency, self.deviation)
                    heatmap_img: Image = self.make_image_of_(normalized_diff_array, self.central_tendency, self.deviation)
                    return heatmap_img
                else:
                    logger.warning("Cannot find central tendency and/or deviation.\n'tensor_to_image style' relies on these to assign colors.\nDefaulting to grayscale...")
                    map_func = plt.cm.gray
            case _:
                logger.warning("Unknown color mode. Defaulting to grayscale...")
                map_func = plt.cm.gray
        colormap_array = map_func(diff_array) 

        logger.debug(f"Colormap output shape: {colormap_array.shape}")

        logger.debug("Ensuring correct shape for colormap output...")
        if colormap_array.ndim == 3 and colormap_array.shape[2] in [3, 4]:
            logger.debug(f"Alpha channel present. Discarding...")
            heatmap_array = colormap_array[..., :3]  # Discarding alpha channel if present
        else:
            raise ValueError("Unexpected shape for color map output")

        logger.debug(f"Converting to 8-bit format...")
        heatmap_array = (heatmap_array * 255).astype(np.uint8)

        logger.debug(f"Converting to PIL Image...")
        if heatmap_array.ndim != 3 or heatmap_array.shape[2] != 3:
            raise ValueError("Difference array must be 3-dimensional with 3 channels for RGB")

        return Image.fromarray(heatmap_array, mode=self._get_colormap(color_mode))

    @staticmethod
    def _get_colormap(color_mode: str) -> str:
        """
        Determine the appropriate color mode for image creation based on the specified color type.

        This function processes a color mode string by removing "false color" text and mapping
        the resulting color type to an appropriate PIL image mode.

        Args:
            color_mode (str): The color mode specification, may include "false color" prefix
                             and various colormap names like "grayscale", "jet", "viridis", etc.
        Returns:
            str: PIL image mode string - either 'L' for grayscale or 'RGB' for color modes.
                 Returns 'L' for grayscale, 'RGB' for supported colormaps (jet, viridis, 
                 plasma, inferno, magma, cividis, twilight, rainbow, seismic, binned coolwarm),
                 and defaults to 'RGB' for unrecognized color modes.
        """
        color_type = re.sub("false color","", color_mode).strip()
        logger.debug(f"color_type: {color_type}")

        mode: str = "RGB"
        match color_type:
            case "grayscale":
                mode = 'L'
            case color_type if color_type in {"jet", "viridis", "plasma", "inferno", "magma", "cividis", "twilight", "rainbow", "seismic"}:
                mode = 'RGB'
            case "binned coolwarm":
                mode = 'RGB'
            case _:
                mode = 'RGB'  # Default to RGB for other color modes
        return mode


    def normalize_tensor_by_central_tendency_and_deviation(self,
                                                          tensor: Float32Array,
                                                          central_tendency: float,
                                                          deviation: float
                                                          ) -> Float32Array:
        """
        Transform a tensor of values into a tensor of normalized standard deviations 
            based on the mean of those values.

        Args:
            tensor (Float32Array): Input tensor to be normalized.
            central_tendency (float): A measure of central tendency (mean, median, etc.)
            deviation (float): A measure of deviancy based on the central tendency (standard deviation, median absolute deviation, etc.)

        Returns:
            Float32Array: Normalized tensor of standard deviations.

        Notes:
            - This method performs the following steps:
                1. Calculate the mean (central tendency) of the input tensor.
                2. Calculate the standard deviation of the input tensor.
                3. Subtract the mean from each value in the tensor.
                4. Divide the result by the standard deviation.
            - The resulting tensor represents how many standard deviations each value is away from the tensor's mean.
        """
        # Avoid division by zero
        if deviation == 0:
            return np.zeros_like(tensor)
        
        logger.info(f"Normalizing tensor: (tensor - {central_tendency}) / {deviation}")
        normalized_tensor = (tensor - central_tendency) / deviation
        return normalized_tensor

    def _map_tensor_to_color_scale(self, x: Float32Array, mu: float, dev: float) -> tuple[Float32Array, float, float]:
        """
        Map tensor values to a color scale for visualization purposes.
        This method normalizes tensor data to a 0-255 range suitable for image representation,
        using statistical thresholds based on mean and standard deviation. The tensor values
        are clipped to a maximum threshold and then scaled to create a 3-channel color representation.

        Args:
            x (Float32Array): Input 2D tensor data to be mapped to color scale
            mu (float): Mean value of the tensor data
            dev (float): Standard deviation of the tensor data

        Returns:
            tuple[Float32Array, float, float]: A tuple containing:
                - tda (Float32Array): 3D array with shape (*x.shape, 3) containing RGB values 
                    scaled to 0-255 range
                - sdp_thresh (float): Positive standard deviation threshold value
                - sdn_thresh (float): Negative standard deviation threshold value
        Notes:
            - Uses CFG_SD_CLIP_THRESHOLD to determine the maximum clipping value
            - Uses CFG_SD_POSITIVE_THRESHOLD and CFG_SD_NEGATIVE_THRESHOLD to calculate
                statistical thresholds for visualization
            - The output tensor is repeated across 3 channels to create RGB representation
        """
        # Map the 2D tensor data to the same range as an image 0-255.
        sdp_max = mu + self.CFG_SD_CLIP_THRESHOLD * dev
            # Set the positive and negative SD thresholds for this specific tensor.
        sdp_thresh = mu + self.CFG_SD_POSITIVE_THRESHOLD * dev
        sdn_thresh = mu - self.CFG_SD_NEGATIVE_THRESHOLD * dev
            # Calculate the absolute difference between the tensor data and the mean.
        tda = np.minimum(np.abs(x), sdp_max).repeat(3, axis=-1).reshape((*x.shape, 3))

        # Scale that range to between 0 and 255.
        tda = 255 * ((tda - np.min(tda)) / np.ptp(tda))
        return tda, sdp_thresh, sdn_thresh


    def make_image_of_(self, 
                       x: Float32Array,
                       central_tendency: float,
                       deviation: float,
                       color_ramp_type: str = "discrete" # FIXME Continuous doesn't work yet.
                       ) -> Image:
        """
        Create an image representation of a given tensor.

        This method processes a tensor and converts it into an RGB image, where the color
        intensity represents the deviation from the central tendency (mean or median).

        The method performs the following steps:
        1. Scale the tensor data to a color scale (0-255) based on the central tendency and deviation.
        2. Maps the scaled tensor data to color intensities:
           - For 'discrete' color ramp:
             * Uses discrete color scales for negative and positive deviations.
             * Darker reds represent more negative deviations.
             * Darker greens represent more positive deviations.
           - For 'continuous' color ramp:
             * Applies continuous color scaling based on deviation thresholds.
             * Red for negative deviations, green for positive, and scaled colors in between.
        3. Converts the resulting color data into a PIL Image.

        Args:
            x (Float32Array): Input tensor to be converted into an image.
            central_tendency (float): A measure of central tendency (mean, median, etc.)
            deviation (float): A measure of deviancy based on the central tendency 
                     (standard deviation, median absolute deviation, etc.)
            color_ramp_type (str): Type of color ramp to use for the image. 
                     Options are 'discrete' or 'continuous'. Default is 'discrete'.

        Returns:
            Image: A PIL Image object representing the tensor data.


        Notes:
            The color mapping is influenced by several class attributes and constants:
            - CFG_SD_CLIP_THRESHOLD: Maximum number of standard deviations for clipping.
            - CFG_SD_POSITIVE_THRESHOLD, CFG_SD_NEGATIVE_THRESHOLD: Thresholds for positive and negative deviations.
            - CFG_NEG_SCALE, CFG_POS_SCALE, CFG_MID_SCALE: Color scaling factors for different ranges.
            The color mapping logic is sensitive to the statistical properties of the input tensor.
        """
        tda, sdp_thresh, sdn_thresh = self._map_tensor_to_color_scale(x, central_tendency, deviation)

        match color_ramp_type :
            case "discrete":  # Discrete Colors
                # Define the boundaries and corresponding colors
                dist_range = [
                    -6, -5, -4, -3, -2, -1, 0, # Negative SD Values use "Reds" color ramp. Darker reds represent more negative SD values.
                    1, 2, 3, 4, 5, 6 # Positive SD Values use "Greens" color ramp. Darker greens represent more positive SD values.
                ]
                colors = np.array([
                    [103, 0, 13],    # < -6σ  -> # 67000d
                    [179, 18, 24],   # -6σ to -5σ -> # b31218
                    [221, 42, 37],   # -5σ to -4σ -> # dd2a25
                    [246, 87, 62],   # -4σ to -3σ -> # f6573e
                    [252, 134, 102], # -3σ to -2σ -> # fc8666
                    [252, 179, 152], # -2σ to -1σ -> # fcb398
                    [254, 220, 205], # -1σ to 0 -> # fedccd
                    [226, 244, 221], # 0 to 1σ -> # e2f4dd
                    [191, 230, 185], # 1σ to 2σ -> # bfe6b9
                    [148, 211, 144], # 2σ to 3σ -> # 94d390
                    [96, 186, 108],  # 3σ to 4σ -> # 60ba6c
                    [50, 155, 81],   # 4σ to 5σ -> # 329b51
                    [13, 120, 53],   # 5σ to 6σ -> # 0d7835
                    [0, 68, 27]      # > 6σ -> # 00441b
                ])
                # Bin the data and apply colors
                bins = np.digitize(x, np.array(dist_range) * deviation + central_tendency)
                tda[...] *= colors[bins][..., np.newaxis]

            case "continuous":  # Continuous Colors
                tda[x <= sdn_thresh, ...] *= self.CFG_NEG_SCALE
                tda[x >= sdp_thresh, ...] *= self.CFG_POS_SCALE
                tda[np.logical_and(x > sdn_thresh, x < sdp_thresh), ...] *= self.CFG_MID_SCALE

            case _:
                raise ValueError(f"Unknown color ramp type '{color_ramp_type}'.")

        return Image.fromarray(tda.astype(np.uint8), "RGB")

    @property
    def arrays_are_equal(self) -> bool:
        """
        Check if the two tensors are equal.

        Returns:
            bool: True if the tensors are equal, False otherwise.
        """
        return np.array_equal(self.x, self.y)


    def tensor_comparison_to_image(self) -> None:
        """
        Compare two tensors and generate a visual representation of their differences.

        This method performs a comparison between two tensors (tensor1 and tensor2) using
        the specified comparison type and outputs the result as either a heatmap image or
        a GeoTIFF file based on the output file extension.

        The method supports multiple comparison types:
        - 'absolute': Absolute difference between tensors
        - 'mean': Mean-based comparison
        - 'median': Median-based comparison

        Raises:
            ValueError: If the tensors are identical, as meaningful comparison requires
                       different tensors.
        Notes:
            - Logs basic statistics (min/max) for both input tensors
            - Validates that tensors are different before proceeding
            - Saves output as GeoTIFF if output filename has .geotiff, .tiff, or .tif extension
            - Otherwise creates and saves a heatmap image using the specified color mode
            - Logs progress and any errors encountered during the process

        Side Effects:
            - Sets self.heatmap_img attribute if creating a heatmap
            - Saves the comparison result to self.output_path
            - Logs information about the comparison process
        """
        # Log basic stats for both tensors
        for idx in [1, 2]:
            logger.info(f'''
        Tensor{idx} ({self.tensor_name} from {getattr(self, f'model_file{idx}')}) stats
            max = {np.max(getattr(self, f'tensor{idx}')):.4f}
            min = {np.min(getattr(self, f'tensor{idx}')):.4f}
        ''',f=True)

        logger.debug("Checking for identical tensors...")
        if self.arrays_are_equal:
            raise ValueError("Tensors are identical. Tensors must be different in order to perform meaningful comparisons.")

        # Perform tensor comparison based on specified type. Default is mean-based comparison.
        logger.info(f"Performing '{self.comparison_type}' comparison between tensors...")
        match self.comparison_type:
            case 'absolute':
                diff_result = self._compare_tensors('absolute')
            case 'mean':
                diff_result = self._compare_tensors('mean')
            case 'median':
                diff_result = self._compare_tensors('median')
            case _:
                logger.warning(f"Unknown comparison type '{self.comparison_type}'. Defaulting to mean-based comparison...")
                diff_result = self._compare_tensors('mean')

        if self.output_name.endswith((".geotiff", ".tiff", ".tif",)):
            logger.debug("Saving difference_comparison_result as a geotiff file...")
            write_array_to_geotiff(diff_result, self.output_path)
        else:
            self.heatmap_img: Image = self._covert_diff_result_to_heatmap(self.color_mode, diff_result)

        if self.heatmap_img is not None:
            try:
                logger.info(f"Saving to '{self.output_path}'...")
                self.heatmap_img.save(self.output_path)
            except Exception as e:
                logger.error(f"Error saving the image: {e}")
        else:
            logger.error("Failed to create the image.")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for tensor comparison visualization.
    This function sets up a command-line argument parser that handles the comparison
    of tensor values between two LLM models (GGUF and PyTorch formats) and generates
    heatmap visualizations of their differences.
    Returns:
        argparse.ArgumentParser: Configured argument parser with the following arguments:
            - model_file1 (str): Path to the first model file (GGUF or PyTorch)
            - model_file2 (str): Path to the second model file (GGUF or PyTorch)
            - tensor_name (str): Name of the tensor to compare between models
            - comparison_type (str, optional): Type of comparison calculation
              ('mean', 'median', 'absolute'). Default: 'mean'
            - color_mode (str, optional): Color scheme for visualization
              ('grayscale', 'false color jet', 'false color vidiris', 'binned coolwarm').
              Default: 'grayscale'
            - color_ramp_type (str, optional): Color ramp style for tensor visualization
              ('discrete', 'continuous'). Default: 'discrete'
            - output_name (str, optional): Custom output filename for the generated heatmap
    Note:
        The models must share the same foundation architecture for meaningful
        tensor comparisons. The parser includes detailed help text explaining
        output modes and color ramp types.
    """
    parser = argparse.ArgumentParser(
        description="Produce heatmaps of differences in tensor values for LLM models (GGUF and PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates differences in tensor values between two models with the same foundation architecture.
                         By default, output will be a grayscale raster that has the same dimensions as the tensors.
                rows   : Same as above, except the calculation is based on rows.
                cols:  : Same as above, except the calculation is based on columns.
            
            Color ramp types:
              discrete   : Uses discrete color bins for different standard deviation ranges
              continuous : Applies continuous color scaling based on deviation thresholds
        """,
        ),
    )
    parser.add_argument(
        "model_file1",
        type=str,
        required=True,
        help="Filename for the first model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "model_file2",
        type=str,
        help="Filename for the second model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "tensor_name",
        type=str,
        help="Tensor name, must be from models with the same foundation architecture for the differences to be valid.",
    )
    parser.add_argument(
        "--comparison_type",
        choices=["mean", "median", "absolute"],
        default="mean",
        help="Comparison types, Default: mean",
    )
    parser.add_argument(
        "--color_mode",
        choices=["grayscale", "false color jet", "false color vidiris", "binned coolwarm"],
        default="grayscale",
        help="Color mode, Default: grayscale",
    )
    parser.add_argument(
        "--color_ramp_type",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Color ramp type for 'tensor_to_image style' color mode. Default: discrete",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help=f"Output file name for the heatmap.",
    )
    return parser


def main() -> None:
    parser = create_parser()

    if len(sys.argv) != 7:
        logger.error("Usage: python tensor_comparison_to_image.py <model_file1> <model_file2> <tensor_name> --comparison_type=<comparison_type> --color_mode=<color_mode> --output_path=<output_path>")
        sys.exit(1)

    logger.info("* Starting tensor_comparison_to_image program...")
    TensorComparisonToImage(parser).tensor_comparison_to_image()
    logger.info("*\nDone.")


if __name__ == "__main__":
    main()