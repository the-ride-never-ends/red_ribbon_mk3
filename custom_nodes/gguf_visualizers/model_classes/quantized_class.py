from typing import Any, Iterable


import numpy as np
import numpy.typing as npt

import logging
logger = logging.getLogger(__name__)


class _Quantized:
    """
    Abstract class for model quantizations.
    NOTE: Since model classes are implemented using classmethods, this class is technically not abstract.
    However, we treat it as such for typing/standardization purposes.
    """
    dtype: np.dtype[Any]
    block_size: int

    def quantize(self, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """
        Quantize a float32 array to uint8.

        This method should be implemented by subclasses to provide specific quantization logic.

        Parameters:
            arr (npt.NDArray[np.float32]): The input array to be quantized.

        Returns:
            npt.NDArray[np.uint8]: The quantized array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def dequantize(self, arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """
        Dequantize a uint8 array back to float32.

        This method should be implemented by subclasses to provide specific dequantization logic.

        Parameters:
            arr (npt.NDArray[np.uint8]): The input array to be dequantized.

        Returns:
            npt.NDArray[np.float32]: The dequantized array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()


class Quantized_Q8_0(_Quantized):  # noqa: N801
    block_size = 32
    dtype = np.dtype([("d", "f2"), ("qs", "i1", (block_size,))])

    # Mini Q8_0 quantization in Python!
    @classmethod
    def quantize(cls, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """
        Quantize a float32 array to uint8 using block-wise quantization.

        This method divides the input array into blocks and quantizes each block
        independently by finding the maximum absolute value in each block and
        scaling all values accordingly. The quantization uses 8-bit signed
        representation (-127 to 127 range).

        Args:
            arr (npt.NDArray[np.float32]): Input float32 array to be quantized.
                The array size must be divisible by the class block_size.
        Returns:
            npt.NDArray[np.uint8]: Quantized array containing tuples of scaling
                factors and quantized values for each block, structured according
                to the class dtype specification.
        Notes:
            - Uses block-wise quantization where each block is quantized independently
            - Scaling factor is computed as max(abs(block)) / 127
            - Handles all-zero blocks by setting quantized values to 0
            - Implementation optimized for performance using vectorized operations
        """
        n_blocks = arr.size // cls.block_size
        blocks = arr.reshape((n_blocks, cls.block_size))

        # Much faster implementation of block quantization contributed by @Cebtenzzre
        def quantize_blocks(blocks: npt.NDArray[Any]) -> Iterable[tuple[Any, Any]]:

            # Find the maximum absolute value in each block, divided by 127
            d = abs(blocks).max(axis=1) / np.float32(127)

            # Divide each block by its scaling factor and round
            with np.errstate(divide="ignore"):
                qs = (blocks / d[:, None]).round()

            # Handle blocks that are all zeros
            qs[d == 0] = 0
            yield from zip(d, qs)

        return np.fromiter(
            quantize_blocks(blocks),
            count=n_blocks,
            dtype=cls.dtype,
        )

    @classmethod
    def dequantize(cls, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """
        Dequantize a quantized array back to float32 values.

        This method converts quantized uint8 data back to its original float32 representation
        by extracting the scale factors (d) and quantized values (qs) from the structured
        array, then performing element-wise multiplication with proper broadcasting.

        Args:
            array (npt.NDArray[np.uint8]): Input quantized array containing structured data
                with 'd' (scale factors) and 'qs' (quantized values) fields.
        Returns:
            npt.NDArray[np.float32]: Dequantized float32 array with shape flattened from
                the original block structure.

        Note:
            The input array is expected to be a structured numpy array that can be viewed
            as the class's dtype, containing at minimum 'd' and 'qs' fields.
        """
        blocks = array.view(dtype=cls.dtype)
        
        # Use proper numpy broadcasting: expand d to (n_blocks, 1) and multiply with qs
        d_expanded = blocks["d"][:, None]  # Shape: (n_blocks, 1)
        qs_float = blocks["qs"].astype(np.float32)  # Shape: (n_blocks, block_size)
        
        # Broadcast multiplication and flatten
        results = (d_expanded * qs_float).flatten()
        
        logger.debug(f"results shape: {results.shape}")
        return results





    # @classmethod
    # def dequantize(
    #     cls,
    #     arr: npt.NDArray[np.uint8],
    # ) -> npt.NDArray[np.float32]:

    #     blocks = arr.view(dtype=cls.dtype)
    #     #return (blocks["d"][:, None] * np.float32(blocks["qs"])).flatten()

    #     # Unoptimized hack-around for broadcasting errors.
    #     # TODO This is trashy and gross. Learn numpy and fix it!
    #     # It also needs to be verified.
    #     results = []
    #     for block in blocks:

    #         block_results = []
    #         for scalar, weights_array in block:
    #             # Multiply the scaling factor with each element in the weights array
    #             scaled_arr = np.array(np.float32(weights_array)) * scalar
    #             block_results.append(scaled_arr)
    #         results.append(block_results)
    #     results = np.array(results)

    #     #logger.debug(f"results: {results}")
    #     logger.debug(f"results shape: {results.shape}",f=True)

    #     results = results.flatten()
    #     #logger.debug(f"results: {results}")
    #     logger.debug(f"results shape: {results.shape}",f=True)
    #     return results




        # # # Reshape d to match the broadcasting requirements
        # # d_expanded = blocks["d"][:, None]  # This should give (4096, 1, 344) MAXIMUM ABSOLUTE DEVIATION
        # # qs_data = np.float32(blocks["qs"])    # This is (4096, 344, 32) # Multiply them all together and you get the total values in the tensor.



        # # # Print shapes for debugging
        # # logger.debug(f"d_expanded shape: {d_expanded.shape}")
        # # logger.debug(f"qs_data shape: {qs_data.shape}")

        # # logger.debug(f"d_expanded:\n{d_expanded}",f=True)
        # # logger.debug(f"qs_data:\n{qs_data}",f=True)
        
        # # # Perform the multiplication with correct broadcasting
        # # results = d_expanded * qs_data
        # # logger.debug(f"result: {result}",f=True,t=30)

        # return results.flatten()
        # # 
