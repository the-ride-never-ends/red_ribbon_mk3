#!/usr/bin/env python3
import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import OrderedDict
import io
import sys


from custom_nodes.gguf_visualizers.model_classes.stable_diffusion_model import (
    StableDiffusionModel,
)

# Check if the StableDiffusionModel class has the required methods and attributes
assert StableDiffusionModel, "StableDiffusionModel class should be defined."
assert StableDiffusionModel.__init__, "StableDiffusionModel class must have an __init__ method."
assert StableDiffusionModel.tensor_names, "StableDiffusionModel class must have a tensor_names method."
assert StableDiffusionModel.valid, "StableDiffusionModel class must have a valid method."
assert StableDiffusionModel.get_as_f32, "StableDiffusionModel class must have a get_as_f32 method."
assert StableDiffusionModel.get_type_name, "StableDiffusionModel class must have a get_type_name method."
#assert StableDiffusionModel.model, "StableDiffusionModel class must have a model attribute."
#assert StableDiffusionModel.tensors, "StableDiffusionModel class must have a tensors attribute."




class TestStableDiffusionModelGetAsF32:
    """Test StableDiffusionModel get_as_f32 method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock StableDiffusionModel instance for testing."""
        # Mock torch.load to avoid actual file loading
        with patch('torch.load') as mock_torch_load:
            # Capture stdout to suppress warning messages during testing
            captured_output = io.StringIO()
            with patch('sys.stdout', captured_output):
                mock_torch_load.return_value = OrderedDict()
                model = StableDiffusionModel("dummy_path.ckpt")
            return model

    @pytest.fixture
    def float32_tensor(self):
        """Create a mock float32 tensor."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float32'
        tensor.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        return tensor

    @pytest.fixture
    def float16_tensor(self):
        """Create a mock float16 tensor."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float16'
        tensor.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        return tensor

    @pytest.fixture
    def bfloat16_tensor(self):
        """Create a mock bfloat16 tensor."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'bfloat16'
        # bfloat16 doesn't exist in numpy, so we'll mock the conversion
        tensor.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        return tensor

    def test_get_as_f32_with_float32_tensor(self, mock_model, float32_tensor):
        """
        GIVEN a tensor key that exists and is already float32
        WHEN get_as_f32(key) is called
        THEN expect:
            - Returns numpy array with dtype float32
            - Array values match original tensor values
            - Shape is preserved
        """
        # Arrange
        key = "test_tensor"
        mock_model.tensors = OrderedDict({key: float32_tensor})
        expected_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected_array)
        assert result.shape == (2, 2)

    def test_get_as_f32_with_float16_tensor(self, mock_model, float16_tensor):
        """
        GIVEN a tensor key that exists and is float16
        WHEN get_as_f32(key) is called
        THEN expect:
            - Returns numpy array with dtype float32
            - Array values are converted from float16 to float32
            - Shape is preserved
            - No precision loss beyond float16 limits
        """
        # Arrange
        key = "test_tensor"
        mock_model.tensors = OrderedDict({key: float16_tensor})
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        # Values should be converted to float32
        expected_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected_values, rtol=1e-5)

    def test_get_as_f32_with_bfloat16_tensor(self, mock_model, bfloat16_tensor):
        """
        GIVEN a tensor key that exists and is bfloat16
        WHEN get_as_f32(key) is called
        THEN expect:
            - Returns numpy array with dtype float32
            - Array values are converted from bfloat16 to float32
            - Shape is preserved
        """
        # Arrange
        key = "test_tensor"
        mock_model.tensors = OrderedDict({key: bfloat16_tensor})
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        expected_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_values)

    def test_get_as_f32_with_nonexistent_key(self, mock_model):
        """
        GIVEN a tensor key that does not exist
        WHEN get_as_f32(key) is called
        THEN expect:
            - KeyError is raised
        """
        # Arrange
        mock_model.tensors = OrderedDict()
        nonexistent_key = "nonexistent_tensor"
        
        # Act & Assert
        with pytest.raises(KeyError):
            mock_model.get_as_f32(nonexistent_key)

    def test_get_as_f32_preserves_sd_tensor_structure(self, mock_model):
        """
        GIVEN a typical SD tensor (e.g., UNet attention weights)
        WHEN get_as_f32(key) is called
        THEN expect:
            - Returns numpy array maintaining original structure
            - Squeezed dimensions are reflected in output
            - Array is contiguous in memory
        """
        # Arrange
        key = "model.diffusion_model.input_blocks.0.0.weight"
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.name = 'float32'
        # Simulate a typical UNet weight tensor shape
        original_shape = (320, 4, 3, 3)
        mock_tensor.numpy.return_value = np.random.randn(*original_shape).astype(np.float32)
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == original_shape
        assert result.flags['C_CONTIGUOUS'] or result.flags['F_CONTIGUOUS']

    def test_get_as_f32_memory_efficiency(self, mock_model):
        """
        GIVEN a large tensor
        WHEN get_as_f32(key) is called
        THEN expect:
            - Conversion happens without excessive memory copies
            - Original tensor is not modified
            - Returned array owns its memory
        """
        # Arrange
        key = "large_tensor"
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.name = 'float32'
        large_array = np.random.randn(1000, 1000).astype(np.float32)
        mock_tensor.numpy.return_value = large_array
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1000, 1000)
        assert result.flags['OWNDATA']  # Array owns its memory
        
        # Verify original tensor mock wasn't modified
        mock_tensor.numpy.assert_called_once()

    def test_get_as_f32_with_tensor_conversion_failure(self, mock_model):
        """
        GIVEN a tensor that fails during numpy conversion
        WHEN get_as_f32(key) is called
        THEN expect:
            - ValueError is raised with descriptive message
        """
        # Arrange
        key = "failing_tensor"
        mock_tensor = Mock()
        mock_tensor.numpy.side_effect = RuntimeError("Conversion failed")
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to convert tensor 'failing_tensor' to numpy array"):
            mock_model.get_as_f32(key)

    def test_get_as_f32_with_dtype_conversion_failure(self, mock_model):
        """
        GIVEN a tensor that fails during dtype conversion to float32
        WHEN get_as_f32(key) is called
        THEN expect:
            - ValueError is raised with descriptive message
        """
        # Arrange
        key = "failing_dtype_tensor"
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.name = 'int32'
        
        # Create a mock numpy array that fails on astype
        mock_array = Mock()
        mock_array.dtype = np.int32
        mock_array.astype.side_effect = RuntimeError("Type conversion failed")
        mock_tensor.numpy.return_value = mock_array
        
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to convert tensor 'failing_dtype_tensor' to float32"):
            mock_model.get_as_f32(key)

    def test_get_as_f32_with_empty_tensor(self, mock_model):
        """
        GIVEN a tensor that is empty
        WHEN get_as_f32(key) is called
        THEN expect:
            - Returns empty numpy array with dtype float32
            - Shape is (0,) for 1D empty array
        """
        # Arrange
        key = "empty_tensor"
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.name = 'float32'
        mock_tensor.numpy.return_value = np.array([], dtype=np.float32)
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act
        result = mock_model.get_as_f32(key)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (0,)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])