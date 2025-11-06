#!/usr/bin/env python3
import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import OrderedDict


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


class TestStableDiffusionModelValid:
    """Test StableDiffusionModel valid method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock StableDiffusionModel instance for testing."""
        mock_torch_model = OrderedDict({
            'first_stage_model.encoder.conv_in.weight': Mock(),
            'first_stage_model.decoder.conv_out.weight': Mock(),
            'model.diffusion_model.input_blocks.0.0.weight': Mock(),
            'model.diffusion_model.out.2.weight': Mock(),
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': Mock(),
        })
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = mock_torch_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            # Mock tensor squeezing
            for tensor in mock_torch_model.values():
                tensor.squeeze.return_value = tensor
            
            with patch('builtins.print'):  # Suppress warning prints during testing
                model = StableDiffusionModel("dummy_path.ckpt")
                return model

    @pytest.fixture
    def float32_tensor(self):
        """Create a mock float32 tensor with 2 dimensions."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float32'
        tensor.ndim = 2
        tensor.shape = (128, 256)
        return tensor

    @pytest.fixture
    def float16_tensor(self):
        """Create a mock float16 tensor with 2 dimensions."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float16'
        tensor.ndim = 2
        tensor.shape = (64, 128)
        return tensor

    @pytest.fixture
    def bfloat16_tensor(self):
        """Create a mock bfloat16 tensor with 2 dimensions."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'bfloat16'
        tensor.ndim = 2
        tensor.shape = (32, 64)
        return tensor

    @pytest.fixture
    def invalid_dtype_tensor(self):
        """Create a mock tensor with invalid dtype."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'int32'
        tensor.ndim = 2
        tensor.shape = (10, 10)
        return tensor

    @pytest.fixture
    def high_dim_tensor(self):
        """Create a mock tensor with 4 dimensions (valid for SD models)."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float32'
        tensor.ndim = 4
        tensor.shape = (1, 3, 224, 224)
        return tensor

    @pytest.fixture
    def very_high_dim_tensor(self):
        """Create a mock tensor with more than 4 dimensions (invalid)."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = 'float32'
        tensor.ndim = 5
        tensor.shape = (1, 3, 3, 224, 224)
        return tensor

    def test_valid_with_existing_valid_tensor(self, mock_model, float32_tensor):
        """
        GIVEN a tensor key that exists and has valid properties:
            - dtype is float32, float16, or bfloat16
            - dimensions <= 4 (for SD models)
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "valid_tensor"
        mock_model.tensors = OrderedDict({key: float32_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] == "OK"

    def test_valid_with_float16_tensor(self, mock_model, float16_tensor):
        """
        GIVEN a tensor key that exists and is float16 with valid dimensions
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "float16_tensor"
        mock_model.tensors = OrderedDict({key: float16_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert result == (True, "OK")

    def test_valid_with_bfloat16_tensor(self, mock_model, bfloat16_tensor):
        """
        GIVEN a tensor key that exists and is bfloat16 with valid dimensions
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "bfloat16_tensor"
        mock_model.tensors = OrderedDict({key: bfloat16_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert result == (True, "OK")

    def test_valid_with_nonexistent_tensor(self, mock_model):
        """
        GIVEN a tensor key that does not exist in the model
        WHEN valid(key) is called
        THEN expect:
            - Returns (False, "Tensor not found")
        """
        # Arrange
        mock_model.tensors = OrderedDict()
        nonexistent_key = "nonexistent_tensor"
        
        # Act
        result = mock_model.valid(nonexistent_key)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is False
        assert result[1] == "Tensor not found"

    def test_valid_with_invalid_dtype(self, mock_model, invalid_dtype_tensor):
        """
        GIVEN a tensor key that exists but has invalid dtype:
            - Not float32, float16, or bfloat16
            - e.g., int32, int64, float64, etc.
        WHEN valid(key) is called
        THEN expect:
            - Returns (False, "Unhandled type")
        """
        # Arrange
        key = "invalid_dtype_tensor"
        mock_model.tensors = OrderedDict({key: invalid_dtype_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is False
        assert result[1] == "Unhandled type"

    def test_valid_with_high_dimensional_tensor(self, mock_model, high_dim_tensor):
        """
        GIVEN a tensor key that exists with 4 dimensions (valid for SD models):
            - dtype is float32, float16, or bfloat16
            - dimensions = 4 (common for conv layers in UNet)
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "high_dim_tensor"
        mock_model.tensors = OrderedDict({key: high_dim_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] == "OK"

    def test_valid_with_nan_values(self, mock_model):
        """
        GIVEN a tensor that contains NaN values
        WHEN valid(key) is called
        THEN expect:
            - Either returns (False, "Contains NaN values") if checking for NaN
            - Or proceeds with dtype/dimension checks only
        """
        # Arrange
        key = "nan_tensor"
        nan_tensor = Mock()
        nan_tensor.dtype = Mock()
        nan_tensor.dtype.name = 'float32'
        nan_tensor.ndim = 2
        nan_tensor.shape = (10, 10)
        
        # Mock numpy array with NaN values
        mock_array = Mock()
        mock_array.dtype = np.float32
        mock_array.shape = (10, 10)
        nan_tensor.numpy.return_value = mock_array
        
        # Mock np.isnan to return True for NaN check
        with patch('numpy.isnan') as mock_isnan:
            with patch('numpy.any') as mock_any:
                mock_isnan.return_value = Mock()
                mock_any.return_value = True
                
                mock_model.tensors = OrderedDict({key: nan_tensor})
                
                # Act
                result = mock_model.valid(key)
                
                # Assert
                # This test depends on implementation - it might check for NaN or not
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], bool)
                assert isinstance(result[1], str)
                
                # If NaN checking is implemented, expect failure
                if result[0] is False and "NaN" in result[1]:
                    assert result[1] == "Contains NaN values"
                else:
                    # If NaN checking is not implemented, should pass dtype/dimension checks
                    assert result == (True, "OK")

    def test_valid_with_inf_values(self, mock_model):
        """
        GIVEN a tensor that contains Inf values
        WHEN valid(key) is called
        THEN expect:
            - Either returns (False, "Contains Inf values") if checking for Inf
            - Or proceeds with dtype/dimension checks only
        """
        # Arrange
        key = "inf_tensor"
        inf_tensor = Mock()
        inf_tensor.dtype = Mock()
        inf_tensor.dtype.name = 'float32'
        inf_tensor.ndim = 2
        inf_tensor.shape = (10, 10)
        
        # Mock numpy array with Inf values
        mock_array = Mock()
        mock_array.dtype = np.float32
        mock_array.shape = (10, 10)
        inf_tensor.numpy.return_value = mock_array
        
        # Mock np.isinf to return True for Inf check
        with patch('numpy.isinf') as mock_isinf:
            with patch('numpy.any') as mock_any:
                mock_isinf.return_value = Mock()
                mock_any.return_value = True
                
                mock_model.tensors = OrderedDict({key: inf_tensor})
                
                # Act
                result = mock_model.valid(key)
                
                # Assert
                # This test depends on implementation - it might check for Inf or not
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], bool)
                assert isinstance(result[1], str)
                
                # If Inf checking is implemented, expect failure
                if result[0] is False and "Inf" in result[1]:
                    assert result[1] == "Contains Inf values"
                else:
                    # If Inf checking is not implemented, should pass dtype/dimension checks
                    assert result == (True, "OK")

    def test_valid_with_1d_tensor(self, mock_model):
        """
        GIVEN a tensor key that exists with 1 dimension (valid)
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "1d_tensor"
        tensor_1d = Mock()
        tensor_1d.dtype = Mock()
        tensor_1d.dtype.name = 'float32'
        tensor_1d.ndim = 1
        tensor_1d.shape = (256,)
        
        mock_model.tensors = OrderedDict({key: tensor_1d})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert result == (True, "OK")

    def test_valid_with_0d_tensor(self, mock_model):
        """
        GIVEN a tensor key that exists with 0 dimensions (scalar)
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "0d_tensor"
        tensor_0d = Mock()
        tensor_0d.dtype = Mock()
        tensor_0d.dtype.name = 'float32'
        tensor_0d.ndim = 0
        tensor_0d.shape = ()
        
        mock_model.tensors = OrderedDict({key: tensor_0d})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert result == (True, "OK")

    def test_valid_with_edge_case_dtypes(self, mock_model):
        """
        GIVEN tensors with edge case dtypes
        WHEN valid(key) is called
        THEN expect:
            - Proper handling of various dtype edge cases
        """
        # Test cases for various dtypes
        test_cases = [
            ('float64', False, "Unhandled type"),
            ('int8', False, "Unhandled type"),
            ('int16', False, "Unhandled type"),
            ('int64', False, "Unhandled type"),
            ('uint8', False, "Unhandled type"),
            ('bool', False, "Unhandled type"),
            ('complex64', False, "Unhandled type"),
        ]
        
        for dtype_name, expected_valid, expected_message in test_cases:
            # Arrange
            key = f"tensor_{dtype_name}"
            tensor = Mock()
            tensor.dtype = Mock()
            tensor.dtype.name = dtype_name
            tensor.ndim = 2
            tensor.shape = (10, 10)
            
            mock_model.tensors = OrderedDict({key: tensor})
            
            # Act
            result = mock_model.valid(key)
            
            # Assert
            assert result == (expected_valid, expected_message), f"Failed for dtype {dtype_name}"

    def test_valid_with_3d_tensor_boundary(self, mock_model):
        """
        GIVEN a tensor with exactly 3 dimensions (valid for SD models)
        WHEN valid(key) is called
        THEN expect:
            - Returns (True, "OK")
        """
        # Arrange
        key = "3d_tensor"
        tensor_3d = Mock()
        tensor_3d.dtype = Mock()
        tensor_3d.dtype.name = 'float32'
        tensor_3d.ndim = 3
        tensor_3d.shape = (10, 10, 10)
        
        mock_model.tensors = OrderedDict({key: tensor_3d})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert result == (True, "OK")

    def test_valid_with_very_high_dimensional_tensor(self, mock_model, very_high_dim_tensor):
        """
        GIVEN a tensor key that exists with more than 4 dimensions (invalid for SD models)
        WHEN valid(key) is called
        THEN expect:
            - Returns (False, "Unhandled dimensions")
        """
        # Arrange
        key = "very_high_dim_tensor"
        mock_model.tensors = OrderedDict({key: very_high_dim_tensor})
        
        # Act
        result = mock_model.valid(key)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is False
        assert result[1] == "Unhandled dimensions"

    def test_valid_return_type_consistency(self, mock_model):
        """
        GIVEN various tensor scenarios
        WHEN valid(key) is called
        THEN expect:
            - Always returns tuple[bool, str]
            - First element is always boolean
            - Second element is always string
        """
        # Test with valid tensor
        valid_tensor = Mock()
        valid_tensor.dtype = Mock()
        valid_tensor.dtype.name = 'float32'
        valid_tensor.ndim = 2
        
        mock_model.tensors = OrderedDict({'valid': valid_tensor})
        
        # Act & Assert for valid tensor
        result = mock_model.valid('valid')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        
        # Act & Assert for nonexistent tensor
        result = mock_model.valid('nonexistent')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])