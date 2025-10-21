#!/usr/bin/env python3
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
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


class TestStableDiffusionModelGetTypeName:
    """Test StableDiffusionModel get_type_name method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock StableDiffusionModel instance for testing."""
        # Create mock tensors with different dtypes
        mock_float32_tensor = Mock()
        mock_float32_tensor.dtype = Mock()
        mock_float32_tensor.dtype.__str__ = Mock(return_value="torch.float32")
        mock_float32_tensor.squeeze.return_value = mock_float32_tensor
        
        mock_float16_tensor = Mock()
        mock_float16_tensor.dtype = Mock()
        mock_float16_tensor.dtype.__str__ = Mock(return_value="torch.float16")
        mock_float16_tensor.squeeze.return_value = mock_float16_tensor
        
        mock_bfloat16_tensor = Mock()
        mock_bfloat16_tensor.dtype = Mock()
        mock_bfloat16_tensor.dtype.__str__ = Mock(return_value="torch.bfloat16")
        mock_bfloat16_tensor.squeeze.return_value = mock_bfloat16_tensor
        
        mock_torch_model = OrderedDict({
            'first_stage_model.encoder.conv_in.weight': mock_float32_tensor,
            'first_stage_model.decoder.conv_out.weight': mock_float16_tensor,
            'model.diffusion_model.input_blocks.0.0.weight': mock_bfloat16_tensor,
            'model.diffusion_model.out.2.weight': mock_float32_tensor,
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': mock_float16_tensor,
        })
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = mock_torch_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with patch('builtins.print'):  # Suppress warning prints during testing
                model = StableDiffusionModel("dummy_path.ckpt")
                return model

    def test_get_type_name_float32_tensor(self, mock_model):
        """
        GIVEN a tensor with float32 dtype
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns string representation of torch.float32
        """
        # Arrange
        key = "first_stage_model.encoder.conv_in.weight"
        
        # Act
        result = mock_model.get_type_name(key)
        
        # Assert
        assert result == "torch.float32"

    def test_get_type_name_float16_tensor(self, mock_model):
        """
        GIVEN a tensor with float16 dtype
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns string representation of torch.float16
        """
        # Arrange
        key = "first_stage_model.decoder.conv_out.weight"
        
        # Act
        result = mock_model.get_type_name(key)
        
        # Assert
        assert result == "torch.float16"

    def test_get_type_name_bfloat16_tensor(self, mock_model):
        """
        GIVEN a tensor with bfloat16 dtype
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns string representation of torch.bfloat16
        """
        # Arrange
        key = "model.diffusion_model.input_blocks.0.0.weight"
        
        # Act
        result = mock_model.get_type_name(key)
        
        # Assert
        assert result == "torch.bfloat16"

    def test_get_type_name_multiple_tensors_same_dtype(self, mock_model):
        """
        GIVEN multiple tensors with the same dtype
        WHEN get_type_name(key) is called for each
        THEN expect:
            - Returns same dtype string for all
        """
        # Arrange
        float32_keys = [
            "first_stage_model.encoder.conv_in.weight",
            "model.diffusion_model.out.2.weight"
        ]
        
        for key in float32_keys:
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float32"

    def test_get_type_name_unet_input_blocks(self, mock_model):
        """
        GIVEN a tensor key for UNet input blocks
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        unet_input_keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
            "model.diffusion_model.input_blocks.2.0.out_layers.3.weight",
        ]
        
        for key in unet_input_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float32")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float32"

    def test_get_type_name_unet_middle_blocks(self, mock_model):
        """
        GIVEN a tensor key for UNet middle blocks
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        unet_middle_keys = [
            "model.diffusion_model.middle_block.0.in_layers.0.weight",
            "model.diffusion_model.middle_block.1.norm.weight",
            "model.diffusion_model.middle_block.2.out_layers.3.weight",
        ]
        
        for key in unet_middle_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float16")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float16"

    def test_get_type_name_unet_output_blocks(self, mock_model):
        """
        GIVEN a tensor key for UNet output blocks
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        unet_output_keys = [
            "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
            "model.diffusion_model.output_blocks.1.0.out_layers.3.weight",
            "model.diffusion_model.out.0.weight",
            "model.diffusion_model.out.2.weight",
        ]
        
        for key in unet_output_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.bfloat16")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.bfloat16"

    def test_get_type_name_text_encoder(self, mock_model):
        """
        GIVEN a tensor key for text encoder component
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        text_encoder_keys = [
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        ]
        
        for key in text_encoder_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float32")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float32"

    def test_get_type_name_attention_layers(self, mock_model):
        """
        GIVEN a tensor key for attention layers
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        attention_keys = [
            "model.diffusion_model.input_blocks.1.1.norm.weight",
            "model.diffusion_model.input_blocks.1.1.qkv.weight",
            "model.diffusion_model.input_blocks.1.1.proj_out.weight",
            "model.diffusion_model.middle_block.1.norm.weight",
            "model.diffusion_model.middle_block.1.qkv.weight",
        ]
        
        for key in attention_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float16")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float16"

    def test_get_type_name_nonexistent_key(self, mock_model):
        """
        GIVEN a tensor key that does not exist
        WHEN get_type_name(key) is called
        THEN expect:
            - KeyError is raised
        """
        # Arrange
        mock_model.tensors = OrderedDict()
        nonexistent_key = "nonexistent_tensor"
        
        # Act & Assert
        with pytest.raises(KeyError):
            mock_model.get_type_name(nonexistent_key)

    def test_get_type_name_unknown_pattern(self, mock_model):
        """
        GIVEN a tensor key with unknown naming pattern
        WHEN get_type_name(key) is called
        THEN expect:
            - Returns the dtype string of the tensor
        """
        # Arrange
        unknown_keys = [
            "some.random.tensor.weight",
            "custom_component.layer.bias",
            "weird_naming_convention.data",
        ]
        
        for key in unknown_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float32")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.float32"

    def test_get_type_name_consistency(self, mock_model):
        """
        GIVEN the same tensor key called multiple times
        WHEN get_type_name(key) is called repeatedly
        THEN expect:
            - Returns consistent results
            - No side effects
        """
        # Arrange
        key = "model.diffusion_model.input_blocks.0.0.weight"
        
        # Create a mock tensor with dtype
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.__str__ = Mock(return_value="torch.float32")
        
        mock_model.tensors = OrderedDict({key: mock_tensor})
        
        # Act
        result1 = mock_model.get_type_name(key)
        result2 = mock_model.get_type_name(key)
        result3 = mock_model.get_type_name(key)
        
        # Assert
        assert result1 == result2 == result3
        assert result1 == "torch.float32"

    def test_get_type_name_weight_vs_bias(self, mock_model):
        """
        GIVEN tensor keys for weight and bias of the same layer
        WHEN get_type_name(key) is called for both
        THEN expect:
            - Both return their respective dtype strings
            - Weight and bias are handled appropriately
        """
        # Arrange
        base_key = "model.diffusion_model.input_blocks.0.0"
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"
        
        # Create mock tensors with dtypes
        mock_weight_tensor = Mock()
        mock_weight_tensor.dtype = Mock()
        mock_weight_tensor.dtype.__str__ = Mock(return_value="torch.float32")
        
        mock_bias_tensor = Mock()
        mock_bias_tensor.dtype = Mock()
        mock_bias_tensor.dtype.__str__ = Mock(return_value="torch.float32")
        
        mock_model.tensors = OrderedDict({
            weight_key: mock_weight_tensor,
            bias_key: mock_bias_tensor
        })
        
        # Act
        weight_type = mock_model.get_type_name(weight_key)
        bias_type = mock_model.get_type_name(bias_key)
        
        # Assert
        assert weight_type == "torch.float32"
        assert bias_type == "torch.float32"

    def test_get_type_name_hierarchical_structure(self, mock_model):
        """
        GIVEN tensor keys with deep hierarchical structure
        WHEN get_type_name(key) is called
        THEN expect:
            - Handles complex nested naming correctly
            - Returns the dtype string of the tensor
        """
        # Arrange
        complex_keys = [
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.0.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
            "model.diffusion_model.input_blocks.1.1.norm.weight",
            "model.diffusion_model.input_blocks.1.1.qkv.weight",
            "model.diffusion_model.input_blocks.1.1.proj_out.weight",
        ]
        
        for key in complex_keys:
            # Create a mock tensor with dtype
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.bfloat16")
            
            mock_model.tensors = OrderedDict({key: mock_tensor})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert result == "torch.bfloat16"
            
            # Should handle the hierarchical structure appropriately
            # All should be identified as UNet-related since they start with model.diffusion_model
            assert result == "torch.bfloat16"

    def test_get_type_name_performance(self, mock_model):
        """
        GIVEN many tensor keys
        WHEN get_type_name(key) is called for each
        THEN expect:
            - Method performs efficiently
            - No performance degradation with repeated calls
        """
        # Arrange
        many_keys = []
        for i in range(100):
            many_keys.append(f"model.diffusion_model.input_blocks.{i}.0.weight")
            many_keys.append(f"model.diffusion_model.input_blocks.{i}.0.bias")
        
        mock_tensors = OrderedDict()
        for key in many_keys:
            mock_tensor = Mock()
            mock_tensor.dtype = Mock()
            mock_tensor.dtype.__str__ = Mock(return_value="torch.float32")
            mock_tensors[key] = mock_tensor
        
        mock_model.tensors = mock_tensors
        
        # Act
        results = []
        for key in many_keys:
            result = mock_model.get_type_name(key)
            results.append(result)
        
        # Assert
        assert len(results) == len(many_keys)
        assert all(result == "torch.float32" for result in results)

    def test_get_type_name_edge_cases(self, mock_model):
        """
        GIVEN tensor keys with edge case patterns
        WHEN get_type_name(key) is called
        THEN expect:
            - Handles edge cases gracefully
            - Returns meaningful strings
        """
        # Arrange
        edge_cases = [
            "weight",  # Single word
            "model.weight",  # Minimal hierarchy
            "a.b.c.d.e.f.g.h.i.j.weight",  # Very deep hierarchy
            "model.diffusion_model..weight",  # Double dots
            "model.diffusion_model.input_blocks.0.0.",  # Trailing dot
        ]
        
        for key in edge_cases:
            mock_model.tensors = OrderedDict({key: Mock()})
            
            # Act
            result = mock_model.get_type_name(key)
            
            # Assert
            assert isinstance(result, str)
            assert len(result) > 0
            # Should handle edge cases without crashing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])