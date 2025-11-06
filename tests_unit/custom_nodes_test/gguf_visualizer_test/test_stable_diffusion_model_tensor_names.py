#!/usr/bin/env python3
import pytest
from unittest.mock import Mock, patch
from collections import OrderedDict
from typing import Iterable


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


class TestStableDiffusionModelTensorNames:
    """Test StableDiffusionModel tensor_names method."""

    @pytest.fixture
    def mock_model_with_tensors(self):
        """Create a mock StableDiffusionModel instance with typical tensors."""
        mock_torch_model = OrderedDict([
            ('first_stage_model.encoder.conv_in.weight', Mock()),
            ('first_stage_model.encoder.conv_in.bias', Mock()),
            ('first_stage_model.decoder.conv_out.weight', Mock()),
            ('model.diffusion_model.input_blocks.0.0.weight', Mock()),
            ('model.diffusion_model.input_blocks.0.0.bias', Mock()),
            ('model.diffusion_model.out.2.weight', Mock()),
            ('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', Mock()),
        ])
        
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
    def empty_model(self):
        """Create a mock StableDiffusionModel instance with no tensors."""
        mock_torch_model = OrderedDict()
        
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

    def test_tensor_names_returns_all_keys(self, mock_model_with_tensors):
        """
        GIVEN a StableDiffusionModel instance with loaded tensors
        WHEN tensor_names() is called
        THEN expect:
            - Returns an Iterable of all tensor keys
            - All keys from the tensors OrderedDict are included
            - Order is preserved from the OrderedDict
        """
        # Arrange
        expected_keys = [
            'first_stage_model.encoder.conv_in.weight',
            'first_stage_model.encoder.conv_in.bias',
            'first_stage_model.decoder.conv_out.weight',
            'model.diffusion_model.input_blocks.0.0.weight',
            'model.diffusion_model.input_blocks.0.0.bias',
            'model.diffusion_model.out.2.weight',
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight',
        ]
        
        # Act
        result = mock_model_with_tensors.tensor_names()
        
        # Assert
        assert isinstance(result, Iterable)
        result_list = list(result)
        assert result_list == expected_keys
        assert len(result_list) == len(expected_keys)

    def test_tensor_names_empty_model(self, empty_model):
        """
        GIVEN a StableDiffusionModel instance with no tensors
        WHEN tensor_names() is called
        THEN expect:
            - Returns an empty Iterable
            - No exceptions raised
        """
        # Act
        result = empty_model.tensor_names()
        
        # Assert
        assert isinstance(result, Iterable)
        result_list = list(result)
        assert result_list == []
        assert len(result_list) == 0

    def test_tensor_names_sd_specific_components(self, mock_model_with_tensors):
        """
        GIVEN a StableDiffusionModel with typical SD components:
            - VAE tensors (first_stage_model.*)
            - UNet tensors (model.diffusion_model.*)
            - Text encoder tensors (cond_stage_model.*)
        WHEN tensor_names() is called
        THEN expect:
            - All component-specific tensor names are included
            - Names follow SD naming conventions
        """
        # Act
        result = mock_model_with_tensors.tensor_names()
        result_list = list(result)
        
        # Assert - Check for VAE components
        vae_tensors = [name for name in result_list if name.startswith('first_stage_model.')]
        assert len(vae_tensors) > 0
        assert 'first_stage_model.encoder.conv_in.weight' in vae_tensors
        assert 'first_stage_model.decoder.conv_out.weight' in vae_tensors
        
        # Assert - Check for UNet components
        unet_tensors = [name for name in result_list if name.startswith('model.diffusion_model.')]
        assert len(unet_tensors) > 0
        assert 'model.diffusion_model.input_blocks.0.0.weight' in unet_tensors
        assert 'model.diffusion_model.out.2.weight' in unet_tensors
        
        # Assert - Check for text encoder components
        text_encoder_tensors = [name for name in result_list if name.startswith('cond_stage_model.')]
        assert len(text_encoder_tensors) > 0
        assert 'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight' in text_encoder_tensors

    def test_tensor_names_returns_iterable_interface(self, mock_model_with_tensors):
        """
        GIVEN a StableDiffusionModel instance
        WHEN tensor_names() is called
        THEN expect:
            - Result can be iterated multiple times
            - Result supports len() if it's a collection
            - Result behaves like an Iterable[str]
        """
        # Act
        result = mock_model_with_tensors.tensor_names()
        
        # Assert - Can be iterated
        first_iteration = list(result)
        assert all(isinstance(name, str) for name in first_iteration)
        
        # Assert - Can be iterated again (if supported)
        try:
            second_iteration = list(result)
            assert first_iteration == second_iteration
        except TypeError:
            # If it's a generator, that's acceptable too
            pass

    def test_tensor_names_order_preservation(self):
        """
        GIVEN a StableDiffusionModel with tensors in a specific order
        WHEN tensor_names() is called
        THEN expect:
            - Order matches the OrderedDict order
            - First tensor name is the first added
            - Last tensor name is the last added
        """
        # Arrange
        mock_torch_model = OrderedDict([
            ('z_tensor', Mock()),
            ('a_tensor', Mock()),
            ('m_tensor', Mock()),
        ])
        
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
        
        # Act
        result = list(model.tensor_names())
        
        # Assert
        expected_order = ['z_tensor', 'a_tensor', 'm_tensor']
        assert result == expected_order

    def test_tensor_names_with_complex_sd_hierarchy(self):
        """
        GIVEN a StableDiffusionModel with complex nested tensor names
        WHEN tensor_names() is called
        THEN expect:
            - All nested component names are preserved
            - Hierarchical structure is maintained in names
            - No truncation of long names
        """
        # Arrange
        complex_names = [
            'model.diffusion_model.input_blocks.0.0.weight',
            'model.diffusion_model.input_blocks.1.0.in_layers.0.weight',
            'model.diffusion_model.input_blocks.1.0.in_layers.2.weight',
            'model.diffusion_model.input_blocks.1.0.out_layers.0.weight',
            'model.diffusion_model.input_blocks.1.0.out_layers.3.weight',
            'model.diffusion_model.input_blocks.1.1.norm.weight',
            'model.diffusion_model.input_blocks.1.1.qkv.weight',
            'model.diffusion_model.input_blocks.1.1.proj_out.weight',
        ]
        
        mock_torch_model = OrderedDict([(name, Mock()) for name in complex_names])
        
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
        
        # Act
        result = list(model.tensor_names())
        
        # Assert
        assert result == complex_names
        assert all(len(name) > 20 for name in result)  # These are long hierarchical names
        assert all('.' in name for name in result)  # All should have hierarchical structure

    def test_tensor_names_performance_with_large_model(self):
        """
        GIVEN a StableDiffusionModel with many tensors (simulating large model)
        WHEN tensor_names() is called
        THEN expect:
            - Method completes efficiently
            - Returns all tensor names
            - No performance degradation
        """
        # Arrange
        # Create many tensors
        large_tensor_dict = OrderedDict()
        for i in range(1000):
            large_tensor_dict[f'layer_{i:03d}.weight'] = Mock()
            large_tensor_dict[f'layer_{i:03d}.bias'] = Mock()
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = large_tensor_dict
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            # Mock tensor squeezing
            for tensor in large_tensor_dict.values():
                tensor.squeeze.return_value = tensor
            
            with patch('builtins.print'):  # Suppress warning prints during testing
                model = StableDiffusionModel("dummy_path.ckpt")
        
        # Act
        result = list(model.tensor_names())
        
        # Assert
        assert len(result) == 2000  # 1000 weights + 1000 biases
        assert result[0] == 'layer_000.weight'
        assert result[1] == 'layer_000.bias'
        assert result[-2] == 'layer_999.weight'
        assert result[-1] == 'layer_999.bias'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])