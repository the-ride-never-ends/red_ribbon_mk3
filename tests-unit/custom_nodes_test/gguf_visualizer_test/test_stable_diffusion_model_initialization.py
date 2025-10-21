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





class TestStableDiffusionModelInitialization:
    """Test StableDiffusionModel initialization and configuration."""

    @pytest.fixture
    def mock_torch_model(self):
        """Create a mock torch model with typical SD structure."""
        return OrderedDict({
            'first_stage_model.encoder.conv_in.weight': Mock(),
            'first_stage_model.decoder.conv_out.weight': Mock(),
            'model.diffusion_model.input_blocks.0.0.weight': Mock(),
            'model.diffusion_model.out.2.weight': Mock(),
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': Mock(),
        })

    @pytest.fixture
    def mock_path(self):
        """Create a mock file path."""
        return Path("/path/to/model.ckpt")

    def test_init_with_valid_sd_checkpoint(self, mock_torch_model, mock_path):
        """
        GIVEN a valid Stable Diffusion checkpoint file path
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - Instance created successfully
            - torch module is imported and stored
            - model is loaded using torch.load with correct parameters
            - tensors OrderedDict is populated with squeezed tensors
            - All SD components are checked and warnings logged for missing ones
            - No exceptions raised
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            # Mock the torch import
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
            
            # Act
            model = StableDiffusionModel(mock_path)
            
            # Assert
            assert isinstance(model, StableDiffusionModel)
            assert hasattr(model, 'model')
            assert hasattr(model, 'tensors')
            assert isinstance(model.tensors, OrderedDict)
            
            # Verify torch.load was called correctly
            mock_torch.load.assert_called_once_with(mock_path, map_location='cpu')
            
            # Verify tensors were squeezed
            for tensor in mock_torch_model.values():
                tensor.squeeze.assert_called_once()

    def test_init_with_missing_torch_import(self, mock_path):
        """
        GIVEN PyTorch is not installed
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - ImportError is caught
            - Error message logged: "! Loading Stable Diffusion models requires the Torch Python module"
            - sys.exit(1) is called
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    raise ImportError("No module named 'torch'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with patch('builtins.print') as mock_print:
                    
                    # Act & Assert
                    with pytest.raises(SystemExit):
                        StableDiffusionModel(mock_path)
                    
                    mock_print.assert_called_with("! Loading Stable Diffusion models requires the Torch Python module")
                    mock_exit.assert_called_once_with(1)

    def test_init_with_nonexistent_file(self, mock_path):
        """
        GIVEN a file path that does not exist
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - FileNotFoundError or appropriate torch loading error
            - Error is logged
            - sys.exit(1) is called
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.side_effect = FileNotFoundError("No such file or directory")
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with patch('builtins.print') as mock_print:
                    
                    # Act & Assert
                    with pytest.raises(SystemExit):
                        StableDiffusionModel(mock_path)
                    
                    mock_exit.assert_called_once_with(1)
                    # Verify some error message was printed
                    assert mock_print.call_count > 0

    def test_init_with_corrupted_checkpoint(self, mock_path):
        """
        GIVEN a corrupted checkpoint file that cannot be loaded by torch
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - Exception from torch.load is caught
            - Error is logged
            - sys.exit(1) is called
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.side_effect = RuntimeError("Corrupted checkpoint")
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with patch('builtins.print') as mock_print:
                    
                    # Act & Assert
                    with pytest.raises(SystemExit):
                        StableDiffusionModel(mock_path)
                    
                    mock_exit.assert_called_once_with(1)
                    assert mock_print.call_count > 0

    def test_init_with_missing_sd_components(self, mock_path):
        """
        GIVEN a checkpoint with some SD components missing (e.g., no VAE)
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - Model loads successfully
            - Warning logged for each missing component
            - Available components are accessible
            - No exceptions raised
        """
        # Arrange - checkpoint with only UNet, no VAE or text encoder
        incomplete_model = OrderedDict({
            'model.diffusion_model.input_blocks.0.0.weight': Mock(),
            'model.diffusion_model.out.2.weight': Mock(),
        })
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = incomplete_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            for tensor in incomplete_model.values():
                tensor.squeeze.return_value = tensor
            
            with patch('builtins.print') as mock_print:
                
                # Act
                model = StableDiffusionModel(mock_path)
                
                # Assert
                assert isinstance(model, StableDiffusionModel)
                assert isinstance(model.tensors, OrderedDict)
                
                # Verify warnings were logged for missing components
                warning_calls = [call for call in mock_print.call_args_list if "No" in str(call) and "found" in str(call)]
                assert len(warning_calls) > 0  # Should have warnings for missing VAE, text encoder, etc.

    def test_init_with_invalid_checkpoint_structure(self, mock_path):
        """
        GIVEN a checkpoint that is not a dict/OrderedDict
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - Error logged about invalid checkpoint structure
            - sys.exit(1) is called
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = "not_a_dict"  # Invalid structure
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with patch('builtins.print') as mock_print:
                    
                    # Act & Assert
                    with pytest.raises(SystemExit):
                        StableDiffusionModel(mock_path)
                    
                    mock_exit.assert_called_once_with(1)
                    # Should have some error message about invalid structure
                    assert mock_print.call_count > 0

    def test_init_stores_model_reference(self, mock_torch_model, mock_path):
        """
        GIVEN a valid checkpoint
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - model attribute stores the loaded checkpoint
            - tensors attribute is an OrderedDict
            - All expected attributes are present
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = mock_torch_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            for tensor in mock_torch_model.values():
                tensor.squeeze.return_value = tensor
            
            # Act
            model = StableDiffusionModel(mock_path)
            
            # Assert
            assert model.model is mock_torch_model
            assert isinstance(model.tensors, OrderedDict)
            assert len(model.tensors) == len(mock_torch_model)

    def test_init_with_string_path(self, mock_torch_model):
        """
        GIVEN a string file path instead of Path object
        WHEN StableDiffusionModel is initialized
        THEN expect:
            - String path is accepted
            - Model loads successfully
        """
        # Arrange
        string_path = "/path/to/model.ckpt"
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.load.return_value = mock_torch_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            for tensor in mock_torch_model.values():
                tensor.squeeze.return_value = tensor
            
            # Act
            model = StableDiffusionModel(string_path)
            
            # Assert
            assert isinstance(model, StableDiffusionModel)
            mock_torch.load.assert_called_once_with(string_path, map_location='cpu')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])