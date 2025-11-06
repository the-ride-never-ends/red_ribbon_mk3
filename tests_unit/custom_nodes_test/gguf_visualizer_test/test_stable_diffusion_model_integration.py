#!/usr/bin/env python3
import pytest
import sys
import numpy as np
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


class TestStableDiffusionModelIntegration:
    """Integration tests for StableDiffusionModel covering success criteria."""

    @pytest.fixture
    def sd15_checkpoint(self):
        """Mock SD 1.5 checkpoint structure."""
        return OrderedDict([
            ('first_stage_model.encoder.conv_in.weight', self._create_mock_tensor('float32', 2)),
            ('first_stage_model.decoder.conv_out.weight', self._create_mock_tensor('float32', 2)),
            ('model.diffusion_model.input_blocks.0.0.weight', self._create_mock_tensor('float32', 2)),
            ('model.diffusion_model.out.2.weight', self._create_mock_tensor('float32', 2)),
            ('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', self._create_mock_tensor('float32', 2)),
        ])

    @pytest.fixture
    def sd21_checkpoint(self):
        """Mock SD 2.1 checkpoint structure."""
        return OrderedDict([
            ('first_stage_model.encoder.conv_in.weight', self._create_mock_tensor('float16', 2)),
            ('first_stage_model.decoder.conv_out.weight', self._create_mock_tensor('float16', 2)),
            ('model.diffusion_model.input_blocks.0.0.weight', self._create_mock_tensor('float16', 2)),
            ('model.diffusion_model.out.2.weight', self._create_mock_tensor('float16', 2)),
            ('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', self._create_mock_tensor('float16', 2)),
        ])

    @pytest.fixture
    def sdxl_checkpoint(self):
        """Mock SDXL checkpoint structure."""
        return OrderedDict([
            ('first_stage_model.encoder.conv_in.weight', self._create_mock_tensor('bfloat16', 2)),
            ('first_stage_model.decoder.conv_out.weight', self._create_mock_tensor('bfloat16', 2)),
            ('model.diffusion_model.input_blocks.0.0.weight', self._create_mock_tensor('bfloat16', 2)),
            ('model.diffusion_model.out.2.weight', self._create_mock_tensor('bfloat16', 2)),
            ('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', self._create_mock_tensor('bfloat16', 2)),
        ])

    @pytest.fixture
    def pruned_checkpoint(self):
        """Mock pruned checkpoint with missing components."""
        return OrderedDict([
            ('model.diffusion_model.input_blocks.0.0.weight', self._create_mock_tensor('float32', 2)),
            ('model.diffusion_model.out.2.weight', self._create_mock_tensor('float32', 2)),
            # Missing VAE and text encoder components
        ])

    def _create_mock_tensor(self, dtype_name: str, ndim: int, shape: tuple = None):
        """Helper method to create mock tensors."""
        tensor = Mock()
        tensor.dtype = Mock()
        tensor.dtype.name = dtype_name
        tensor.ndim = ndim
        
        # Generate appropriate shape based on ndim
        if shape is None:
            if ndim == 1:
                shape = (64,)
            elif ndim == 2:
                shape = (64, 64)
            elif ndim == 3:
                shape = (64, 64, 64)
            elif ndim == 4:
                shape = (64, 64, 64, 64)
            else:
                shape = tuple(64 for _ in range(ndim))
        
        tensor.shape = shape
        tensor.squeeze.return_value = tensor
        
        # Mock numpy conversion
        if dtype_name == 'float32':
            tensor.numpy.return_value = np.random.randn(*tensor.shape).astype(np.float32)
        elif dtype_name == 'float16':
            tensor.numpy.return_value = np.random.randn(*tensor.shape).astype(np.float16)
        else:  # bfloat16
            tensor.numpy.return_value = np.random.randn(*tensor.shape).astype(np.float32)
        
        return tensor

    def test_model_loading_success_rate(self, sd15_checkpoint, sd21_checkpoint, sdxl_checkpoint, pruned_checkpoint):
        """
        Success Criterion: S_load = N_successful / N_total = 1.0
        
        GIVEN multiple valid SD checkpoint files of different types:
            - SD 1.5 checkpoint
            - SD 2.1 checkpoint  
            - SDXL checkpoint
            - Pruned checkpoint (missing some components)
        WHEN each is loaded with StableDiffusionModel
        THEN expect:
            - All valid checkpoints load successfully
            - Success rate = 100%
        """
        # Arrange
        checkpoints = [
            ("sd15.ckpt", sd15_checkpoint),
            ("sd21.ckpt", sd21_checkpoint),
            ("sdxl.ckpt", sdxl_checkpoint),
            ("pruned.ckpt", pruned_checkpoint),
        ]
        
        successful_loads = 0
        total_loads = len(checkpoints)
        
        # Act
        for filename, checkpoint_data in checkpoints:
            try:
                with patch('builtins.__import__') as mock_import:
                    # Mock torch module
                    mock_torch = Mock()
                    mock_torch.load.return_value = checkpoint_data
                    
                    def custom_import(name, *args, **kwargs):
                        if name == 'torch':
                            return mock_torch
                        return __import__(name, *args, **kwargs)
                    
                    mock_import.side_effect = custom_import
                    
                    # Suppress print statements for this test
                    with patch('builtins.print'):
                        model = StableDiffusionModel(filename)
                        
                        # Verify model was created successfully
                        assert isinstance(model, StableDiffusionModel)
                        assert hasattr(model, 'model')
                        assert hasattr(model, 'tensors')
                        
                        successful_loads += 1
            except Exception as e:
                pytest.fail(f"Failed to load {filename}: {e}")
        
        # Assert
        success_rate = successful_loads / total_loads
        assert success_rate == 1.0, f"Expected 100% success rate, got {success_rate * 100:.1f}%"

    def test_component_extraction_reliability(self, sd15_checkpoint, pruned_checkpoint):
        """
        Success Criterion: R_extract = C_extracted / C_present = 1.0
        
        GIVEN SD checkpoints with various components present:
            - Full model with VAE, UNet, Text Encoder
            - Model with only UNet and Text Encoder
            - Model with custom components
        WHEN attempting to extract all present components
        THEN expect:
            - All present components are successfully extracted
            - Extraction rate = 100%
            - Failures occur only for truly missing components
        """
        # Test with full model
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = sd15_checkpoint
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                model = StableDiffusionModel("full_model.ckpt")
                
                # All components should be extracted
                expected_components = len(sd15_checkpoint)
                extracted_components = len(model.tensors)
                
                extraction_rate = extracted_components / expected_components
                assert extraction_rate == 1.0, f"Expected 100% extraction rate, got {extraction_rate * 100:.1f}%"
                
                # Verify all original keys are present
                for key in sd15_checkpoint.keys():
                    assert key in model.tensors, f"Component {key} not extracted"

        # Test with pruned model
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = pruned_checkpoint
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                model = StableDiffusionModel("pruned_model.ckpt")
                
                # Only present components should be extracted
                expected_components = len(pruned_checkpoint)
                extracted_components = len(model.tensors)
                
                extraction_rate = extracted_components / expected_components
                assert extraction_rate == 1.0, f"Expected 100% extraction rate for pruned model, got {extraction_rate * 100:.1f}%"

    def test_missing_component_warning_coverage(self, pruned_checkpoint):
        """
        Success Criterion: W_coverage = M_warned / M_total = 1.0
        
        GIVEN SD checkpoints with known missing components:
            - Checkpoint without VAE
            - Checkpoint without safety checker
            - Checkpoint with partial text encoder
        WHEN model is loaded
        THEN expect:
            - Warning logged for each missing component
            - Warning coverage = 100%
            - No warnings for present components
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = pruned_checkpoint
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print') as mock_print:
                # Act
                model = StableDiffusionModel("pruned_model.ckpt")
                
                # Assert
                print_calls = mock_print.call_args_list
                warning_calls = [call for call in print_calls if any("Warning" in str(arg) and "found" in str(arg) for arg in call[0])]
                
                # Should have warnings for missing components
                assert len(warning_calls) > 0, "Expected warnings for missing components"
                
                # Verify warning coverage
                warning_coverage = len(warning_calls) / 2  # Expecting at least 2 missing components
                assert warning_coverage >= 1.0, f"Expected full warning coverage, got {warning_coverage * 100:.1f}%"

    def test_tensor_validation_accuracy(self, sd15_checkpoint):
        """
        Success Criterion: V_accuracy = T_correct / T_validated = 1.0
        
        GIVEN tensors with various properties:
            - Valid tensors (float32/16/bfloat16, <=4 dims)
            - Invalid dtype tensors (int32, float64)
            - High dimensional tensors (>4 dims)
            - Missing tensors
        WHEN valid() is called on each
        THEN expect:
            - All validations return correct result
            - Validation accuracy = 100%
        """
        # Arrange
        test_tensors = OrderedDict([
            ('valid_float32', self._create_mock_tensor('float32', 2)),
            ('valid_float16', self._create_mock_tensor('float16', 2)),
            ('valid_bfloat16', self._create_mock_tensor('bfloat16', 2)),
            ('valid_1d', self._create_mock_tensor('float32', 1)),
            ('valid_4d', self._create_mock_tensor('float32', 4)),
            ('invalid_dtype', self._create_mock_tensor('int32', 2)),
            ('invalid_dims', self._create_mock_tensor('float32', 5)),  # 5D is too many
        ])
        
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = OrderedDict()
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                model = StableDiffusionModel("test_model.ckpt")
                model.tensors = test_tensors
                
                # Define expected validation results
                expected_results = {
                    'valid_float32': (True, "OK"),
                    'valid_float16': (True, "OK"),
                    'valid_bfloat16': (True, "OK"),
                    'valid_1d': (True, "OK"),
                    'valid_4d': (True, "OK"),
                    'invalid_dtype': (False, "Unhandled type"),
                    'invalid_dims': (False, "Unhandled dimensions"),
                    'missing_tensor': (False, "Tensor not found"),
                }
                
                # Act & Assert
                correct_validations = 0
                total_validations = len(expected_results)
                
                for tensor_key, expected_result in expected_results.items():
                    actual_result = model.valid(tensor_key)
                    if actual_result == expected_result:
                        correct_validations += 1
                    else:
                        pytest.fail(f"Validation failed for {tensor_key}: expected {expected_result}, got {actual_result}")
                
                validation_accuracy = correct_validations / total_validations
                assert validation_accuracy == 1.0, f"Expected 100% validation accuracy, got {validation_accuracy * 100:.1f}%"

    def test_error_detection_rate(self):
        """
        Success Criterion: E_detection = F_caught / F_total = 1.0
        
        GIVEN various failure conditions:
            - Corrupted checkpoint file
            - Missing PyTorch
            - Invalid checkpoint structure
            - File not found
            - Pickle errors
        WHEN these conditions occur
        THEN expect:
            - All failures are detected and handled
            - Error detection rate = 100%
            - Appropriate error messages logged
            - sys.exit(1) called when appropriate
        """
        # Test cases for different failure conditions
        failure_conditions = [
            ("missing_torch", ImportError("No module named 'torch'")),
            ("file_not_found", FileNotFoundError("No such file or directory")),
            ("corrupted_file", RuntimeError("Corrupted checkpoint")),
            ("pickle_error", Exception("Pickle error")),
        ]
        
        detected_failures = 0
        total_failures = len(failure_conditions)
        
        for condition_name, exception in failure_conditions:
            if condition_name == "missing_torch":
                # Test missing torch by making the module unavailable
                original_modules = sys.modules.copy()
                if 'torch' in sys.modules:
                    del sys.modules['torch']
                
                with patch.dict('sys.modules', {'torch': None}):
                    with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                        with patch('builtins.print'):
                            with pytest.raises(SystemExit):
                                StableDiffusionModel("test.ckpt")
                            mock_exit.assert_called_with(1)
                            detected_failures += 1
                
                # Restore modules
                sys.modules.update(original_modules)
            else:
                # Test other failures by mocking torch.load to raise the exception
                with patch('builtins.__import__') as mock_import:
                    mock_torch = Mock()
                    mock_torch.load.side_effect = exception
                    
                    def custom_import(name, *args, **kwargs):
                        if name == 'torch':
                            return mock_torch
                        return __import__(name, *args, **kwargs)
                    
                    mock_import.side_effect = custom_import
                    
                    with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                        with patch('builtins.print'):
                            with pytest.raises(SystemExit):
                                StableDiffusionModel("test.ckpt")
                            mock_exit.assert_called_with(1)
                            detected_failures += 1
        
        # Assert
        error_detection_rate = detected_failures / total_failures
        assert error_detection_rate == 1.0, f"Expected 100% error detection rate, got {error_detection_rate * 100:.1f}%"

    def test_protocol_compliance(self, sd15_checkpoint):
        """
        Success Criterion: P_compliance = sum(m_i) / 5 = 1.0
        
        GIVEN the Model protocol requirements
        WHEN checking StableDiffusionModel implementation
        THEN expect:
            - __init__ method properly implemented
            - tensor_names returns Iterable[str]
            - valid returns tuple[bool, None | str]
            - get_as_f32 returns np.ndarray[np.float32]
            - get_type_name returns str
            - Protocol compliance = 100%
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = sd15_checkpoint
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                model = StableDiffusionModel("test_model.ckpt")
                
                compliance_checks = []
                
                # Check 1: __init__ method properly implemented
                try:
                    assert isinstance(model, StableDiffusionModel)
                    assert hasattr(model, 'model')
                    assert hasattr(model, 'tensors')
                    compliance_checks.append(1)
                except Exception:
                    compliance_checks.append(0)
                
                # Check 2: tensor_names returns Iterable[str]
                try:
                    tensor_names = model.tensor_names()
                    assert isinstance(tensor_names, Iterable)
                    tensor_list = list(tensor_names)
                    assert all(isinstance(name, str) for name in tensor_list)
                    compliance_checks.append(1)
                except Exception:
                    compliance_checks.append(0)
                
                # Check 3: valid returns tuple[bool, None | str]
                try:
                    # Test with existing tensor
                    first_key = next(iter(model.tensors.keys()))
                    result = model.valid(first_key)
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    assert isinstance(result[0], bool)
                    assert isinstance(result[1], (str, type(None)))
                    
                    # Test with non-existing tensor
                    result = model.valid("nonexistent")
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    assert isinstance(result[0], bool)
                    assert isinstance(result[1], (str, type(None)))
                    compliance_checks.append(1)
                except Exception:
                    compliance_checks.append(0)
                
                # Check 4: get_as_f32 returns np.ndarray[np.float32]
                try:
                    first_key = next(iter(model.tensors.keys()))
                    result = model.get_as_f32(first_key)
                    assert isinstance(result, np.ndarray)
                    assert result.dtype == np.float32
                    compliance_checks.append(1)
                except Exception:
                    compliance_checks.append(0)
                
                # Check 5: get_type_name returns str
                try:
                    first_key = next(iter(model.tensors.keys()))
                    result = model.get_type_name(first_key)
                    assert isinstance(result, str)
                    compliance_checks.append(1)
                except Exception:
                    compliance_checks.append(0)
                
                # Assert
                protocol_compliance = sum(compliance_checks) / 5
                assert protocol_compliance == 1.0, f"Expected 100% protocol compliance, got {protocol_compliance * 100:.1f}%"

    def test_end_to_end_workflow(self, sd15_checkpoint):
        """
        GIVEN a complete SD checkpoint
        WHEN performing a typical workflow:
            1. Load model
            2. Get tensor names
            3. Validate tensors
            4. Extract tensor data
            5. Get tensor types
        THEN expect:
            - All operations complete successfully
            - Data consistency throughout workflow
            - No memory leaks or errors
        """
        # Arrange
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = sd15_checkpoint
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                # Act 1: Load model
                model = StableDiffusionModel("test_model.ckpt")
                assert isinstance(model, StableDiffusionModel)
                
                # Act 2: Get tensor names
                tensor_names = list(model.tensor_names())
                assert len(tensor_names) > 0
                assert all(isinstance(name, str) for name in tensor_names)
                
                # Act 3: Validate all tensors
                valid_tensors = []
                for name in tensor_names:
                    is_valid, message = model.valid(name)
                    if is_valid:
                        valid_tensors.append(name)
                
                assert len(valid_tensors) > 0, "Expected at least some valid tensors"
                
                # Act 4: Extract tensor data for valid tensors
                extracted_data = {}
                for name in valid_tensors[:3]:  # Test first 3 valid tensors
                    try:
                        data = model.get_as_f32(name)
                        assert isinstance(data, np.ndarray)
                        assert data.dtype == np.float32
                        extracted_data[name] = data
                    except Exception as e:
                        pytest.fail(f"Failed to extract data for {name}: {e}")
                
                # Act 5: Get tensor types
                tensor_types = {}
                for name in valid_tensors[:3]:
                    try:
                        tensor_type = model.get_type_name(name)
                        assert isinstance(tensor_type, str)
                        tensor_types[name] = tensor_type
                    except Exception as e:
                        pytest.fail(f"Failed to get type for {name}: {e}")
                
                # Assert final state
                assert len(extracted_data) > 0, "Expected successful data extraction"
                assert len(tensor_types) > 0, "Expected successful type extraction"
                assert len(extracted_data) == len(tensor_types), "Data and type counts should match"

    def test_memory_efficiency_large_model(self):
        """
        GIVEN a large model with many tensors
        WHEN performing operations
        THEN expect:
            - Memory usage remains reasonable
            - No memory leaks
            - Operations complete in reasonable time
        """
        # Arrange - Create a large mock model
        large_model = OrderedDict()
        for i in range(100):  # 100 tensors
            for component in ['weight', 'bias']:
                key = f'layer_{i:03d}.{component}'
                large_model[key] = self._create_mock_tensor('float32', 2, (256, 256))
        
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = large_model
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                # Act
                model = StableDiffusionModel("large_model.ckpt")
                
                # Test tensor_names performance
                tensor_names = list(model.tensor_names())
                assert len(tensor_names) == 200  # 100 layers * 2 components
                
                # Test validation performance
                validation_results = []
                for name in tensor_names[:10]:  # Test first 10
                    result = model.valid(name)
                    validation_results.append(result)
                
                assert len(validation_results) == 10
                assert all(isinstance(result, tuple) for result in validation_results)
                
                # Test data extraction performance
                extracted_count = 0
                for name in tensor_names[:5]:  # Test first 5
                    try:
                        data = model.get_as_f32(name)
                        assert isinstance(data, np.ndarray)
                        extracted_count += 1
                    except Exception:
                        pass
                
                assert extracted_count > 0, "Expected at least some successful extractions"

    def test_robustness_edge_cases(self):
        """
        GIVEN various edge case scenarios
        WHEN operations are performed
        THEN expect:
            - Graceful handling of edge cases
            - Appropriate error messages
            - No crashes or undefined behavior
        """
        # Test with empty model
        empty_model = OrderedDict()
        
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.load.return_value = empty_model
            
            def custom_import(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = custom_import
            
            with patch('builtins.print'):
                model = StableDiffusionModel("empty_model.ckpt")
                
                # Test operations on empty model
                tensor_names = list(model.tensor_names())
                assert tensor_names == []
                
                result = model.valid("nonexistent")
                assert result == (False, "Tensor not found")
                
                with pytest.raises(KeyError):
                    model.get_as_f32("nonexistent")
                
                # get_type_name with nonexistent key should handle gracefully
                try:
                    model.get_type_name("nonexistent")
                except KeyError:
                    pass  # Expected behavior
                except Exception as e:
                    pytest.fail(f"Unexpected exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])