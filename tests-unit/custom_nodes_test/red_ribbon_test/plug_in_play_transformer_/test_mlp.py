import torch
import pytest
from custom_nodes.red_ribbon.plug_in_play_transformer.mlp import (
    MLPExpansionNode, 
    ActivationFunctionNode, 
    MLPContractionNode, 
    DropoutNode, 
    MLP
)

@pytest.fixture
def test_data():
    """Fixture providing test data for MLP nodes."""
    batch_size = 2
    seq_len = 16
    embed_dim = 64
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    expansion_factor = 4.0
    activation_type = "gelu"
    dropout_rate = 0.1
    target_dim = embed_dim
    
    return {
        'x': x,
        'expansion_factor': expansion_factor,
        'activation_type': activation_type,
        'dropout_rate': dropout_rate,
        'target_dim': target_dim
    }

def test_expansion(test_data):
    """Test expansion node."""
    x = test_data['x']
    expansion_factor = test_data['expansion_factor']
    
    node = MLPExpansionNode()
    expanded, = node.expand(x, expansion_factor)
    
    # Check shape
    B, T, C = x.shape
    expected_expanded_dim = int(C * expansion_factor)
    assert expanded.shape == (B, T, expected_expanded_dim)

def test_activation(test_data):
    """Test activation node."""
    x = test_data['x']
    expansion_factor = test_data['expansion_factor']
    activation_type = test_data['activation_type']
    
    # First expand the input
    expansion_node = MLPExpansionNode()
    expanded, = expansion_node.expand(x, expansion_factor)
    
    # Test activation
    activation_node = ActivationFunctionNode()
    activated, = activation_node.activate(expanded, activation_type)
    
    # Check shape (should be unchanged)
    assert activated.shape == expanded.shape
    
    # Test different activations
    for act_type in ["relu", "silu", "swish"]:
        activated, = activation_node.activate(expanded, act_type)
        assert activated.shape == expanded.shape

def test_contraction(test_data):
    """Test contraction node."""
    x = test_data['x']
    expansion_factor = test_data['expansion_factor']
    activation_type = test_data['activation_type']
    
    # Prepare expanded and activated input
    expansion_node = MLPExpansionNode()
    expanded, = expansion_node.expand(x, expansion_factor)
    
    activation_node = ActivationFunctionNode()
    activated, = activation_node.activate(expanded, activation_type)
    
    # Test contraction
    B, T, C = x.shape
    contraction_node = MLPContractionNode()
    contracted, = contraction_node.contract(activated, C)
    
    # Check shape (should be back to original)
    assert contracted.shape == x.shape

def test_dropout(test_data):
    """Test dropout node."""
    x = test_data['x']
    expansion_factor = test_data['expansion_factor']
    activation_type = test_data['activation_type']
    dropout_rate = test_data['dropout_rate']
    
    # Prepare fully processed input before dropout
    expansion_node = MLPExpansionNode()
    expanded, = expansion_node.expand(x, expansion_factor)
    
    activation_node = ActivationFunctionNode()
    activated, = activation_node.activate(expanded, activation_type)
    
    B, T, C = x.shape
    contraction_node = MLPContractionNode()
    contracted, = contraction_node.contract(activated, C)
    
    # Test dropout
    dropout_node = DropoutNode()
    output, = dropout_node.apply_dropout(contracted, dropout_rate)
    
    # Check shape (should be unchanged)
    assert output.shape == contracted.shape

def test_full_mlp(test_data):
    """Test the full MLP implementation."""
    x = test_data['x']
    expansion_factor = test_data['expansion_factor']
    activation_type = test_data['activation_type']
    dropout_rate = test_data['dropout_rate']
    
    # Test the original implementation
    class MockConfig:
        def __init__(self):
            self.n_embd = x.shape[-1]
            self.resid_pdrop = dropout_rate
            
    config = MockConfig()
    original_model = MLP(config)
    original_output = original_model(x)
    
    # Test our modular implementation
    B, T, C = x.shape
    
    expansion_node = MLPExpansionNode()
    expanded, = expansion_node.expand(x, expansion_factor)
    
    activation_node = ActivationFunctionNode()
    activated, = activation_node.activate(expanded, activation_type)
    
    contraction_node = MLPContractionNode()
    contracted, = contraction_node.contract(activated, C)
    
    dropout_node = DropoutNode()
    modular_output, = dropout_node.apply_dropout(contracted, dropout_rate)
    
    # The outputs won't be identical due to random initialization
    # but shapes should match
    assert original_output.shape == modular_output.shape