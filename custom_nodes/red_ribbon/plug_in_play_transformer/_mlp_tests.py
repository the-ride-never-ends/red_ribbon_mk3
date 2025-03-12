import torch
import unittest
from .mlp import MLPExpansionNode, ActivationFunctionNode, MLPContractionNode, DropoutNode, MLP

class TestMLPNodes(unittest.TestCase):
    def setUp(self):
        # Mock input data
        batch_size = 2
        seq_len = 16
        embed_dim = 64
        
        self.x = torch.randn(batch_size, seq_len, embed_dim)
        self.expansion_factor = 4.0
        self.activation_type = "gelu"
        self.dropout_rate = 0.1
        self.target_dim = embed_dim  # Contract back to original dimension
        
    def test_expansion(self):
        # Test expansion node
        node = MLPExpansionNode()
        expanded, = node.expand(self.x, self.expansion_factor)
        
        # Check shape
        B, T, C = self.x.shape
        expected_expanded_dim = int(C * self.expansion_factor)
        self.assertEqual(expanded.shape, (B, T, expected_expanded_dim))
        
    def test_activation(self):
        # First expand the input
        expansion_node = MLPExpansionNode()
        expanded, = expansion_node.expand(self.x, self.expansion_factor)
        
        # Test activation
        activation_node = ActivationFunctionNode()
        activated, = activation_node.activate(expanded, self.activation_type)
        
        # Check shape (should be unchanged)
        self.assertEqual(activated.shape, expanded.shape)
        
        # Test different activations
        for act_type in ["relu", "silu", "swish"]:
            activated, = activation_node.activate(expanded, act_type)
            self.assertEqual(activated.shape, expanded.shape)
        
    def test_contraction(self):
        # Prepare expanded and activated input
        expansion_node = MLPExpansionNode()
        expanded, = expansion_node.expand(self.x, self.expansion_factor)
        
        activation_node = ActivationFunctionNode()
        activated, = activation_node.activate(expanded, self.activation_type)
        
        # Test contraction
        B, T, C = self.x.shape
        contraction_node = MLPContractionNode()
        contracted, = contraction_node.contract(activated, C)
        
        # Check shape (should be back to original)
        self.assertEqual(contracted.shape, self.x.shape)
        
    def test_dropout(self):
        # Prepare fully processed input before dropout
        expansion_node = MLPExpansionNode()
        expanded, = expansion_node.expand(self.x, self.expansion_factor)
        
        activation_node = ActivationFunctionNode()
        activated, = activation_node.activate(expanded, self.activation_type)
        
        B, T, C = self.x.shape
        contraction_node = MLPContractionNode()
        contracted, = contraction_node.contract(activated, C)
        
        # Test dropout
        dropout_node = DropoutNode()
        output, = dropout_node.apply_dropout(contracted, self.dropout_rate)
        
        # Check shape (should be unchanged)
        self.assertEqual(output.shape, contracted.shape)
        
        # Test with training mode vs eval mode
        # In training mode, some values will be zeroed
        # In eval mode, all values should remain
        
    def test_full_mlp(self):
        # Test the original implementation
        class MockConfig:
            def __init__(self):
                self.n_embd = self.x.shape[-1]
                self.resid_pdrop = self.dropout_rate
                
        from .mlp import MLP
        
        config = MockConfig()
        original_model = MLP(config)
        original_output = original_model(self.x)
        
        # Test our modular implementation
        B, T, C = self.x.shape
        
        expansion_node = MLPExpansionNode()
        expanded, = expansion_node.expand(self.x, self.expansion_factor)
        
        activation_node = ActivationFunctionNode()
        activated, = activation_node.activate(expanded, self.activation_type)
        
        contraction_node = MLPContractionNode()
        contracted, = contraction_node.contract(activated, C)
        
        dropout_node = DropoutNode()
        modular_output, = dropout_node.apply_dropout(contracted, self.dropout_rate)
        
        # The outputs won't be identical due to random initialization
        # but shapes should match
        self.assertEqual(original_output.shape, modular_output.shape)

if __name__ == "__main__":
    unittest.main()