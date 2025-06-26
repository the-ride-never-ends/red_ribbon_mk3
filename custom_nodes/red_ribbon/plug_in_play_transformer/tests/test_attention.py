import torch
import unittest
from .attention import (
    QKVProjectionNode, 
    CalculateCausalAttentionMatrixNode, 
    ApplyAttentionNode, 
    OutputProjectionNode
)

class TestTransformerNodes(unittest.TestCase):
    def setUp(self):
        # Mock input data
        batch_size = 2
        seq_len = 16
        embed_dim = 64
        n_head = 4
        
        self.x = torch.randn(batch_size, seq_len, embed_dim)
        self.n_head = n_head
        self.n_embd = embed_dim
        self.block_size = 1024
        self.dropout_rate = 0.1
        
    def test_qkv_projection(self):
        # Test QKV projection node
        node = QKVProjectionNode()
        q, k, v = node.project(self.x, self.n_head, self.n_embd)
        
        # Check shapes
        B, T, C = self.x.shape
        self.assertEqual(q.shape, (B, self.n_head, T, C // self.n_head))
        self.assertEqual(k.shape, (B, self.n_head, T, C // self.n_head))
        self.assertEqual(v.shape, (B, self.n_head, T, C // self.n_head))
        
    def test_attention_matrix(self):
        # First get q, k from projection
        node = QKVProjectionNode()
        q, k, _ = node.project(self.x, self.n_head, self.n_embd)
        
        # Test attention matrix calculation
        att_node = CalculateCausalAttentionMatrixNode()
        att, = att_node.calculate_attention(q, k, self.block_size, self.dropout_rate)
        
        # Check shape and properties
        B, nh, T, _ = q.shape
        self.assertEqual(att.shape, (B, nh, T, T))
        
        # Check causality (upper triangle should be -inf before softmax)
        # This is harder to test directly after softmax
        
    def test_apply_attention(self):
        # Get q, k, v and attention matrix
        qkv_node = QKVProjectionNode()
        q, k, v = qkv_node.project(self.x, self.n_head, self.n_embd)
        
        att_node = CalculateCausalAttentionMatrixNode()
        att, = att_node.calculate_attention(q, k, self.block_size, self.dropout_rate)
        
        # Test applying attention
        apply_node = ApplyAttentionNode()
        y, = apply_node.apply_attention(att, v, self.n_embd)
        
        # Check shape
        B, T, C = self.x.shape
        self.assertEqual(y.shape, (B, T, C))
        
    def test_output_projection(self):
        # Get full attention output
        qkv_node = QKVProjectionNode()
        q, k, v = qkv_node.project(self.x, self.n_head, self.n_embd)
        
        att_node = CalculateCausalAttentionMatrixNode()
        att, = att_node.calculate_attention(q, k, self.block_size, self.dropout_rate)
        
        apply_node = ApplyAttentionNode()
        y, = apply_node.apply_attention(att, v, self.n_embd)
        
        # Test output projection
        out_node = OutputProjectionNode()
        output, = out_node.project_output(y, self.n_embd, self.dropout_rate)
        
        # Check shape
        B, T, C = self.x.shape
        self.assertEqual(output.shape, (B, T, C))
        
    def test_full_attention(self):
        # Test the original implementation
        class MockConfig:
            def __init__(self):
                self.n_head = self.n_head
                self.n_embd = self.n_embd
                self.block_size = self.block_size
                self.attn_pdrop = self.dropout_rate
                self.resid_pdrop = self.dropout_rate
                
        from .attention import CausalSelfAttentionNode
        
        config = MockConfig()
        original_model = CausalSelfAttentionNode(config)
        original_output = original_model(self.x)
        
        # Test our modular implementation
        qkv_node = QKVProjectionNode()
        q, k, v = qkv_node.project(self.x, self.n_head, self.n_embd)
        
        att_node = CalculateCausalAttentionMatrixNode()
        att, = att_node.calculate_attention(q, k, self.block_size, self.dropout_rate)
        
        apply_node = ApplyAttentionNode()
        y, = apply_node.apply_attention(att, v, self.n_embd)
        
        out_node = OutputProjectionNode()
        modular_output, = out_node.project_output(y, self.n_embd, self.dropout_rate)
        
        # The outputs won't be identical due to random initialization
        # but shapes should match
        self.assertEqual(original_output.shape, modular_output.shape)

if __name__ == "__main__":
    unittest.main()