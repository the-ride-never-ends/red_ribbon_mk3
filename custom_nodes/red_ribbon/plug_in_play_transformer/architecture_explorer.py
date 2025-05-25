


class PositionalEncodingExplorerNode:
    """
    Node that generates different types of positional encodings.
    
    Inputs:
      - encoding_type: Type of positional encoding
      - embedding_dim: Dimension of the embeddings
      - max_seq_len: Maximum sequence length
      - parameters: Additional parameters
    
    Outputs:
      - function_code: Positional encoding function
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoding_type": (["sinusoidal", "learned", "rotary", "alibi", "custom"], {"default": "sinusoidal"}),
                "embedding_dim": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "max_seq_len": ("INT", {"default": 1024, "min": 16, "max": 32768}),
            },
            "optional": {
                "parameters": ("STRING", {
                    "multiline": True,
                    "default": "{}"
                }),
                "custom_code": ("STRING", {
                    "multiline": True,
                    "default": "# Your custom positional encoding here"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "create_positional_encoding"
    CATEGORY = "transformer/positional"
    
    def create_positional_encoding(self, encoding_type, embedding_dim, max_seq_len, parameters="{}", custom_code=None):
        import json
        
        try:
            params = json.loads(parameters)
        except:
            params = {}
        
        encodings = {
            "sinusoidal": self._sinusoidal_encoding(embedding_dim, max_seq_len, params),
            "learned": self._learned_encoding(embedding_dim, max_seq_len, params),
            "rotary": self._rotary_encoding(embedding_dim, max_seq_len, params),
            "alibi": self._alibi_encoding(embedding_dim, max_seq_len, params),
            "custom": custom_code if custom_code else "# No custom code provided",
        }
        
        return (encodings.get(encoding_type, self._sinusoidal_encoding(embedding_dim, max_seq_len, params)),)
    
    def _sinusoidal_encoding(self, embedding_dim, max_seq_len, params):
        freq = params.get("freq", 10000.0)
        return f"""
def create_sinusoidal_encoding(seq_len, d_model={embedding_dim}, max_len={max_seq_len}, freq={freq}):
    # Implementation of sinusoidal positional encodings from "Attention Is All You Need"
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log({freq}) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
    
    # Only return up to seq_len positions
    return pe[:seq_len, :]

def apply_positional_encoding(x):
    # x has shape [batch_size, seq_len, embedding_dim]
    seq_len = x.size(1)
    pe = create_sinusoidal_encoding(seq_len)
    pe = pe.unsqueeze(0).to(x.device)  # Add batch dimension
    return x + pe
"""
    
    def _rotary_encoding(self, embedding_dim, max_seq_len, params):
        freq = params.get("freq", 10000.0)
        return f"""
def apply_rotary_position_embeddings(x, freq={freq}):
    # Rotary Position Embedding (RoPE) implementation
    # x has shape [batch_size, seq_len, n_heads, head_dim] or [batch_size, seq_len, embedding_dim]
    
    orig_shape = x.shape
    if len(orig_shape) == 3:
        batch_size, seq_len, embedding_dim = orig_shape
        # Assume this is pre-multihead projection, reshape is needed later
        n_heads = 1
        head_dim = embedding_dim
        x = x.view(batch_size, seq_len, n_heads, head_dim)
    else:
        batch_size, seq_len, n_heads, head_dim = orig_shape
    
    # Compute positional encodings
    positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
    dim = torch.arange(0, head_dim, 2, device=x.device).float()
    freqs = 1.0 / (freq ** (dim / head_dim))
    theta = positions * freqs
    
    # Compute sin and cos for alternating dimensions
    emb = torch.zeros(seq_len, head_dim, device=x.device)
    emb[:, 0::2] = torch.sin(theta)
    emb[:, 1::2] = torch.cos(theta)
    
    # Apply rotary embeddings
    cos = emb[:, None, None, 0::2].expand(-1, batch_size, n_heads, -1)
    sin = emb[:, None, None, 1::2].expand(-1, batch_size, n_heads, -1)
    
    x_reshaped = x.permute(1, 0, 2, 3)  # [seq_len, batch, heads, dim]
    
    # Extract even and odd dimensions
    x_even = x_reshaped[..., 0::2]
    x_odd = x_reshaped[..., 1::2]
    
    # Apply rotation
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos
    
    # Interleave the rotated dimensions
    x_rotated = torch.zeros_like(x_reshaped)
    x_rotated[..., 0::2] = x_rotated_even
    x_rotated[..., 1::2] = x_rotated_odd
    
    # Return to original shape
    x_rotated = x_rotated.permute(1, 0, 2, 3)  # [batch, seq, heads, dim]
    
    if len(orig_shape) == 3:
        # Reshape back if needed
        x_rotated = x_rotated.reshape(batch_size, seq_len, head_dim)
    
    return x_rotated

def apply_positional_encoding(x):
    return apply_rotary_position_embeddings(x)
"""
    
    def _alibi_encoding(self, embedding_dim, max_seq_len, params):
        return f"""
def create_alibi_slopes(n_heads):
    # Implementation of ALiBi (Attention with Linear Biases)
    # Create slopes according to the ALiBi paper
    m = torch.arange(1, n_heads + 1)
    m = 1.0 / (2 ** (m / 2))
    return m

def apply_alibi_to_attention(attn_scores, n_heads={embedding_dim // 64}):
    # Apply ALiBi to attention scores
    # attn_scores has shape [batch_size, n_heads, seq_len, seq_len]
    batch_size, n_heads, seq_len, _ = attn_scores.shape
    
    # Create distance matrix
    distances = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    # Make it causal - set upper triangular to a large negative value
    distances = torch.tril(distances)
    distances = distances.to(attn_scores.device)
    
    # Get slopes for each head
    slopes = create_alibi_slopes(n_heads).to(attn_scores.device)
    slopes = slopes.view(1, n_heads, 1, 1)
    
    # Apply slopes to distances to get the bias
    alibi = distances.view(1, 1, seq_len, seq_len) * slopes
    
    # Add the bias to attention scores
    return attn_scores + alibi

def attention_with_alibi(q, k, v):
    # Calculate attention scores
    attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply ALiBi bias
    attn_scores = apply_alibi_to_attention(attn_scores)
    
    # Apply causal mask
    seq_len = attn_scores.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(attn_scores.device)
    attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
    
    # Apply softmax and attend
    attn_probs = F.softmax(attn_scores, dim=-1)
    return attn_probs @ v
"""
    
    def _learned_encoding(self, embedding_dim, max_seq_len, params):
        return f"""
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len={max_seq_len}, d_model={embedding_dim}):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len, embedding_dim]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# Instantiate module
learned_pe = LearnedPositionalEncoding()

def apply_positional_encoding(x):
    return learned_pe(x)
"""


class ModelArchitectureExplorerNode:
    """
    Node that allows exploring different transformer model architectures.
    
    Inputs:
      - embedding_dim: Dimension of the embeddings
      - num_heads: Number of attention heads
      - num_layers: Number of transformer layers
      - attention_type: Type of attention mechanism
      - mlp_type: Type of MLP
      - normalization_type: Type of normalization
      - positional_encoding: Type of positional encoding
    
    Outputs:
      - model_config: Complete model configuration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding_dim": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "num_heads": ("INT", {"default": 8, "min": 1, "max": 128}),
                "num_layers": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "attention_type": (["standard", "flash", "linear", "local", "sparse"], {"default": "standard"}),
                "mlp_type": (["standard", "gated", "swiglu", "geglu"], {"default": "standard"}),
                "normalization_type": (["layernorm", "rmsnorm", "scalednorm"], {"default": "layernorm"}),
                "positional_encoding": (["sinusoidal", "learned", "rotary", "alibi"], {"default": "sinusoidal"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "generate_architecture"
    CATEGORY = "transformer/exploration"
    
    def generate_architecture(self, embedding_dim, num_heads, num_layers, attention_type, 
                             mlp_type, normalization_type, positional_encoding):
        # Create model configuration
        config = {
            "architecture": {
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "head_dim": embedding_dim // num_heads,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
            },
            "attention": {
                "type": attention_type,
                "params": {}
            },
            "mlp": {
                "type": mlp_type,
                "params": {}
            },
            "normalization": {
                "type": normalization_type,
                "params": {}
            },
            "positional_encoding": {
                "type": positional_encoding,
                "params": {}
            }
        }
        
        # Add specific parameters based on selected types
        if attention_type == "local":
            config["attention"]["params"]["window_size"] = 128
        elif attention_type == "sparse":
            config["attention"]["params"]["sparsity"] = 0.9
        
        if positional_encoding == "rotary":
            config["positional_encoding"]["params"]["freq"] = 10000.0
        
        # Format as JSON
        import json
        formatted_config = json.dumps(config, indent=2)
        
        return (formatted_config,)