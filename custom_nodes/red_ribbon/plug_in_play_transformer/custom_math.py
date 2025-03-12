import ast
from easy_nodes import Choice


class FunctionLibraryNode:
    """
    Node that provides a library of predefined mathematical functions
    that can be used in custom function nodes.
    
    Inputs:
      - function_type: Category of functions to access
      - function_name: Specific function to retrieve
    
    Outputs:
      - function_code: String representation of the selected function
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function_type": (["attention", "mlp", "layer_norm", "activations", "advanced"], {"default": "attention"}),
                "function_name": (lambda function_type: cls.get_function_names(function_type)),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "get_function"
    CATEGORY = "transformer/custom"
    
    @classmethod
    def get_function_names(cls, function_type):
        function_library = {
            "attention": ["standard_attention", "flash_attention", "sliding_window", "local_attention", "linear_attention"],
            "mlp": ["standard_mlp", "gated_mlp", "swiglu", "geglu"],
            "layer_norm": ["layer_norm", "rms_norm", "scaled_norm"],
            "activations": ["gelu", "relu", "silu", "swish", "mish"],
            "advanced": ["rotary_embeddings", "alibi", "mixture_of_experts"]
        }
        return function_library.get(function_type, ["unknown"])
    
    def get_function(self, function_type, function_name):
        # Library of predefined functions
        function_library = {
            "attention": {
                "standard_attention": "def attention(q, k, v):\n    # Standard scaled dot-product attention\n    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))\n    mask = torch.tril(torch.ones(att.size(-2), att.size(-1))).to(att.device)\n    att = att.masked_fill(mask == 0, float('-inf'))\n    att = F.softmax(att, dim=-1)\n    return att @ v",
                
                "flash_attention": "def attention(q, k, v):\n    # Simplified flash attention simulation\n    # (Real flash attention requires CUDA kernels)\n    scale = 1.0 / math.sqrt(k.size(-1))\n    q = q * scale\n    # Causal mask within chunked attention\n    T = q.size(2)\n    chunk_size = min(128, T)  # Simulate chunking\n    output = torch.zeros_like(v)\n    \n    for i in range(0, T, chunk_size):\n        end_idx = min(i + chunk_size, T)\n        q_chunk = q[:, :, i:end_idx]\n        k_chunk = k[:, :, :end_idx]  # Causal: only attend to past\n        v_chunk = v[:, :, :end_idx]\n        \n        att = q_chunk @ k_chunk.transpose(-2, -1)\n        mask = torch.ones_like(att).triu_(diagonal=1).bool()\n        att.masked_fill_(mask, float('-inf'))\n        att = F.softmax(att, dim=-1)\n        output[:, :, i:end_idx] = att @ v_chunk\n    \n    return output",
                
                "linear_attention": "def attention(q, k, v):\n    # Linear attention with ReLU feature map\n    # O(N) instead of O(N²) complexity\n    q = F.relu(q) + 1e-6  # Feature map + numerical stability\n    k = F.relu(k) + 1e-6\n    \n    # Linear attention\n    k_cumsum = torch.cumsum(k, dim=-2)  # Causal cumulative sum\n    D_inv = 1.0 / torch.einsum('...nd,...nd->...n', q, k_cumsum)\n    context = torch.einsum('...nd,...ne->...nde', k, v)\n    context_cumsum = torch.cumsum(context, dim=-3)  # Causal\n    out = torch.einsum('...nd,...nde,...n->...ne', q, context_cumsum, D_inv)\n    return out"
            },
            
            "mlp": {
                "standard_mlp": "def mlp(x, n_embd):\n    # Standard MLP with GELU\n    h = F.gelu(torch.nn.Linear(n_embd, 4 * n_embd)(x))\n    return torch.nn.Linear(4 * n_embd, n_embd)(h)",
                
                "swiglu": "def mlp(x, n_embd):\n    # SwiGLU activation (from PaLM)\n    fc1 = torch.nn.Linear(n_embd, 4 * n_embd)\n    fc2 = torch.nn.Linear(n_embd, 4 * n_embd)\n    fc3 = torch.nn.Linear(4 * n_embd, n_embd)\n    \n    gate = F.silu(fc1(x))\n    value = fc2(x)\n    return fc3(gate * value)",
                
                "geglu": "def mlp(x, n_embd):\n    # GEGLU activation (from GELU variants)\n    fc1 = torch.nn.Linear(n_embd, 4 * n_embd)\n    fc2 = torch.nn.Linear(n_embd, 4 * n_embd)\n    fc3 = torch.nn.Linear(4 * n_embd, n_embd)\n    \n    gate = F.gelu(fc1(x))\n    value = fc2(x)\n    return fc3(gate * value)"
            },
            
            "advanced": {
                "rotary_embeddings": "def apply_rotary_emb(x, freq=10000.0):\n    # Simplified rotary embeddings implementation\n    d = x.shape[-1]\n    dim_t = torch.arange(0, d, 2).to(x.device)\n    freqs = 1.0 / (freq ** (dim_t / d))\n    \n    seq_len = x.shape[-2]\n    pos = torch.arange(seq_len).to(x.device)\n    sincos = torch.outer(pos, freqs)\n    sin, cos = torch.sin(sincos), torch.cos(sincos)\n    \n    x1, x2 = x[..., ::2], x[..., 1::2]\n    rotated_x1 = x1 * cos - x2 * sin\n    rotated_x2 = x1 * sin + x2 * cos\n    \n    result = torch.zeros_like(x)\n    result[..., ::2] = rotated_x1\n    result[..., 1::2] = rotated_x2\n    return result"
            }
        }
        
        try:
            return (function_library[function_type][function_name],)
        except KeyError:
            return ("# Function not found in library",)



class MathExpressionNode:
    """
    Node that creates a mathematical expression that can be used in other function nodes.
    
    Inputs:
      - expression: Mathematical expression string
      - variables: List of variables used in the expression
      - wrapper_template: Template for wrapping the expression
    
    Outputs:
      - function_code: Complete function code with the expression integrated
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {
                    "multiline": True, 
                    "default": "x * torch.sin(x)"
                }),
                "variables": ("STRING", {
                    "default": "x"
                }),
                "function_name": ("STRING", {
                    "default": "custom_math_fn"
                }),
            },
            "optional": {
                "wrapper_template": ("STRING", {
                    "multiline": True,
                    "default": "def {function_name}({variables}):\n    return {expression}"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "generate_function"
    CATEGORY = "transformer/custom"
    
    def generate_function(self, expression, variables, function_name, wrapper_template=None):
        if wrapper_template is None:
            wrapper_template = "def {function_name}({variables}):\n    return {expression}"
        
        try:
            # Validate expression by trying to parse it
            ast.parse(expression)
            
            # Generate function code
            function_code = wrapper_template.format(
                function_name=function_name,
                variables=variables,
                expression=expression
            )
            
            return (function_code,)
        except SyntaxError as e:
            error_msg = f"# Error in expression: {str(e)}\n# Please correct the syntax"
            return (error_msg,)
        


class FunctionCombinerNode:
    """
    Node that combines multiple functions into a single function.
    
    Inputs:
      - function1: First function to combine
      - function2: Second function to combine
      - combination_type: How to combine the functions
    
    Outputs:
      - combined_function: Combined function code
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function1": ("STRING", {
                    "multiline": True, 
                    "default": "def f1(x):\n    return x"
                }),
                "function2": ("STRING", {
                    "multiline": True, 
                    "default": "def f2(x):\n    return x"
                }),
                "combination_type": (["compose", "add", "multiply", "custom"], {"default": "compose"}),
            },
            "optional": {
                "custom_combiner": ("STRING", {
                    "multiline": True,
                    "default": "def combine(f1_result, f2_result):\n    return f1_result + f2_result * 0.5"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_function",)
    FUNCTION = "combine_functions"
    CATEGORY = "transformer/custom"
    
    def combine_functions(self, 
                          function1: str, 
                          function2: str, 
                          combination_type: Choice, 
                          custom_combiner=None
                          ):
        # Extract function names
        f1_name = self._extract_function_name(function1)
        f2_name = self._extract_function_name(function2)
        
        # Create combined function based on combination type
        if combination_type == "compose":
            combined = f"{function1}\n\n{function2}\n\ndef combined_function(x):\n    return {f2_name}({f1_name}(x))"
        elif combination_type == "add":
            combined = f"{function1}\n\n{function2}\n\ndef combined_function(x):\n    return {f1_name}(x) + {f2_name}(x)"
        elif combination_type == "multiply":
            combined = f"{function1}\n\n{function2}\n\ndef combined_function(x):\n    return {f1_name}(x) * {f2_name}(x)"
        elif combination_type == "custom" and custom_combiner:
            combined = f"{function1}\n\n{function2}\n\n{custom_combiner}\n\ndef combined_function(x):\n    f1_result = {f1_name}(x)\n    f2_result = {f2_name}(x)\n    return combine(f1_result, f2_result)"
        else:
            combined = f"# Invalid combination type or missing custom combiner\n\ndef combined_function(x):\n    return x"
        
        return (combined,)
    
    def _extract_function_name(self, function_code):
        try:
            tree = ast.parse(function_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
            return "unknown_function"
        except:
            return "unknown_function"
        

class PiecewiseFunctionNode:
    """
    Node that creates a piecewise function based on conditions.
    
    Inputs:
      - conditions: List of conditions separated by semicolons
      - expressions: List of expressions corresponding to conditions, separated by semicolons
      - else_expression: Expression for when no conditions are met
      - variables: Variables used in the function
    
    Outputs:
      - function_code: Piecewise function code
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditions": ("STRING", {
                    "multiline": True, 
                    "default": "x < 0; x >= 0 and x < 1; x >= 1"
                }),
                "expressions": ("STRING", {
                    "multiline": True, 
                    "default": "torch.sin(x); x; x**2"
                }),
                "variables": ("STRING", {
                    "default": "x"
                }),
                "function_name": ("STRING", {
                    "default": "piecewise_function"
                }),
            },
            "optional": {
                "else_expression": ("STRING", {
                    "default": "x"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "generate_piecewise"
    CATEGORY = "transformer/custom"
    
    def generate_piecewise(self, conditions, expressions, variables, function_name, else_expression="x"):
        # Split conditions and expressions
        condition_list = [c.strip() for c in conditions.split(";")]
        expression_list = [e.strip() for e in expressions.split(";")]
        
        # Ensure equal length or handle mismatch
        if len(condition_list) != len(expression_list):
            return ("# Error: Number of conditions does not match number of expressions",)
        
        # Generate function code
        function_code = [f"def {function_name}({variables}):"]
        
        for i, (condition, expression) in enumerate(zip(condition_list, expression_list)):
            if i == 0:
                function_code.append(f"    if {condition}:")
            else:
                function_code.append(f"    elif {condition}:")
            function_code.append(f"        return {expression}")
        
        function_code.append(f"    else:")
        function_code.append(f"        return {else_expression}")
        
        return ("\n".join(function_code),)
    

class ActivationFunctionExplorerNode:
    """
    Node that allows exploring different activation functions.
    
    Inputs:
      - base_activation: Base activation function to modify
      - alpha: Parameter for scaling
      - beta: Parameter for shifting
    
    Outputs:
      - function_code: Custom activation function
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_activation": (["relu", "gelu", "silu", "mish", "custom"], {"default": "gelu"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "beta": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "custom_expression": ("STRING", {
                    "multiline": True,
                    "default": "torch.sin(x) * torch.sigmoid(x)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "create_activation"
    CATEGORY = "transformer/activations"
    
    def create_activation(self, base_activation, alpha, beta, custom_expression=None):
        activations = {
            "relu": "F.relu(x)",
            "gelu": "F.gelu(x)",
            "silu": "F.silu(x)",
            "mish": "x * torch.tanh(F.softplus(x))",
        }
        
        if base_activation == "custom" and custom_expression:
            base_expr = custom_expression
        else:
            base_expr = activations.get(base_activation, "x")
        
        function_code = f"""def custom_activation(x):
    # Custom activation with alpha={alpha} and beta={beta}
    return {alpha} * ({base_expr}) + {beta}
"""
        return (function_code,)
    



class AttentionPatternExplorerNode:
    """
    Node that creates attention patterns with different characteristics.
    
    Inputs:
      - pattern_type: Type of attention pattern to generate
      - window_size: Size of attention window for local attention
      - sparsity: Sparsity factor for sparse attention
      - custom_mask: Custom mask expression
    
    Outputs:
      - function_code: Attention function with the specified pattern
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_type": (["full", "local", "dilated", "sparse", "custom"], {"default": "local"}),
                "window_size": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "sparsity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "custom_mask": ("STRING", {
                    "multiline": True,
                    "default": "abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)) <= window_size"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("function_code",)
    FUNCTION = "create_attention_pattern"
    CATEGORY = "transformer/attention_patterns"
    
    def create_attention_pattern(self, pattern_type, window_size, sparsity, custom_mask=None):
        patterns = {
            "full": self._full_attention(window_size),
            "local": self._local_attention(window_size),
            "dilated": self._dilated_attention(window_size),
            "sparse": self._sparse_attention(sparsity),
            "custom": self._custom_attention(custom_mask) if custom_mask else self._local_attention(window_size),
        }
        
        return (patterns.get(pattern_type, self._full_attention(window_size)),)
    
    def _full_attention(self, window_size):
        return """def attention(q, k, v):
    # Full causal attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply causal mask
    seq_len = att.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(att.device)
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    
    att = F.softmax(att, dim=-1)
    return att @ v"""
    
    def _local_attention(self, window_size):
        return f"""def attention(q, k, v):
    # Local windowed causal attention with window_size = {window_size}
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply causal mask
    seq_len = att.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(att.device)
    
    # Apply window mask
    window_mask = torch.zeros(seq_len, seq_len).to(att.device)
    for i in range(seq_len):
        start = max(0, i - {window_size})
        window_mask[i, start:i+1] = 1.0
    
    # Combine masks
    combined_mask = causal_mask * window_mask
    att = att.masked_fill(combined_mask == 0, float('-inf'))
    
    att = F.softmax(att, dim=-1)
    return att @ v"""
    
    def _dilated_attention(self, window_size):
        return f"""def attention(q, k, v):
    # Dilated causal attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply causal mask
    seq_len = att.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(att.device)
    
    # Apply dilated mask
    dilated_mask = torch.zeros(seq_len, seq_len).to(att.device)
    for i in range(seq_len):
        # Nearby tokens
        start_local = max(0, i - 16)
        dilated_mask[i, start_local:i+1] = 1.0
        
        # Dilated tokens - attend to every {window_size//4}th token
        for j in range(i-16-{window_size//4}, -1, -{window_size//4}):
            if j < 0:
                break
            dilated_mask[i, j] = 1.0
    
    # Combine masks
    combined_mask = causal_mask * dilated_mask
    att = att.masked_fill(combined_mask == 0, float('-inf'))
    
    att = F.softmax(att, dim=-1)
    return att @ v"""
    
    def _sparse_attention(self, sparsity):
        return f"""def attention(q, k, v):
    # Sparse causal attention with sparsity = {sparsity}
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply causal mask
    seq_len = att.size(-1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(att.device)
    
    # Create sparse random mask
    sparse_mask = torch.rand(seq_len, seq_len).to(att.device) > {sparsity}
    
    # Always attend to current token
    for i in range(seq_len):
        sparse_mask[i, i] = True
    
    # Combine masks
    combined_mask = causal_mask * sparse_mask.float()
    att = att.masked_fill(combined_mask == 0, float('-inf'))
    
    att = F.softmax(att, dim=-1)
    return att @ v"""
    
    def _custom_attention(self, custom_mask):
        return f"""def attention(q, k, v):
    # Custom attention pattern
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply custom mask
    seq_len = att.size(-1)
    
    # Create custom mask based on provided expression
    custom_mask = {custom_mask}
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(att.device)
    
    # Combine masks
    combined_mask = causal_mask * custom_mask.float()
    att = att.masked_fill(combined_mask == 0, float('-inf'))
    
    att = F.softmax(att, dim=-1)
    return att @ v"""






class MathematicalTransformerNode:
    """
    Node that creates a transformer defined entirely by mathematical expressions.
    
    Inputs:
      - attention_fn: Mathematical expression for attention
      - mlp_fn: Mathematical expression for MLP
      - layer_norm_fn: Mathematical expression for layer normalization
      - residual_fn: Mathematical expression for residual connections
    
    Outputs:
      - transformer_code: Complete transformer implementation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_fn": ("STRING", {
                    "multiline": True, 
                    "default": "(q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))"
                }),
                "mlp_fn": ("STRING", {
                    "multiline": True, 
                    "default": "F.gelu(fc1(x)) @ fc2.weight.t() + fc2.bias"
                }),
                "layer_norm_fn": ("STRING", {
                    "multiline": True, 
                    "default": "(x - x.mean(-1, keepdim=True)) / (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * weight + bias"
                }),
                "residual_fn": ("STRING", {
                    "multiline": True, 
                    "default": "x + residual"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transformer_code",)
    FUNCTION = "generate_transformer"
    CATEGORY = "transformer/mathematical"
    
    def generate_transformer(self, attention_fn, mlp_fn, layer_norm_fn, residual_fn):
        # Create a transformer implementation based on mathematical expressions
        transformer_code = f"""class MathematicalTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Attention parameters
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)
        
        # MLP parameters
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        
        # Layer normalization parameters
        self.ln1_weight = nn.Parameter(torch.ones(dim))
        self.ln1_bias = nn.Parameter(torch.zeros(dim))
        self.ln2_weight = nn.Parameter(torch.ones(dim))
        self.ln2_bias = nn.Parameter(torch.zeros(dim))
    
    def _layer_norm(self, x, weight, bias):
        # Custom layer norm based on user expression
        return {layer_norm_fn}
    
    def _attention(self, q, k, v):
        # Prepare inputs for attention
        B, T, C = q.size()
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply custom attention expression
        att = {attention_fn}
        
        # Apply causal mask
        mask = torch.tril(torch.ones(T, T)).to(att.device).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y
    
    def _mlp(self, x):
        # Apply custom MLP expression
        fc1 = self.fc1
        fc2 = self.fc2
        return {mlp_fn}
    
    def _residual(self, x, residual):
        # Apply custom residual expression
        return {residual_fn}
    
    def forward(self, x):
        # First sub-layer: Attention
        ln1 = self._layer_norm(x, self.ln1_weight, self.ln1_bias)
        qkv = self.c_attn(ln1)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_out = self._attention(q, k, v)
        attn_out = self.c_proj(attn_out)
        x = self._residual(x, attn_out)
        
        # Second sub-layer: MLP
        ln2 = self._layer_norm(x, self.ln2_weight, self.ln2_bias)
        mlp_out = self._mlp(ln2)
        x = self._residual(x, mlp_out)
        
        return x

class MathematicalTransformer(nn.Module):
    def __init__(self, n_layers=12, n_heads=8, dim=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            MathematicalTransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
"""
        return (transformer_code,)
    
class ExecuteArbitraryCodeNode:
    """
    Write and execute arbitrary code. 
    This code is evaluated and the outputs of it injected into a workflow.

    Inputs:
        - inputs(Optional[Any]|list[Optional[Any]]): Inputs necessary to run a given code block.

    Outputs:
      - outputs(Optional[Any]|Optional[list[Any]]): Outputs from the code, if any. 
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_fn": ("STRING", {
                    "multiline": True, 
                    "default": "(q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))"
                }),
                "mlp_fn": ("STRING", {
                    "multiline": True, 
                    "default": "F.gelu(fc1(x)) @ fc2.weight.t() + fc2.bias"
                }),
                "layer_norm_fn": ("STRING", {
                    "multiline": True, 
                    "default": "(x - x.mean(-1, keepdim=True)) / (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * weight + bias"
                }),
                "residual_fn": ("STRING", {
                    "multiline": True, 
                    "default": "x + residual"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transformer_code",)
    FUNCTION = "generate_transformer"
    CATEGORY = "transformer/mathematical"