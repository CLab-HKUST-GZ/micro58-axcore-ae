opt_125m = [
    # layer 0
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 1
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 2
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 3
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 4
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 5
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 6
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 7
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 8
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 9
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 10
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2
    # layer 11
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # q_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # k_proj
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # v_proj
# [[2048, 768], [2048, 768], [2048, 2048], [], [], 4, 1], # s=q*k'
# [[2048, 2048], [768, 2048], [2048, 768], [], [], 4, 1], # o=s*v
[[2048, 768], [768, 768], [2048, 768], [], [], 4, 1],   # out_proj
[[2048, 768], [3072, 768], [2048, 3072], [], [], 4, 1], # fc1
[[2048, 3072], [768, 3072], [2048, 768], [], [], 4, 1], # fc2

[[1, 768], [50272, 768], [1, 50272], [], [], 4, 1], # lm_head    
]

def generate_opt_layers(hidden_size, ffn_dim, seq_len=2048, num_layers=1, vocab_size=50272, wbits=4):
    """
    Automatically generate layer configurations for OPT models
    
    Args:
        hidden_size: Model's hidden dimension size
        intermediate_size: Size of the intermediate layer in FFN
        batch_size: Batch size for matrix multiplication
        num_layers: Number of layers to generate
        vocab_size: Vocabulary size for lm_head
        wbits: Weight bits (default: 4)
        abits: Activation bits (default: 1)
        
    Returns:
        List of layer configurations
    """
    layers = []
    
    # Generate configurations for each layer
    for layer_idx in range(num_layers):
        
        # Attention projections
        q_proj = [[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1]
        k_proj = [[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1]
        v_proj = [[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1]
        
        # Add commented attention computation (as in your original code)
        # s_comp = f"# [[{seq_len}, {hidden_size}], [{seq_len}, {hidden_size}], [{seq_len}, {seq_len}], [], [], {wbits}, {1}]"
        # o_comp = f"# [[{seq_len}, {seq_len}], [{hidden_size}, {seq_len}], [{seq_len}, {hidden_size}], [], [], {wbits}, {1}]"
        
        # Output projection and FFN
        out_proj = [[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1]
        fc1 =      [[seq_len, hidden_size], [ffn_dim, hidden_size],     [seq_len, ffn_dim],     [], [], wbits, 1]
        fc2 =      [[seq_len, ffn_dim],     [hidden_size, ffn_dim],     [seq_len, hidden_size], [], [], wbits, 1]
        
        # Add to layers list with comments
        # layers.append([[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1],)   # q_proj"
        # layers.append([[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1],)   # k_proj"
        # layers.append([[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1],)   # v_proj"
        # layers.append([[seq_len, hidden_size], [seq_len, hidden_size],     [seq_len, seq_len],     [], [], wbits, 1],)   # s=q*k'
        # layers.append([[seq_len, seq_len],     [hidden_size, seq_len],     [seq_len, hidden_size], [], [], wbits, 1],)   # o=s*v
        # layers.append([[seq_len, hidden_size], [hidden_size, hidden_size], [seq_len, hidden_size], [], [], wbits, 1],)   # out_proj"
        # layers.append([[seq_len, hidden_size], [ffn_dim, hidden_size],     [seq_len, ffn_dim],     [], [], wbits, 1],)   # fc1
        # layers.append([[seq_len, ffn_dim],     [hidden_size, ffn_dim],     [seq_len, hidden_size], [], [], wbits, 1],)   # fc2
        layers.append(q_proj)
        layers.append(k_proj)
        layers.append(v_proj)
        layers.append(out_proj)
        layers.append(fc1)
        layers.append(fc2)
    
    # Add lm_head
    layers.append([[1, hidden_size], [vocab_size, hidden_size], [1, vocab_size], [], [], wbits, 1],) # lm_head")
    
    return layers




opt_350m = generate_opt_layers(hidden_size=1024, ffn_dim=4096, num_layers=24, seq_len=1)

opt_1_3b = generate_opt_layers(hidden_size=2048, ffn_dim=8192, num_layers=24, seq_len=1)

opt_2_7b = generate_opt_layers(hidden_size=2560, ffn_dim=10240, num_layers=32, seq_len=1)

opt_6_7b = generate_opt_layers(hidden_size=4096, ffn_dim=16384, num_layers=32, seq_len=1)

opt_13b = generate_opt_layers(hidden_size=5120, ffn_dim=20480, num_layers=40, seq_len=1)

opt_30b = generate_opt_layers(hidden_size=7168, ffn_dim=28672, num_layers=48, seq_len=1)

opt_66b = generate_opt_layers(hidden_size=9216, ffn_dim=36864, num_layers=64, seq_len=1)

opt_175b = generate_opt_layers(hidden_size=12288, ffn_dim=49152, num_layers=96, seq_len=1)