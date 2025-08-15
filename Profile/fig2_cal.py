
batch_size = 32
token = 1
models = ["OPT-175B", "LLaMA-3.1-405B"]

for model in models:
    if model == "OPT-175B":
        hidden_size = 12288
        ffn_dim = 49152
        num_layers = 96
    elif model == "LLaMA-3.1-405B":
        hidden_size = 16384
        ffn_dim = 53248
        num_layers = 126
    
    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768]:
        linear_ops = 0
        attn_ops = 0
        if model == "OPT-175B":
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # q_proj   [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # k_proj   [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # v_proj   [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # out_proj [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * ffn_dim      # fc1      [bs, 1, hidden_size] @ [hidden_size, ffn_dim]
            linear_ops += 2 * batch_size * token * ffn_dim * hidden_size      # fc2      [bs, 1, ffn_dim] @ [ffn_dim, hidden_size]
            
            attn_ops += 2 * batch_size * token * hidden_size * seq_len        # q @ k'   [bs, 1, hidden_size] @ [hidden_size, seq_len]
            attn_ops += 2 * batch_size * token * seq_len * hidden_size        # s @ v    [bs, 1, seq_len] @ [seq_len, hidden_size]
            
        elif model == "LLaMA-3.1-405B":
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # q_proj     [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # k_proj     [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # v_proj     [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * hidden_size  # out_proj   [bs, 1, hidden_size] @ [hidden_size, hidden_size]
            linear_ops += 2 * batch_size * token * hidden_size * ffn_dim      # gate_proj  [bs, 1, hidden_size] @ [hidden_size, ffn_dim]
            linear_ops += 2 * batch_size * token * hidden_size * ffn_dim      # up_proj    [bs, 1, hidden_size] @ [hidden_size, ffn_dim]
            linear_ops += 2 * batch_size * token * ffn_dim * hidden_size      # down_proj  [bs, 1, ffn_dim] @ [ffn_dim, hidden_size]
            
            attn_ops += 2 * batch_size * token * hidden_size * seq_len        # q @ k'     [bs, 1, hidden_size] @ [hidden_size, seq_len]
            attn_ops += 2 * batch_size * token * seq_len * hidden_size        # s @ v      [bs, 1, seq_len] @ [seq_len, hidden_size]
        
        total_ops = linear_ops + attn_ops
        linear_ops_percentage = linear_ops / total_ops
        attn_ops_percentage = attn_ops / total_ops
        print("Model: {}, Sequence Length: {}, Linear Ops: {:.2f}%, Attn Ops: {:.2f}%".format(model, seq_len, linear_ops_percentage * 100, attn_ops_percentage * 100))
        
    