import torch
from .quantization.quant_funcs import *
from .utils import get_module_by_name_suffix

from .quantization.qlinear import WLinear, DSEWLinear, AxCoreDSEWLinear
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import math


def print_vram_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(torch.cuda.memory_summary())

def approximation_wrapper(model, args, torch_dtype):
    
    if args.kvquant:
        if args.quant_method == "RTN":
            if args.attention_approx:
                from .baseline_approx.baseline_approx import BaselineApproxLinearFP16, BaselineApproxLinearBF16
                from .L_Mul.LMul_approx import LMulApproxLinearFP16, LMulApproxLinearBF16
                from .FPMA.FPMA_approx import FPMAApproxLinearFP16, FPMAApproxLinearBF16
                if 'qwen2' in model.config._name_or_path.lower():
                    from .attn_ap.sw.qwen2_attn_ap import Qwen2Attention
                    for i, block in tqdm(enumerate(model.model.layers)):
                        new_attn = Qwen2Attention(args, model.config, block.self_attn.layer_idx).to(torch_dtype).to(block.self_attn.q_proj.weight.device)
                        new_attn.load_state_dict(block.self_attn.state_dict())
                        block.self_attn = new_attn
                elif 'opt' in model.config._name_or_path.lower():
                    from .attn_ap.sw.opt_attn_ap import AP_OPTAttention
                    for i, block in tqdm(enumerate(model.model.decoder.layers)):
                        new_attn = AP_OPTAttention(args, model.config, block.self_attn.layer_idx).to('cpu')
                        new_attn.load_state_dict({k: v.to('cpu') for k, v in block.self_attn.state_dict().items()})
                        new_attn = new_attn.to(torch_dtype).to(block.self_attn.q_proj.weight.device)
                        block.self_attn = new_attn
                        torch.cuda.empty_cache()  
                        import gc
                        gc.collect()
                elif 'llama' in model.config._name_or_path.lower():
                    from .attn_ap.sw.llama_attn_ap import AP_LlamaAttention
                    for i, block in tqdm(enumerate(model.model.layers)):
                        new_attn = AP_LlamaAttention(args, model.config, block.self_attn.layer_idx).to('cpu')
                        new_attn.load_state_dict({k: v.to('cpu') for k, v in block.self_attn.state_dict().items()})
                        new_attn = new_attn.to(torch_dtype).to(block.self_attn.q_proj.weight.device)
                        block.self_attn = new_attn
                        torch.cuda.empty_cache()  
                        import gc
                        gc.collect()
                        
                torch.cuda.empty_cache()  
    elif args.attention_approx:
        if 'llama' in model.config.architectures[0].lower():
            from .attn_ap.sw.llama_attn_ap import ApproxLlamaAttention
            for i, block in tqdm(enumerate(model.model.layers)):
                new_attn = ApproxLlamaAttention(args, model.config, block.self_attn.layer_idx).to(torch_dtype).to(block.self_attn.q_proj.weight.device)
                new_attn.load_state_dict(block.self_attn.state_dict())
                block.self_attn = new_attn
        elif 'mistral' in model.config.architectures[0].lower():
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    # Weight-only quantization
    if args.wquant:
        from .baseline_approx.baseline_approx import BaselineApproxLinearFP16, BaselineApproxLinearBF16
        from .L_Mul.LMul_approx import LMulApproxLinearFP16, LMulApproxLinearBF16
        from .FPMA.FPMA_approx import FPMAApproxLinearFP16, FPMAApproxLinearBF16
        from .AxCore.AxCore import AxCoreLinearFP16
        if args.dtype == "bfloat16":
            if args.approx_kernel == 'baseline':
                aplinear = BaselineApproxLinearBF16
            elif args.approx_kernel == 'LMul':
                aplinear = LMulApproxLinearBF16
            elif args.approx_kernel == 'FPMA':
                aplinear = FPMAApproxLinearBF16
            else:
                raise ValueError("Unknown approximation kernel")
        elif args.dtype == "float16":
            if args.approx_kernel == 'baseline':
                aplinear = BaselineApproxLinearFP16
            elif args.approx_kernel == 'LMul':
                aplinear = LMulApproxLinearFP16
            elif args.approx_kernel == 'FPMA':
                aplinear = FPMAApproxLinearFP16
            else:
                raise ValueError("Unknown approximation kernel")
        if args.quant_method == "RTN":
            if args.linear_approx:
                for name, module in tqdm(model.named_modules()):
                    if (isinstance(module, torch.nn.Linear) or isinstance(module, DSEWLinear)) and 'lm_head' not in name and 'output_layer' not in name:
                        # print(f'name: {name}')
                        # if '.0.' in name:
                        #     if 'self_attn.o_proj' in name:
                        #         # save original weight
                        #         original_weight = module.weight.data
                        #         torch.save(original_weight, f"/data-ssd/home/jiaxiang/codes/draw/AxCore/data/llama2_7b_0_attn_o.pt")
                        #         continue
                        # if '.29.' in name:
                        #     if 'self_attn.o_proj' in name:
                        #         # save original weight
                        #         original_weight = module.weight.data
                        #         torch.save(original_weight, f"/data-ssd/home/jiaxiang/codes/draw/AxCore/data/llama2_7b_29_attn_o.pt")
                        #         continue
                        # if '.39.' in name:
                        #     if 'self_attn.out_proj' in name:
                        #         # save original weight
                        #         original_weight = module.weight.data
                        #         torch.save(original_weight, f"/data-ssd/home/jiaxiang/codes/draw/AxCore/data/opt_13b_39_attn_o.pt")
                        #         continue
                        # module.weight.data, scales_w = pseudo_quantize_tensor_approx_mxAP(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, mantissa_bit=1)
                        # module.weight.data = pseudo_quantize_tensor(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=False, mantissa_bit=1)
                        # module.weight.data = pseudo_quantize_tensor_approx_dse(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=True, mantissa_bit=1)
                        # module.weight.data = pseudo_quantize_tensor_approx(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=True, mantissa_bit=1)
                        # module.weight.data = pseudo_quantize_tensor_dse(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size)
                        device = next(module.parameters()).device
                        if args.dtype == "float16":
                            module.weight.data, scales_w = pseudo_quantize_tensor_approx_mxAP(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, mantissa_bit=1)
                            module.weight.data = module.weight.data.to('cpu')
                            scales_w = scales_w.to('cpu')
                            new_linear = AxCoreLinearFP16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev='cpu')
                            with torch.no_grad():
                                new_linear.weight.data = (module.weight.data.to(torch.float16)).contiguous()
                                new_linear.scales.data = (scales_w.to(torch.float16)).contiguous()
                                if module.bias is not None:
                                    new_linear.bias.data = (module.bias.data.to(torch.float16)).contiguous()
                            new_linear = new_linear.to(device)
                            father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                            setattr(father_module, name.split('.')[-1], new_linear)
                            del new_linear, module
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise NotImplementedError
                        torch.cuda.empty_cache() 
                        # try:
                        #     # module.weight.data = pseudo_quantize_tensor_approx(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=True, mantissa_bit=1)
                        #     # module.weight.data = pseudo_quantize_tensor(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=True, mantissa_bit=1)
                        #     rtn_linear = aplinear(module.in_features, 
                        #                             module.out_features, 
                        #                             module.bias is not None, 
                        #                             dev=next(module.parameters()).device)
                        #     with torch.no_grad():
                        #         rtn_linear.weight.data = (module.weight.data.to(torch_dtype)).contiguous()
                        #         if module.bias is not None:
                        #             rtn_linear.bias.data = (module.bias.data.to(torch_dtype)).contiguous()
                        #     father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        #     setattr(father_module, name.split('.')[-1], rtn_linear)
                        #     del rtn_linear, module
                        #     torch.cuda.empty_cache()  
                        #     torch.cuda.synchronize()     
                        # except Exception as e:
                        #     print(f"Failed to quantize {name}, error: {e}")  
            else:
                # if (args.w_bit is not None and args.w_bit < 16) and (args.a_bit is None or args.a_bit >= 16):
                #     assert args.w_bit > 0 and args.w_bit < 16, "Weight bitwidth should be an integer between [1, 16] for weigth-only quantization, please check."
                # Use original Linear module
                for name, module in tqdm(model.named_modules()):
                    if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                        try:
                            module.weight.data = pseudo_quantize_tensor(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size, zero_point=False, fpq=False, mantissa_bit=1)
                            torch.cuda.empty_cache()       
                        except Exception as e:
                            print(f"Failed to quantize {name}, error: {e}")  
       
        else:
            raise NotImplementedError
        
    elif args.waquant:
        
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import gc
            
        print("Using pileval calibration for DSE...")
        
        # Load calibration data in a simple, direct way
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
            if data == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
                # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            else:
                raise NotImplementedError
            dataset = dataset.shuffle(seed=42)
            samples = []
            n_run = 0
            for data in dataset:
                line = data["text"]
                line = line.strip()
                line_encoded = tokenizer.encode(line)
                if len(line_encoded) > 512:
                    continue
                sample = torch.tensor([line_encoded])
                if sample.numel() == 0:
                    continue
                samples.append(sample)
                n_run += 1
                if n_run == n_samples:
                    break
            # now concatenate all samples and split according to block size
            cat_samples = torch.cat(samples, dim=1)
            n_split = cat_samples.shape[1] // block_size
            print(f" * Split into {n_split} blocks")
            return [
                cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
            ]

        samples = get_calib_dataset(
            data="pileval", tokenizer=tokenizer, n_samples=128, block_size=512
        )
        print(f'samples.shape: {samples[0].shape}')
        # Process calibration data for DSE quantization
        print("Processing calibration data for DSE...")
        
        # Dictionary to store calibration inputs for each module
        calibration_inputs = {}
        
        # Register forward hooks to collect calibration data
        hooks = []
        
        def save_input_hook(name):
            def hook(module, inputs, _):
                # Ensure input is valid and not empty
                if inputs[0] is None or inputs[0].numel() == 0:
                    return
                    
                # Handle tensor device placement
                input_tensor = inputs[0].detach().to('cpu')
                
                if name not in calibration_inputs:
                    calibration_inputs[name] = input_tensor
                elif calibration_inputs[name].size(0) <= 256:  # Limit data size to avoid OOM
                    # Ensure consistent device placement
                    if input_tensor.device != calibration_inputs[name].device:
                        input_tensor = input_tensor.to(calibration_inputs[name].device)
                    # Ensure consistent dtype
                    if input_tensor.dtype != calibration_inputs[name].dtype:
                        input_tensor = input_tensor.to(dtype=calibration_inputs[name].dtype)
                    # Concatenate tensors
                    calibration_inputs[name] = torch.cat([calibration_inputs[name], input_tensor], dim=0)
            return hook
        
        # Register hooks for all linear layers (except output layers)
        linear_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                hooks.append(module.register_forward_hook(save_input_hook(name)))
                linear_count += 1
        
        print(f"Registered hooks for {linear_count} linear layers")
        
        # Run forward pass with samples to collect calibration data
        print("Collecting calibration data with forward passes")
        with torch.no_grad():
            # Use up to 4 batches of samples for better calibration coverage
            # num_batches = min(4, len(samples))
            # num_batches = min(10, len(samples))
            num_batches = len(samples)
            for i in range(num_batches):
                input_batch = samples[i].to(model.device)
                print(f"Processing calibration batch {i+1}/{num_batches}, shape: {input_batch.shape}")
                _ = model(input_batch, use_cache=False)
                torch.cuda.empty_cache()
                gc.collect()
        del samples
        gc.collect()
        torch.cuda.empty_cache()
        
        # Remove hooks after collecting data
        for hook in hooks:
            hook.remove()
        
        # Store the original device for each module
        module_original_devices = {}
        
        # First, move all linear modules to CPU to free up GPU memory
        print("Moving linear modules to CPU to conserve GPU memory...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                # Store original device
                module_original_devices[name] = next(module.parameters()).device
                # Move to CPU
                module.to('cpu')
                calibration_inputs[name] = calibration_inputs[name].to('cpu')
                torch.cuda.empty_cache()  
                gc.collect()
        
        # Counter for successful quantizations
        quantized_count = 0
        M_counts = [0, 0, 0]
        
        model_name = args.model.lower()
        if ("opt-2.7b" in model_name) or ("llama-2-7b" in model_name) or ("llama2_7b" in model_name):
            use_approx_dequant = False
        else:
            use_approx_dequant = True
        # Check if sequential calibration mode is enabled
        if hasattr(args, 'sequential_calib') and args.sequential_calib:
            print("Using true sequential error-propagation calibration mode...")
            
            # True sequential calibration: properly handle transformer blocks with residuals and layer norms
            
            # Organize layers by transformer blocks
            transformer_blocks = {}
            layer_norm_modules = {}
            
            # Collect transformer blocks and layer norms
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                    if name in calibration_inputs and calibration_inputs[name].size(0) > 0:
                        # Extract block index from layer name
                        parts = name.split('.')
                        block_idx = None
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                block_idx = int(part)
                                break
                        
                        if block_idx is not None:
                            if block_idx not in transformer_blocks:
                                transformer_blocks[block_idx] = {'attention': [], 'mlp': []}
                            
                            # Categorize layer type
                            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
                                transformer_blocks[block_idx]['attention'].append((name, module))
                            elif any(x in name for x in ['o_proj', 'out_proj', 'dense']):
                                transformer_blocks[block_idx]['attention'].append((name, module))
                            elif any(x in name for x in ['gate_proj', 'up_proj', 'fc1', 'fc2', 'down_proj']):
                                transformer_blocks[block_idx]['mlp'].append((name, module))
                
                # Collect layer norm modules
                elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
                    layer_norm_modules[name] = module
            
            print(f"Found {len(transformer_blocks)} transformer blocks for sequential calibration")
            
            # Initialize with first block's first layer calibration data
            current_hidden_states = None
            
            # Process blocks sequentially
            for block_idx in tqdm(sorted(transformer_blocks.keys())):
                block_layers = transformer_blocks[block_idx]
                print(f" * Processing transformer block {block_idx}")
                
                # Find appropriate layer norm for this block
                input_layernorm = None
                post_attention_layernorm = None
                
                for ln_name, ln_module in layer_norm_modules.items():
                    if f'.{block_idx}.' in ln_name:
                        if any(x in ln_name for x in ['input_layernorm', 'ln_1', 'self_attn_layer_norm']):
                            input_layernorm = ln_module
                        elif any(x in ln_name for x in ['post_attention_layernorm', 'ln_2', 'final_layer_norm']):
                            post_attention_layernorm = ln_module
                
                # Get input for this block
                if current_hidden_states is None:
                    # Use first layer's original calibration data
                    first_layer_name = block_layers['attention'][0][0] if block_layers['attention'] else block_layers['mlp'][0][0]
                    current_hidden_states = calibration_inputs[first_layer_name].clone()
                
                # Store residual for attention
                residual = current_hidden_states.clone()
                
                # Apply input layer norm
                if input_layernorm is not None:
                    target_device = next(input_layernorm.parameters()).device
                    current_hidden_states = current_hidden_states.to(target_device)
                    input_layernorm.to(target_device)
                    current_hidden_states = input_layernorm(current_hidden_states)
                    current_hidden_states = current_hidden_states.cpu()
                
                # Process attention layers
                attention_input = current_hidden_states
                
                # Separate Q, K, V, and output projection layers
                q_layers = [(n, m) for n, m in block_layers['attention'] if any(x in n for x in ['q_proj', 'query'])]
                k_layers = [(n, m) for n, m in block_layers['attention'] if any(x in n for x in ['k_proj', 'key'])]
                v_layers = [(n, m) for n, m in block_layers['attention'] if any(x in n for x in ['v_proj', 'value'])]
                o_layers = [(n, m) for n, m in block_layers['attention'] if any(x in n for x in ['o_proj', 'out_proj', 'dense'])]
                
                # Process Q, K, V projections
                q_outputs, k_outputs, v_outputs = [], [], []
                
                # Quantize Q projections
                for name, module in q_layers:
                    print(f"   -> Quantizing Q projection {name}")
                    target_device = module_original_devices[name]
                    module.to(target_device)
                    q_input = attention_input.to(target_device)
                    
                    with torch.no_grad():
                        new_linear, m_counts = AxCoreDSEWLinear.from_float(
                            module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                            use_approx_dequant=use_approx_dequant,
                            calibration_data=q_input
                        )
                        m_counts = m_counts.to('cpu')
                        M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                    
                    # Replace module
                    parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                    old_module = getattr(parent_module, name.split('.')[-1])
                    setattr(parent_module, name.split('.')[-1], new_linear)
                    del old_module
                    
                    # Generate Q output
                    new_linear.to(target_device)
                    with torch.no_grad():
                        q_output = new_linear(q_input)
                    new_linear.to('cpu')
                    q_outputs.append(q_output.cpu())
                    quantized_count += 1
                
                # Quantize K projections
                for name, module in k_layers:
                    print(f"   -> Quantizing K projection {name}")
                    target_device = module_original_devices[name]
                    module.to(target_device)
                    k_input = attention_input.to(target_device)
                    
                    with torch.no_grad():
                        new_linear, m_counts = AxCoreDSEWLinear.from_float(
                            module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                            use_approx_dequant=use_approx_dequant,
                            calibration_data=k_input
                        )
                        m_counts = m_counts.to('cpu')
                        M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                    
                    # Replace module
                    parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                    old_module = getattr(parent_module, name.split('.')[-1])
                    setattr(parent_module, name.split('.')[-1], new_linear)
                    del old_module
                    
                    # Generate K output
                    new_linear.to(target_device)
                    with torch.no_grad():
                        k_output = new_linear(k_input)
                    new_linear.to('cpu')
                    k_outputs.append(k_output.cpu())
                    quantized_count += 1
                
                # Quantize V projections
                for name, module in v_layers:
                    print(f"   -> Quantizing V projection {name}")
                    target_device = module_original_devices[name]
                    module.to(target_device)
                    v_input = attention_input.to(target_device)
                    
                    with torch.no_grad():
                        new_linear, m_counts = AxCoreDSEWLinear.from_float(
                            module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                            use_approx_dequant=use_approx_dequant,
                            calibration_data=v_input
                        )
                        m_counts = m_counts.to('cpu')
                        M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                    
                    # Replace module
                    parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                    old_module = getattr(parent_module, name.split('.')[-1])
                    setattr(parent_module, name.split('.')[-1], new_linear)
                    del old_module
                    
                    # Generate V output
                    new_linear.to(target_device)
                    with torch.no_grad():
                        v_output = new_linear(v_input)
                    new_linear.to('cpu')
                    v_outputs.append(v_output.cpu())
                    quantized_count += 1
                
                # Perform attention computation with quantized Q, K, V
                if q_outputs and k_outputs and v_outputs:
                    print(f"   -> Computing attention with quantized Q, K, V")
                    
                    # Get single Q, K, V outputs (not concatenation for multi-head)
                    Q = q_outputs[0]  # Single Q projection output
                    K = k_outputs[0]  # Single K projection output  
                    V = v_outputs[0]  # Single V projection output
                    
                    # Move to GPU for attention computation
                    device = next(model.parameters()).device
                    Q, K, V = Q.to(device), K.to(device), V.to(device)
                    
                    # Get attention configuration from model
                    batch_size, seq_len = Q.shape[:2]
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
                    
                    # CRITICAL FIX 5: Conditionally apply RoPE for LLaMA-style models
                    if "llama" in model.config.model_type.lower() or "mistral" in model.config.model_type.lower():
                        print(f"   -> Applying RoPE for {model.config.model_type}")
                        try:
                            # Find the rotary embedding module from the original model
                            rotary_emb = None
                            for name, module in model.named_modules():
                                if name.endswith('rotary_emb'):
                                    rotary_emb = module
                                    break
                            
                            if rotary_emb is not None:
                                rotary_emb.to(device)
                                cos, sin = rotary_emb(V, seq_len=seq_len)
                                
                                # Import the RoPE application function
                                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
                            else:
                                print("   -> Warning: Could not find rotary_emb module.")

                        except Exception as e:
                            print(f"   -> Warning: RoPE application failed: {e}")

                    # Get attention configuration from model
                    batch_size, seq_len = Q.shape[:2]
                    
                    # Try to get num_heads from model config
                    try:
                        if hasattr(model.config, 'num_attention_heads'):
                            num_heads = model.config.num_attention_heads
                        elif hasattr(model.config, 'n_head'):
                            num_heads = model.config.n_head
                        else:
                            # Fallback: assume head_dim = 64
                            num_heads = Q.shape[-1] // 64
                        
                        head_dim = Q.shape[-1] // num_heads
                        scaling = head_dim ** -0.5
                        
                        # CRITICAL FIX 1: Apply scaling to Q immediately after projection (like official implementation)
                        Q = Q * scaling
                        
                        # Reshape for multi-head attention
                        Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        
                        with torch.no_grad():
                            # Scaled dot-product attention (scaling already applied to Q)
                            attention_scores = torch.matmul(Q, K.transpose(-2, -1))
                            
                            # CRITICAL FIX 2: Improved causal mask handling
                            if seq_len > 1:
                                # Create proper 4D causal mask like official implementation
                                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                                # Expand to match attention_scores shape [batch, heads, seq, seq]
                                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
                                attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
                            
                            # CRITICAL FIX 3: Numerical stability - upcast to fp32 for softmax
                            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                            
                            # CRITICAL FIX 4: Add dropout for training consistency (even if eval mode)
                            # Note: During calibration, we should simulate training-time behavior
                            dropout_p = getattr(model.config, 'attention_dropout', 0.0)
                            if dropout_p > 0.0:
                                attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout_p, training=False)
                            
                            attention_output = torch.matmul(attention_weights, V)
                            
                            # Reshape back to original format
                            attention_output = attention_output.transpose(1, 2).contiguous()
                            attention_output = attention_output.view(batch_size, seq_len, -1)
                        
                        print(f"   -> Multi-head attention: {num_heads} heads, head_dim={head_dim}, dropout={dropout_p}")
                        
                    except Exception as e:
                        print(f"   -> Warning: Multi-head attention failed, using simple attention: {e}")
                        # Fallback to simple attention with fixes
                        with torch.no_grad():
                            d_k = Q.size(-1)
                            scaling = d_k ** -0.5
                            Q = Q * scaling  # Apply scaling to Q
                            attention_scores = torch.matmul(Q, K.transpose(-2, -1))
                            
                            # Apply causal mask
                            if seq_len > 1:
                                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                                attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
                            
                            # Numerical stability
                            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                            attention_output = torch.matmul(attention_weights, V)
                    
                    attention_output = attention_output.cpu()
                    
                    # Clean up GPU memory immediately
                    del Q, K, V, attention_scores, attention_weights
                    torch.cuda.empty_cache()
                    
                else:
                    # Fallback: use input if no proper Q,K,V found
                    attention_output = attention_input
                
                # Process output projection layers
                for name, module in o_layers:
                    print(f"   -> Quantizing output projection {name}")
                    target_device = module_original_devices[name]
                    module.to(target_device)
                    o_input = attention_output.to(target_device)
                    
                    with torch.no_grad():
                        new_linear, m_counts = AxCoreDSEWLinear.from_float(
                            module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                            use_approx_dequant=use_approx_dequant,
                            calibration_data=o_input
                        )
                        m_counts = m_counts.to('cpu')
                        M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                    
                    # Replace module
                    parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                    old_module = getattr(parent_module, name.split('.')[-1])
                    setattr(parent_module, name.split('.')[-1], new_linear)
                    del old_module
                    
                    # Generate final attention output
                    new_linear.to(target_device)
                    with torch.no_grad():
                        attention_output = new_linear(o_input)
                    new_linear.to('cpu')
                    attention_output = attention_output.cpu()
                    quantized_count += 1
                    
                    # Clean up immediately
                    del o_input
                    torch.cuda.empty_cache()
                
                # CRITICAL FIX 6: Add dropout after attention (like official implementation)
                dropout_p = getattr(model.config, 'dropout', 0.0)
                if dropout_p > 0.0:
                    attention_output = torch.nn.functional.dropout(attention_output, p=dropout_p, training=False)
                
                # Apply residual connection for attention
                attention_output = attention_output + residual
                
                # Apply post-attention layer norm BEFORE MLP (Pre-LayerNorm architecture)
                if post_attention_layernorm is not None:
                    print(f"   -> Applying post-attention layer norm BEFORE MLP")
                    target_device = next(post_attention_layernorm.parameters()).device
                    attention_output = attention_output.to(target_device)
                    post_attention_layernorm.to(target_device)
                    mlp_input = post_attention_layernorm(attention_output)
                    mlp_input = mlp_input.cpu()
                    attention_output = attention_output.cpu()  # Keep for residual
                else:
                    mlp_input = attention_output
                
                # Store residual for MLP
                residual = attention_output.clone()
                
                # Process MLP layers
                mlp_input = attention_output
                
                # Separate MLP layers by type
                gate_layers = [(n, m) for n, m in block_layers['mlp'] if 'gate_proj' in n]
                up_layers = [(n, m) for n, m in block_layers['mlp'] if 'up_proj' in n]
                down_layers = [(n, m) for n, m in block_layers['mlp'] if 'down_proj' in n]
                fc1_layers = [(n, m) for n, m in block_layers['mlp'] if 'fc1' in n]
                fc2_layers = [(n, m) for n, m in block_layers['mlp'] if 'fc2' in n]
                
                # Handle LLaMA-style MLP (gate_proj + up_proj + down_proj)
                if gate_layers and up_layers and down_layers:
                    print(f"   -> Processing LLaMA-style MLP (gate + up + down)")
                    
                    # Process gate_proj
                    gate_outputs = []
                    for name, module in gate_layers:
                        print(f"   -> Quantizing gate projection {name}")
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        gate_input = mlp_input.to(target_device)
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=gate_input
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate gate output with SiLU activation
                        new_linear.to(target_device)
                        with torch.no_grad():
                            gate_output = new_linear(gate_input)
                            gate_output = torch.nn.functional.silu(gate_output)  # Apply SiLU
                        new_linear.to('cpu')
                        gate_outputs.append(gate_output.cpu())
                        quantized_count += 1
                    
                    # Process up_proj (parallel to gate_proj)
                    up_outputs = []
                    for name, module in up_layers:
                        print(f"   -> Quantizing up projection {name}")
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        up_input = mlp_input.to(target_device)  # Same input as gate_proj
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=up_input
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate up output (no activation)
                        new_linear.to(target_device)
                        with torch.no_grad():
                            up_output = new_linear(up_input)
                        new_linear.to('cpu')
                        up_outputs.append(up_output.cpu())
                        quantized_count += 1
                    
                    # Element-wise multiplication of gate and up outputs
                    if gate_outputs and up_outputs:
                        gate_combined = torch.cat(gate_outputs, dim=-1) if len(gate_outputs) > 1 else gate_outputs[0]
                        up_combined = torch.cat(up_outputs, dim=-1) if len(up_outputs) > 1 else up_outputs[0]
                        mlp_intermediate = gate_combined * up_combined  # Element-wise multiplication
                        print(f"   -> Applied element-wise multiplication: gate * up")
                    else:
                        mlp_intermediate = mlp_input
                    
                    # Process down_proj
                    mlp_output = mlp_intermediate
                    for name, module in down_layers:
                        print(f"   -> Quantizing down projection {name}")
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        down_input = mlp_output.to(target_device)
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=down_input
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate final MLP output
                        new_linear.to(target_device)
                        with torch.no_grad():
                            mlp_output = new_linear(down_input)
                        new_linear.to('cpu')
                        mlp_output = mlp_output.cpu()
                        quantized_count += 1
                
                # Handle OPT-style MLP (fc1 + fc2)
                elif fc1_layers and fc2_layers:
                    print(f"   -> Processing OPT-style MLP (fc1 + fc2)")
                    
                    mlp_output = mlp_input
                    # Process fc1 with ReLU activation
                    for name, module in fc1_layers:
                        print(f"   -> Quantizing fc1 {name}")
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        fc1_input = mlp_output.to(target_device)
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=fc1_input
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate fc1 output with ReLU activation
                        new_linear.to(target_device)
                        with torch.no_grad():
                            mlp_output = new_linear(fc1_input)
                            mlp_output = torch.nn.functional.relu(mlp_output)  # Apply ReLU
                        new_linear.to('cpu')
                        mlp_output = mlp_output.cpu()
                        quantized_count += 1
                    
                    # Process fc2 (no activation)
                    for name, module in fc2_layers:
                        print(f"   -> Quantizing fc2 {name}")
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        fc2_input = mlp_output.to(target_device)
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=fc2_input
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate final MLP output
                        new_linear.to(target_device)
                        with torch.no_grad():
                            mlp_output = new_linear(fc2_input)
                        new_linear.to('cpu')
                        mlp_output = mlp_output.cpu()
                        quantized_count += 1
                
                else:
                    # Fallback: sequential processing for unknown MLP structure
                    print(f"   -> Fallback: sequential MLP processing")
                    mlp_output = mlp_input
                    for name, module in block_layers['mlp']:
                        print(f"   -> Quantizing MLP layer {name}")
                        
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        mlp_output = mlp_output.to(target_device)
                        
                        with torch.no_grad():
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module, w_bit=args.w_bit, weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=mlp_output
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        
                        # Replace module
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        
                        # Generate quantized output with activation
                        new_linear.to(target_device)
                        with torch.no_grad():
                            mlp_output = new_linear(mlp_output)
                            
                            # Apply activation function based on layer type
                            if 'fc1' in name or 'gate_proj' in name:
                                if 'llama' in type(model).__name__.lower() or 'mistral' in type(model).__name__.lower():
                                    mlp_output = torch.nn.functional.silu(mlp_output)
                                elif 'opt' in type(model).__name__.lower():
                                    mlp_output = torch.nn.functional.relu(mlp_output)
                        
                        new_linear.to('cpu')
                        mlp_output = mlp_output.cpu()
                        quantized_count += 1
                
                # Apply residual connection for MLP
                current_hidden_states = mlp_output + residual
                
                # Clean up memory
                torch.cuda.empty_cache()
                gc.collect()
                
                print(f"   -> Block {block_idx} output shape: {current_hidden_states.shape}, std: {torch.std(current_hidden_states):.6f}")
            
        else:
            # Original calibration mode (unchanged)
            for name, module in tqdm(model.named_modules()):
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                    if name in calibration_inputs and calibration_inputs[name].size(0) > 0:
                        print(f" * Quantizing {name} with {calibration_inputs[name].size(0)} calibration samples")
                        
                        # Move current module to proper device for processing
                        target_device = module_original_devices[name]
                        module.to(target_device)
                        
                        with torch.no_grad():
                            current_calib_data = calibration_inputs[name].to(target_device)
                            
                            # Create quantized linear layer using DSE
                            new_linear, m_counts = AxCoreDSEWLinear.from_float(
                                module=module,
                                w_bit=args.w_bit,
                                weight_group=args.w_group_size,
                                use_approx_dequant=use_approx_dequant,
                                calibration_data=current_calib_data
                            )
                            m_counts = m_counts.to('cpu')
                            M_counts = [M_counts[i] + m_counts[i] for i in range(3)]
                        # Replace the module
                        # parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        # setattr(parent_module, name.split('.')[-1], new_linear)
                        parent_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                        old_module = getattr(parent_module, name.split('.')[-1])
                        setattr(parent_module, name.split('.')[-1], new_linear)
                        del old_module
                        quantized_count += 1
                        # Clean up memory after successful quantization
                        del current_calib_data
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Move the new module back to CPU temporarily to free GPU memory
                        new_linear.to('cpu')
                        torch.cuda.synchronize()
                    else:
                        print(f"Warning: No valid calibration data for {name}, skipping quantization")
                    
                    # Clean up references to ensure proper memory management
                    torch.cuda.empty_cache()
                    gc.collect()
        del calibration_inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print M bit distribution
        print(f'M bit distribution in {args.model}: M=0: {M_counts[0]}, M=1: {M_counts[1]}, M=2: {M_counts[2]}')
        
        # After all layers are quantized, move them back to their original devices incrementally
        print("Moving all modules back to their original devices (incrementally)...")
        for name, device in module_original_devices.items():
            try:
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent_module = get_module_by_name_suffix(model, parent_name)
                current_module = getattr(parent_module, module_name)
                current_module.to(device)
                
                # Clean up memory after each module is moved
                torch.cuda.empty_cache()
                gc.collect()
                print(f"  - Moved {name} back to {device}")
            except (AttributeError, RuntimeError) as e:
                print(f"Warning: Could not move {name} back to {device}: {e}")
        
        print(f"Successfully quantized {quantized_count} linear layers")
    
        ## ---------------------------------------------------------- ##
        ## ---------------------- AxCore Test ----------------------- ##
        ## ---------------------------------------------------------- ##
    
    else:
        if args.linear_approx:
            if args.approx_kernel == "baseline":
                from .baseline_approx.baseline_approx import BaselineApproxLinearBF16, BaselineApproxLinearFP16
                from .fp_kernel.fp_kernel import GEMMLinearFP16
                import time
                import gc
                for name, module in tqdm(model.named_modules()):
                        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                        # if isinstance(module, torch.nn.Linear):
                            # if not quant_mix_gate and 'gate' in name and 'mixtral' in model.config.architectures[0].lower():
                            #     pass
                            if args.dtype == "bfloat16":
                                new_linear = BaselineApproxLinearBF16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=next(module.parameters()).device)
                                with torch.no_grad():
                                    new_linear.weight.copy_(module.weight.to(torch.bfloat16))
                                    if module.bias is not None:
                                        new_linear.bias.copy_(module.bias.to(torch.bfloat16))
                            elif args.dtype == "float16":
                                device = next(module.parameters()).device
                                # new_linear = BaselineApproxLinearFP16(module.in_features, 
                                #                                     module.out_features, 
                                #                                     module.bias is not None, 
                                #                                     dev=device)
                                new_linear = GEMMLinearFP16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=device)
                                # new_linear = nn.Linear(module.in_features, 
                                #                                     module.out_features, 
                                #                                     module.bias is not None,)
                                with torch.no_grad():
                                    new_linear.weight.data = (module.weight.data.to(torch.float16)).contiguous()
                                    if module.bias is not None:
                                        new_linear.bias.data = (module.bias.data.to(torch.float16)).contiguous()
                            # module.cpu()
                            # new_linear.to(device)
                            father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                            setattr(father_module, name.split('.')[-1], new_linear)
                            del new_linear, module
                            torch.cuda.empty_cache()
                            # print_vram_usage()
                            gc.collect()
                            # time.sleep(0.01)
                            
            elif args.approx_kernel == "LMul":
                from .L_Mul.LMul_approx import LMulApproxLinearBF16, LMulApproxLinearFP16
                for name, module in tqdm(model.named_modules()):
                        # if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                        if isinstance(module, torch.nn.Linear):
                            # if not quant_mix_gate and 'gate' in name and 'mixtral' in model.config.architectures[0].lower():
                            #     pass
                            if args.dtype == "bfloat16":
                                new_linear = LMulApproxLinearBF16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=next(module.parameters()).device)
                                with torch.no_grad():
                                    new_linear.weight.copy_(module.weight.to(torch.bfloat16))
                                    if module.bias is not None:
                                        new_linear.bias.copy_(module.bias.to(torch.bfloat16))
                            elif args.dtype == "float16":
                                new_linear = LMulApproxLinearFP16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=next(module.parameters()).device)
                                with torch.no_grad():
                                    new_linear.weight.data = (module.weight.data.to(torch.float16)).contiguous()
                                    if module.bias is not None:
                                        new_linear.bias.data = (module.bias.data.to(torch.float16)).contiguous()
                            father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                            setattr(father_module, name.split('.')[-1], new_linear)
                            del new_linear, module
                            torch.cuda.empty_cache()
            
            elif args.approx_kernel == "FPMA":
                from .FPMA.FPMA_approx import FPMAApproxLinearBF16, FPMAApproxLinearFP16
                for name, module in tqdm(model.named_modules()):
                        # if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                        if isinstance(module, torch.nn.Linear):
                            # if not quant_mix_gate and 'gate' in name and 'mixtral' in model.config.architectures[0].lower():
                            #     pass
                            if args.dtype == "bfloat16":
                                new_linear = FPMAApproxLinearBF16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=next(module.parameters()).device)
                                with torch.no_grad():
                                    new_linear.weight.copy_(module.weight.to(torch.bfloat16))
                                    if module.bias is not None:
                                        new_linear.bias.copy_(module.bias.to(torch.bfloat16))
                            elif args.dtype == "float16":
                                new_linear = FPMAApproxLinearFP16(module.in_features, 
                                                                    module.out_features, 
                                                                    module.bias is not None, 
                                                                    dev=next(module.parameters()).device)
                                with torch.no_grad():
                                    new_linear.weight.data = (module.weight.data.to(torch.float16)).contiguous()
                                    if module.bias is not None:
                                        new_linear.bias.data = (module.bias.data.to(torch.float16)).contiguous()
                            father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                            setattr(father_module, name.split('.')[-1], new_linear)
                            del new_linear, module
                            torch.cuda.empty_cache()
        
            else:
                raise ValueError("approx_kernel should be in ['baseline', 'LMul', 'FPMA']")
    
    print(model)
    return model
                