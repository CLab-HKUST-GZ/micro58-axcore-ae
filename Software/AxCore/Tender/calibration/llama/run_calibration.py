import torch
import os

import argparse
from calibration import get_scale_factor, get_bias

def build_model_and_tokenizer(model_name, seq_len):
    if os.path.exists('../../models/modeling_llama.py'):
        os.system('rm ../../models/modeling_llama.py')
    cwd = os.getcwd()
    os.chdir('../../models/')
    os.system('ln -s modeling_llama_tender.py modeling_llama.py')
    os.chdir(cwd)

    from transformers import (
        LlamaForCausalLM,
        LlamaTokenizer,
    )

    kwargs = {"torch_dtype": torch.float16, "device_map": "cpu"}
    model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length=seq_len)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--target', type=str, choices=['scale', 'bias'], required=True,
                        help='Calibrate scale factor or bias')
    parser.add_argument('--output-path', type=str, default='biases/llama-7b.pt',
                        help='where to save the result')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--q_bits', type=int, default=8, 
                        help='Number of bits for quantization')
    parser.add_argument('--decomp_factor', type=int, default=8, 
                        help='Number of column groups')
    parser.add_argument('--chunk_size', type=int, default=256, 
                        help='Size of row chunk')
    parser.add_argument('--quant_mha', action='store_true', 
                        help='Whether to quantize multi-head-attention')
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.seq_len)
    
    for layer in model.model.layers:
        layer.self_attn.quant_mha = args.quant_mha

        layer.self_attn.q_bits = args.q_bits
        layer.mlp.q_bits = args.q_bits
    
        layer.self_attn.decomp_factor = args.decomp_factor
        layer.mlp.decomp_factor = args.decomp_factor

        layer.self_attn.chunk_size = args.chunk_size
        layer.mlp.chunk_size = args.chunk_size
    
    if args.target == 'scale':
        result = get_scale_factor(model, tokenizer, args.dataset_path,
                                    args.num_samples, args.seq_len, args.quant_mha)
    else: 
        result = get_bias(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len, args.quant_mha)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(result, args.output_path)


if __name__ == '__main__':
    main()
