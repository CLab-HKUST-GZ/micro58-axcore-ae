import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import argparse

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from approximation_computation import approximation_wrapper

TASK_GROUPS = {
    "group1": {"tasks": ["piqa", "hellaswag", "winogrande", "arc_easy"], "batch_size": 32, "num_fewshot": 0},
    "group2": {"tasks": ["gsm8k"], "batch_size": 32, "num_fewshot": 8},
    "group3": {"tasks": ["mmlu"], "batch_size": 8, "num_fewshot": 5},
    "group4": {"tasks": ["wikitext"], "batch_size": 2, "num_fewshot": 0},
}

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("* Quantization Format: kv_{}_w_{}".format(args.kv_bit, args.w_bit))

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise NotImplementedError
    if args.attention_approx or args.kvquant:
        attn_implementation = "eager"
    else:
        attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    # # quantize model
    
    model = approximation_wrapper(model, args, torch_dtype)

    task_group = TASK_GROUPS[args.task_group]
    tasks = task_group["tasks"]
    batch_size = task_group["batch_size"]
    num_fewshot = task_group["num_fewshot"]
    lm = HFLM(model, backend="causal", batch_size=batch_size)

    results = lm_eval.simple_evaluate(lm, tasks=tasks, num_fewshot=num_fewshot, batch_size=batch_size)
    
    print("Evaluation Results:")
    print(make_table(results))
    # Write results to file
    with open('./results/lm_eval_results.txt', 'a') as f:
        f.write(f"model: {args.model}\n")
        f.write(make_table(results))
        f.write("\n")
    
    if args.wquant and args.kvquant:
        print("* Quantization Format: kv_{}_w_{}".format(args.kv_bit, args.w_bit))
    elif args.wquant:
        print("* Quantization Format: w_{}".format(args.w_bit))
    elif args.kvquant:
        print("* Quantization Format: kv_{}".format(args.kv_bit))
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path of the hf model")
    parser.add_argument("--output_path", type=str, help="path to save the quantized model")
    parser.add_argument("--task_group", type=str, default="group1", choices=["group1", "group2", "group3", "group4"])
    parser.add_argument("--metrics", type=str, default="mc1,mc2")
    parser.add_argument("--dtype", "-t", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--approx_kernel", "-a", type=str, choices=['baseline', 'LMul', 'FPMA'])
    parser.add_argument("--quant_method", "-qm", type=str, choices=['RTN', 'HQQ'])
    parser.add_argument("--wquant", "-wq", action='store_true', default=False)
    parser.add_argument("--w_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=4, help='Weight quantization bits (4/8/16)')
    parser.add_argument("--kvquant", "-kvq", action='store_true', default=False)
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_bit", type=int, default=16, help='KV cache quantization bits (8/16)') 
    parser.add_argument("--linear_approx", "-linea", action='store_true', default=False, help='Enable linear layer approximation')
    parser.add_argument("--sequential_calib", action='store_true', default=False, help='Enable sequential error-aware calibration (each layer uses previous quantized output as input)')
    parser.add_argument("--attention_approx", "-attna", action='store_true', default=False, help='Enable attention approximation')
    parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code for model loading')
    parser.add_argument('--use_fast_tokenizer', action='store_true', help='Use fast tokenizer implementation')
    parser.add_argument("--waquant", action='store_true', default=False, help='Enable Weight-Activation quantization')
    parser.add_argument("--a_bit", type=int, default=16, help='Activation quantization bits')
    args = parser.parse_args()

    main(args)