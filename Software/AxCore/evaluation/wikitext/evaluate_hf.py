import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from tqdm import tqdm

from approximation_computation import approximation_wrapper

def main(args):
    print(args)
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda device found")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = approximation_wrapper(model, args, torch_dtype)
    
    # args.wquant = True
    # args.waquant = False
    # model = approximation_wrapper(model, args, torch_dtype)
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=device_id)
    
    # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    
    # Write results to file
    with open('./results/ppl_results.txt', 'a') as f:
        f.write(f"model: {args.model}, ppl: {ppl.item():.4f}\n")

    
if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--dtype", "-t", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_length", "-l", type=int, default=4096)
    parser.add_argument("--stride", "-r", type=int, default=512)
    parser.add_argument("--approx_kernel", "-a", type=str, choices=['baseline', 'LMul', 'FPMA'])
    parser.add_argument("--quant_method", "-qm", type=str, choices=['RTN', 'HQQ'])
    parser.add_argument("--wquant", "-wq", action='store_true', default=False)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--w_group_size", type=int, default=128)
    parser.add_argument("--kvquant", "-kvq", action='store_true', default=False)
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_bit", type=int, default=16, help='KV cache quantization bits (8/16)') 
    parser.add_argument("--linear_approx", "-linea", action='store_true', default=False)
    parser.add_argument("--sequential_calib", action='store_true', default=False, help='Enable sequential error-aware calibration (each layer uses previous quantized output as input)')
    parser.add_argument("--attention_approx", "-attna", action='store_true', default=False)
    parser.add_argument("--waquant", action='store_true', default=False, help='Enable Weight-Activation quantization')
    parser.add_argument("--a_bit", type=int, default=16, help='Activation quantization bits')
    # ablation
    parser.add_argument("--ablation", action='store_true', default=False)
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2]) # mpFPMA, mpFPMA+S, mpFPMA+C
    args = parser.parse_args()
    print("helloworld")
    main(args)