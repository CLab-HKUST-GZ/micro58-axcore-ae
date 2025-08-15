import time
import torch
import torch.nn as nn

import argparse
from datautils import *


def get_opt_base(model, seq_len):
    def skip(*args, **kwargs):
        pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='cpu')
    
    model.seqlen = seq_len
    if model.config.max_position_embeddings < seq_len:
        print(f"Warning: Given seqlen {model.seqlen} is larger than max length {model.config.max_position_embeddings}")

    return model

@torch.no_grad()
def opt_eval(model, testenc, eval_samples):
    print('Evaluating ', end='')
    
    dev = torch.device('cuda:0')
    testenc = testenc.input_ids
    if eval_samples:
        nsamples = eval_samples
    else:
        nsamples = min(1000, testenc.numel() // model.seqlen)
    print("nsamples: ", nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(i, end=' ',flush=True)
    print()

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("Perplexity: ", ppl.item(), flush=True)

    model.config.use_cache = use_cache


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--eval_dataset', type=str,
        help='evaluation dataset'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='model sequence length'
    )
    parser.add_argument(
        '--eval_samples', type=int, default=0,
        help='number of sample evaluation dataset'
    )
    parser.add_argument('--seed', type=int, default=0, 
        help='Random seed for data load'
    )
    parser.add_argument(
        '--q_bits', type=int, default=0, 
        help='Number of bits for quantization'
    )
    parser.add_argument(
        '--decomp_factor', type=int, default=0, 
        help='Number of channel groups'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=256,
        help='Size of row chunk'
    )
    parser.add_argument(
        '--quant_mha', action='store_true',
        help='Whether to quantize multi-head-attention'
    )
    parser.add_argument(
        '--scale_factor', type=str, default="",
        help='path to scale factor learned from calibration data.'
    )
    parser.add_argument(
        '--bias', type=str, default="",
        help='path to bias learned from calibration data.'
    )
    
    args = parser.parse_args()
    model = get_opt_base(args.model, args.seq_len)

    if args.scale_factor:
        scale_factor = torch.load(args.scale_factor)
        for layer in model.model.decoder.layers:
            attn = layer.self_attn
            prefix = "model.decoder.layers." + str(attn.layer_idx)
            
            name = prefix + ".self_attn" + "h_tmax"
            attn.h_tmax = scale_factor[name]
            name = prefix + ".self_attn" + "h_cmax"
            attn.h_group_index = scale_factor[name]
            name = prefix + ".self_attn" + "o_tmax"
            attn.o_tmax = scale_factor[name]
            name = prefix + ".self_attn" + "o_cmax"
            attn.o_group_index = scale_factor[name]

            if args.quant_mha:
                name = prefix + ".self_attn" + "q_tmax"
                attn.q_tmax = scale_factor[name]
                name = prefix + ".self_attn" + "q_cmax"
                attn.q_group_index = scale_factor[name]
                name = prefix + ".self_attn" + "s_tmax"
                attn.s_tmax = scale_factor[name]
                name = prefix + ".self_attn" + "s_cmax"
                attn.s_group_index = scale_factor[name]

                name = prefix + ".self_attn" + "k_scale"
                attn.k_scale = scale_factor[name]
                name = prefix + ".self_attn" + "v_scale"
                attn.v_scale = scale_factor[name]

            name = prefix + "fc1_tmax"
            layer.fc1_tmax = scale_factor[name]
            name = prefix + "fc1_cmax"
            layer.fc1_group_index = scale_factor[name]
            name = prefix + "fc2_tmax"
            layer.fc2_tmax = scale_factor[name]
            name = prefix + "fc2_cmax"
            layer.fc2_group_index = scale_factor[name]

    if args.bias:
        bias = torch.load(args.bias)
        for layer in model.model.decoder.layers:
            attn = layer.self_attn
            prefix = "model.decoder.layers." + str(attn.layer_idx)
            
            name = prefix + ".self_attn" + "h_ch_bias"
            attn.h_ch_bias = bias[name]
            name = prefix + ".self_attn" + "o_ch_bias"
            attn.o_ch_bias = bias[name]

            if args.quant_mha:
                name = prefix + ".self_attn" + "q_ch_bias"
                attn.q_ch_bias = bias[name]
                name = prefix + ".self_attn" + "k_ch_bias"
                attn.k_ch_bias = bias[name]

            name = prefix + "h_ch_bias"
            layer.h_ch_bias = bias[name]
            
    model.eval()

    for layer in model.model.decoder.layers:
        layer.self_attn.quant_mha = args.quant_mha

        layer.self_attn.q_bits = args.q_bits
        layer.q_bits = args.q_bits

        layer.self_attn.decomp_factor = args.decomp_factor
        layer.decomp_factor = args.decomp_factor

        layer.self_attn.chunk_size = args.chunk_size
        layer.chunk_size = args.chunk_size
    
    testloader = get_loaders(
        args.eval_dataset, seed = args.seed, seqlen = model.seqlen, model = args.model
    )

    opt_eval(model, testloader, args.eval_samples)
