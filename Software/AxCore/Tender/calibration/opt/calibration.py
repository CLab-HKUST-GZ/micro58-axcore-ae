import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

def forward_by_layer(model, inputs, num_samples, seqlen):
    
    dev = torch.device("cuda:0")

    inputs = inputs.input_ids
    
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
    inps = torch.zeros((num_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    #inps = [0 for _ in range(num_samples)]
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp # 1, seqlen, dim
            cache['i'] += 1
            #if kwargs['attention_mask'].shape[-1] == seqlen:
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    # Layer 0 -> Attention mask, position embedding
    layers[0] = Catcher(layers[0])
    for i in range(num_samples):
        batch = inputs[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
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

    # Entire layer
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(num_samples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(i, end=' ', flush=True)
    print()

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    # Final layer norm and lm_head
    nlls = []
    inputs = inputs.to(dev)
    for i in range(num_samples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    model.config.use_cache = use_cache

    return torch.stack(nlls).sum()

def select_best_scheme(scale_factors, model, inputs, quant_mha = False):
    nll_sum = []
    for i, scale_factor in enumerate(scale_factors):
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

            if quant_mha:
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

        nll = forward_by_layer(model, inputs, 1, model.seqlen)
        ppl = torch.exp(nll / (1 * model.seqlen)).item()
        print("index %d ppl %f"%(i, ppl))
        nll_sum.append(nll.item())

    idx = np.argmin(np.array(nll_sum))
    if idx==0:
        scheme = "rdn"
    elif idx==1:
        scheme = "rup"

    print("scheme %s selected"%(scheme),flush=True)

    return scale_factors[idx]

def get_scale_factor(model, tokenizer, dataset_path, num_samples=512, seq_len=512, quant_mha = False):
    model.eval()
    model.seqlen = seq_len
    scale_factor = {}

    def stat_tensor(attn, name):
        h_tmax = attn.h_tmax_cal # chunks
        o_tmax = attn.o_tmax_cal

        h_cmax = attn.h_cmax_cal # chunks, hidden_dim
        o_cmax = attn.o_cmax_cal
        
        tmaxes = [h_tmax, o_tmax]
        cmaxes = [h_cmax, o_cmax]
        names = ["h", "o"]

        if quant_mha:
            k_scale = attn.k_scale_cal
            v_scale = attn.v_scale_cal

            q_tmax = attn.q_tmax_cal # b*h, chunks
            s_tmax = attn.s_tmax_cal # b*h, chunks
            q_cmax = attn.q_cmax_cal # b*h, chunks, head_dim
            s_cmax = attn.s_cmax_cal # b*h, chunks, head_dim
            tmaxes.extend([q_tmax, s_tmax])
            cmaxes.extend([q_cmax, s_cmax])
            names.extend(["q", "s"])
        
        if name in scale_factor:
            for i in range(len(names)):
                old_tmax = scale_factor[name + names[i] + "_tmax"]
                new_tmax = tmaxes[i]
                old_cmax = scale_factor[name + names[i] + "_cmax"]
                new_cmax = cmaxes[i]

                scale_factor[name + names[i] + "_tmax"] = torch.where(old_tmax > new_tmax, old_tmax, new_tmax)
                scale_factor[name + names[i] + "_cmax"] = torch.where(old_cmax > new_cmax, old_cmax, new_cmax)
            if quant_mha:
                old_k_scale = scale_factor[name + "k_scale"]
                old_v_scale = scale_factor[name + "v_scale"]
                scale_factor[name + "k_scale"] = torch.where(old_k_scale > k_scale, old_k_scale, k_scale)
                scale_factor[name + "v_scale"] = torch.where(old_v_scale > v_scale, old_v_scale, v_scale)
        else:
            for i in range(len(names)):
                scale_factor[name + names[i] + "_tmax"] = tmaxes[i]
                scale_factor[name + names[i] + "_cmax"] = cmaxes[i]
            if quant_mha:
                scale_factor[name + "k_scale"] = k_scale
                scale_factor[name + "v_scale"] = v_scale
            scale_factor[name] = True
    
    def decoder_layer_stat_tensor(decoder, name):
        fc1_tmax = decoder.fc1_tmax_cal
        fc2_tmax = decoder.fc2_tmax_cal

        fc1_cmax = decoder.fc1_cmax_cal # chunks, hidden_dim
        fc2_cmax = decoder.fc2_cmax_cal
        
        tmaxes = [fc1_tmax, fc2_tmax]
        cmaxes = [fc1_cmax, fc2_cmax]
        names = ["fc1", "fc2"]
        
        if name in scale_factor:
            for i in range(len(names)):
                old_tmax = scale_factor[name + names[i] + "_tmax"]
                new_tmax = tmaxes[i]
                old_cmax = scale_factor[name + names[i] + "_cmax"]
                new_cmax = cmaxes[i]

                scale_factor[name + names[i] + "_tmax"] = torch.where(old_tmax > new_tmax, old_tmax, new_tmax)
                scale_factor[name + names[i] + "_cmax"] = torch.where(old_cmax > new_cmax, old_cmax, new_cmax)
        else:
            for i in range(len(names)):
                scale_factor[name + names[i] + "_tmax"] = tmaxes[i]
                scale_factor[name + names[i] + "_cmax"] = cmaxes[i]
            scale_factor[name] = True

    def stat_input_hook(m, hidden_states, output_attentions, name):
        stat_tensor(m, name)
    def decoder_layer_stat_input_hook(m, hidden_states, output_attentions, name):
        decoder_layer_stat_tensor(m, name)

    hooks = []
    for name, m in model.named_modules():
        if name.endswith('self_attn'):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
        if name.endswith('layers'):
            layer_index = 0
            for layer in m:
                hooks.append(
                        layer.register_forward_hook(
                            functools.partial(decoder_layer_stat_input_hook, name=name+"."+str(layer_index)))
                )
                layer_index += 1

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    
    inputs = tokenizer("\n\n".join(dataset['text'][:1000]), return_tensors='pt')

    forward_by_layer(model, inputs, num_samples, seq_len)

    for h in hooks:
        h.remove()
    
    decomp_factor = model.model.decoder.layers[0].decomp_factor
    
    # Static calibration: Chooses between round up and round down
    # Runtime: Round up 
    import copy
    scale_factor_rdn = copy.deepcopy(scale_factor)
    scale_factor_rup = copy.deepcopy(scale_factor)

    for name in scale_factor:
        if "tmax" in name:
            tmax = scale_factor[name] # chunks
            cmax = scale_factor[name.replace("tmax", "cmax")] # chunks, hidden_dim

            thresholds = []
            for i in range(decomp_factor):
                thresh = (tmax / (2**(decomp_factor-1-i))).unsqueeze(-1) # chunks, 1
                thresholds.append(thresh)
            
            group_index_rdn = torch.zeros_like(cmax) # chunks, hidden_dim
            group_index_rup = torch.zeros_like(cmax)
            
            for i in range(decomp_factor):
                if i == 0:
                    mask = cmax <= thresholds[i]
                    group_index_rup = torch.where(mask, i, group_index_rup)
                    group_index_rdn = torch.where(mask, i, group_index_rdn)
                else:
                    mask = torch.logical_and((thresholds[i-1] < cmax), (cmax <= thresholds[i]))
                    group_index_rup = torch.where(mask, i, group_index_rup)

                    group_index_rdn = torch.where(mask, i-1, group_index_rdn)

            scale_factor_rdn[name.replace("tmax", "cmax")] = group_index_rdn
            scale_factor_rup[name.replace("tmax", "cmax")] = group_index_rup

    scale_factors = [scale_factor_rdn, scale_factor_rup]
    scale_factor = select_best_scheme(scale_factors, model, inputs, quant_mha)

    return scale_factor

def get_bias(model, tokenizer, dataset_path, num_samples = 512, seq_len = 512, quant_mha = False):
    model.eval()
    model.seqlen = seq_len
    device = next(model.parameters()).device
    bias = {}

    def stat_tensor(attn, name):
        h_ch_bias = attn.h_ch_bias_cal # chunks, 1, hidden_dim
        o_ch_bias= attn.o_ch_bias_cal
        
        biases = [h_ch_bias, o_ch_bias]
        bias_names = ["h_ch_bias", "o_ch_bias"]

        if quant_mha:
            q_ch_bias = attn.q_ch_bias_cal # heads, chunks, 1, head_dim
            k_ch_bias = attn.k_ch_bias_cal # heads, 1, head_dim
            biases.extend([q_ch_bias, k_ch_bias])
            bias_names.extend(['q_ch_bias', 'k_ch_bias'])
        
        if name in bias:
            for i in range(len(biases)):
                bias[name + bias_names[i]] = bias[name + bias_names[i]] + biases[i]
        else:
            for i in range(len(biases)):
                bias[name + bias_names[i]] = biases[i]
            bias[name] = True
    
    def decoder_layer_stat_tensor(decoder, name):
        h_ch_bias = decoder.h_ch_bias_cal

        biases = [h_ch_bias]
        bias_names = ["h_ch_bias"]

        if name in bias:
            for i in range(len(biases)):
                bias[name + bias_names[i]] = bias[name + bias_names[i]] + biases[i]
        else:
            for i in range(len(biases)):
                bias[name + bias_names[i]] = biases[i]
            bias[name] = True

    def stat_input_hook(m, hidden_states, output_attentions, name):
        stat_tensor(m, name)

    def decoder_layer_stat_input_hook(m, hidden_states, output_attentions, name):
        decoder_layer_stat_tensor(m, name)

    hooks = []
    for name, m in model.named_modules():
        if name.endswith('self_attn'):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
        if name.endswith('layers'):
            layer_index = 0
            for layer in m:
                hooks.append(
                        layer.register_forward_hook(
                            functools.partial(decoder_layer_stat_input_hook, name=name+"."+str(layer_index)))
                )
                layer_index += 1

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    
    inputs = tokenizer("\n\n".join(dataset['text'][:1000]), return_tensors='pt')
    forward_by_layer(model, inputs, num_samples, seq_len)

    for h in hooks:
        h.remove()

    for name in bias:
        bias[name] = bias[name] / num_samples
    
    return bias
