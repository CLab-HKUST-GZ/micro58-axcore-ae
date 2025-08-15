import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def get_wikitext2(seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    if 'llama' in model:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc

def get_ptb(seqlen, model):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    if 'llama' in model:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    return testenc

def get_loaders(
    name, seed=0, seqlen=2048, model='', it=0, c4_num=0
):
    set_seed(seed)
    if 'wikitext2' in name:
        return get_wikitext2(seqlen, model)
    if 'ptb' in name:
        return get_ptb(seqlen, model)
