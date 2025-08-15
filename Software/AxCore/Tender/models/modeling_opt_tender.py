# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
import random
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig

import math


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

# linear-symmetric integer quantization (round to nearest, ties to even)
def sym_quant(fp, scale, q_bits):
    fp = fp.to(torch.float32)
    scale = scale.to(torch.float32) # in case of overflow / underflow

    if(type(scale) == torch.Tensor):
        assert torch.sum(scale == 0.0).item() == 0.0, "Zero is given as a scale factor"
        scale = torch.clamp(scale.abs(), min = 1e-9)
    else:
        assert scale != 0.0, "Zero is given as a scale factor"

    q_max = (2 ** (q_bits - 1) - 1)
    q_int = torch.round(fp / scale).clamp(min = -(q_max + 1), max = q_max)
    q_int = q_int

    return q_int

def sym_dequant(q_int, scale, dtype):

    fp = q_int * scale
    fp = fp.to(dtype)

    return fp

def quant_bfloat(t):
    assert t.dtype == torch.float32 # should not convert between bf16 and fp16

    import mx
    specs = {"bfloat16": 16, "round" : "even"}
    mx_specs = mx.finalize_mx_specs(specs)
    
    return mx.quantize_bfloat(t, mx_specs)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        layer_idx: int = 0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.layer_idx = layer_idx

        self.quant_mha = False
        self.quant_out_bf16 = False
        self.q_bits = 4
        self.q_bits_kv = 4
        self.decomp_factor = 8
        self.chunk_size = 256

        # quantized weights
        self.WQ = None
        self.WK = None
        self.WV = None
        self.WO = None

        ## calibrated metadata
        # channel bias
        self.h_ch_bias_cal = None
        self.k_ch_bias_cal = None
        self.q_ch_bias_cal = None
        self.o_ch_bias_cal = None
        
        self.h_ch_bias = None
        self.k_ch_bias = None
        self.q_ch_bias = None
        self.o_ch_bias = None

        # tensor-max
        self.h_tmax_cal = None
        self.q_tmax_cal = None
        self.s_tmax_cal = None
        self.o_tmax_cal = None
        
        self.h_tmax = None
        self.q_tmax = None
        self.s_tmax = None
        self.o_tmax = None
        
        # channel-max & grp idx
        self.h_cmax_cal = None
        self.q_cmax_cal = None
        self.s_cmax_cal = None
        self.o_cmax_cal = None
        
        self.h_group_index = None
        self.q_group_index = None
        self.s_group_index = None
        self.o_group_index = None
        
        # scale
        self.k_scale_cal = None
        self.v_scale_cal = None
        self.k_scale = None
        self.v_scale = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, hidden_dim = hidden_states.size()
        dtype = hidden_states.dtype

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        # Weight Quant =====================
        q_weight = self.q_proj.weight.data
        k_weight = self.k_proj.weight.data
        v_weight = self.v_proj.weight.data
        o_weight = self.out_proj.weight.data

        q_max = (2**(self.q_bits-1))-1

        if self.WQ == None:
            scale = (torch.max(torch.abs(q_weight.clone()), dim = -1, keepdim = True)[0]) / q_max
            WQ = sym_dequant(sym_quant(q_weight, scale, self.q_bits), scale, dtype)
            self.WQ = WQ.transpose(-1,-2)

            scale = (torch.max(torch.abs(k_weight.clone()), dim = -1, keepdim = True)[0]) / q_max
            WK = sym_dequant(sym_quant(k_weight, scale, self.q_bits), scale, dtype)
            self.WK = WK.transpose(-1,-2)

            scale = (torch.max(torch.abs(v_weight.clone()), dim = -1, keepdim = True)[0]) / q_max
            WV = sym_dequant(sym_quant(v_weight, scale, self.q_bits), scale, dtype)
            self.WV = WV.transpose(-1,-2)

            scale = (torch.max(torch.abs(o_weight.clone()), dim = -1, keepdim = True)[0]) / q_max
            WO = sym_dequant(sym_quant(o_weight, scale, self.q_bits), scale, dtype)
            self.WO = WO.transpose(-1,-2)

        q_bias = self.q_proj.bias.data
        k_bias = self.k_proj.bias.data
        v_bias = self.v_proj.bias.data
        o_bias = self.out_proj.bias.data
        #==================================
    
        bsz, tgt_len, hidden_dim = hidden_states.shape
        orig_tgt_len = tgt_len
        hidden_states = hidden_states.view(tgt_len * bsz, hidden_dim)
        tgt_len, hidden_dim = hidden_states.size()
        zero = torch.zeros((1,1), dtype = hidden_states.dtype, device = hidden_states.device)
        decomp_factor = self.decomp_factor

        #QKV ==============================
        chunks = int(math.ceil(tgt_len / self.chunk_size))
        assert(chunks != 0)

        padded_rows = chunks * self.chunk_size
        pad_num = padded_rows - tgt_len
        
        if pad_num > 0:
            padding = torch.zeros((pad_num, hidden_dim), dtype = hidden_states.dtype, device = hidden_states.device) 
            hidden_chunks = torch.cat((hidden_states, padding),dim=0)
        else:
            hidden_chunks = hidden_states

        hidden_chunks = hidden_chunks.reshape(chunks, self.chunk_size, hidden_dim)
        result = torch.zeros_like(hidden_chunks)

        # Act normalize
        if self.h_ch_bias is None:
            ch_max = torch.max(hidden_chunks, dim=1, keepdim=True)[0] # chunks, 1, hidden_dim
            ch_min = torch.min(hidden_chunks, dim=1, keepdim=True)[0]
            ch_bias = torch.div(ch_max + ch_min, 2)
            self.h_ch_bias_cal = ch_bias
        else:
            ch_bias = self.h_ch_bias[:chunks]

        q_ch_bias = torch.matmul(ch_bias, q_weight.transpose(-1, -2)) # chunks, 1, out_dim
        k_ch_bias = torch.matmul(ch_bias, k_weight.transpose(-1, -2))
        v_ch_bias = torch.matmul(ch_bias, v_weight.transpose(-1, -2))
        hidden_chunks = hidden_chunks.clone() - ch_bias
        
        if self.h_tmax is None:
            h_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # chunks, chunk_size
            h_tmax = torch.max(h_tmax, dim=-1)[0] # chunks
            self.h_tmax_cal = h_tmax
        else:
            h_tmax = self.h_tmax[:chunks]

        thresholds = []
        for i in range(decomp_factor):
            thresholds.append((h_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # chunks, 1
        
        if self.h_group_index is None:
            h_cmax = torch.max(torch.abs(hidden_chunks),dim=-2)[0] # chunks, hidden_dim
            self.h_cmax_cal = h_cmax
        
        count = []
        for i in range(decomp_factor):
            if self.h_group_index is None:
                if i==0:
                    mask = (h_cmax <= thresholds[i]) #chunks, hidden_dim
                else: 
                    mask = torch.logical_and((thresholds[i-1] < h_cmax), (h_cmax <= thresholds[i]))
            else:
                mask = (self.h_group_index[:chunks] == i)

            count.append(torch.sum(mask).item())
            if count[i] == 0:
                continue

            mask = mask.unsqueeze(1).repeat(1, self.chunk_size, 1) #chunks, chunk_size, hidden_dim
            scale = thresholds[i].unsqueeze(-1) / q_max #chunks, 1, 1
            decomp_fp = torch.where(mask, hidden_chunks, zero)
            decomp_i = sym_quant(decomp_fp, scale, self.q_bits)
            result += sym_dequant(decomp_i, scale, dtype)
            
        assert sum(count) == (self.embed_dim * chunks)

        Q = torch.matmul(result, self.WQ) + q_ch_bias 
        Q = Q.reshape(padded_rows, hidden_dim)[:orig_tgt_len * bsz] + q_bias
        K = torch.matmul(result, self.WK) + k_ch_bias 
        K = K.reshape(padded_rows, hidden_dim)[:orig_tgt_len * bsz] + k_bias
        V = torch.matmul(result, self.WV) + v_ch_bias 
        V = V.reshape(padded_rows, hidden_dim)[:orig_tgt_len * bsz] + v_bias

        if self.quant_out_bf16:
            Q = quant_bfloat(Q)
            K = quant_bfloat(K)
            V = quant_bfloat(V)
        #========================================

        Q = Q * self.scaling
        Q = self._shape(Q, -1, bsz)
        bsz, num_heads, tgt_len, _ = Q.shape
        Q = Q.view(*proj_shape)

        K = self._shape(K, -1, bsz)
        _, _, src_len, _ = K.shape
        K = K.view(*proj_shape)

        if self.quant_mha == False:
            attn_weights = torch.bmm(Q, K.transpose(1,2))
        else:
            # Q x K^T============================
            # q_max = (2**(self.q_bits-1)-1)
            q_max = (2**(self.q_bits_kv-1)-1)

            # key normalize
            orig_k = K.clone() 
            if self.k_ch_bias is None:
                ch_max = torch.max(K, dim=1, keepdim=True)[0]
                ch_min = torch.min(K, dim=1, keepdim=True)[0]
                k_ch_bias = torch.div(ch_max + ch_min, 2) # b*h, 1, head_dim
                self.k_ch_bias_cal = k_ch_bias
            else:
                k_ch_bias = self.k_ch_bias

            K = K - k_ch_bias
            k_ch_bias = k_ch_bias.unsqueeze(2) # b*h, 1, 1, head_dim

            # key - row-wise symmetric
            if self.k_scale is None:
                scale = (torch.max(torch.abs(K), dim=-1, keepdim=True)[0]) / q_max #b*h, src_len, 1
                self.k_scale_cal = scale
            else:
                scale = self.k_scale[:,:src_len]

            K = sym_quant(K, scale, self.q_bits_kv)
            K = sym_dequant(K, scale, dtype)
            K = K.transpose(-1, -2)

            # chunking
            chunks = int(math.ceil(tgt_len / self.chunk_size))
            assert (chunks != 0)

            padded_rows = chunks * self.chunk_size
            pad_num = padded_rows - tgt_len
            
            if pad_num > 0:
                padding = torch.zeros((bsz * num_heads, pad_num, self.head_dim), dtype = K.dtype, device = K.device) 
                hidden_chunks = torch.cat((Q, padding),dim=1)
            else:
                hidden_chunks = Q 

            hidden_chunks = hidden_chunks.reshape(bsz * num_heads, chunks, self.chunk_size, self.head_dim)
            result = torch.zeros_like(hidden_chunks)

            # Acts normalize
            if self.q_ch_bias is None:
                ch_max = torch.max(hidden_chunks, dim = 2, keepdim=True)[0] #b*h, chunks, 1, head_dim
                ch_min = torch.min(hidden_chunks, dim = 2, keepdim=True)[0]
                q_ch_bias = torch.div(ch_max + ch_min, 2)
                self.q_ch_bias_cal = q_ch_bias
            else:
                q_ch_bias = self.q_ch_bias[:, :chunks]

            o1_ch_bias = torch.matmul(q_ch_bias, orig_k.unsqueeze(1).repeat(1,chunks,1,1).transpose(-1,-2)) #b*h, chunks, 1, src_len (overhead)
            o2_ch_bias = torch.matmul(hidden_chunks, k_ch_bias.transpose(-1, -2)) #b*h, chunks, chunk_size, 1 (overhead)
            o3_ch_bias = - torch.matmul(q_ch_bias, k_ch_bias.transpose(-1, -2)) #b*h, chunks, 1, 1 

            hidden_chunks -= q_ch_bias

            if self.q_tmax is None:
                q_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # b*h, chunks, chunk_size
                q_tmax = torch.max(q_tmax, dim=-1)[0] # b*h, chunks
                self.q_tmax_cal = q_tmax
            else:
                q_tmax = self.q_tmax[:, :chunks]

            thresholds = []
            for i in range(decomp_factor):
                thresholds.append((q_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # b*h, chunks, 1

            if self.q_group_index is None:
                q_cmax = torch.max(torch.abs(hidden_chunks),dim=-2)[0] # b*h, chunks, head_dim
                self.q_cmax_cal = q_cmax
            
            count = []
            for i in range(decomp_factor):
                if self.q_group_index is None:
                    if i==0:
                        mask = (q_cmax <= thresholds[i]) # b*h, chunks, head_dim
                    else:
                        mask = torch.logical_and((thresholds[i-1] < q_cmax), (q_cmax <= thresholds[i]))
                else:
                    mask = (self.q_group_index[:, :chunks] == i)
            
                count.append(torch.sum(mask).item())
                if count[i] == 0:
                    continue
            
                mask = mask.unsqueeze(2).repeat(1, 1, self.chunk_size, 1) #b*h, chunks, chunk_size, head_dim
                scale = thresholds[i].unsqueeze(-1) / q_max  # b*h, chunks, 1, 1
                decomp_fp = torch.where(mask, hidden_chunks, zero)
                decomp_i = sym_quant(decomp_fp, scale, self.q_bits_kv)
                result += sym_dequant(decomp_i, scale, dtype)
            
            assert sum(count) == (self.num_heads * self.head_dim * chunks)

            result = result.reshape(bsz * self.num_heads, padded_rows, self.head_dim)
            o1_ch_bias = o1_ch_bias.repeat(1, 1, self.chunk_size, 1).reshape(bsz * self.num_heads, padded_rows, src_len)
            o2_ch_bias = o2_ch_bias.reshape(bsz*self.num_heads, padded_rows, 1)
            o3_ch_bias = o3_ch_bias.repeat(1, 1, self.chunk_size, 1).reshape(bsz * self.num_heads, padded_rows, 1)
            
            attn_weights = torch.matmul(result, K) + o1_ch_bias + o2_ch_bias + o3_ch_bias # b*h, padded_rows, src_len
            attn_weights = attn_weights[:, :tgt_len] # b*h, tgt_len, src_len

            if self.quant_out_bf16:
                attn_weights = quant_bfloat(attn_weights)
            # ===================================

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        V = self._shape(V, -1, bsz)
        V = V.view(*proj_shape)

        if self.quant_mha == False:
            attn_output = torch.bmm(attn_probs, V)
        else:
            # S x V ==============================
            _, tgt_len, src_len = attn_weights.shape
            q_max = (2 ** (self.q_bits_kv - 1) - 1)
            prob_q_max = (2 ** (self.q_bits_kv) - 1) #for score, quantize to uint

            if self.v_scale is None:
                scale = (torch.max(torch.abs(V), dim = -2, keepdim = True)[0]) / q_max #heads, 1, head_dim
                self.v_scale_cal = scale
            else:
                scale = self.v_scale

            V = sym_quant(V, scale, self.q_bits_kv)
            V = sym_dequant(V, scale, dtype)

            chunks = int(math.ceil(tgt_len/self.chunk_size))
            assert (chunks != 0)

            padded_rows = chunks * self.chunk_size
            pad_num = padded_rows - tgt_len
            
            if pad_num > 0:
                padding = torch.zeros((bsz*num_heads, pad_num, src_len), dtype = hidden_states.dtype, device = hidden_states.device) 
                hidden_chunks = torch.cat((attn_weights, padding),dim=1)
            else:
                hidden_chunks = attn_weights 

            hidden_chunks = hidden_chunks.reshape(bsz * num_heads, chunks, self.chunk_size, src_len)
            result = torch.zeros_like(hidden_chunks)

            if self.s_tmax is None:
                s_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # b*h, chunks, chunk_size
                s_tmax = torch.max(s_tmax, dim=-1)[0] # b*h, chunks
                self.s_tmax_cal = s_tmax
            else:
                s_tmax = self.s_tmax[:, :chunks]

            thresholds=[]
            for i in range(decomp_factor):
                thresholds.append((s_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # b*h, chunks, 1

            if self.s_group_index is None:
                s_cmax = torch.max(torch.abs(hidden_chunks), dim = -2)[0] # b*h, chunks, src_len
                self.s_cmax_cal = s_cmax

            count = []
            for i in range(decomp_factor):
                if self.s_group_index is None:
                    if i==0:
                        mask = (s_cmax <= thresholds[i]) # b*h, chunks, src_len
                    elif i == decomp_factor - 1:
                        mask = (s_cmax > thresholds[i-1])
                    else:
                        mask = torch.logical_and((thresholds[i-1] < s_cmax), (s_cmax <= thresholds[i]))
                else:
                    mask = (self.s_group_index[:, :chunks, :src_len] == i) 
            
                count.append(torch.sum(mask).item())
                if count[i] == 0:
                    continue
            
                mask = mask.unsqueeze(2).repeat(1, 1, self.chunk_size, 1) # b*h, chunks, chunk_size, src_len
                scale = thresholds[i].unsqueeze(-1) / prob_q_max # b*h, chunks, 1, 1
                decomp_fp = torch.where(mask, hidden_chunks, zero)
                decomp_i = torch.round(decomp_fp / scale).clamp(min = 0, max = prob_q_max)
                result += sym_dequant(decomp_i, scale, dtype)
            
            assert sum(count) == (self.num_heads * src_len * chunks)

            result = result.to(attn_weights.dtype)
            result = result.reshape(bsz * self.num_heads, padded_rows, src_len)[:, :tgt_len, ...]

            attn_output = torch.matmul(result, V).reshape(bsz * self.num_heads, tgt_len, self.head_dim)

            if self.quant_out_bf16:
                attn_output = quant_bfloat(attn_output)
            #=====================================

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

#        attn_output = self.out_proj(attn_output)
        # O Proj ============================
        bsz, tgt_len, hidden_dim = attn_output.shape
        orig_tgt_len = tgt_len
        attn_output = attn_output.view(tgt_len * bsz, hidden_dim)
        tgt_len, hidden_dim = attn_output.size()

        chunks = int(math.ceil(tgt_len/self.chunk_size))
        assert(chunks != 0)

        padded_rows = chunks * self.chunk_size
        pad_num = padded_rows - tgt_len
        
        if pad_num > 0:
            padding = torch.zeros((pad_num, hidden_dim), dtype = attn_output.dtype, device = attn_output.device) 
            hidden_chunks = torch.cat((attn_output, padding),dim=0)
        else:
            hidden_chunks = attn_output

        hidden_chunks = hidden_chunks.reshape(chunks, self.chunk_size, hidden_dim)
        result = torch.zeros_like(hidden_chunks)

        # Act normalize
        if self.o_ch_bias is None:
            ch_max = torch.max(hidden_chunks, dim=1, keepdim=True)[0] # chunks, 1, hidden_dim
            ch_min = torch.min(hidden_chunks, dim=1, keepdim=True)[0]
            ch_bias = torch.div(ch_max + ch_min,2)
            self.o_ch_bias_cal = ch_bias
        else:
            ch_bias = self.o_ch_bias[:chunks]

        o_ch_bias = torch.matmul(ch_bias, o_weight.transpose(-1, -2))
        hidden_chunks = hidden_chunks.clone() - ch_bias

        if self.o_tmax is None:
            h_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # chunks, chunk_size
            h_tmax = torch.max(h_tmax, dim=-1)[0] # chunks
            self.o_tmax_cal = h_tmax
        else:
            h_tmax = self.o_tmax[:chunks]

        thresholds=[]
        for i in range(decomp_factor):
            thresholds.append((h_tmax/(2**(decomp_factor-1-i))).unsqueeze(-1)) # chunks, 1

        if self.o_group_index is None:
            h_cmax = torch.max(torch.abs(hidden_chunks),dim=-2)[0] # chunks, hidden_dim
            self.o_cmax_cal = h_cmax
        
        count = []
        for i in range(decomp_factor):
            if self.o_group_index is None:
                if i==0:
                    mask = (h_cmax <= thresholds[i]) #chunks, hidden_dim
                else: 
                    mask = torch.logical_and((thresholds[i-1] < h_cmax), (h_cmax <= thresholds[i]))
            else:
                mask = (self.o_group_index[:chunks] == i).to(hidden_states.device)

            count.append(torch.sum(mask).item())
            if count[i] == 0:
                continue

            mask = mask.unsqueeze(1).repeat(1, self.chunk_size, 1) #chunks, chunk_size, hidden_dim
            scale = thresholds[i].unsqueeze(-1)/q_max #chunks, 1, 1
            decomp_fp = torch.where(mask, hidden_chunks, zero)
            decomp_i = sym_quant(decomp_fp, scale, self.q_bits)
            result += sym_dequant(decomp_i, scale, dtype)
            
        assert sum(count) == (self.embed_dim * chunks)

        out = torch.matmul(result, self.WO) + o_ch_bias
        out = out.reshape(padded_rows, hidden_dim)[:bsz * orig_tgt_len] + o_bias

        attn_output = out.reshape(bsz, orig_tgt_len, hidden_dim)

        if self.quant_out_bf16:
            attn_output = quant_bfloat(attn_output)
        #===================================

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx:int = 0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.ffn_dim = config.ffn_dim
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            layer_idx=layer_idx,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.layer_idx = layer_idx

        self.quant_out_bfloat = False
        self.q_bits = 4
        self.decomp_factor = 8
        self.chunk_size = 256

        # calibrated metadata
        self.h_ch_bias_cal = None
        self.h_ch_bias = None

        self.fc1_tmax_cal = None
        self.fc2_tmax_cal = None
        
        self.fc1_tmax = None
        self.fc2_tmax = None

        self.fc1_cmax_cal = None
        self.fc2_cmax_cal = None
        
        self.fc1_group_index = None
        self.fc2_group_index = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # Weight Quant ================================
        dtype = hidden_states.dtype
        fc1_weight = self.fc1.weight.data
        fc1_bias = self.fc1.bias.data
        fc2_weight = self.fc2.weight.data
        fc2_bias = self.fc2.bias.data

        q_max = (2**(self.q_bits-1))-1

        scale = (torch.max(torch.abs(fc1_weight.clone()), dim = -1, keepdim = True)[0]) / q_max
        Wfc1 = sym_dequant(sym_quant(fc1_weight.clone(), scale, self.q_bits), scale, dtype)
        Wfc1 = Wfc1.transpose(-1,-2)

        scale = (torch.max(torch.abs(fc2_weight.clone()),dim = -1, keepdim = True)[0]) / q_max
        Wfc2 = sym_dequant(sym_quant(fc2_weight.clone(), scale, self.q_bits), scale, dtype)
        Wfc2 = Wfc2.transpose(-1,-2)
        #==============================================

        tgt_len, hidden_dim = hidden_states.size()
        orig_tgt_len = tgt_len
        data_type = hidden_states.dtype
        zero = torch.zeros((1,1), dtype = hidden_states.dtype, device = hidden_states.device)
        decomp_factor = self.decomp_factor

        #hidden_states = self.fc1(hidden_states)
        # FC1 ========================================
        chunks = int(math.ceil(tgt_len / self.chunk_size))
        assert(chunks != 0)

        padded_rows = chunks * self.chunk_size
        pad_num = padded_rows - tgt_len
        
        if pad_num > 0:
            padding = torch.zeros((pad_num, hidden_dim), dtype = hidden_states.dtype, device = hidden_states.device) 
            hidden_chunks = torch.cat((hidden_states, padding),dim=0)
        else:
            hidden_chunks = hidden_states

        hidden_chunks = hidden_chunks.reshape(chunks, self.chunk_size, hidden_dim)
        result = torch.zeros_like(hidden_chunks)
        
        # Act normalize
        if self.h_ch_bias is None:
            ch_max = torch.max(hidden_chunks, dim=1, keepdim=True)[0] # chunks, 1, hidden_dim
            ch_min = torch.min(hidden_chunks, dim=1, keepdim=True)[0]
            ch_bias = torch.div(ch_max+ch_min,2)
            self.h_ch_bias_cal = ch_bias
        else:
            ch_bias = self.h_ch_bias[:chunks]

        fc1_ch_bias = torch.matmul(ch_bias, fc1_weight.transpose(-1,-2)) # chunks, 1, out_dim
        hidden_chunks -= ch_bias
        
        if self.fc1_tmax is None:
            h_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # chunks, chunk_size
            h_tmax = torch.max(h_tmax, dim=-1)[0] # chunks
            self.fc1_tmax_cal = h_tmax
        else:
            h_tmax = self.fc1_tmax[:chunks]

        thresholds=[]
        for i in range(decomp_factor):
            thresholds.append((h_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # chunks, 1

        if self.fc1_group_index is None:
            h_cmax = torch.max(torch.abs(hidden_chunks), dim=-2)[0] # chunks, hidden_dim
            self.fc1_cmax_cal = h_cmax
        
        count = []
        for i in range(decomp_factor):
            if self.fc1_group_index is None:
                if i==0:
                    mask = (h_cmax <= thresholds[i]) #chunks, hidden_dim
                else: 
                    mask = torch.logical_and((thresholds[i-1] < h_cmax),(h_cmax <= thresholds[i]))
            else:
                mask = (self.fc1_group_index[:chunks] == i).to(hidden_states.device)
        
            count.append(torch.sum(mask).item())
            if count[i] == 0:
                continue

            mask = mask.unsqueeze(1).repeat(1, self.chunk_size, 1) #chunks, chunk_size, hidden_dim
            scale = thresholds[i].unsqueeze(-1) / q_max #chunks, 1, 1
            decomp_fp = torch.where(mask, hidden_chunks, zero)
            decomp_i = sym_quant(decomp_fp, scale, self.q_bits)
            result += sym_dequant(decomp_i, scale, dtype)

        assert sum(count) == (self.embed_dim * chunks)

        fc1_out = torch.matmul(result, Wfc1) + fc1_ch_bias
        fc1_out = fc1_out.reshape(padded_rows, self.ffn_dim)[:orig_tgt_len]
        hidden_states = fc1_out + fc1_bias 

        if self.quant_out_bfloat:
            hidden_states = quant_bfloat(hidden_states)
        #=============================================

        hidden_states = self.activation_fn(hidden_states)

        #hidden_states = self.fc2(hidden_states)
        # FC2 ========================================
        tgt_len, hidden_dim = hidden_states.size()
        q_max = (2**self.q_bits-1) #quant to uint after ReLU 

        chunks = int(math.ceil(tgt_len / self.chunk_size))
        assert(chunks != 0)

        padded_rows = chunks * self.chunk_size
        pad_num = padded_rows - tgt_len

        if pad_num > 0:
            padding = torch.zeros((pad_num, self.ffn_dim), dtype = hidden_states.dtype, device = hidden_states.device) 
            hidden_chunks = torch.cat((hidden_states, padding),dim=0)
        else:
            hidden_chunks = hidden_states

        hidden_chunks = hidden_chunks.reshape(chunks, self.chunk_size, self.ffn_dim)
        result = torch.zeros_like(hidden_chunks)

        if self.fc2_tmax is None:
            h_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # chunks, chunk_size
            h_tmax = torch.max(h_tmax, dim=-1)[0] # chunks
            self.fc2_tmax_cal = h_tmax
        else:
            h_tmax = self.fc2_tmax[:chunks]

        thresholds=[]
        for i in range(decomp_factor):
            thresholds.append((h_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # chunks, 1

        if self.fc2_group_index is None:
            h_cmax = torch.max(torch.abs(hidden_chunks), dim = -2)[0] # chunks, hidden_dim
            self.fc2_cmax_cal = h_cmax
    
        count = []
        for i in range(decomp_factor):
            if self.fc2_group_index is None:
                if i==0:
                    mask = (h_cmax <= thresholds[i]) #chunks, hidden_dim
                else: 
                    mask = torch.logical_and((thresholds[i-1] < h_cmax), (h_cmax <= thresholds[i]))
            else:
                mask = (self.fc2_group_index[:chunks] == i).to(hidden_states.device)
    
            count.append(torch.sum(mask).item())
            if count[i] == 0:
                continue

            mask = mask.unsqueeze(1).repeat(1, self.chunk_size, 1) #chunks, chunk_size, hidden_dim
            scale = thresholds[i].unsqueeze(-1) / q_max
            decomp_fp = torch.where(mask, hidden_chunks, zero)
            decomp_i = torch.round(decomp_fp / scale).clamp(min=0,max=q_max)
            result += sym_dequant(decomp_i, scale, dtype)
    
        assert sum(count) == (self.ffn_dim * chunks)

        fc2_out = torch.matmul(result, Wfc2)
        fc2_out = fc2_out.reshape(padded_rows, self.embed_dim)[:orig_tgt_len]

        hidden_states = fc2_out + fc2_bias
        if self.quant_out_bfloat:
            hidden_states = quant_bfloat(hidden_states)
        #=============================================
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def get_sparsity(self):
        sparsity = 0.0
        for layer in self.decoder.layers:
            sparsity += (layer.self_attn.sparsity)/config.num_hidden_layers

        return sparsity

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

class LmHeadTender(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_out_bf16 = False
        self.q_bits = 4
        self.decomp_factor = 8 
        self.chunk_size = 256

    def forward(self, hidden_states, lm_weight):
        dtype = hidden_states.dtype
        decomp_factor = self.decomp_factor
        q_max = (2**(self.q_bits-1))-1

        scale = (torch.max(torch.abs(lm_weight.clone()),dim=-1,keepdim=True)[0])/q_max
        Wl = sym_dequant(sym_quant(lm_weight.clone(), scale, self.q_bits), scale, dtype)
        Wl = Wl.transpose(-1,-2)

        # LM Head =========================
        bsz, tgt_len, hidden_dim = hidden_states.shape
        orig_tgt_len = tgt_len
        hidden_states = hidden_states.view(tgt_len*bsz, hidden_dim)
        tgt_len, hidden_dim = hidden_states.size()
        zero = torch.zeros((1,1), dtype = hidden_states.dtype, device = hidden_states.device)

        chunks = int(math.ceil(tgt_len/self.chunk_size))
        assert(chunks != 0)

        padded_rows = chunks * self.chunk_size
        pad_num = padded_rows - tgt_len
        
        if pad_num > 0:
            padding = torch.zeros((pad_num, hidden_dim), dtype = hidden_states.dtype, device = hidden_states.device) 
            hidden_chunks = torch.cat((hidden_states, padding),dim=0)
        else:
            hidden_chunks = hidden_states

        hidden_chunks = hidden_chunks.reshape(chunks, self.chunk_size, hidden_dim)
        result = torch.zeros_like(hidden_chunks)

        h_tmax = torch.max(torch.abs(hidden_chunks), dim=-1)[0]  # chunks, chunk_size
        h_tmax = torch.max(h_tmax, dim=-1)[0] # chunks

        thresholds = []
        for i in range(decomp_factor):
            thresholds.append((h_tmax / (2 ** (decomp_factor - 1 - i))).unsqueeze(-1)) # chunks, 1
        
        h_cmax = torch.max(torch.abs(hidden_chunks), dim = -2)[0] # chunks, hidden_dim
        
        count = []
        for i in range(decomp_factor):
            if i==0:
                mask = (h_cmax <= thresholds[i]) #chunks, hidden_dim
            elif i == decomp_factor - 1:
                mask = (h_cmax > thresholds[i-1])
            else: 
                mask = torch.logical_and((thresholds[i - 1] < h_cmax), (h_cmax <= thresholds[i]))
        
            count.append(torch.sum(mask).item())
            if count[i] == 0:
                continue

            mask = mask.unsqueeze(1).repeat(1, self.chunk_size, 1) #chunks, chunk_size, hidden_dim
            scale = thresholds[i].unsqueeze(-1) / q_max #chunks, 1, 1
            decomp_fp = torch.where(mask, hidden_chunks, zero)
            decomp_i = sym_quant(decomp_fp, scale, self.q_bits)
            result += sym_dequant(decomp_i, scale, dtype)

        assert sum(count) == (hidden_dim * chunks)

        lm_out = torch.matmul(result, Wl)
        logits = lm_out.reshape(padded_rows, -1)[:bsz * orig_tgt_len]
        logits = logits.reshape(bsz, orig_tgt_len, -1)
        if self.quant_out_bf16:
            logits = quant_bfloat(logits)
        return logits


class OPTForCausalLM(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.quant_lm_head = False
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.lm_head_tender = LmHeadTender()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        if self.quant_lm_head:
            lm_weight = self.lm_head.weight.data.clone()
            logits = self.lm_head_tender(outputs[0], lm_weight).contiguous()
        else:
            logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
