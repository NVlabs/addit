# Copyright 2024 The HuggingFace Team. All rights reserved.
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
#
# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

from collections import defaultdict
from diffusers.models.attention_processor import Attention, apply_rope
from typing import Callable, List, Optional, Tuple, Union

from addit_attention_store import AttentionStore
from visualization_utils import show_tensors

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import brentq

def apply_standard_attention(query, key, value, attn, attention_probs=None):
    batch_size, attn_heads, _, head_dim = query.shape

    # Do normal attention, to cache the attention scores
    query = query.reshape(batch_size*attn_heads, -1, head_dim)
    key = key.reshape(batch_size*attn_heads, -1, head_dim)
    value = value.reshape(batch_size*attn_heads, -1, head_dim)
    
    if attention_probs is None:
        attention_probs = attn.get_attention_scores(query, key)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = hidden_states.view(batch_size, attn_heads, -1, head_dim)
    
    return hidden_states, attention_probs

def apply_extended_attention(query, key, value, attention_store, attn, layer_name, step_index, extend_type="pixels",
                             extended_scale=1., record_attention=False):
    batch_size = query.size(0)
    extend_query = query[1:]

    if extend_type == "full":
        added_key = key[0] * extended_scale
        added_value = value[0]
    elif extend_type == "text":
        added_key = key[0, :, :512] * extended_scale
        added_value = value[0, :, :512]
    elif extend_type == "pixels":
        added_key =  key[0, :, 512:]
        added_value =  value[0, :, 512:]

        key[1]  = key[1] * extended_scale

    extend_key = torch.cat([added_key, key[1]], dim=1).unsqueeze(0)
    extend_value = torch.cat([added_value, value[1]], dim=1).unsqueeze(0)

    hidden_states_0 = F.scaled_dot_product_attention(query[:1], key[:1], value[:1], dropout_p=0.0, is_causal=False)

    if record_attention or attention_store.is_cache_attn_ratio(step_index):
        hidden_states_1, attention_probs_1 = apply_standard_attention(extend_query, extend_key, extend_value, attn)
    else:
        hidden_states_1 = F.scaled_dot_product_attention(extend_query, extend_key, extend_value, dropout_p=0.0, is_causal=False)

    if record_attention:
        # Store Attention
        seq_len = attention_probs_1.size(2) - attention_probs_1.size(1)
        self_attention_probs_1 = attention_probs_1[:,:,seq_len:]
        attention_store.store_attention(self_attention_probs_1, layer_name, 1, attn.heads)

    if attention_store.is_cache_attn_ratio(step_index):
        attention_store.store_attention_ratios(attention_probs_1, step_index, layer_name)
            
    hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)

    return hidden_states

def apply_attention(query, key, value, attention_store, attn, layer_name, step_index,
                    record_attention, extended_attention, extended_scale):
    if extended_attention:
        hidden_states = apply_extended_attention(query, key, value, attention_store, attn, layer_name, step_index,
                                                     extended_scale=extended_scale, 
                                                     record_attention=record_attention)
    else:
        if record_attention:
            hidden_states_0 = F.scaled_dot_product_attention(query[:1], key[:1], value[:1], dropout_p=0.0, is_causal=False)
            hidden_states_1, attention_probs_1 = apply_standard_attention(query[1:], key[1:], value[1:], attn)
            attention_store.store_attention(attention_probs_1, layer_name, 1, attn.heads)

            hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

    return hidden_states

class AdditFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer_name: str, attention_store: AttentionStore, 
                 extended_steps: Tuple[int, int] = (0, 30), **kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.layer_name = layer_name
        self.layer_idx = int(layer_name.split(".")[-1])
        self.attention_store = attention_store

        self.extended_steps = (0, extended_steps) if isinstance(extended_steps, int) else extended_steps

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,

        step_index: Optional[int] = None,
        extended_scale: Optional[float] = 1.0,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        record_attention = self.attention_store.is_record_attention(self.layer_name, step_index)
        extend_start, extend_end = self.extended_steps
        extended_attention = extend_start <= step_index <= extend_end

        hidden_states = apply_attention(query, key, value, self.attention_store, attn, self.layer_name, step_index,
                        record_attention, extended_attention, extended_scale)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
    
class AdditFluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, layer_name: str, attention_store: AttentionStore, 
                 extended_steps: Tuple[int, int] = (0, 30), **kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.layer_name = layer_name
        self.layer_idx = int(layer_name.split(".")[-1])
        self.attention_store = attention_store

        self.extended_steps = (0, extended_steps) if isinstance(extended_steps, int) else extended_steps

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        extended_scale: Optional[float] = 1.0,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        record_attention = self.attention_store.is_record_attention(self.layer_name, step_index)
        extend_start, extend_end = self.extended_steps
        extended_attention = extend_start <= step_index <= extend_end

        hidden_states = apply_attention(query, key, value, self.attention_store, attn, self.layer_name, step_index,
                        record_attention, extended_attention, extended_scale)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states