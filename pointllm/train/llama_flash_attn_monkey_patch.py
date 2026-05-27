"""Robust flash-attn monkey patch aligned with top-level implementation.

Handles 2D and 4D attention masks, returns (attn_out, None) to match HF
decoder expectations, and supports a naive-attention fallback via env:
POINTLLM_FORCE_NAIVE_ATTENTION=1 or DISABLE_FLASH_ATTNN=1.
"""

from typing import Optional, Tuple
import os
import math

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

_FLASH_ATTN_AVAILABLE = True
_FLASH_ATTN_IMPORT_ERROR = None
try:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    except Exception:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
        )
    from flash_attn.bert_padding import unpad_input, pad_input
except Exception as exc:
    _FLASH_ATTN_AVAILABLE = False
    _FLASH_ATTN_IMPORT_ERROR = exc


def _is_primary_process() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    rank = os.getenv("RANK", None)
    local_rank = os.getenv("LOCAL_RANK", None)
    if rank is not None:
        return str(rank) == "0"
    if local_rank is not None:
        return str(local_rank) == "0"
    return True


def _naive_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    try:
        n_heads = self.num_heads
    except AttributeError:
        n_heads = getattr(getattr(self, 'config', object()), 'num_attention_heads', 32)
    try:
        head_dim = self.head_dim
    except AttributeError:
        hidden_size = getattr(self, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self, 'config', object()), 'hidden_size', hidden_states.size(-1))
        head_dim = hidden_size // n_heads

    q = self.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)

    kv_seq_len = k.shape[-2]
    offset = 0
    if past_key_value is not None:
        offset = past_key_value[0].shape[-2]
        kv_seq_len += offset
    try:
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
    except AttributeError:
        try:
            cos, sin = self.rope(v, seq_len=kv_seq_len)
        except AttributeError:
            cos, sin = None, None
    if cos is not None and sin is not None:
        q, k = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)

    key_padding_mask = attention_mask
    if key_padding_mask is not None and key_padding_mask.dim() == 4:
        b, one, tlen, slen = key_padding_mask.shape
        if one == 1:
            m = key_padding_mask.squeeze(1)
            keep = (m.max(dim=1).values == 0)
            key_padding_mask = keep
        else:
            raise AssertionError(f"Unsupported 4D attention_mask shape: {tuple(key_padding_mask.shape)}")
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)

    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal = torch.full((q_len, q_len), float('-inf'), device=attn_scores.device, dtype=attn_scores.dtype)
    causal = torch.triu(causal, diagonal=1)
    attn_scores = attn_scores + causal
    if key_padding_mask is not None:
        pad_mask = ~key_padding_mask
        pad_mask = pad_mask.view(bsz, 1, 1, q_len)
        attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_probs, v)
    out = context.transpose(1, 2).contiguous().view(bsz, q_len, n_heads * head_dim)
    attn_out = self.o_proj(out)
    return attn_out, None, None


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    try:
        n_heads = self.num_heads
    except AttributeError:
        n_heads = getattr(getattr(self, 'config', object()), 'num_attention_heads', 32)
    try:
        head_dim = self.head_dim
    except AttributeError:
        hidden_size = getattr(self, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self, 'config', object()), 'hidden_size', hidden_states.size(-1))
        head_dim = hidden_size // n_heads

    query_states = self.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    offset = 0
    if past_key_value is not None:
        offset = past_key_value[0].shape[-2]
        kv_seq_len += offset
    try:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    except AttributeError:
        try:
            cos, sin = self.rope(value_states, seq_len=kv_seq_len)
        except AttributeError:
            cos, sin = None, None
    if cos is not None and sin is not None:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)

    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"
    assert past_key_value is None, "past_key_value is not supported"

    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # b, nh, 3, s, d
    qkv = qkv.transpose(1, 3)  # b, s, 3, nh, d
    key_padding_mask = attention_mask

    if key_padding_mask is not None and key_padding_mask.dim() == 4:
        b, one, tlen, slen = key_padding_mask.shape
        if one == 1:
            m = key_padding_mask.squeeze(1)
            keep = (m.max(dim=1).values == 0)
            key_padding_mask = keep
        else:
            raise AssertionError(f"Unsupported 4D attention_mask shape: {tuple(key_padding_mask.shape)}")
    if key_padding_mask is not None:
        assert key_padding_mask.dim() == 2 and key_padding_mask.shape[1] == q_len
        key_padding_mask = key_padding_mask.to(torch.bool)

    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        output = flash_attn_unpadded_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_unpadded_qkvpacked_func(x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices, bsz, q_len), 'b s (h d) -> b s h d', h=nheads)

    attn_out = self.o_proj(rearrange(output, 'b s h d -> b s (h d)'))
    return attn_out, None, None


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    force_naive = (
        os.getenv("POINTLLM_FORCE_NAIVE_ATTENTION", "0") == "1"
        or os.getenv("DISABLE_FLASH_ATTN", "0") == "1"
        or not _FLASH_ATTN_AVAILABLE
    )
    if force_naive:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = _naive_attn_forward
        if _is_primary_process():
            reason = "env override" if _FLASH_ATTN_AVAILABLE else f"flash-attn unavailable: {_FLASH_ATTN_IMPORT_ERROR}"
            print(f"[ATTN] Using naive attention ({reason})")
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
        if _is_primary_process():
            print("[ATTN] Using flash-attn (with fp32 fallback)")
