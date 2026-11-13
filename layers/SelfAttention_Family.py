import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from layers.Embed import RotaryEmbedding, apply_rotary_pos_emb_blhD


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayerWithRoPE(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 rope=True, rope_theta=10000, rope_max_pos=2048):
        super(AttentionLayerWithRoPE, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.attn_dropout = getattr(attention, 'dropout', nn.Dropout(0.0)).p
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.use_rope = rope
        if self.use_rope:
            if d_keys % 2 != 0:
                raise ValueError(f"RoPE requires head_dim (d_keys) to be even, got d_keys={d_keys}")
            self.rotary_emb = RotaryEmbedding(
                dim=d_keys,
                max_position_embeddings=rope_max_pos,
                base=rope_theta,
            )

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, position_ids=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        if self.use_rope:
            cos, sin = self.rotary_emb(values, seq_len=S)
            if L != S:
                q_cos, q_sin = self.rotary_emb(values, seq_len=L)
                queries, _ = apply_rotary_pos_emb_blhD(queries, queries, q_cos, q_sin)
                _, keys    = apply_rotary_pos_emb_blhD(keys, keys, cos, sin)
            else:
                queries, keys = apply_rotary_pos_emb_blhD(queries, keys, cos, sin)

        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_projection(out), None
