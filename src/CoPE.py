import math

import torch
import torch.nn as nn
import flip
import cumsum
import flipcumsum

from einops import einsum


class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(
            torch.zeros(1, head_dim, npos_max)
        )

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        flipped_gates = flip.flip(gates, -1)
        pos = cumsum.cumsum(flipped_gates, -1)
        # pos = flipcumsum.flip_cumsum(gates, -1)
        pos = flip.flip(pos, -1)  # Flip back
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor  # Interpolation factor
        return logits_ceil * w + logits_floor * (1 - w)

class SelfAttn(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.cope = CoPE(npos_max, head_dim)
        self.head_dim = head_dim

    def forward(self, query, key, val, mask=None):
        # q, k, v have dimensions batch x seq_len x head_dim
        attn_logits = einsum(query, key, 'b i d, b j d -> b i j')  # QK^T
        attn_logits = attn_logits / math.sqrt(self.head_dim)  # QK^T
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))  # Add positional encodings (CoPE)
        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)
        out = einsum(attn, val, 'b i j, b j d -> b i d')
        return out

