"""Cross-attention compressor: k learnable queries attend over a block's token embeddings -> k summary vectors.

Per the design doc (Eq. 3) this is one nn.MultiheadAttention layer with a bank of k learnable queries
(AutoCompressors / Perceiver-resampler style). Two small additions make it work on raw embedding-layer
outputs (which carry no position information):
  * a learned absolute position embedding is added to the keys (otherwise the block is a bag of tokens)
  * the output is LayerNorm'ed and rescaled to the base model's embedding scale, so summary vectors are
    statistically similar to real token embeddings
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionCompressor(nn.Module):
    def __init__(self, d_model: int, k: int = 4, n_heads: int = 8, max_block_len: int = 320,
                 embed_std: float = 0.02):
        super().__init__()
        self.d_model, self.k, self.max_block_len = d_model, k, max_block_len
        self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.pos = nn.Embedding(max_block_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.norm_in = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_scale = nn.Parameter(torch.tensor(float(embed_std)))  # per-element std of real embeddings

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        """blocks: list of (n_i, d) token-embedding tensors (n_i > 0). Returns (len(blocks), k, d)."""
        if not blocks:
            return torch.zeros(0, self.k, self.d_model, device=self.queries.device)
        dev = self.queries.device
        n_max = min(max(b.shape[0] for b in blocks), self.max_block_len)
        B = len(blocks)
        x = torch.zeros(B, n_max, self.d_model, device=dev, dtype=torch.float32)
        pad = torch.ones(B, n_max, dtype=torch.bool, device=dev)  # True = ignore
        for i, b in enumerate(blocks):
            n = min(b.shape[0], n_max)
            x[i, :n] = b[:n].to(dev, torch.float32)
            pad[i, :n] = False
        with torch.autocast(device_type=dev.type, enabled=False):
            x = self.norm_in(x + self.pos(torch.arange(n_max, device=dev))[None])
            q = self.queries[None].expand(B, -1, -1)
            out, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)
            out = self.norm_out(out) * self.out_scale
        return out
