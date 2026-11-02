"""SplitCache: mixed-resolution context assembly.

Implementation note (state this in the report): SplitCache is realised as ``inputs_embeds`` +
interpolated ``position_ids`` fed to the frozen model, NOT as a custom per-layer DynamicCache subclass.
The effect is the same for the transformer (it attends over one contiguous sequence in a single fused SDPA
call), and the memory/compute savings are real because the sequence is physically shorter. Exact tokens keep
the RoPE positions they would have had in the uncompressed sequence; each block's k summary vectors get
positions evenly spaced across that block's original position range (Eq. 4 of the design doc), so the live
tokens that follow see exactly the same relative distances as in the uncompressed case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class SplitInputs:
    inputs_embeds: torch.Tensor          # (1, L, d)
    position_ids: torch.Tensor           # (1, L) float32 (fractional for summary vectors)
    attention_mask: torch.Tensor         # (1, L) long, all ones
    n_prefix: int                        # sequence index where the target tokens start (== L if no target)
    next_position: float                 # position id of the first generated token
    ctx_tokens_before: int               # cross-file context tokens if nothing were compressed
    ctx_tokens_after: int                # context entries actually fed (exact tokens + k per compressed block)
    left_tokens: int
    n_compressed_blocks: int = 0
    n_exact_blocks: int = 0
    block_kinds: list[str] = field(default_factory=list)  # "exact" / "compress" per input chunk


def interpolated_positions(start: float, n_tokens: int, k: int) -> list[float]:
    """Eq. 4: p_j = p_start + j * (p_end - p_start) / (k - 1), with p_end = p_start + n - 1."""
    if k == 1:
        return [start + (n_tokens - 1) / 2.0]
    return [start + j * (n_tokens - 1) / (k - 1) for j in range(k)]


def assemble(embed_fn: Callable[[list[int]], torch.Tensor], compressor, chunk_ids: list[list[int]],
             compress_flags: list[bool], left_ids: list[int], target_ids: list[int] | None, k: int,
             dtype: torch.dtype) -> SplitInputs:
    """Build the hybrid sequence: [chunk blocks in the given order] + [live left context] (+ [target]).

    chunk_ids / compress_flags are in PLACEMENT order (least relevant first, best chunk nearest the live code).
    A block with n <= k tokens is never compressed (nothing to gain).
    """
    assert len(chunk_ids) == len(compress_flags)
    flags = [bool(f) and len(ids) > k and compressor is not None for ids, f in zip(chunk_ids, compress_flags)]
    comp_idx = [i for i, f in enumerate(flags) if f]
    summaries = None
    if comp_idx:
        with torch.no_grad():
            blocks = [embed_fn(chunk_ids[i]).float() for i in comp_idx]
        summaries = compressor(blocks)  # (m, k, d), grad flows into the compressor only

    parts: list[torch.Tensor] = []
    pos: list[float] = []
    cursor = 0.0
    before = after = 0
    comp_seen = 0
    for i, ids in enumerate(chunk_ids):
        n = len(ids)
        before += n
        if flags[i]:
            parts.append(summaries[comp_seen].to(dtype))
            pos.extend(interpolated_positions(cursor, n, k))
            comp_seen += 1
            after += k
        else:
            with torch.no_grad():
                parts.append(embed_fn(ids).to(dtype))
            pos.extend(cursor + j for j in range(n))
            after += n
        cursor += n

    n_left = len(left_ids)
    if n_left:
        with torch.no_grad():
            parts.append(embed_fn(left_ids).to(dtype))
        pos.extend(cursor + j for j in range(n_left))
        cursor += n_left
    n_prefix = sum(p.shape[0] for p in parts)
    if target_ids:
        with torch.no_grad():
            parts.append(embed_fn(target_ids).to(dtype))
        pos.extend(cursor + j for j in range(len(target_ids)))
        cursor += len(target_ids)

    embeds = torch.cat(parts, dim=0)[None]
    dev = embeds.device
    return SplitInputs(
        inputs_embeds=embeds,
        position_ids=torch.tensor([pos], dtype=torch.float32, device=dev),
        attention_mask=torch.ones(1, embeds.shape[1], dtype=torch.long, device=dev),
        n_prefix=n_prefix,
        next_position=cursor if not target_ids else cursor,
        ctx_tokens_before=before, ctx_tokens_after=after, left_tokens=n_left,
        n_compressed_blocks=len(comp_idx), n_exact_blocks=len(chunk_ids) - len(comp_idx),
        block_kinds=["compress" if f else "exact" for f in flags],
    )


def truncate_to_budget(chunk_ids_best_first: list[list[int]], budget: int) -> list[list[int]]:
    """Truncated baseline: keep exact tokens, best chunk first, until `budget` tokens, then cut off."""
    out, used = [], 0
    for ids in chunk_ids_best_first:
        if used >= budget:
            break
        take = ids[: budget - used]
        out.append(take)
        used += len(take)
    return out
