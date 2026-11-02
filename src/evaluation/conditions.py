"""Routing for the evaluation conditions. Each returns (chunk_ids_best_first, compress_flags_best_first)."""
from __future__ import annotations

from src.cache.split_cache import truncate_to_budget

CONDITIONS = ["no_context", "full_context", "truncated", "gated", "oracle_retrieval"]


def plan(example: dict, condition: str, tau: float | None = None, budget: int | None = None):
    if condition == "no_context":
        return [], []
    if condition == "full_context":            # all retrieved chunks, uncompressed (design doc's "Oracle")
        ids = [c["ids"] for c in example["chunks_rg1"]]
        return ids, [False] * len(ids)
    if condition == "oracle_retrieval":        # reference-peeking retrieval, uncompressed (extra upper bound)
        ids = [c["ids"] for c in example["chunks_oracle"]]
        return ids, [False] * len(ids)
    if condition == "truncated":               # same token budget as the gated system, exact tokens, cut off
        ids = truncate_to_budget([c["ids"] for c in example["chunks_rg1"]], int(budget))
        return ids, [False] * len(ids)
    if condition == "gated":
        ids = [c["ids"] for c in example["chunks_rg1"]]
        scores = example["gate_rg1"]
        return ids, [s < tau for s in scores]
    raise ValueError(condition)
