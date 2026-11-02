"""Shared evaluation loop used by run_eval.py and sweep_gate.py."""
from __future__ import annotations

import torch

from src.evaluation.conditions import plan
from src.evaluation.metrics import compression_ratio, edit_similarity, exact_match, kv_bytes_per_token


def run_condition(model, rows: list[dict], condition: str, tau: float | None = None,
                  budgets: list[int] | None = None, max_new_tokens: int = 48, desc: str = "") -> list[dict]:
    """Greedy-decode every row under one condition. Returns one record per example."""
    try:
        from tqdm import tqdm
        it = tqdm(list(enumerate(rows)), desc=desc or condition, leave=False)
    except ImportError:  # pragma: no cover
        it = list(enumerate(rows))
    cuda = model.device.type == "cuda"
    base_alloc = torch.cuda.memory_allocated() if cuda else 0
    kvb = kv_bytes_per_token(model.lm.config, 2)
    recs = []
    for i, e in it:
        chunks, flags = plan(e, condition, tau, None if budgets is None else budgets[i])
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        out = model.generate(e["left_ids"], chunks, flags, max_new_tokens)
        peak = (torch.cuda.max_memory_allocated() if cuda else 0)
        full_ctx = sum(len(c["ids"]) for c in e["chunks_oracle" if condition == "oracle_retrieval" else "chunks_rg1"])
        recs.append({
            "task_id": e["task_id"], "condition": condition, "tau": tau if tau is not None else "",
            "pred": out["text"], "gt": e["groundtruth"],
            "em": exact_match(out["text"], e["groundtruth"]),
            "es": edit_similarity(out["text"], e["groundtruth"]),
            "ctx_full": full_ctx, "ctx_fed": out["ctx_after"],
            "compression": compression_ratio(out["ctx_after"], full_ctx) if full_ctx else 0.0,
            "seq_len": out["seq_len"], "n_compressed": out["n_compressed"], "n_exact": out["n_exact"],
            "kv_cache_mb": out["seq_len"] * kvb / 2**20,
            "latency_s": out["latency_s"],
            "peak_vram_gb": peak / 2**30, "peak_extra_gb": max(peak - base_alloc, 0) / 2**30,
        })
    return recs
