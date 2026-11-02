"""Metrics: exact match, edit similarity, compression ratio, KV-cache size, bootstrap CIs."""
from __future__ import annotations

import numpy as np


def exact_match(pred: str, gt: str) -> float:
    return float(pred.strip() == gt.strip())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(pred: str, gt: str) -> float:
    p, g = pred.strip(), gt.strip()
    if not p and not g:
        return 1.0
    return 1.0 - levenshtein(p, g) / max(len(p), len(g))


def compression_ratio(tokens_after: float, tokens_before: float) -> float:
    """Eq. 6: 1 - after/before, on cross-file context tokens (summary vectors count as k tokens)."""
    return 1.0 - tokens_after / tokens_before if tokens_before > 0 else 0.0


def kv_bytes_per_token(lm_config, dtype_bytes: int = 2) -> int:
    n_kv = getattr(lm_config, "num_key_value_heads", lm_config.num_attention_heads)
    head_dim = getattr(lm_config, "head_dim", None) or lm_config.hidden_size // lm_config.num_attention_heads
    return 2 * lm_config.num_hidden_layers * n_kv * head_dim * dtype_bytes


def bootstrap_ci(values, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n: int = 1000, seed: int = 0):
    """Mean(a - b) with a paired bootstrap CI (same examples in both)."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n, seed)
