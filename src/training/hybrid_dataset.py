"""Hybrid training sequences: each example's retrieved chunks are routed to exact / compress and the model
is trained to predict the ground-truth line. The (cheap) routing is decided here; the compression itself
happens inside the model so gradients reach the compressor. Loss covers ONLY the target-line tokens.
"""
from __future__ import annotations

import random

from torch.utils.data import Dataset


class HybridDataset(Dataset):
    def __init__(self, rows: list[dict], tau_range=(0.2, 0.8), p_all_compress: float = 0.2,
                 fixed_tau: float | None = None, seed: int = 0):
        self.rows, self.tau_range, self.p_all, self.fixed_tau, self.seed = rows, tau_range, p_all_compress, fixed_tau, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        e = self.rows[i]
        chunks = [c["ids"] for c in e["chunks_rg1"]]
        scores = e["gate_rg1"]
        if self.fixed_tau is not None:
            flags = [s < self.fixed_tau for s in scores]
        else:
            rng = random.Random(f"{self.seed}-{self.epoch}-{i}")  # deterministic per (epoch, index) -> resumable
            if rng.random() < self.p_all:
                flags = [True] * len(chunks)
            else:
                tau = rng.uniform(*self.tau_range)
                flags = [s < tau for s in scores]
        return {"task_id": e["task_id"], "left_ids": e["left_ids"], "target_ids": e["target_ids"],
                "chunks": chunks, "compress": flags}
