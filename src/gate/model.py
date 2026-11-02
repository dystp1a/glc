"""Confidence gate: small MLP over static-analysis features -> P(chunk must stay exact)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConfidenceGate(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))

    @torch.no_grad()
    def score(self, feats: np.ndarray) -> np.ndarray:
        """feats: (n, F) -> (n,) probabilities in [0, 1]. High = keep exact; chunk is exact iff score >= tau."""
        if len(feats) == 0:
            return np.zeros((0,), dtype=np.float32)
        was = self.training
        self.eval()
        p = self.forward(torch.as_tensor(feats, dtype=torch.float32, device=self.mean.device))
        self.train(was)
        return p.cpu().numpy()

    @classmethod
    def load(cls, path: str, device="cpu") -> "ConfidenceGate":
        ck = torch.load(path, map_location=device, weights_only=False)
        g = cls(ck["n_features"], ck["hidden"], ck.get("dropout", 0.1))
        g.load_state_dict(ck["state_dict"])
        return g.to(device).eval()
