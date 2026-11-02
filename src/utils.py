"""Shared helpers: config loading, seeding, jsonl IO, device selection."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(*names: str, root: Path | None = None) -> dict:
    """Load and merge configs/<name>.yaml files (later files win on key clashes, top level only)."""
    root = root or ROOT
    cfg: dict[str, Any] = {}
    for n in names:
        with open(root / "configs" / f"{n}.yaml", "r", encoding="utf-8") as f:
            part = yaml.safe_load(f) or {}
        for k, v in part.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    ov = os.environ.get("GLC_OVERRIDE")  # optional yaml merged last (used by the smoke test)
    if ov:
        with open(ov, "r", encoding="utf-8") as f:
            part = yaml.safe_load(f) or {}
        for k, v in part.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg


def resolve(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root or ROOT) / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32}[name]
