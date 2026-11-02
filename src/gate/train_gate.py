"""Train the confidence gate, report honest metrics, and write gate scores back into the processed splits.

    python -m src.gate.train_gate

Labels come from ground-truth identifier overlap (labels.py). Features come from features.py.
Reports validation AND test accuracy/AUC next to trivial baselines (majority class, retrieval-score-only)
so you can show the gate learns something beyond the retrieval score.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from src.gate.features import FEATURE_NAMES, N_FEATURES, example_features
from src.gate.labels import example_labels
from src.gate.model import ConfidenceGate
from src.utils import load_config, read_jsonl, resolve, seed_everything, write_jsonl


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties get average rank). Returns 0.5 if only one class is present."""
    y = np.asarray(y).astype(int)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def build_xy(rows: list[dict], max_chunks: int):
    X, y, src = [], [], []
    for e in rows:
        for variant in ("rg1", "oracle"):
            chunks = e[f"chunks_{variant}"]
            if not chunks:
                continue
            X.append(example_features(chunks, e["left_text"], max_chunks))
            y.extend(example_labels(chunks, e["groundtruth"], e["left_text"]))
            src.extend([variant] * len(chunks))
    X = np.concatenate(X) if X else np.zeros((0, N_FEATURES), dtype=np.float32)
    return X, np.asarray(y, dtype=np.float32), np.asarray(src)


def evaluate(g: ConfidenceGate, X, y):
    p = g.score(X)
    pred = p >= 0.5
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    tnr = float((~pred)[y == 0].mean()) if (y == 0).any() else 0.0
    return {"acc": float((pred == (y == 1)).mean()), "bal_acc": (tpr + tnr) / 2, "recall_exact": tpr,
            "auc": auc(y, p), "pos_rate": float(y.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=None)
    a = ap.parse_args()
    cfg = {**load_config("data", "model", "training")}
    gcfg = cfg["gate"]  # training.yaml's gate section overrides model.yaml's (merged below)
    mcfg_gate = load_config("model")["gate"]
    seed_everything(cfg["seed"])
    pdir = resolve(a.processed_dir or cfg["processed_dir"])
    splits = {s: read_jsonl(pdir / f"{s}.jsonl") for s in ("train", "val", "test")}
    mc = cfg["max_chunks"]
    data = {s: build_xy(r, mc) for s, r in splits.items()}
    Xtr, ytr, _ = data["train"]
    print(f"gate data: train {len(ytr)} chunks (exact-rate {ytr.mean():.2%}), val {len(data['val'][1])}, test {len(data['test'][1])}")

    mean, std = Xtr.mean(0), Xtr.std(0)
    g = ConfidenceGate(N_FEATURES, mcfg_gate["hidden"], mcfg_gate["dropout"])
    g.set_norm(mean, std)
    opt = torch.optim.AdamW(g.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"])
    pos_w = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6)).clamp(0.2, 5.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xv, yv = data["val"][0], data["val"][1]
    best, best_state, bad = -1.0, None, 0
    for ep in range(gcfg["epochs"]):
        g.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), gcfg["batch_size"]):
            idx = perm[i:i + gcfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(g.logits(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        m = evaluate(g, Xv, yv)
        if m["auc"] > best:
            best, best_state, bad = m["auc"], copy.deepcopy(g.state_dict()), 0
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"epoch {ep:3d} loss {loss.item():.4f} val acc {m['acc']:.3f} auc {m['auc']:.3f}")
        if bad >= gcfg["patience"]:
            print(f"early stop at epoch {ep}")
            break
    g.load_state_dict(best_state)

    report = {}
    for s in ("train", "val", "test"):
        X, y, _ = data[s]
        r = evaluate(g, X, y)
        r["majority_acc"] = float(max(y.mean(), 1 - y.mean()))
        r["retrieval_score_auc"] = auc(y, X[:, FEATURE_NAMES.index("retrieval_score")])
        report[s] = r
        print(f"{s:5s}: AUC gate {r['auc']:.3f} vs retrieval-score-only {r['retrieval_score_auc']:.3f} | "
              f"balanced acc {r['bal_acc']:.3f} | recall of exact-needed chunks @0.5 {r['recall_exact']:.2f} | "
              f"exact-rate {r['pos_rate']:.2%}")

    path = resolve(cfg["paths"]["gate_ckpt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": g.state_dict(), "n_features": N_FEATURES, "hidden": mcfg_gate["hidden"],
                "dropout": mcfg_gate["dropout"], "feature_names": FEATURE_NAMES, "report": report}, path)
    print("saved", path)

    # write per-chunk gate scores back into the processed splits (used by training routing and eval sweeps)
    for s, rows in splits.items():
        for e in rows:
            for variant in ("rg1", "oracle"):
                ch = e[f"chunks_{variant}"]
                e[f"gate_{variant}"] = [float(x) for x in g.score(example_features(ch, e["left_text"], mc))]
        write_jsonl(pdir / f"{s}.jsonl", rows)
    print("gate scores written to", pdir)


if __name__ == "__main__":
    main()
