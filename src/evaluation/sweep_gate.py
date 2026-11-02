"""Sweep the gate threshold tau -> accuracy / compression Pareto data + plot (the project's central result).

    python -m src.evaluation.sweep_gate [--split val] [--n 100]

For every tau it runs the gated system AND the truncated baseline with the same per-example token budget, so
the plot compares like with like. Horizontal lines show the full-context and no-context references.
Writes results/sweep_<split>.csv, results/pareto_<split>.png and (for split=val) results/chosen_tau.json.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import bootstrap_ci
from src.evaluation.runner import run_condition
from src.model.gated_model import GatedCompressionModel
from src.utils import get_device, load_config, read_jsonl, resolve, seed_everything


def pareto_plot(sweep: pd.DataFrame, refs: dict, path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (m, lab) in zip(axes, (("es", "Edit similarity"), ("em", "Exact match"))):
        g = sweep[sweep.condition == "gated"].sort_values("compression")
        t = sweep[sweep.condition == "truncated"].sort_values("compression")
        ax.plot(g["compression"], g[m], "o-", label="gated (ours)")
        ax.plot(t["compression"], t[m], "s--", label="truncated (same budget)")
        for _, r in g.iterrows():
            ax.annotate(f"τ={r['tau']:g}", (r["compression"], r[m]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(refs["full_context"][m], color="g", ls=":", label="full context (uncompressed)")
        ax.axhline(refs["no_context"][m], color="r", ls=":", label="no context")
        ax.set_xlabel("cross-file context compression ratio")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config("data", "model", "training", "sweep")
    seed_everything(cfg["seed"])
    res = resolve(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    n = a.n or (cfg["n_val_sweep"] if a.split == "val" else cfg["n_test"])
    rows = read_jsonl(resolve(cfg["processed_dir"]) / f"{a.split}.jsonl")[:n]
    p = {k: resolve(v) for k, v in cfg["paths"].items()}
    model = GatedCompressionModel.from_pretrained(cfg["base_model"], str(p["gate_ckpt"]), str(p["compressor_ckpt"]),
                                                  str(p["lora_dir"]), cfg=cfg, device=get_device())
    mnt = cfg["max_new_tokens"]
    refs = {}
    for c in ("full_context", "no_context"):
        r = run_condition(model, rows, c, None, None, mnt)
        refs[c] = {"em": sum(x["em"] for x in r) / len(r), "es": sum(x["es"] for x in r) / len(r)}
        print(f"{c:13s} EM {refs[c]['em']:.3f}  ES {refs[c]['es']:.3f}")

    out = []
    for tau in cfg["taus"]:
        g = run_condition(model, rows, "gated", tau, None, mnt, desc=f"gated tau={tau}")
        t = run_condition(model, rows, "truncated", tau, [x["ctx_fed"] for x in g], mnt, desc=f"truncated tau={tau}")
        for cond, recs in (("gated", g), ("truncated", t)):
            em, es = [x["em"] for x in recs], [x["es"] for x in recs]
            row = {"condition": cond, "tau": tau, "n": len(recs),
                   "em": sum(em) / len(em), "es": sum(es) / len(es),
                   "es_ci_lo": bootstrap_ci(es, cfg["bootstrap_samples"])[1], "es_ci_hi": bootstrap_ci(es, cfg["bootstrap_samples"])[2],
                   "compression": sum(x["compression"] for x in recs) / len(recs),
                   "seq_len": sum(x["seq_len"] for x in recs) / len(recs),
                   "latency_s": sum(x["latency_s"] for x in recs) / len(recs),
                   "peak_vram_gb": max(x["peak_vram_gb"] for x in recs)}
            out.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    sweep = pd.DataFrame(out)
    sweep.to_csv(res / f"sweep_{a.split}.csv", index=False)
    json.dump(refs, open(res / f"refs_{a.split}.json", "w"), indent=2)
    pareto_plot(sweep, refs, res / f"pareto_{a.split}.png", f"Accuracy vs compression ({a.split}, n={len(rows)})")
    print("saved", res / f"pareto_{a.split}.png")

    if a.split == "val":  # choose the operating point on VALIDATION data only
        g = sweep[sweep.condition == "gated"]
        ok = g[g["es"] >= refs["full_context"]["es"] - cfg["tolerance_es"]]
        pick = (ok.sort_values("compression", ascending=False).iloc[0] if len(ok)
                else g.sort_values("es", ascending=False).iloc[0])
        json.dump({"tau": float(pick["tau"]), "val_es": float(pick["es"]), "val_compression": float(pick["compression"]),
                   "rule": f"max compression with val edit-sim >= full_context - {cfg['tolerance_es']}" if len(ok)
                           else "fallback: best val edit-sim (no tau within tolerance)"},
                  open(res / "chosen_tau.json", "w"), indent=2)
        print(f"chosen tau = {pick['tau']} (val ES {pick['es']:.3f}, compression {pick['compression']:.1%})")


if __name__ == "__main__":
    main()
