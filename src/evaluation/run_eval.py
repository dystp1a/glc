"""Evaluate the four conditions (+ optional oracle-retrieval upper bound) on the same held-out examples.

    python -m src.evaluation.run_eval --split test [--tau 0.5] [--n 250]

no_context       lower bound
full_context     all retrieved chunks, uncompressed  (= the design doc's "Oracle" for compression)
truncated        same per-example cross-file token budget as the gated system, exact tokens, cut off
gated            exact if gate score >= tau, k summary vectors otherwise
oracle_retrieval extra: retrieval that peeks at the reference, uncompressed (skipped if file missing)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.evaluation.metrics import bootstrap_ci, paired_diff_ci
from src.evaluation.runner import run_condition
from src.model.gated_model import GatedCompressionModel
from src.utils import get_device, load_config, read_jsonl, resolve, seed_everything


def summarize(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        em, em_lo, em_hi = bootstrap_ci(g["em"], n_boot)
        es, es_lo, es_hi = bootstrap_ci(g["es"], n_boot)
        rows.append({
            "condition": cond, "n": len(g),
            "exact_match": em, "em_ci_lo": em_lo, "em_ci_hi": em_hi,
            "edit_sim": es, "es_ci_lo": es_lo, "es_ci_hi": es_hi,
            "compression_ratio": g["compression"].mean(),
            "ctx_tokens_fed": g["ctx_fed"].mean(), "seq_len": g["seq_len"].mean(),
            "kv_cache_mb": g["kv_cache_mb"].mean(),
            "peak_vram_gb_mean": g["peak_vram_gb"].mean(), "peak_vram_gb_max": g["peak_vram_gb"].max(),
            "peak_extra_gb_mean": g["peak_extra_gb"].mean(),
            "latency_s": g["latency_s"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=None, help="default: results/chosen_tau.json, else 0.5")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config("data", "model", "training", "sweep")
    seed_everything(cfg["seed"])
    res = resolve(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    tau = a.tau
    if tau is None:
        f = res / "chosen_tau.json"
        tau = json.loads(f.read_text())["tau"] if f.exists() else 0.5
    n = a.n or cfg["n_test"]
    rows = read_jsonl(resolve(cfg["processed_dir"]) / f"{a.split}.jsonl")[:n]
    print(f"{a.split}: {len(rows)} examples, tau={tau}")

    p = {k: resolve(v) for k, v in cfg["paths"].items()}
    model = GatedCompressionModel.from_pretrained(cfg["base_model"], str(p["gate_ckpt"]), str(p["compressor_ckpt"]),
                                                  str(p["lora_dir"]), cfg=cfg, device=get_device())
    mnt = cfg["max_new_tokens"]
    recs = run_condition(model, rows, "gated", tau, max_new_tokens=mnt)
    budgets = [r["ctx_fed"] for r in recs]                      # per-example budget = what the gated system used
    recs += run_condition(model, rows, "truncated", tau, budgets, mnt)
    recs += run_condition(model, rows, "full_context", None, None, mnt)
    recs += run_condition(model, rows, "no_context", None, None, mnt)
    if rows and rows[0].get("chunks_oracle"):
        recs += run_condition(model, rows, "oracle_retrieval", None, None, mnt)
    df = pd.DataFrame(recs)
    df.to_csv(res / f"{a.split}_per_example.csv", index=False)
    order = ["no_context", "truncated", "gated", "full_context", "oracle_retrieval"]
    summ = summarize(df, cfg["bootstrap_samples"])
    summ["condition"] = pd.Categorical(summ["condition"], [c for c in order if c in set(summ["condition"])])
    summ = summ.sort_values("condition")
    summ.to_csv(res / f"{a.split}_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False))

    g, t, f = (df[df.condition == c].set_index("task_id") for c in ("gated", "truncated", "full_context"))
    idx = g.index
    print(f"\ntau={tau}  compression={g['compression'].mean():.1%}")
    for name, other in (("gated - truncated", t), ("gated - full_context", f)):
        for m in ("em", "es"):
            d, lo, hi = paired_diff_ci(g.loc[idx, m], other.loc[idx, m], cfg["bootstrap_samples"])
            print(f"  {name:22s} {m.upper()}: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    (res / f"{a.split}_run.json").write_text(json.dumps({"tau": tau, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
