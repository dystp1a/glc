"""Turn results/*.csv into results/RESULTS.md (markdown tables for the report/README)."""
from pathlib import Path
import json
import pandas as pd


def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


R = Path(__file__).resolve().parents[1] / "results"
out = ["# Results\n"]
f = R / "test_summary.csv"
if f.exists():
    s = pd.read_csv(f)
    tau = json.loads((R / "test_run.json").read_text())["tau"] if (R / "test_run.json").exists() else "?"
    out.append(f"## Test set (tau = {tau}, chosen on validation)\n")
    cols = ["condition", "n", "exact_match", "edit_sim", "compression_ratio", "ctx_tokens_fed", "kv_cache_mb",
            "peak_vram_gb_max", "latency_s"]
    t = s[cols].copy()
    t["exact_match"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(s.exact_match, s.em_ci_lo, s.em_ci_hi)]
    t["edit_sim"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(s.edit_sim, s.es_ci_lo, s.es_ci_hi)]
    out.append(md(t.round(3)))
f = R / "sweep_val.csv"
if f.exists():
    out.append("\n## Validation tau sweep\n")
    d = pd.read_csv(f).round(3)
    out.append(md(d))
(R / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", R / "RESULTS.md")
