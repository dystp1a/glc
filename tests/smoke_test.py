"""End-to-end smoke test on a tiny random model + synthetic data (CPU, ~1-3 minutes).

    python tests/smoke_test.py

Runs the real scripts: prepare -> train_gate -> train_compressor -> sanity_check -> sweep_gate -> run_eval.
It checks that the PLUMBING works (shapes, positions, routing, checkpoints, metrics, plots). The tiny random
model learns nothing, so the numbers it prints are meaningless - only the absence of errors matters.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from make_tiny import make_fake_cceval, make_tiny_model  # noqa: E402


def run(mod: str, env: dict, *args: str) -> None:
    print(f"\n$ python -m {mod} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {mod}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="glc_smoke_"))
    model_dir = make_tiny_model(tmp / "model")
    make_fake_cceval(tmp / "raw")
    ov = {
        "base_model": str(model_dir), "dtype": "float32", "k": 4,
        "compressor": {"n_heads": 4, "max_block_len": 96},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "raw_dir": str(tmp / "raw"), "processed_dir": str(tmp / "processed"),
        "max_chunks": 3, "chunk_tokens": 48, "left_tokens": 96, "max_target_tokens": 24, "train_max_tokens": 600,
        "split": {"n_test": 12, "n_val": 6, "n_train": 30, "seed": 0},
        "grad_accum": 2, "eval_every": 2, "save_every": 2, "val_examples": 4, "log_every": 1, "warmup_steps": 1,
        "gate": {"epochs": 30, "patience": 10, "hidden": 16},
        "paths": {"gate_ckpt": str(tmp / "ck/gate.pt"), "compressor_ckpt": str(tmp / "ck/compressor.pt"),
                  "lora_dir": str(tmp / "ck/lora"), "train_state": str(tmp / "ck/state.pt"),
                  "log_csv": str(tmp / "logs/train_log.csv")},
        "taus": [0.0, 0.5, 1.01], "n_val_sweep": 6, "n_test": 8, "max_new_tokens": 6,
        "bootstrap_samples": 50, "results_dir": str(tmp / "results"), "amp": False,
    }
    ovp = tmp / "override.yaml"
    ovp.write_text(yaml.safe_dump(ov))
    env = {**os.environ, "GLC_OVERRIDE": str(ovp), "PYTHONPATH": str(ROOT), "TOKENIZERS_PARALLELISM": "false"}
    run("src.data.prepare_cceval", env, "--inspect")
    run("src.data.prepare_cceval", env)
    run("src.gate.train_gate", env)
    run("src.training.train_compressor", env, "--max_steps", "4")
    run("src.training.train_compressor", env, "--max_steps", "6", "--resume")  # exercises resume
    run("src.sanity_check", env)
    run("src.evaluation.sweep_gate", env, "--split", "val")
    run("src.evaluation.run_eval", env, "--split", "test")
    for f in ("pareto_val.png", "chosen_tau.json", "test_summary.csv"):
        assert (tmp / "results" / f).exists(), f"missing result {f}"
    print(f"\nSMOKE TEST PASSED (artifacts in {tmp})")


if __name__ == "__main__":
    main()
