#!/usr/bin/env bash
# Full pipeline (Kaggle / Linux / WSL2). Usage: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/smoke_test.py                       # plumbing check on a tiny random model (~1-3 min)
python scripts/download_data.py
python -m src.data.prepare_cceval
python -m src.gate.train_gate
python -m src.sanity_check
python -m src.training.train_compressor --max_steps 2   # timing probe: look at seconds/step, then run the real one
python -m src.training.train_compressor
python -m src.evaluation.sweep_gate --split val
python -m src.evaluation.run_eval --split test
python scripts/make_report.py
