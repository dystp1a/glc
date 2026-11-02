# Full pipeline on Windows PowerShell (needs Python 3.10+ and a CUDA PyTorch for GPU steps).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python tests/smoke_test.py
python scripts/download_data.py
python -m src.data.prepare_cceval
python -m src.gate.train_gate
python -m src.sanity_check
python -m src.training.train_compressor
python -m src.evaluation.sweep_gate --split val
python -m src.evaluation.run_eval --split test
python scripts/make_report.py
