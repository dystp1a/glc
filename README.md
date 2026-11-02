# Syntax-Aware Gated Latent Compression for Repository-Level Code Completion

A frozen Qwen2.5-Coder-1.5B completes one line of Python using cross-file context from CrossCodeEval.
A **confidence gate** keeps precision-critical chunks as exact tokens and sends the rest through a
**cross-attention compressor** (k learnable queries -> k summary vectors per chunk). **SplitCache** assembles the
mixed sequence with interpolated RoPE positions. Central result: the accuracy-vs-compression **Pareto curve** from a tau sweep.

```
retrieved chunks -> features -> Gate --score>=tau--> exact tokens ----\
                                  \--score<tau---> Compressor (k vecs) +--> SplitCache -> frozen LLM(+LoRA) -> completion
```

## Quick start

**Kaggle (training + eval, T4 16 GB):** open `notebooks/kaggle_run.ipynb`, set GPU T4 + Internet on, run top to bottom.
**Windows / Linux:** `pip install -r requirements.txt`, then `scripts/run_all.ps1` (PowerShell) or `bash scripts/run_all.sh`.
On Windows, use the Kaggle run for training; the laptop is for the smoke test, sanity check and Gradio demo
(`python demo/app.py` after copying the checkpoints/ and data/processed/ folders from Kaggle).

Always start with `python tests/smoke_test.py` (tiny random model, CPU, 1-3 min) - it prints `SMOKE TEST PASSED` if the plumbing works.

| Step | Command |
|---|---|
| download CrossCodeEval | `python scripts/download_data.py` |
| tokenize, split by repo, cache | `python -m src.data.prepare_cceval` (`--inspect` shows the raw schema) |
| gate features, labels, training | `python -m src.gate.train_gate` |
| zero-compression check | `python -m src.sanity_check` |
| compressor + LoRA | `python -m src.training.train_compressor [--resume] [--max_steps N]` |
| tau sweep + Pareto plot | `python -m src.evaluation.sweep_gate --split val` |
| final evaluation | `python -m src.evaluation.run_eval --split test` |
| markdown tables | `python scripts/make_report.py` |

All hyperparameters live in `configs/*.yaml`.

## Evaluation conditions (same held-out examples)

| Condition | Meaning |
|---|---|
| `no_context` | lower bound |
| `full_context` | all retrieved chunks, uncompressed (the design doc's "Oracle" for compression) |
| `truncated` | exact tokens, best chunk first, cut at **the same per-example token budget** as the gated run |
| `gated` | exact if gate score >= tau, else k=4 summary vectors |
| `oracle_retrieval` | extra upper bound: retrieval that peeks at the reference, uncompressed |

Metrics: exact match, edit similarity (95% bootstrap CIs; paired differences vs truncated/full), compression ratio
(Eq. 6, on cross-file context tokens), sequence length, estimated KV-cache MB, peak VRAM (`torch.cuda.max_memory_allocated`),
latency. Greedy decoding, max 48 new tokens, stop at newline. tau is **chosen on validation**, never on test.

## Data

CrossCodeEval Python (official release): 2,665 examples, 471 repositories. The rg1/oracle/no-context files list the same
tasks in different orders, so they are joined by `task_id`. Split **by repository** (no repo in two splits; asserted).
Each chunk is truncated to 256 tokens, the live file keeps its last 512 tokens; token ids are cached once.

## Gate labels (why it is not circular)

A chunk is labelled *exact* if it contains an identifier of the ground-truth line that is **not already visible** in the
live context, else *compress*. The ground truth is only used for labels; features (tree-sitter structure counts, identifier
overlap with the last 20/3 lines, whether the chunk's file is imported, retrieval score/rank) are all available at inference.
`train_gate` prints validation/test AUC next to a retrieval-score-only baseline. Accuracy at 0.5 is not reported as the headline
because the loss is class-weighted (only ~17% of chunks are "exact").

## Deviations from the design doc (state these in the report)

* **Parser:** the tree-sitter dependency traversal is replaced by CrossCodeEval's pre-retrieved chunks; tree-sitter is used for gate features only.
* **SplitCache** is implemented as `inputs_embeds` + interpolated `position_ids` (one contiguous sequence, one fused SDPA call), not a custom per-layer `DynamicCache` subclass. Savings are real because the sequence is shorter. Exact tokens keep their uncompressed RoPE positions; a block's k vectors are spread over its original range (Eq. 4).
* **Compressor** adds a learned position embedding on the keys and a LayerNorm + learned scale on the outputs (summary vectors match the embedding scale).
* **Attention backend:** PyTorch SDPA, not flash-attn (T4/Windows friendly). **Optimizer:** AdamW. **Logging:** CSV (no W&B). **Gate:** 2 hidden layers of width 64 (doc said 128 in one place, 64 in another).
* The training routing is random: tau ~ U(0.2, 0.8) per example, plus 20% of examples with every chunk compressed.

## Honest limitations

* Contamination: Qwen2.5-Coder was probably trained on public GitHub code overlapping CrossCodeEval.
* One epoch over ~1,000 examples is small; the compressor reads raw embedding-layer outputs. Expect results between truncated and full context - the 60% / <5% target in the design doc is a goal, not a guarantee.
* With a 1.5B GQA model the KV cache is only ~28 KB/token, so **peak VRAM is dominated by weights**; the report includes estimated KV-cache MB, sequence length and latency so the saving is visible.
* The gate is a weak-but-real predictor (see `train_gate` output). Secrets in context are not detected.
* The demo uses CrossCodeEval's retrieved chunks, not retrieval over your own repo.

## Layout

`configs/` yaml configs | `src/data` prep | `src/gate` features, labels, MLP | `src/compressor` | `src/cache/split_cache.py` |
`src/model/gated_model.py` (HF-style `from_pretrained`, `generate`, `complete`) | `src/training` | `src/evaluation` | `tests/` smoke + gradient tests | `demo/app.py` | `notebooks/kaggle_run.ipynb`

## Push to GitHub

```
git init && git add . && git commit -m "Gated latent compression"
git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
```
