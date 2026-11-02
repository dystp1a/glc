"""Gradio demo: side-by-side completions with live memory/latency readouts.

    python demo/app.py            # needs trained checkpoints + data/processed/*.jsonl (run the pipeline first)

Pick a held-out example and a threshold tau. Three completions are produced from the same frozen model:
full context (uncompressed), truncated (same token budget as the gated run) and gated (ours).
Context chunks are colour-coded: green = kept exact, orange = compressed to k vectors.
Retrieval is CrossCodeEval's pre-computed chunks (the demo does not retrieve from your own repo).
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.evaluation.conditions import plan
from src.model.gated_model import GatedCompressionModel
from src.utils import get_device, load_config, read_jsonl, resolve


def compare(model: GatedCompressionModel, ex: dict, tau: float, max_new_tokens: int = 48) -> dict:
    """Run the three conditions on one example. Pure function (no UI) so it can be tested headlessly."""
    cuda = model.device.type == "cuda"
    res = {}
    for cond in ("gated", "full_context", "truncated"):
        budget = res["gated"]["ctx_after"] if cond == "truncated" else None
        chunks, flags = plan(ex, cond, tau, budget)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        out = model.generate(ex["left_ids"], chunks, flags, max_new_tokens)
        out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        res[cond] = out
    return res


def chunks_html(ex: dict, kinds: list[str]) -> str:
    parts = []
    for c, kind in zip(ex["chunks_rg1"], kinds):
        color, label = ("#d8f3dc", "EXACT") if kind == "exact" else ("#ffe5b4", "COMPRESSED -> k vectors")
        parts.append(f"<div style='background:{color};color:#111;padding:6px;margin:4px 0;border-radius:6px;font-size:12px'>"
                     f"<b>{label}</b> &nbsp; {html.escape(c['filename'])} ({len(c['ids'])} tokens)"
                     f"<pre style='white-space:pre-wrap;margin:2px 0'>{html.escape(c['text'][:500])}</pre></div>")
    return "".join(parts)


def main() -> None:
    import gradio as gr

    cfg = load_config("data", "model", "training", "sweep")
    p = {k: resolve(v) for k, v in cfg["paths"].items()}
    rows = read_jsonl(resolve(cfg["processed_dir"]) / "test.jsonl")[:100]
    model = GatedCompressionModel.from_pretrained(cfg["base_model"], str(p["gate_ckpt"]), str(p["compressor_ckpt"]),
                                                  str(p["lora_dir"]), cfg=cfg, device=get_device())
    by_id = {r["task_id"]: r for r in rows}

    def run(task_id: str, tau: float):
        ex = by_id[task_id]
        r = compare(model, ex, tau)
        g, f, t = r["gated"], r["full_context"], r["truncated"]
        stats = (f"gated: {g['ctx_after']}/{g['ctx_before']} context tokens "
                 f"({1 - g['ctx_after'] / max(g['ctx_before'], 1):.0%} compressed), {g['latency_s']:.2f}s, peak {g['peak_gb']:.2f} GB | "
                 f"full: {f['ctx_after']} tokens, {f['latency_s']:.2f}s, peak {f['peak_gb']:.2f} GB | "
                 f"truncated: {t['ctx_after']} tokens, {t['latency_s']:.2f}s")
        return (ex["left_text"][-800:], chunks_html(ex, g["block_kinds"]), f["text"], t["text"], g["text"],
                ex["groundtruth"], stats)

    with gr.Blocks(title="Gated latent compression") as ui:
        gr.Markdown("## Syntax-aware gated latent compression - side-by-side completions")
        with gr.Row():
            sel = gr.Dropdown(list(by_id), value=list(by_id)[0], label="Held-out example")
            tau = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="tau (higher = more compression)")
            btn = gr.Button("Complete", variant="primary")
        code = gr.Code(label="Live file (tail)", language="python")
        ctx = gr.HTML(label="Retrieved context")
        with gr.Row():
            o_full, o_trunc, o_gate = (gr.Textbox(label=l) for l in ("Full context", "Truncated (same budget)", "Gated (ours)"))
        gt = gr.Textbox(label="Ground truth")
        stats = gr.Textbox(label="Live readouts")
        btn.click(run, [sel, tau], [code, ctx, o_full, o_trunc, o_gate, gt, stats])
    ui.launch()


if __name__ == "__main__":
    main()
