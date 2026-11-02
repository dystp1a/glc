"""Zero-compression sanity check (the design doc's reliability milestone).

With no block routed to the compressor, the SplitCache path (inputs_embeds + position_ids + KV-cache decode loop)
must reproduce the plain base model. Checks, with the LoRA adapter disabled:
  1. logits: hybrid path vs. plain model(input_ids) on [chunks + left context]
  2. greedy generation: custom loop vs. HF generate()
  3. compress path runs, produces k vectors/block, and positions are monotone and interpolated
    python -m src.sanity_check [--model Qwen/Qwen2.5-Coder-1.5B]
"""
from __future__ import annotations

import argparse

import torch

from src.cache.split_cache import assemble, interpolated_positions
from src.model.gated_model import GatedCompressionModel
from src.utils import get_device, load_config, read_jsonl, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    cfg = load_config("data", "model", "training")
    device = get_device()
    model = GatedCompressionModel.from_pretrained(a.model or cfg["base_model"], cfg=cfg, device=device)
    tok, lm = model.tokenizer, model.lm
    pf = resolve(cfg["processed_dir"]) / "val.jsonl"
    if pf.exists():
        ex = read_jsonl(pf)[0]
        chunks = [c["ids"] for c in ex["chunks_rg1"]][:3]
        left = ex["left_ids"][-200:]
    else:
        chunks = [tok("# a.py\ndef foo(x):\n    return x + 1\n\n", add_special_tokens=False)["input_ids"]]
        left = tok("import a\nvalue = foo(", add_special_tokens=False)["input_ids"]
    tol = 5e-2 if model.dtype == torch.float16 else 1e-4
    ok = True
    with torch.no_grad(), model._peft[0].disable_adapter():
        placed = list(reversed(chunks))
        si = assemble(model.embed, None, placed, [False] * len(placed), left, None, model.k, model.dtype)
        h = model._hidden(si).last_hidden_state[:, -1]
        l_hybrid = lm.lm_head(h).float()
        flat = [t for c in placed for t in c] + left
        l_plain = lm(input_ids=torch.tensor([flat], device=device)).logits[:, -1].float()
        diff = (l_hybrid - l_plain).abs().max().item()
        print(f"[1] logits max|diff| hybrid vs plain: {diff:.2e} (tol {tol:g}) -> {'PASS' if diff < tol else 'FAIL'}")
        ok &= diff < tol

        gen = model.generate(left, [], [], max_new_tokens=8, stop_newline=False)["text"]
        ref_ids = lm.generate(input_ids=torch.tensor([left], device=device), max_new_tokens=8, do_sample=False,
                              pad_token_id=tok.eos_token_id or 0)[0, len(left):]
        ref = tok.decode(ref_ids, skip_special_tokens=True)
        same = gen.strip() == ref.split("\n")[0].strip() or gen.strip() == ref.strip()
        print(f"[2] greedy generation custom loop vs HF generate: {'PASS' if same else 'FAIL'}\n    custom={gen!r}\n    hf    ={ref!r}")
        ok &= same

    big = [list(range(10, 10 + 40)), list(range(60, 60 + 12))]
    big_flags = [True, True]
    si = assemble(model.embed, model.compressor, big, big_flags, left[:5], None, model.k, model.dtype)
    pos = si.position_ids[0].tolist()
    mono = all(b >= a_ for a_, b in zip(pos, pos[1:]))
    exp0 = interpolated_positions(0.0, 40, model.k)
    good = si.ctx_tokens_after == 2 * model.k and mono and abs(pos[0] - exp0[0]) < 1e-6 and abs(pos[model.k - 1] - exp0[-1]) < 1e-6
    print(f"[3] compress path: {si.n_compressed_blocks} blocks -> {si.ctx_tokens_after} vectors, positions monotone={mono}, "
          f"first block positions {pos[:model.k]} -> {'PASS' if good else 'FAIL'}")
    ok &= good
    print("SANITY CHECK", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
