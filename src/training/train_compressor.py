"""Train the cross-attention compressor (+ LoRA) with next-token prediction on hybrid sequences.

    python -m src.training.train_compressor [--resume] [--max_steps N]

* base LLM weights frozen; only the compressor and LoRA adapters are updated
* fp16 autocast + GradScaler (T4 has no bf16), gradient checkpointing, batch size 1 x grad_accum
* checkpoints every `save_every` steps (Kaggle sessions can die) -> --resume continues
* validation loss at fixed tau every `eval_every` steps, best checkpoint kept, early stopping
* CSV log at logs/train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from src.model.gated_model import GatedCompressionModel
from src.training.hybrid_dataset import HybridDataset
from src.utils import get_device, load_config, read_jsonl, resolve, seed_everything


@torch.no_grad()
def validate(model, ds, amp, device):
    model.eval()
    tot, n = 0.0, 0
    for i in range(len(ds)):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = model.hybrid_loss(ds[i])
        tot += float(loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    a = ap.parse_args()

    cfg = load_config("data", "model", "training")
    seed_everything(cfg["seed"])
    device = get_device()
    amp = bool(cfg["amp"]) and device.type == "cuda"
    pdir = resolve(cfg["processed_dir"])
    train_rows = read_jsonl(pdir / "train.jsonl")
    val_rows = read_jsonl(pdir / "val.jsonl")[: cfg["val_examples"]]
    if "gate_rg1" not in train_rows[0]:
        raise SystemExit("gate scores missing - run `python -m src.gate.train_gate` first")
    train_ds = HybridDataset(train_rows, tuple(cfg["train_tau_range"]), cfg["p_all_compress"], seed=cfg["seed"])
    val_ds = HybridDataset(val_rows, fixed_tau=cfg["val_tau"])

    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    model = GatedCompressionModel.from_pretrained(cfg["base_model"], cfg=cfg, device=device, trainable_lora=True)
    if cfg["grad_checkpointing"]:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    comp_params = [p for n, p in named.items() if n.startswith("compressor.")]
    lora_params = [p for n, p in named.items() if not n.startswith("compressor.")]
    print(f"trainable: compressor {sum(p.numel() for p in comp_params)/1e6:.2f}M, LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M")
    opt = torch.optim.AdamW([{"params": comp_params, "lr": cfg["lr_compressor"]},
                             {"params": lora_params, "lr": cfg["lr_lora"]}], weight_decay=cfg["weight_decay"])
    accum = cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train_ds) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_scale(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        prog = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best_val, bad = 0, float("inf"), 0
    if a.resume and paths["train_state"].exists():
        st = torch.load(paths["train_state"], map_location="cpu", weights_only=False)
        for n, p in named.items():
            p.data.copy_(st["params"][n].to(p.device, p.dtype))
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best_val, bad = st["step"], st["best_val"], st["bad"]
        print(f"resumed from step {step}")

    log_path = resolve(cfg["paths"]["log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not (a.resume and log_path.exists())
    logf = open(log_path, "a" if not new_log else "w", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["step", "train_loss", "val_loss", "lr_comp", "lr_lora", "seconds", "peak_vram_gb"])

    def save_best():
        model.save(str(paths["compressor_ckpt"]), str(paths["lora_dir"]))

    def save_state():
        paths["train_state"].parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": {n: p.detach().cpu() for n, p in named.items()}, "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_val": best_val, "bad": bad}, paths["train_state"])

    if step == 0:
        v0 = validate(model, val_ds, amp, device)
        print(f"step 0 | val loss (untrained compressor) {v0:.4f}")
        writer.writerow([0, "", f"{v0:.5f}", 0, 0, 0, 0])
        logf.flush()
        best_val = v0
        save_best()  # a valid (untrained) checkpoint always exists

    t0, run_loss, run_n = time.time(), 0.0, 0
    epoch = step // steps_per_epoch if steps_per_epoch else 0
    stop = False
    while step < total_steps and not stop:
        train_ds.set_epoch(epoch)
        order = list(range(len(train_ds)))
        import random as _r
        _r.Random(cfg["seed"] + epoch).shuffle(order)
        start = (step % steps_per_epoch) * accum
        for micro_start in range(start, len(order), accum):
            for idx in order[micro_start: micro_start + accum]:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = model.hybrid_loss(train_ds[idx]) / accum
                scaler.scale(loss).backward()
                run_loss += float(loss.detach()) * accum
                run_n += 1
            scale = lr_scale(step)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * scale
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(named.values()), cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            val = ""
            if step % cfg["log_every"] == 0 or step == total_steps:
                peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                print(f"step {step}/{total_steps} | train loss {run_loss/max(run_n,1):.4f} | {time.time()-t0:.0f}s | peak {peak:.1f} GB")
            if step % cfg["eval_every"] == 0 or step == total_steps:
                v = validate(model, val_ds, amp, device)
                val = f"{v:.5f}"
                print(f"  val loss {v:.4f} (best {best_val:.4f})")
                if v < best_val - 1e-4:
                    best_val, bad = v, 0
                    save_best()
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        print("early stopping")
                        stop = True
            if step % cfg["log_every"] == 0 or val or step == total_steps:
                writer.writerow([step, f"{run_loss/max(run_n,1):.5f}", val, f"{opt.param_groups[0]['lr']:.2e}",
                                 f"{opt.param_groups[1]['lr']:.2e}", f"{time.time()-t0:.0f}",
                                 f"{torch.cuda.max_memory_allocated()/2**30:.2f}" if device.type == "cuda" else 0])
                logf.flush()
                run_loss, run_n = 0.0, 0
            if step % cfg["save_every"] == 0:
                save_state()
            if step >= total_steps or stop:
                break
        epoch += 1
    save_state()
    logf.close()
    print(f"done. best val loss {best_val:.4f}. checkpoints: {paths['compressor_ckpt']}, {paths['lora_dir']}")


if __name__ == "__main__":
    main()
