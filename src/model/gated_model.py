"""GatedCompressionModel: frozen Qwen2.5-Coder + LoRA + gate + cross-attention compressor + SplitCache.

HF-style entry point:
    model = GatedCompressionModel.from_pretrained("Qwen/Qwen2.5-Coder-1.5B",
                gate_ckpt="checkpoints/gate.pt", compressor_ckpt="checkpoints/compressor.pt",
                lora_dir="checkpoints/lora")
    out = model.generate(left_ids, chunks, flags)            # explicit routing
    out = model.complete(example, tau=0.5)                   # gate-routed (uses/computes gate scores)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.split_cache import SplitInputs, assemble
from src.compressor.cross_attn import CrossAttentionCompressor
from src.gate.features import example_features
from src.gate.model import ConfidenceGate
from src.utils import load_config, torch_dtype


class GatedCompressionModel(nn.Module):
    def __init__(self, lm, tokenizer, compressor: CrossAttentionCompressor, k: int, peft_model=None,
                 gate: ConfidenceGate | None = None, max_chunks: int = 5):
        super().__init__()
        self.lm = lm                      # the frozen Qwen2ForCausalLM (LoRA layers injected in place)
        self.compressor = compressor
        self.tokenizer = tokenizer
        self.k = k
        self.gate = gate
        self.max_chunks = max_chunks
        self._peft = [peft_model]         # list: keeps the PeftModel wrapper out of the module registry

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, base_model_id: str, gate_ckpt: str | None = None, compressor_ckpt: str | None = None,
                        lora_dir: str | None = None, cfg: dict | None = None, device=None,
                        trainable_lora: bool = False) -> "GatedCompressionModel":
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = cfg or load_config("model")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch_dtype(cfg["dtype"]) if device.type == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(base_model_id)
        import transformers

        dkw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        lm = AutoModelForCausalLM.from_pretrained(base_model_id, attn_implementation=cfg.get("attn_implementation", "sdpa"),
                                                  **{dkw: dtype})
        if lm.dtype != dtype:
            lm = lm.to(dtype)
        for p in lm.parameters():
            p.requires_grad = False
        if lora_dir and Path(lora_dir).exists():
            peft_model = PeftModel.from_pretrained(lm, lora_dir, is_trainable=trainable_lora)
        else:
            lc = cfg["lora"]
            peft_model = get_peft_model(lm, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                                       target_modules=lc["target_modules"], bias="none",
                                                       task_type="CAUSAL_LM"))
        for p in lm.parameters():         # trainable (LoRA) params in fp32 so GradScaler works with fp16 base
            if p.requires_grad:
                p.data = p.data.float()
        emb = lm.get_input_embeddings().weight
        comp = CrossAttentionCompressor(emb.shape[1], cfg["k"], cfg["compressor"]["n_heads"],
                                        cfg["compressor"]["max_block_len"], embed_std=float(emb.float().std()))
        if compressor_ckpt and Path(compressor_ckpt).exists():
            ck = torch.load(compressor_ckpt, map_location="cpu", weights_only=False)
            comp.load_state_dict(ck["state_dict"])
        gate = ConfidenceGate.load(gate_ckpt, "cpu") if gate_ckpt and Path(gate_ckpt).exists() else None
        m = cls(lm, tok, comp, cfg["k"], peft_model, gate)
        m.to(device)
        m.compressor.float()
        m.eval()
        return m

    def save(self, compressor_path: str, lora_dir: str) -> None:
        Path(compressor_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.compressor.state_dict(), "k": self.k}, compressor_path)
        self._peft[0].save_pretrained(lora_dir)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    def embed(self, ids: list[int]) -> torch.Tensor:
        return self.lm.get_input_embeddings()(torch.as_tensor(ids, dtype=torch.long, device=self.device))

    def gate_scores(self, example: dict, variant: str = "rg1") -> list[float]:
        key = f"gate_{variant}"
        if key in example:
            return example[key]
        chunks = example[f"chunks_{variant}"]
        if self.gate is None:
            raise ValueError("no precomputed gate scores and no gate loaded")
        return [float(x) for x in self.gate.score(example_features(chunks, example["left_text"], self.max_chunks))]

    def _assemble(self, left_ids, chunks_ids_best_first, flags_best_first, target_ids) -> SplitInputs:
        # placement order: least relevant first, best chunk closest to the live code
        ids = list(reversed(chunks_ids_best_first))
        flags = list(reversed(flags_best_first))
        return assemble(self.embed, self.compressor, ids, flags, left_ids, target_ids, self.k, self.dtype)

    def _hidden(self, si: SplitInputs, past=None, use_cache=False):
        return self.lm.model(inputs_embeds=si.inputs_embeds, position_ids=si.position_ids,
                             attention_mask=si.attention_mask, past_key_values=past, use_cache=use_cache)

    # ------------------------------------------------------------------ training
    def hybrid_loss(self, spec: dict) -> torch.Tensor:
        """spec: left_ids, target_ids, chunks (list of id lists, best first), compress (list[bool], best first)."""
        si = self._assemble(spec["left_ids"], spec["chunks"], spec["compress"], spec["target_ids"])
        h = self._hidden(si).last_hidden_state[0]
        nt = len(spec["target_ids"])
        sel = h[si.n_prefix - 1: si.n_prefix - 1 + nt]
        logits = self.lm.lm_head(sel).float()
        tgt = torch.as_tensor(spec["target_ids"], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, tgt)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, left_ids: list[int], chunks_ids_best_first: list[list[int]], flags_best_first: list[bool],
                 max_new_tokens: int = 48, stop_newline: bool = True) -> dict:
        from transformers import DynamicCache

        t0 = time.perf_counter()
        si = self._assemble(left_ids, chunks_ids_best_first, flags_best_first, None)
        cache = DynamicCache()
        out = self._hidden(si, cache, True)
        logits = self.lm.lm_head(out.last_hidden_state[:, -1])
        eos = self.tokenizer.eos_token_id
        new: list[int] = []
        pos, total = si.next_position, si.inputs_embeds.shape[1]
        for _ in range(max_new_tokens):
            nxt = int(logits.argmax(-1))
            if eos is not None and nxt == eos:
                break
            new.append(nxt)
            if stop_newline and "\n" in self.tokenizer.decode(new):
                break
            total += 1
            step = SplitInputs(self.embed([nxt])[None].to(self.dtype),
                               torch.tensor([[pos]], dtype=torch.float32, device=self.device),
                               torch.ones(1, total, dtype=torch.long, device=self.device),
                               0, pos, 0, 0, 0)
            out = self._hidden(step, cache, True)
            logits = self.lm.lm_head(out.last_hidden_state[:, -1])
            pos += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        text = self.tokenizer.decode(new)
        return {"text": text.split("\n")[0], "new_tokens": len(new), "latency_s": time.perf_counter() - t0,
                "ctx_before": si.ctx_tokens_before, "ctx_after": si.ctx_tokens_after,
                "seq_len": int(si.inputs_embeds.shape[1]), "left_tokens": si.left_tokens,
                "n_compressed": si.n_compressed_blocks, "n_exact": si.n_exact_blocks,
                "block_kinds": list(reversed(si.block_kinds))}  # back to best-first order

    def complete(self, example: dict, tau: float = 0.5, variant: str = "rg1", max_new_tokens: int = 48) -> dict:
        chunks = example[f"chunks_{variant}"]
        scores = self.gate_scores(example, variant)
        flags = [s < tau for s in scores]
        return self.generate(example["left_ids"], [c["ids"] for c in chunks], flags, max_new_tokens)
