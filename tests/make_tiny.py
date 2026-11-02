"""Build a tiny random Qwen2 + byte-level tokenizer + synthetic CrossCodeEval-schema data for smoke tests.
No downloads needed; runs on CPU in seconds."""
from __future__ import annotations

import json
import random
from pathlib import Path


def make_tiny_model(out: Path) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out.mkdir(parents=True, exist_ok=True)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<|endoftext|>"])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>")
    fast.save_pretrained(out)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=fast.eos_token_id, bos_token_id=None)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return out


def make_fake_cceval(raw_dir: Path, n_repos: int = 14, per_repo: int = 6, seed: int = 0) -> None:
    rng = random.Random(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rg1, oracle, nocx = [], [], []
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "lambda", "zeta"]
    for r in range(n_repos):
        repo = f"user-repo{r}-abc{r:03d}"
        for j in range(per_repo):
            fn = f"{rng.choice(words)}_{rng.randrange(100)}"
            arg = f"{rng.choice(words)}_{rng.randrange(100)}"
            prompt = "".join(f"import {rng.choice(words)}\n" for _ in range(rng.randrange(2, 12))) + \
                     f"\ndef main():\n    {arg} = 3\n    out = "
            gt = f"{fn}({arg}, mode='fast')"
            body = f"def {fn}(x, mode='slow'):\n    return x * 2\n"
            filler = lambda: "\n".join(f"{rng.choice(words)}_{rng.randrange(9)} = {rng.randrange(99)}" for _ in range(6))
            tid = f"project_cc_python/{r * per_repo + j}"
            md = {"task_id": tid, "repository": repo, "file": "main.py", "context_start_lineno": 0,
                  "groundtruth_start_lineno": 10, "right_context_start_lineno": 11}

            def cc(chunks):
                chunks = [{"retrieved_chunk": c, "filename": f"mod{i}.py", "score": round(0.9 - 0.1 * i, 3)}
                          for i, c in enumerate(chunks)]
                return {"text": "# Here are some relevant code fragments\n" + "\n".join(c["retrieved_chunk"] for c in chunks),
                        "list": chunks}
            rg_chunks = [filler(), body, filler(), filler(), filler()]
            rng.shuffle(rg_chunks)
            base = {"prompt": prompt, "groundtruth": gt, "right_context": "\n    return out\n", "metadata": md}
            rg1.append({**base, "crossfile_context": cc(rg_chunks)})
            oracle.append({**base, "crossfile_context": cc([body, filler(), filler(), filler(), filler()])})
            nocx.append(base)
    rng.shuffle(oracle)  # real files list the same tasks in a different order
    for name, rows in (("line_completion_rg1_unixcoder_cosine_sim.jsonl", rg1),
                       ("line_completion_oracle_unixcoder_cosine_sim.jsonl", oracle), ("line_completion.jsonl", nocx)):
        with open(raw_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
