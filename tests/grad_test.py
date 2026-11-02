"""Gradient-flow + learnability test on the tiny model.
1. loss.backward() must give non-zero grads to the compressor and to LoRA-B (and none to frozen weights)
2. the compressor must actually influence the logits
3. overfit test: the answer exists ONLY inside a compressed chunk -> training must beat the no-context loss
"""
import sys, tempfile, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import torch
from make_tiny import make_tiny_model
from src.model.gated_model import GatedCompressionModel
from src.utils import load_config

cfg = load_config("model")
tmp = Path(tempfile.mkdtemp())
md = make_tiny_model(tmp / "m")
cfg.update(dtype="float32", compressor={"n_heads": 4, "max_block_len": 96},
           lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]})
torch.manual_seed(0)
m = GatedCompressionModel.from_pretrained(str(md), cfg=cfg, device=torch.device("cpu"), trainable_lora=True)
tok = m.tokenizer
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

spec = {"left_ids": enc("x = 1\nout = "), "target_ids": enc("alpha_7(x)\n"),
        "chunks": [enc("# a.py\ndef alpha_7(x):\n    return x\n\n"), enc("# b.py\nfoo = 3\n")], "compress": [True, True]}
m.train()
loss = m.hybrid_loss(spec); loss.backward()
g = {n: p.grad.abs().sum().item() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
comp = sum(v for n, v in g.items() if n.startswith("compressor."))
loraB = sum(v for n, v in g.items() if "lora_B" in n)
frozen = [n for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None]
print(f"[1] grad sum compressor={comp:.3e} lora_B={loraB:.3e} frozen-with-grad={len(frozen)}")
assert comp > 0 and loraB > 0 and not frozen
m.zero_grad()

m.eval()
with torch.no_grad():
    l1 = m.hybrid_loss(spec).item()
    with torch.no_grad():
        m.compressor.queries.add_(torch.randn_like(m.compressor.queries))
    l2 = m.hybrid_loss(spec).item()
print(f"[2] loss before/after perturbing compressor queries: {l1:.4f} -> {l2:.4f}")
assert abs(l1 - l2) > 1e-6

# [3] overfit: answer only inside the compressed chunk
m2 = GatedCompressionModel.from_pretrained(str(md), cfg=cfg, device=torch.device("cpu"), trainable_lora=True)
rng = random.Random(0)
names = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
data = []
for i in range(8):
    fn = f"{names[i]}_{i}"
    data.append({"left_ids": enc("v = 1\nout = "), "target_ids": enc(f"{fn}(v)\n"),
                 "chunks": [enc(f"# m.py\ndef {fn}(x):\n    return x\n\n")], "compress": [True]})
def avg(model, flag):
    model.eval()
    with torch.no_grad():
        return sum(model.hybrid_loss({**d, "compress": [True] if flag else [], "chunks": d["chunks"] if flag else []}).item() for d in data) / len(data)
before_ctx, before_noctx = avg(m2, True), avg(m2, False)
m2.train()
opt = torch.optim.AdamW(m2.trainable_parameters(), lr=3e-3)
for step in range(800):
    opt.zero_grad()
    loss = sum(m2.hybrid_loss(d) for d in data) / len(data)
    loss.backward(); opt.step()
after_ctx = avg(m2, True)
# no-context loss of the SAME trained model: can it memorise the answers without the chunk?
nc = avg(m2, False)
print(f"[3] loss with compressed chunk: {before_ctx:.3f} -> {after_ctx:.3f} | same model WITHOUT the chunk: {nc:.3f}")
assert after_ctx < before_ctx - 1.0 and after_ctx < nc - 0.5, 'compressed chunk should carry information the model uses'
print("GRAD TEST PASSED")
