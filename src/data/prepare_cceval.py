"""Prepare CrossCodeEval (Python) for the gated latent-compression pipeline.

Input : official jsonl files (rg1 retrieval = realistic, oracle retrieval = reference-peeking upper bound).
        Verified schema per row: prompt, groundtruth, right_context, metadata{task_id, repository, ...},
        crossfile_context{text, list[{retrieved_chunk, filename, score}]}.
Output: data/processed/{train,val,test}.jsonl with cached token ids, so training never re-tokenizes.

Key design choices
  * the three variants list the same tasks in different orders -> joined by metadata.task_id
  * split BY REPOSITORY (no repo in two splits) to avoid leakage
  * each retrieved chunk is truncated to `chunk_tokens`; current file keeps its last `left_tokens` tokens
  * the target is the ground-truth line + newline
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from src.utils import ROOT, load_config, read_jsonl, resolve, write_jsonl


# ----------------------------------------------------------------------------- loading
def _chunks_of(row: dict, max_chunks: int) -> list[dict]:
    cc = row.get("crossfile_context") or {}
    lst = cc.get("list") if isinstance(cc, dict) else None
    if not lst:
        return []
    lst = sorted(lst, key=lambda c: -float(c.get("score", 0.0)))  # best first
    return lst[:max_chunks]


def load_variant(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    out = {}
    for r in rows:
        tid = r["metadata"]["task_id"]
        out[tid] = r
    return out


def inspect(path: Path) -> None:
    rows = read_jsonl(path)
    r = rows[0]
    print(f"{path.name}: {len(rows)} rows, keys={list(r.keys())}")
    print("  metadata:", r["metadata"])
    cc = r.get("crossfile_context")
    if cc:
        print("  crossfile_context keys:", list(cc.keys()), "| n chunks:", len(cc.get("list", [])))
        print("  chunk keys:", list(cc["list"][0].keys()))


# ----------------------------------------------------------------------------- tokenization
def truncate_left(tok, prompt: str, max_tokens: int) -> tuple[list[int], str]:
    """Keep (about) the last `max_tokens` tokens of the current file, starting at a line boundary."""
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return ids, prompt
    text = tok.decode(ids[-max_tokens:])
    nl = text.find("\n")
    if 0 <= nl < 2000:
        text = text[nl + 1:]
    return tok(text, add_special_tokens=False)["input_ids"], text


def encode_chunk(tok, filename: str, body: str, chunk_tokens: int) -> dict:
    body_ids = tok(body, add_special_tokens=False)["input_ids"][:chunk_tokens]
    body_text = tok.decode(body_ids)
    full = f"# {filename}\n{body_text.rstrip()}\n\n"
    return {"filename": filename, "text": body_text,
            "ids": tok(full, add_special_tokens=False)["input_ids"]}


def build_example(tok, row: dict, row_oracle: dict | None, cfg: dict) -> dict:
    md = row["metadata"]
    left_ids, left_text = truncate_left(tok, row["prompt"], cfg["left_tokens"])
    gt = row["groundtruth"]
    target_ids = tok(gt + "\n", add_special_tokens=False)["input_ids"][: cfg["max_target_tokens"]]

    def chunks(r):
        out = []
        for c in _chunks_of(r, cfg["max_chunks"]):
            e = encode_chunk(tok, c["filename"], c["retrieved_chunk"], cfg["chunk_tokens"])
            e["score"] = float(c.get("score", 0.0))
            out.append(e)
        return out

    ex = {
        "task_id": md["task_id"],
        "repo": md["repository"],
        "file": md.get("file", ""),
        "left_ids": left_ids,
        "left_text": left_text,
        "groundtruth": gt,
        "target_ids": target_ids,
        "chunks_rg1": chunks(row),
        "chunks_oracle": chunks(row_oracle) if row_oracle is not None else [],
    }
    ex["total_len_rg1"] = len(left_ids) + len(target_ids) + sum(len(c["ids"]) for c in ex["chunks_rg1"])
    return ex


# ----------------------------------------------------------------------------- splitting
def split_by_repo(examples: list[dict], n_test: int, n_val: int, n_train: int, seed: int):
    """Whole repositories are assigned to a single split. Test first, then val, then train."""
    by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for e in examples:
        by_repo[e["repo"]].append(e)
    repos = sorted(by_repo)
    random.Random(seed).shuffle(repos)
    splits = {"test": [], "val": [], "train": []}
    quota = {"test": n_test, "val": n_val, "train": n_train}
    order = ["test", "val", "train"]
    cur = 0
    for r in repos:
        while cur < len(order) and len(splits[order[cur]]) >= quota[order[cur]]:
            cur += 1
        if cur >= len(order):
            break
        splits[order[cur]].extend(by_repo[r])
    # a repo can push a split slightly past its quota; trim to the quota only for train
    # (test/val keep whole repos; trimming them would not create leakage but keeps things simple)
    splits["train"] = splits["train"][: n_train] if n_train else splits["train"]
    return splits


def check_no_repo_overlap(splits: dict[str, list[dict]]) -> None:
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for e in rows:
            if e["repo"] in seen and seen[e["repo"]] != name:
                raise AssertionError(f"repo {e['repo']} appears in both {seen[e['repo']]} and {name}")
            seen[e["repo"]] = name


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None, help="tokenizer name/path (default: base_model from configs/model.yaml)")
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N examples (smoke tests)")
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_val", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--inspect", action="store_true", help="print the schema of the raw files and exit")
    a = ap.parse_args()

    cfg = load_config("data")
    mcfg = load_config("model")
    raw = resolve(a.raw_dir or cfg["raw_dir"])
    out = resolve(a.out_dir or cfg["processed_dir"])
    rg1_path, oracle_path = raw / cfg["rg1_file"], raw / cfg["oracle_file"]
    if not rg1_path.exists():
        raise SystemExit(f"{rg1_path} not found - run `python scripts/download_data.py` first "
                         f"(and `ls {raw}` to check the file names)")
    if a.inspect:
        inspect(rg1_path)
        if oracle_path.exists():
            inspect(oracle_path)
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or mcfg["base_model"])
    rg1 = load_variant(rg1_path)
    oracle = load_variant(oracle_path) if oracle_path.exists() else {}
    if not oracle:
        print("WARNING: oracle file missing - the extra `oracle_retrieval` condition will be skipped")
    ids = list(rg1)
    if a.limit:
        ids = ids[: a.limit]
    print(f"tokenizing {len(ids)} examples ...")
    try:
        from tqdm import tqdm
        it = tqdm(ids)
    except ImportError:  # pragma: no cover
        it = ids
    examples = [build_example(tok, rg1[t], oracle.get(t), cfg) for t in it]

    sp = cfg["split"]
    splits = split_by_repo(examples, a.n_test or sp["n_test"], a.n_val or sp["n_val"],
                           a.n_train or sp["n_train"], sp["seed"])
    # training-only filter: not too long, at least one retrieved chunk
    before = len(splits["train"])
    splits["train"] = [e for e in splits["train"]
                       if e["total_len_rg1"] <= cfg["train_max_tokens"] and e["chunks_rg1"]]
    print(f"train filter: {before} -> {len(splits['train'])} examples")
    check_no_repo_overlap(splits)

    meta = {"tokenizer": a.tokenizer or mcfg["base_model"], "config": cfg, "splits": {}}
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        n_rep = len({e["repo"] for e in rows})
        ctx = [sum(len(c["ids"]) for c in e["chunks_rg1"]) for e in rows] or [0]
        meta["splits"][name] = {"examples": len(rows), "repos": n_rep,
                                "mean_context_tokens": sum(ctx) / len(ctx)}
        print(f"{name:5s}: {len(rows):5d} examples from {n_rep:4d} repos | mean cross-file context {sum(ctx)/len(ctx):.0f} tokens")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("repo-level split verified: no repository appears in two splits.")


if __name__ == "__main__":
    main()
