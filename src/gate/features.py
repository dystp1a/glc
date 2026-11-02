"""Static-analysis features for the confidence gate (one vector per retrieved chunk).

Everything here is available at inference time: the chunk text, the retrieval score/rank, and the
tail of the current file. The ground truth is NEVER used for features (only for labels, see labels.py).
Tree-sitter provides the structural counts; identifier overlap with the last ~20 lines of the live
file is the key signal that lets the gate predict something real (is this chunk's content already
visible / about to be used?).
"""
from __future__ import annotations

import keyword
import math
import re
from functools import lru_cache

import numpy as np

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
STOP_IDENTS = set(keyword.kwlist) | {"self", "cls", "True", "False", "None", "print", "len", "range", "str", "int",
                                     "float", "list", "dict", "set", "tuple", "bool", "type", "isinstance",
                                     "super", "object", "open", "min", "max", "sum", "any", "all", "zip", "map",
                                     "enumerate", "sorted", "reversed", "getattr", "setattr", "hasattr", "Exception"}

FEATURE_NAMES = [
    "log_tokens", "n_lines", "comment_ratio",
    "n_type_annotations", "n_str_literals", "n_num_literals", "n_const_literals",
    "n_unique_idents", "ident_unique_ratio",
    "ast_depth", "n_funcdef", "n_classdef", "n_import", "n_call", "n_decorator", "n_parse_errors",
    "overlap_left_frac_chunk", "overlap_left_jaccard", "overlap_left_frac_left", "n_defs_used_in_left",
    "retrieval_score", "retrieval_rank_norm",
    "file_imported_in_left", "overlap_last3_frac_left", "n_defs_used_in_last3", "n_defs_used_in_full_left",
]
N_FEATURES = len(FEATURE_NAMES)


def identifiers(text: str) -> set[str]:
    return {t for t in IDENT_RE.findall(text) if t not in STOP_IDENTS and len(t) > 1}


@lru_cache(maxsize=1)
def _parser():
    try:
        import tree_sitter_python as tsp
        from tree_sitter import Language, Parser

        lang = Language(tsp.language())
        try:
            return Parser(lang)
        except TypeError:  # older py-tree-sitter API
            p = Parser()
            p.set_language(lang)
            return p
    except Exception:  # pragma: no cover - tree-sitter missing: structural counts fall back to regex
        return None


def ast_stats(text: str) -> dict[str, float]:
    """Counts of node types and max depth. Falls back to regex approximations without tree-sitter."""
    parser = _parser()
    s = dict(type=0, str=0, num=0, const=0, func=0, cls=0, imp=0, call=0, deco=0, err=0, depth=0)
    if parser is None:
        s["func"] = len(re.findall(r"^\s*(?:async\s+)?def\s", text, re.M))
        s["cls"] = len(re.findall(r"^\s*class\s", text, re.M))
        s["imp"] = len(re.findall(r"^\s*(?:import|from)\s", text, re.M))
        s["call"] = len(re.findall(r"[A-Za-z_]\w*\s*\(", text))
        s["deco"] = len(re.findall(r"^\s*@", text, re.M))
        s["str"] = len(re.findall(r"\"[^\"\n]*\"|'[^'\n]*'", text))
        s["num"] = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        s["const"] = len(re.findall(r"\b(?:True|False|None)\b", text))
        s["depth"] = max((len(l) - len(l.lstrip())) // 4 for l in text.splitlines()) if text.strip() else 0
        return s
    tree = parser.parse(text.encode("utf8", errors="ignore"))
    stack = [(tree.root_node, 1)]
    while stack:
        node, d = stack.pop()
        s["depth"] = max(s["depth"], d)
        t = node.type
        if t == "type":
            s["type"] += 1
        elif t == "string":
            s["str"] += 1
        elif t in ("integer", "float"):
            s["num"] += 1
        elif t in ("true", "false", "none"):
            s["const"] += 1
        elif t == "function_definition":
            s["func"] += 1
        elif t == "class_definition":
            s["cls"] += 1
        elif t in ("import_statement", "import_from_statement"):
            s["imp"] += 1
        elif t == "call":
            s["call"] += 1
        elif t == "decorator":
            s["deco"] += 1
        elif t == "ERROR":
            s["err"] += 1
        for c in node.children:
            stack.append((c, d + 1))
    return s


def last_lines(text: str, n: int = 20) -> str:
    return "\n".join(text.splitlines()[-n:])


def chunk_features(chunk_text: str, left_text: str, n_tokens: int, score: float, rank: int,
                   max_chunks: int = 5, left_lines: int = 20, filename: str = "") -> np.ndarray:
    tail = last_lines(left_text, left_lines)
    last3 = last_lines(left_text, 3)
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    imported = float(bool(stem) and re.search(r"(?:import|from)[^\n]*\b" + re.escape(stem) + r"\b", left_text) is not None)
    ids_last3 = identifiers(last3)
    ids_c, ids_l = identifiers(chunk_text), identifiers(tail)
    all_ids = [t for t in IDENT_RE.findall(chunk_text) if t not in STOP_IDENTS and len(t) > 1]
    lines = chunk_text.splitlines() or [""]
    st = ast_stats(chunk_text)
    inter = len(ids_c & ids_l)
    defs = set(DEF_RE.findall(chunk_text))
    f = [
        math.log1p(n_tokens), float(len(lines)), sum(1 for l in lines if l.lstrip().startswith("#")) / len(lines),
        st["type"], st["str"], st["num"], st["const"],
        float(len(ids_c)), len(ids_c) / max(1, len(all_ids)),
        st["depth"], st["func"], st["cls"], st["imp"], st["call"], st["deco"], st["err"],
        inter / max(1, len(ids_c)), inter / max(1, len(ids_c | ids_l)), inter / max(1, len(ids_l)),
        float(len(defs & ids_l)),
        float(score), rank / max(1, max_chunks - 1),
        imported, len(ids_c & ids_last3) / max(1, len(ids_last3)), float(len(defs & ids_last3)),
        float(len(defs & identifiers(left_text))),
    ]
    assert len(f) == N_FEATURES
    return np.asarray(f, dtype=np.float32)


def example_features(chunks: list[dict], left_text: str, max_chunks: int = 5) -> np.ndarray:
    """(n_chunks, N_FEATURES) for an example's chunks (given best-first, as stored by prepare_cceval)."""
    if not chunks:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.stack([chunk_features(c["text"], left_text, len(c["ids"]), c.get("score", 0.0), i, max_chunks,
                                    filename=c.get("filename", ""))
                     for i, c in enumerate(chunks)])
