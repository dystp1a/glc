"""Gate labels derived from the ground-truth completion (training-time only).

A chunk is labelled EXACT (1) if it contains an identifier that appears in the ground-truth line and that
is NOT already visible in the kept live context: i.e. the chunk holds exact symbol information the model
would otherwise have to guess. Otherwise it is COMPRESS (0): its content is either irrelevant to the
target or already covered by the live file, so a lossy summary is enough.

This is deliberately NOT a function of the gate's input features (the ground truth is unseen at inference),
so the gate is learning a real prediction problem rather than a thresholded formula of its own inputs.
"""
from __future__ import annotations

from src.gate.features import identifiers


def chunk_label(chunk_text: str, groundtruth: str, left_text: str) -> int:
    needed = identifiers(groundtruth) - identifiers(left_text)
    return int(len(needed & identifiers(chunk_text)) > 0)


def example_labels(chunks: list[dict], groundtruth: str, left_text: str) -> list[int]:
    return [chunk_label(c["text"], groundtruth, left_text) for c in chunks]
