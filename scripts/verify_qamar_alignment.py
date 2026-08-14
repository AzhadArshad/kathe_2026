#!/usr/bin/env python3
"""
KATHE 2026 — per-line alignment verification and repair for the SMUQamar 30K.

WHY THIS EXISTS
---------------
The corpus as published is misaligned, in two separate ways, and using it
naively would teach the model to translate each English sentence as the
PREVIOUS sentence's Kashmiri:

  1. English has a blank line at index 12219 that Kashmiri does not, so the two
     sides differ in length (30,000 vs 29,999) and everything after it shifts.
  2. Independently, a band of ~7,200 lines has `K[i]` holding the translation
     of `E[i-1]`. Visible by eye:

         E[15001] "I need a loan"   K[15001] = "I must resist"   (= E[15000])
         E[18002] "Tom worked there" K[18002] = "Tom won't starve" (= E[18001])

HOW IT IS VERIFIED — and why the obvious methods fail
-----------------------------------------------------
* **Sentence-length correlation is NOT reliable here.** It has false positives
  (BPCC `bpcc-seed-latest`, definitely parallel, scores r=0.463) and it is
  confounded by length variance — the misaligned band is short dialogue
  (19 chars mean vs 136 elsewhere), so low correlation there is expected even
  when correct. It flagged the right region for the wrong reason.
* **A raw LaBSE cosine threshold is NOT reliable either**, because LaBSE
  represents Kashmiri poorly (PLANNING.md 2026-08-07); absolute values are low
  everywhere.

What IS reliable is a **relative** comparison: for each line, score `E[i]`
against `K[i-1]`, `K[i]` and `K[i+1]` with the same encoder and take the best.
Encoder weakness, sentence length and vocabulary all cancel, because every arm
uses the same sentences and only the pairing changes.

The regime is then decided by a ROLLING MAJORITY over a window rather than
per-line argmax, because a single short sentence's argmax is noisy. Lines whose
chosen pairing still scores below `--min-cos`, or that disagree with their local
regime, are dropped rather than guessed at.

Calibration: BPCC `daily` (known-parallel, short) gives a true-vs-random cosine
gap of 0.408. The 30K band scored 0.102 before repair and 0.285 after.

Usage:
    uv run python scripts/verify_qamar_alignment.py \\
        --eng ".../HuggingFace 30K/English.txt" \\
        --kas ".../HuggingFace 30K/Kashmiri.txt" \\
        --output data/processed/qamar30k_repaired.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", required=True, type=Path)
    ap.add_argument("--kas", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--window", type=int, default=21,
                    help="rolling majority window for deciding the local regime")
    ap.add_argument("--min-cos", type=float, default=0.15,
                    help="drop a pair whose best pairing still scores below this")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="drop a line whose regime pairing loses to another "
                         "offset by more than this")
    args = ap.parse_args()

    E = args.eng.read_text(encoding="utf-8").splitlines()
    K = args.kas.read_text(encoding="utf-8").splitlines()
    print(f"  raw: {len(E):,} English / {len(K):,} Kashmiri")

    # --- defect 1: blank lines on one side only -----------------------------
    be = [i for i, x in enumerate(E) if not x.strip()]
    bk = [i for i, x in enumerate(K) if not x.strip()]
    if len(E) != len(K):
        if len(E) - len(be) != len(K) - len(bk):
            raise SystemExit("side mismatch not explained by blank lines — do NOT zip")
        E = [x for x in E if x.strip()]
        K = [x for x in K if x.strip()]
        print(f"  dropped blank lines (English {be}, Kashmiri {bk}) -> {len(E):,} each")
    assert len(E) == len(K)

    # --- encode ONCE per side, then shift ------------------------------------
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model, device="cpu")
    print(f"  encoding {2 * len(E):,} sentences with {args.model} (CPU)...", flush=True)
    A = m.encode(E, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    B = m.encode(K, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    n = len(E)
    cos = {}
    for off in (-1, 0, 1):
        c = np.full(n, -1.0, dtype=np.float32)
        lo, hi = max(0, -off), min(n, n - off)
        c[lo:hi] = (A[lo:hi] * B[lo + off:hi + off]).sum(1)
        cos[off] = c

    # --- decide the local regime by rolling majority -------------------------
    per_line = np.stack([cos[-1], cos[0], cos[1]]).argmax(0) - 1
    w = args.window
    regime = np.zeros(n, dtype=int)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        vals, counts = np.unique(per_line[lo:hi], return_counts=True)
        regime[i] = vals[counts.argmax()]

    chosen = np.stack([cos[o][i] for i, o in enumerate(regime)])
    best = np.stack([cos[-1], cos[0], cos[1]]).max(0)
    keep = (chosen >= args.min_cos) & (best - chosen <= args.margin)

    print(f"\n  regime distribution: " + ", ".join(
        f"offset {o:+d}: {int((regime == o).sum()):,}" for o in (-1, 0, 1)))
    print(f"  dropped: {int((~keep).sum()):,} lines "
          f"({int((chosen < args.min_cos).sum()):,} below min-cos, "
          f"{int((best - chosen > args.margin).sum()):,} disagree with their regime)")

    rows = []
    for i in range(n):
        if not keep[i]:
            continue
        j = i + regime[i]
        if 0 <= j < n:
            rows.append({"s": E[i], "t": K[j], "offset": int(regime[i]),
                         "cos": round(float(chosen[i]), 4)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(rows):,} verified pairs -> {args.output}")
    print(f"  mean cosine of kept pairs: {np.mean([r['cos'] for r in rows]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
