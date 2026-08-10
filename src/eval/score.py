#!/usr/bin/env python3
"""
KATHE 2026 — score a hypothesis file against a reference file.

Replicates the official scorer exactly by calling the one shared implementation
in `data.normalize.score`. Nothing about the metric is reimplemented here; this
module only loads files, reports, and adds the length diagnostics that R6 needs.

Always name the dev set when reporting a number (PROJECT_NOTES.md §6). Tokenizer is
13a, always.

Usage:
    uv run python -m eval.score \\
        --hyp data/dev/r0/r0.hyp.zeroshot-1b \\
        --ref data/dev/r0/r0.kas_Arab \\
        --name "R0 register-matched" --src data/dev/r0/r0.eng_Latn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import NormConfig, normalize, score  # noqa: E402


def load(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--src", type=Path, help="source file, for length-ratio diagnostics")
    ap.add_argument("--name", default="", help="dev set name — always report it")
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    args = ap.parse_args()

    hyps, refs = load(args.hyp), load(args.ref)
    if len(hyps) != len(refs):
        print(f"FATAL: {len(hyps)} hypotheses vs {len(refs)} references. Scoring is "
              f"positional; a length mismatch means the files are not aligned.")
        return 1

    res = score(hyps, refs)

    # Length diagnostics are measured on the normalized text, since that is what
    # the metric sees. R6 tunes toward the test set's ~39 chars / 7.3 words.
    cfg = NormConfig(scorer_normalizer=True)
    nh = [normalize(h, cfg) for h in hyps]
    nr = [normalize(r, cfg) for r in refs]
    hyp_chars = sum(len(x) for x in nh) / max(1, len(nh))
    ref_chars = sum(len(x) for x in nr) / max(1, len(nr))

    label = args.name or f"{args.hyp.name} vs {args.ref.name}"
    print(f"\n=== {label} ===")
    print(f"  lines            {len(hyps):,}")
    print(f"  BLEU (13a)       {res['bleu']:.2f}")
    print(f"  chrF++           {res['chrf_plus_plus']:.2f}")
    print(f"  GEOMETRIC MEAN   {res['geometric_mean']:.2f}")
    print(f"\n  mean chars/line  hyp {hyp_chars:.1f}   ref {ref_chars:.1f}   "
          f"ratio {hyp_chars / max(1e-9, ref_chars):.3f}")
    if args.src:
        src = load(args.src)
        sw = sum(len(s.split()) for s in src) / max(1, len(src))
        print(f"  mean src words   {sw:.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "name": label, "lines": len(hyps), **res,
            "hyp_chars_per_line": round(hyp_chars, 2),
            "ref_chars_per_line": round(ref_chars, 2),
            "hyp_ref_char_ratio": round(hyp_chars / max(1e-9, ref_chars), 4),
            "hyp_file": str(args.hyp), "ref_file": str(args.ref),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
