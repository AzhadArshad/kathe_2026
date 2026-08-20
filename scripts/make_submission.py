#!/usr/bin/env python3
"""
KATHE 2026 — build a submission CSV from a hypothesis file.

Takes the source CSV (whose row ORDER is authoritative) and a plain-text file
of one hypothesis per line, applies the shared post-processing normalization,
and writes `ID,kashmiri_text`.

Row order is copied from the source, never sorted. The official scorer deletes
the ID column from both frames and zips what remains positionally, so a
correctly-ID'd but reordered file scores near zero while looking fine
(PROJECT_NOTES.md §3).

Post-processing settings come from a YAML config, not from flags, so the file
recorded in `submissions/<n>/` fully determines the output (PROJECT_NOTES.md §6).

Usage:
    python scripts/make_submission.py \
        --source     data/raw/englishdev.csv \
        --hypotheses data/dev/baseline_kas.txt \
        --postproc   config/postproc_zeroshot.yaml \
        --output     submissions/001/submission.csv

Then, always:
    python scripts/validate_submission.py --source data/raw/englishdev.csv \
                                          --submission submissions/001/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.normalize import NormConfig, normalize  # noqa: E402


def load_postproc(path: Path) -> NormConfig:
    """Build a NormConfig from YAML, rejecting unknown keys loudly.

    A typo'd key that silently defaults would change the submitted text without
    changing the recorded config — exactly the kind of drift that makes a
    leaderboard reading unreproducible.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(NormConfig)}
    unknown = set(raw) - known
    if unknown:
        sys.exit(f"FATAL: unknown post-processing keys in {path}: {sorted(unknown)}")
    return NormConfig(**raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="englishdev.csv — defines row order")
    ap.add_argument("--hypotheses", required=True, help="one translation per line, source order")
    ap.add_argument("--postproc", required=True, help="YAML of NormConfig fields")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = pd.read_csv(args.source, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "ID" not in src.columns:
        sys.exit(f"FATAL: {args.source} has no ID column (got {list(src.columns)})")

    # rstrip("\n") only: a trailing blank line is a file-format artifact, but a
    # blank line in the middle is a missing translation and must not be dropped.
    text = Path(args.hypotheses).read_text(encoding="utf-8")
    hyps = text.rstrip("\n").split("\n") if text else []

    if len(hyps) != len(src):
        sys.exit(
            f"FATAL: {len(hyps)} hypotheses vs {len(src)} source rows. "
            "Refusing to write a misaligned submission."
        )

    cfg = load_postproc(Path(args.postproc))
    out = [normalize(h, cfg) for h in hyps]

    blank = [i for i, s in enumerate(out) if not s.strip()]
    if blank:
        sys.exit(
            f"FATAL: {len(blank)} row(s) empty after post-processing (first: {blank[:5]}). "
            "The scorer raises on this and rejects the entire submission."
        )

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": src["ID"], "kashmiri_text": out}).to_csv(
        dest, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )

    print(f"wrote {dest} — {len(out)} rows")
    print(f"post-processing: {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
