#!/usr/bin/env python3
"""
KATHE 2026 — submission validator.

Nothing goes to Kaggle without passing this (PROJECT_NOTES.md §2.1). Submissions are
unlimited, but every check here corresponds to a real failure mode in the
official scorer, and an unlimited budget does not make a rejected file cheap.

Usage:
    python scripts/validate_submission.py --source data/raw/englishdev.csv \
                                          --submission submissions/001/submission.csv

Exit code 0 = safe to submit. Non-zero = do not submit.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata as ud
from pathlib import Path

import pandas as pd

# Resolve src/ from this file, not from the cwd, so the validator works when
# invoked from anywhere. Harmless when the editable install already provides it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.normalize import NormConfig, normalize  # noqa: E402

SCORER_CFG = NormConfig(scorer_normalizer=True)

DEVANAGARI = ("\u0900", "\u097f")
ARABIC = ("\u0600", "\u06ff")

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def load(path: str, what: str) -> pd.DataFrame:
    # keep_default_na=False stops pandas turning literal strings into NaN;
    # dtype=str stops it inferring numeric IDs inconsistently between files.
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not read {what} at {path}: {exc}")
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="englishdev.csv")
    ap.add_argument("--submission", required=True)
    ap.add_argument(
        "--min-diacritic-density",
        type=float,
        default=1.0,
        help="warn below this many diacritics per 100 chars",
    )
    args = ap.parse_args()

    src = load(args.source, "source")
    sub = load(args.submission, "submission")

    # --- structure -------------------------------------------------------
    if list(sub.columns) != ["ID", "kashmiri_text"]:
        err(f"columns must be exactly ['ID', 'kashmiri_text'], got {list(sub.columns)}")
        # everything downstream depends on this
        report()
        return 1

    if len(sub) != len(src):
        err(f"row count {len(sub)} != source row count {len(src)}")

    # --- ID alignment (the scorer deletes IDs and zips positionally) -----
    if "ID" in src.columns and len(sub) == len(src):
        src_ids = src["ID"].tolist()
        sub_ids = sub["ID"].tolist()
        if src_ids != sub_ids:
            if sorted(src_ids) == sorted(sub_ids):
                err(
                    "IDs are present but REORDERED. The scorer aligns rows by "
                    "position, not by ID — this scores near zero while looking fine."
                )
                first = next(i for i, (a, b) in enumerate(zip(src_ids, sub_ids)) if a != b)
                err(f"  first divergence at row {first}: source={src_ids[first]!r} submission={sub_ids[first]!r}")
            else:
                missing = set(src_ids) - set(sub_ids)
                extra = set(sub_ids) - set(src_ids)
                err(f"ID sets differ. missing={len(missing)} extra={len(extra)}")

    # --- emptiness (the scorer raises and rejects the whole file) --------
    raw = sub["kashmiri_text"].tolist()
    normed = [normalize(v, SCORER_CFG) for v in raw]

    empty_rows = [i for i, s in enumerate(normed) if not s.strip()]
    if empty_rows:
        err(
            f"{len(empty_rows)} row(s) empty AFTER normalization — the scorer "
            f"raises on this and rejects the entire submission. "
            f"First at row index {empty_rows[:5]}"
        )

    nan_like = [i for i, v in enumerate(raw) if str(v).strip().lower() in {"", "nan", "none", "null"}]
    if nan_like:
        err(f"{len(nan_like)} row(s) contain NaN-like placeholder text, first at {nan_like[:5]}")

    # --- script sanity ---------------------------------------------------
    joined = "".join(normed)
    if not joined:
        report()
        return 1

    deva = sum(1 for c in joined if DEVANAGARI[0] <= c <= DEVANAGARI[1])
    arab = sum(1 for c in joined if ARABIC[0] <= c <= ARABIC[1])
    latin = sum(1 for c in joined if c.isascii() and c.isalpha())

    if deva:
        err(f"{deva} Devanagari characters present — target script is kas_Arab, this scores ~0")
    if arab / len(joined) < 0.5:
        err(f"only {100 * arab / len(joined):.1f}% of characters are Perso-Arabic — check the output")
    if latin > 0.02 * len(joined):
        warn(f"{100 * latin / len(joined):.1f}% Latin letters — likely untranslated source copied through")

    # --- diacritics (the dominant scoring factor) ------------------------
    from KashmiriNormalizer.constants import KASHMIRI_DIACRITICS

    dia = sum(1 for c in joined if c in KASHMIRI_DIACRITICS)
    density = 100 * dia / len(joined)
    if density < args.min_diacritic_density:
        warn(
            f"diacritic density {density:.2f}/100 chars is low. Stripped diacritics "
            f"score ~13 vs 100 on otherwise-identical text. Verify against reference density."
        )
    else:
        print(f"  diacritic density: {density:.2f} per 100 chars")

    # --- round-trip ------------------------------------------------------
    if joined != ud.normalize("NFC", joined):
        warn("output is not NFC — the scorer applies no Unicode normalization, so composition must match references")

    dupes = len(raw) - len(set(raw))
    if dupes > 0.05 * len(raw):
        warn(f"{dupes} duplicate translations ({100 * dupes / len(raw):.1f}%) — possible degenerate decoding")

    return report()


def report() -> int:
    print()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print()
    if errors:
        print(f"FAILED — {len(errors)} error(s). DO NOT SUBMIT.")
        return 1
    print(f"PASSED{' with ' + str(len(warnings)) + ' warning(s)' if warnings else ''}. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
