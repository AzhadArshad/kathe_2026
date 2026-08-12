#!/usr/bin/env python3
"""
KATHE 2026 — assemble the monolingual Kashmiri text the char restorer trains on.

Training data for this task is free: take any Kashmiri sentence, strip the three
marks, and (stripped, original) is a supervised example. No English side, no
alignment. What is NOT free is provenance discipline, and this module is mostly
that.

Three things it enforces, each for a reason already paid for in this project:

1. **Scorer normalization first.** The restorer runs on IndicTrans2 output,
   which has been through `KashmiriNormalizer`. Training on un-normalized text
   would teach it a character distribution (Arabic yeh, presentation forms,
   Kashmiri digits) that never reaches it at inference.

2. **R0 and the eval slice are excluded by exact stripped-and-normalized
   string.** The lexicon was built from train targets only and still could not
   have leaked R0, because pairs were held out by key. Here the unit is a bare
   sentence from three corpora that were never filtered against each other, so
   the check has to be done on the text itself. `--report-only-leaks` prints
   what was dropped.

3. **Per-source counts are reported and the source tag is kept per line**, so
   the provenance ablation (BPCC-only vs all sources) is a filter over one
   built file rather than a second build.

Sub 005 is the standing warning: a 260k-entry table learned from BPCC n-grams
scored HIGHER on R0 (cut from BPCC) and LOWER on the leaderboard. Three
provenances is the structural answer to that.

Usage:
    uv run python -m restore.build_text \\
        --bpcc     data/processed/r3_corpus/train/eng_Latn-kas_Arab/train.kas_Arab \\
        --external data/processed/external_nawabhussain.kas_Arab \\
        --qamar    "$HOME/.cache/.../HuggingFace 30K/Kashmiri.txt" \\
        --exclude  data/dev/r0/r0.kas_Arab data/processed/r3_corpus/dev/eng_Latn-kas_Arab/dev.kas_Arab \\
        --output   data/processed/restore_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.diacritize import RESTORABLE, strip_key  # noqa: E402
from data.normalize import NormConfig, normalize  # noqa: E402

SCORER_ONLY = NormConfig(scorer_normalizer=True)

# Perso-Arabic block plus the Kashmiri extensions the corpus actually uses.
def _arabic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if "؀" <= c <= "ۿ" or "ﭐ" <= c <= "﻿")
    return arabic / len(letters)


def load_source(path: Path, min_chars: int, min_arabic: float,
                min_restorable: int) -> tuple[list[str], Counter]:
    """Read, scorer-normalize, and drop what cannot teach anything."""
    kept: list[str] = []
    stats: Counter = Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stats["read"] += 1
            text = normalize(line.rstrip("\n"), SCORER_ONLY)
            if len(text) < min_chars:
                stats["drop_short"] += 1
                continue
            if _arabic_share(text) < min_arabic:
                stats["drop_script"] += 1  # Latin/Devanagari leakage
                continue
            if sum(1 for c in text if c in RESTORABLE) < min_restorable:
                stats["drop_undiacritized"] += 1
                continue
            kept.append(text)
            stats["kept"] += 1
    return kept, stats


def profile(lines: list[str]) -> dict:
    chars = sum(len(x) for x in lines)
    marks: Counter = Counter()
    for x in lines:
        for c in x:
            if c in RESTORABLE:
                marks[c] += 1
    return {
        "lines": len(lines),
        "chars": chars,
        "mean_chars": round(chars / max(1, len(lines)), 1),
        "restorable_per_100c": round(100 * sum(marks.values()) / max(1, chars), 2),
        "kasra_per_100c": round(100 * marks["ِ"] / max(1, chars), 2),
        "damma_per_100c": round(100 * marks["ُ"] / max(1, chars), 2),
        "fatha_per_100c": round(100 * marks["َ"] / max(1, chars), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", nargs=2, metavar=("NAME", "PATH"),
                    required=True, help="repeatable: --source bpcc path/to/train.kas_Arab")
    ap.add_argument("--exclude", nargs="*", type=Path, default=[],
                    help="dev references that must NOT appear in training text")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-chars", type=int, default=8)
    ap.add_argument("--min-arabic", type=float, default=0.8)
    ap.add_argument("--min-restorable", type=int, default=0,
                    help="drop lines carrying fewer than N restorable marks. "
                         "DEFAULT 0 (keep everything) — measured 2026-08-12: "
                         "22.1%% of R0's own human references carry no kasra, "
                         "damma or fatha at all, so mark-free lines are part of "
                         "the reference distribution, not corpus noise. Set to 1 "
                         "to train on diacritized lines only and let `none_bias` "
                         "handle the precision/recall trade instead.")
    args = ap.parse_args()

    # Exclusion keys are stripped AND normalized: the restorer's input is the
    # stripped form, so two lines differing only in diacritics are the same
    # example as far as leakage is concerned.
    excluded: set[str] = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                excluded.add(strip_key(normalize(line.rstrip("\n"), SCORER_ONLY)))
    print(f"  exclusion set: {len(excluded):,} stripped dev lines "
          f"from {len(args.exclude)} file(s)")

    seen: set[str] = set()
    rows: list[dict] = []
    per_source: dict[str, dict] = {}
    for name, path in args.source:
        lines, stats = load_source(Path(path), args.min_chars, args.min_arabic,
                                   args.min_restorable)
        leaked = dup = 0
        kept: list[str] = []
        for text in lines:
            key = strip_key(text)
            if key in excluded:
                leaked += 1
                continue
            if key in seen:
                dup += 1  # cross-source duplicate, or a repeat within one file
                continue
            seen.add(key)
            kept.append(text)
            rows.append({"text": text, "source": name})
        per_source[name] = {
            "path": str(path), **dict(stats),
            "dropped_dev_leak": leaked, "dropped_duplicate": dup,
            **profile(kept),
        }
        s = per_source[name]
        print(f"  {name:10} read {s['read']:>7,} -> kept {s['lines']:>7,}  "
              f"(short {s.get('drop_short', 0):,}, script {s.get('drop_script', 0):,}, "
              f"undiacritized {s.get('drop_undiacritized', 0):,}, "
              f"dup {dup:,}, DEV-LEAK {leaked:,})  "
              f"{s['restorable_per_100c']}/100c")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = profile([r["text"] for r in rows])
    manifest = {"per_source": per_source, "total": total,
                "excluded_files": [str(p) for p in args.exclude],
                "min_chars": args.min_chars, "min_arabic": args.min_arabic,
                "min_restorable": args.min_restorable}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  TOTAL {total['lines']:,} lines  {total['chars']:,} chars  "
          f"{total['restorable_per_100c']}/100c "
          f"(kasra {total['kasra_per_100c']}, damma {total['damma_per_100c']}, "
          f"fatha {total['fatha_per_100c']})")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
