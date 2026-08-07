#!/usr/bin/env python3
"""
KATHE 2026 — dev-set leakage check.

BPCC is partly web-mined and FLORES is on the web, so FLORES sentences DO end
up in mined corpora. Any leaked pair makes the dev score optimistic and the
leaderboard correlation meaningless. PROJECT_NOTES.md §2.3 makes this mandatory before
every training run.

Three detectors, cheapest first:
  1. exact match, on the scorer-normalized text
  2. near-duplicate, on an aggressively normalized key (diacritics and
     punctuation stripped, whitespace collapsed) -- catches the same sentence
     re-typed with a different orthographic convention
  3. near-duplicate, character 5-gram Jaccard >= threshold, blocked by shared
     rare n-grams so it stays O(n) rather than O(n*m)

Matching is done on BOTH sides independently. A pair whose Kashmiri target
matches a FLORES reference is leaked even if the English differs (and vice
versa) -- mined data often pairs a FLORES sentence with a different translation.

Usage:
    uv run python -m data.leakage \\
        --pairs data/processed/bpcc_kas_filtered.jsonl \\
        --dev-kas data/dev/flores_kas.txt \\
        --dev-en  data/dev/flores_en.txt \\
        --output-clean data/processed/bpcc_kas_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import NormConfig, normalize  # noqa: E402

# Scorer-normalized only: this is the space the competition actually compares in.
_SCORER = NormConfig(scorer_normalizer=True)

_PUNCT_WS = re.compile(r"[^\w]+", re.UNICODE)


def scorer_key(text: str) -> str:
    return normalize(text, _SCORER)


def loose_key(text: str) -> str:
    """
    Aggressive key for near-duplicate detection: strip every combining mark
    (all Kashmiri vowel diacritics are Mn), drop punctuation, collapse space,
    casefold. Two sentences that differ only in orthographic convention collide.
    """
    s = normalize(text, _SCORER)
    s = ud.normalize("NFD", s)
    s = "".join(c for c in s if not ud.combining(c))
    s = ud.normalize("NFC", s)
    s = _PUNCT_WS.sub(" ", s)
    return " ".join(s.split()).casefold()


def ngrams(s: str, n: int = 5) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


class NearDupIndex:
    """
    Inverted index over character 5-grams. For each query we only compare
    against dev sentences sharing at least one n-gram, which keeps this linear
    in practice instead of |pairs| x |dev|.
    """

    def __init__(self, dev_lines, n=5):
        self.n = n
        self.dev = dev_lines
        self.grams = [ngrams(loose_key(x), n) for x in dev_lines]
        self.index = defaultdict(list)
        for i, gs in enumerate(self.grams):
            for g in gs:
                self.index[g].append(i)

    def best_match(self, text: str, threshold: float):
        q = ngrams(loose_key(text), self.n)
        if not q:
            return None
        candidates = defaultdict(int)
        for g in q:
            for i in self.index.get(g, ()):
                candidates[i] += 1
        best_i, best_j = None, 0.0
        for i, shared in candidates.items():
            # Jaccard upper bound: if even a perfect union can't clear the bar,
            # skip the exact computation.
            if shared / len(q) < threshold:
                continue
            union = len(q | self.grams[i])
            j = shared / union if union else 0.0
            if j > best_j:
                best_i, best_j = i, j
        if best_i is not None and best_j >= threshold:
            return best_i, best_j
        return None


def load_lines(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--dev-kas", required=True, type=Path)
    ap.add_argument("--dev-en", required=True, type=Path)
    ap.add_argument("--output-clean", type=Path,
                    help="write the non-leaked pairs here")
    ap.add_argument("--output-report", type=Path,
                    help="write the leaked pairs + what they matched, as JSONL")
    ap.add_argument("--jaccard", type=float, default=0.80)
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    pairs = read_jsonl(args.pairs)
    dev_kas = load_lines(args.dev_kas)
    dev_en = load_lines(args.dev_en)

    print(f"\n=== LEAKAGE CHECK ===")
    print(f"  pairs      {len(pairs):>9,}  ({args.pairs})")
    print(f"  dev kas    {len(dev_kas):>9,}  ({args.dev_kas})")
    print(f"  dev en     {len(dev_en):>9,}  ({args.dev_en})")
    if len(dev_kas) != len(dev_en):
        print(f"  !! dev files are NOT parallel ({len(dev_kas)} vs {len(dev_en)}). "
              f"Checking both anyway, but verify the files.")

    kas_exact = {scorer_key(x) for x in dev_kas}
    en_exact = {scorer_key(x) for x in dev_en}
    kas_loose = {loose_key(x) for x in dev_kas}
    en_loose = {loose_key(x) for x in dev_en}
    kas_loose.discard("")
    en_loose.discard("")

    print("  building near-dup index ...", file=sys.stderr, flush=True)
    kas_idx = NearDupIndex(dev_kas)
    en_idx = NearDupIndex(dev_en)

    clean, leaked = [], []
    counts = {
        "exact_kas": 0, "exact_en": 0,
        "loose_kas": 0, "loose_en": 0,
        "near_kas": 0, "near_en": 0,
    }

    for p in pairs:
        src, tgt = p["src"], p["tgt"]
        reasons = []

        if scorer_key(tgt) in kas_exact:
            counts["exact_kas"] += 1
            reasons.append(("exact_kas", 1.0, None))
        elif loose_key(tgt) in kas_loose:
            counts["loose_kas"] += 1
            reasons.append(("loose_kas", 1.0, None))
        else:
            m = kas_idx.best_match(tgt, args.jaccard)
            if m:
                counts["near_kas"] += 1
                reasons.append(("near_kas", round(m[1], 3), dev_kas[m[0]]))

        if scorer_key(src) in en_exact:
            counts["exact_en"] += 1
            reasons.append(("exact_en", 1.0, None))
        elif loose_key(src) in en_loose:
            counts["loose_en"] += 1
            reasons.append(("loose_en", 1.0, None))
        else:
            m = en_idx.best_match(src, args.jaccard)
            if m:
                counts["near_en"] += 1
                reasons.append(("near_en", round(m[1], 3), dev_en[m[0]]))

        if reasons:
            leaked.append({**p, "leak_reasons": [
                {"kind": k, "score": s, "dev_line": d} for k, s, d in reasons
            ]})
        else:
            clean.append(p)

    print("\n--- detections (a pair can trip more than one) ---")
    for k, v in counts.items():
        print(f"  {k:12} {v:>9,}")

    pct = 100 * len(leaked) / len(pairs) if pairs else 0.0
    print(f"\n--- OVERLAP ---")
    print(f"  leaked pairs (union)   {len(leaked):>9,}  ({pct:.3f}% of input)")
    print(f"  clean pairs            {len(clean):>9,}")
    uniq_dev = len({r['dev_line'] for p in leaked for r in p['leak_reasons']
                    if r['dev_line']})
    print(f"  distinct dev lines hit by near-dup: {uniq_dev}")

    if leaked:
        print(f"\n--- first {min(args.examples, len(leaked))} leaked examples ---")
        for p in leaked[:args.examples]:
            kinds = ",".join(f"{r['kind']}@{r['score']}" for r in p["leak_reasons"])
            print(f"  [{p.get('config', '?')}] {kinds}")
            print(f"    EN : {p['src'][:110]}")
            print(f"    KAS: {p['tgt'][:110]}")

    if args.output_clean:
        args.output_clean.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_clean, "w", encoding="utf-8") as fh:
            for r in clean:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  wrote {len(clean):,} clean pairs -> {args.output_clean}")

    if args.output_report and leaked:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as fh:
            for r in leaked:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(leaked):,} leaked pairs -> {args.output_report}")


if __name__ == "__main__":
    main()
