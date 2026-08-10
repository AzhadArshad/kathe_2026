#!/usr/bin/env python3
"""
KATHE 2026 — build the register-matched dev sets (R0).

Why this exists
---------------
Every offline decision so far was made against FLORES devtest, which is
encyclopedic Wikipedia prose: 21.6 words per line, 0.8% of lines at eight words
or fewer. The KATHE test set is short everyday sentences — 7.3 words, 74.3% at
eight or fewer, measured directly on `data/raw/englishdev.csv`. They are
different tasks, and FLORES is demoted to a regression check (PLANNING.md, Q3).

This script cuts two held-out sets from the same cleaned BPCC pool the trainer
uses, both length-matched to the test set:

  R0        the primary decision set. Checkpoint selection, beam size, length
            penalty, MBR pool and post-processing are all judged here.
  eval      the in-training eval slice. As previously built it was sampled at
            random and averaged 15.7 words, so it would have selected
            checkpoints optimized for the wrong sentence length.

They are disjoint, so a knob tuned on R0 is not also the thing that picked the
checkpoint.

How the matching works
----------------------
Sources are drawn stratified by English word count, in the proportions
`englishdev.csv` shows over the same 1..--max-words range. That reproduces the
test set's length *shape*, not merely its ceiling: a flat "<=10 words" filter
would have given a 7.95-word mean skewed toward the cap, because the corpus has
7,791 ten-word pairs against 34 one-word pairs.

Two composition rules:

  * `daily` is over-weighted, up to --daily-share of each stratum. It is the
    closest BPCC subcorpus to the test register (10.0 words / 54.9 chars). It
    is capped rather than exhausted because R9 wants to upsample `daily` in
    training, and a dev set cannot eat the lever it is measuring.
  * Mined pairs are excluded outright. These are reference translations; the
    LaBSE >=0.80 filter that admitted them measures LaBSE's weak Kashmiri
    coverage, not pair quality (PLANNING.md, 2026-08-07), and a noisy reference
    costs score on every system equally while telling you nothing.

Selection runs on the output of `build_corpus.load_and_clean`, i.e. exactly the
rows the trainer would otherwise have trained on, so the pair keys written here
are the ones `build_corpus --exclude` removes.

Usage:
    uv run python -m data.build_devsets \\
        --pairs    data/processed/bpcc_kas_clean.jsonl \\
        --test-csv data/raw/englishdev.csv \\
        --out      data/dev \\
        --dev-kas  data/dev/flores_kas.txt \\
        --dev-en   data/dev/flores_en.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_corpus import load_and_clean, pair_key, run_leakage_check, sha256  # noqa: E402

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

# Ordered best-register-first. `daily` is handled separately; this is the
# fill order once its per-stratum cap is reached.
HUMAN_FILL_ORDER = ["nllb-seed", "bpcc-seed-latest", "bpcc-seed-v1", "bpcc-seed-v2"]


def words(s: str) -> int:
    return len(s.split())


def test_length_histogram(csv_path: Path, max_words: int) -> dict[int, float]:
    """Word-count proportions of the real test input, over 1..max_words.

    Read with the csv module rather than pandas: the source column is English,
    but the file is the one artifact we cannot afford to have silently mangled.
    """
    counts: Counter = Counter()
    total_all = 0
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            n = words(str(row["sentence"]))
            total_all += 1
            if 1 <= n <= max_words:
                counts[n] += 1
    kept = sum(counts.values())
    print(f"  test length reference: {csv_path}  ({total_all:,} rows, "
          f"{kept:,} within 1..{max_words} = {100 * kept / total_all:.1f}%)")
    return {n: c / kept for n, c in sorted(counts.items())}


def allocate(hist: dict[int, float], total: int) -> dict[int, int]:
    """Largest-remainder apportionment, so the strata sum to exactly `total`."""
    raw = {n: p * total for n, p in hist.items()}
    base = {n: int(v) for n, v in raw.items()}
    short = total - sum(base.values())
    for n in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:short]:
        base[n] += 1
    return base


def select(
    pool: list[dict],
    want: dict[int, int],
    daily_share: float,
    rng: random.Random,
) -> tuple[list[dict], Counter, list[str]]:
    """Draw `want[n]` pairs of each word count, preferring `daily`."""
    by_len: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in pool:
        by_len[words(r["src"])][r["config"]].append(r)

    picked: list[dict] = []
    taken: Counter = Counter()
    warnings: list[str] = []

    for n in sorted(want):
        need = want[n]
        if need == 0:
            continue
        buckets = by_len.get(n, {})
        chosen: list[dict] = []

        daily = list(buckets.get("daily", ()))
        rng.shuffle(daily)
        chosen.extend(daily[: min(int(round(need * daily_share)), len(daily))])

        for cfg in HUMAN_FILL_ORDER:
            if len(chosen) >= need:
                break
            rest = list(buckets.get(cfg, ()))
            rng.shuffle(rest)
            chosen.extend(rest[: need - len(chosen)])

        # Only if the preferred sources are exhausted do we dip further into
        # `daily`; it keeps the cap meaningful without leaving a stratum short.
        if len(chosen) < need and len(daily) > len(
            [r for r in chosen if r["config"] == "daily"]
        ):
            already = {id(r) for r in chosen}
            chosen.extend(
                [r for r in daily if id(r) not in already][: need - len(chosen)]
            )

        if len(chosen) < need:
            warnings.append(
                f"word count {n}: wanted {need}, pool had only {len(chosen)}"
            )
        picked.extend(chosen)
        taken[n] = len(chosen)
    return picked, taken, warnings


def deal(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Split into two sets with identical length distributions.

    Alternating within each word-count stratum, rather than cutting a shuffled
    list in half, so both sets are length-matched exactly and not just in
    expectation.
    """
    a: list[dict] = []
    b: list[dict] = []
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_len[words(r["src"])].append(r)
    for n in sorted(by_len):
        group = by_len[n]
        rng.shuffle(group)
        a.extend(group[0::2])
        b.extend(group[1::2])
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def describe(name: str, rows: list[dict]) -> dict:
    w = [words(r["src"]) for r in rows]
    chars = [len(r["tgt"]) for r in rows]
    cfg = Counter(r["config"] for r in rows)
    stats = {
        "name": name,
        "pairs": len(rows),
        "mean_src_words": round(sum(w) / max(1, len(w)), 2),
        "pct_src_le_8_words": round(100 * sum(1 for x in w if x <= 8) / max(1, len(w)), 1),
        "max_src_words": max(w) if w else 0,
        "mean_tgt_chars": round(sum(chars) / max(1, len(chars)), 1),
        "config_counts": dict(cfg.most_common()),
    }
    print(f"\n  {name}: {stats['pairs']:,} pairs")
    print(f"    src words   mean {stats['mean_src_words']:>5}   "
          f"<=8 words {stats['pct_src_le_8_words']:>5}%   max {stats['max_src_words']}")
    print(f"    tgt chars   mean {stats['mean_tgt_chars']:>5}")
    print(f"    sources     {stats['config_counts']}")
    return stats


def write_set(out: Path, name: str, rows: list[dict]) -> dict:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    src_p, tgt_p = d / f"{name}.{SRC_LANG}", d / f"{name}.{TGT_LANG}"
    with open(src_p, "w", encoding="utf-8") as sf, open(tgt_p, "w", encoding="utf-8") as tf:
        for r in rows:
            sf.write(r["src"] + "\n")
            tf.write(r["tgt"] + "\n")
    jsonl_p = d / f"{name}.jsonl"
    with open(jsonl_p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "src_file": str(src_p), "tgt_file": str(tgt_p), "pairs_file": str(jsonl_p),
        "src_sha256": sha256(src_p), "tgt_sha256": sha256(tgt_p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--test-csv", required=True, type=Path,
                    help="data/raw/englishdev.csv — the length reference. READ ONLY.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dev-kas", type=Path, help="FLORES kas reference, for the leakage check")
    ap.add_argument("--dev-en", type=Path, help="FLORES en source, for the leakage check")
    ap.add_argument("--jaccard", type=float, default=0.80)
    ap.add_argument("--size", type=int, default=1000, help="pairs per dev set")
    ap.add_argument("--max-words", type=int, default=10)
    ap.add_argument("--daily-share", type=float, default=0.40,
                    help="target share of each stratum drawn from `daily`, capped "
                         "by availability. Kept below 1.0 so R9 still has `daily` "
                         "to upsample in training.")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--allow-leakage", action="store_true",
                    help="do not use; present only so the failure is a deliberate act")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("\n=== BUILD DEV SETS (R0 + in-training eval slice) ===")
    cleaned, _ = load_and_clean(args.pairs, verbose=True)
    pool_size = len(cleaned)

    hist = test_length_histogram(args.test_csv, args.max_words)

    # References must be human. See module docstring.
    pool = [
        r for r in cleaned
        if r.get("provenance") == "human" and 1 <= words(r["src"]) <= args.max_words
    ]
    print(f"  eligible pool: {len(pool):,} human pairs with 1..{args.max_words}-word "
          f"sources  (of {pool_size:,} cleaned)")

    total = args.size * 2
    want = allocate(hist, total)
    print(f"\n  stratified allocation for {total:,} pairs "
          f"(-> {args.size:,} R0 + {args.size:,} eval):")
    avail = Counter(words(r["src"]) for r in pool)
    for n in sorted(want):
        print(f"    {n:>2} words  test {100 * hist[n]:>5.1f}%   "
              f"want {want[n]:>4}   pool {avail[n]:>6}")

    picked, taken, warnings = select(pool, want, args.daily_share, rng)
    for w in warnings:
        print(f"  !! {w}")
    if len(picked) != total:
        print(f"\nFATAL: selected {len(picked):,} of {total:,} requested. The "
              f"eligible pool cannot support a length-matched draw this size; "
              f"lower --size or raise --max-words.")
        return 1

    # A pair key must be unique, or --exclude in build_corpus miscounts.
    keys = {pair_key(r) for r in picked}
    if len(keys) != len(picked):
        print("\nFATAL: duplicate pair keys among the selected rows.")
        return 1

    r0_rows, eval_rows = deal(picked, rng)
    if {pair_key(r) for r in r0_rows} & {pair_key(r) for r in eval_rows}:
        print("\nFATAL: R0 and the eval slice overlap.")
        return 1

    # --- leakage -------------------------------------------------------------
    # PROJECT_NOTES.md §2.3. These sets are *drawn from* BPCC, so the question is not
    # whether they leak into training — build_corpus excludes them by key — but
    # whether they overlap FLORES, which would make the two dev sets correlated
    # and the regression check circular.
    if args.dev_kas and args.dev_en:
        for name, rows in (("R0", r0_rows), ("eval", eval_rows)):
            print(f"\n  leakage check: {name} vs FLORES ...", flush=True)
            leaked = run_leakage_check(rows, args.dev_kas, args.dev_en, args.jaccard)
            print(f"    leaked pairs: {leaked}")
            if leaked and not args.allow_leakage:
                print(f"\nFATAL: {name} overlaps FLORES. Re-run data.leakage.")
                return 1
    else:
        print("\n  !! leakage check SKIPPED — --dev-kas/--dev-en not given. "
              "PROJECT_NOTES.md §2.3 requires it.")

    # The test set's English sources are public. If BPCC contains them, R0 is
    # not the only thing affected — the training corpus would be contaminated
    # against the live leaderboard. Cheap to check, so check.
    with open(args.test_csv, encoding="utf-8", newline="") as fh:
        test_src = {" ".join(str(r["sentence"]).split()).casefold()
                    for r in csv.DictReader(fh)}
    hits = sum(1 for r in cleaned if " ".join(r["src"].split()).casefold() in test_src)
    print(f"\n  test-input overlap: {hits} of {pool_size:,} cleaned pairs share an "
          f"English source with {args.test_csv.name}")

    # --- write ---------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_by": "data.build_devsets",
        "source_pairs_file": str(args.pairs),
        "source_pairs_sha256": sha256(args.pairs),
        "cleaned_pool_pairs": pool_size,
        "eligible_pool_pairs": len(pool),
        "test_length_reference": str(args.test_csv),
        "params": {
            "size": args.size, "max_words": args.max_words,
            "daily_share": args.daily_share, "seed": args.seed,
        },
        "stratified_allocation": {str(k): v for k, v in sorted(want.items())},
        "test_input_source_overlap": hits,
        "sets": {},
    }
    for name, rows in (("r0", r0_rows), ("r3_eval", eval_rows)):
        stats = describe("R0" if name == "r0" else "eval slice", rows)
        manifest["sets"][name] = {**stats, **write_set(args.out, name, rows)}

    (args.out / "devsets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  wrote {args.out}/r0/ and {args.out}/r3_eval/")
    print(f"  held out {total:,} pairs total -> build_corpus train should be "
          f"{pool_size:,} - {total:,} = {pool_size - total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
