#!/usr/bin/env python3
"""
KATHE 2026 — build the fine-tuning corpus (R3).

Takes the leakage-cleaned JSONL from the R2 pipeline and emits the directory
layout AI4Bharat's fine-tuning script expects, so that their loader is used
verbatim rather than reimplemented (PROJECT_NOTES.md §2.8):

    <out>/train/eng_Latn-kas_Arab/train.eng_Latn
    <out>/train/eng_Latn-kas_Arab/train.kas_Arab
    <out>/dev/eng_Latn-kas_Arab/dev.eng_Latn
    <out>/dev/eng_Latn-kas_Arab/dev.kas_Arab

Three things happen here that matter to the score:

1. **Targets are normalized with `KashmiriNormalizer`** (PROJECT_NOTES.md §3). The
   scorer normalizes both sides, so training on un-normalized targets teaches
   the model distinctions the metric cannot see. The shared implementation in
   `data.normalize` is used — never a second copy.

2. **The in-training eval split is held out from BPCC, not FLORES.** FLORES dev
   is the knob-tuning set and devtest is the reporting set; neither is ever
   trained on, and neither is needed inside the training session, which keeps a
   gated download off the critical path. The held-out slice is drawn from human
   sources only, so checkpoint selection is not steered by mined noise.

3. **The leakage check is re-run**, because PROJECT_NOTES.md §2.3 requires it before
   every training run and whenever the mix changes. It fails loudly rather than
   warning.

Usage:
    uv run python -m data.build_corpus \\
        --pairs   data/processed/bpcc_kas_clean.jsonl \\
        --out     data/processed/r3_corpus \\
        --dev-kas data/dev/flores_kas.txt \\
        --dev-en  data/dev/flores_en.txt

Stage 2 of the two-stage plan (a short finishing pass on human seed data only):
    uv run python -m data.build_corpus ... --provenance human --out data/processed/r3_corpus_human
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import NormConfig, normalize  # noqa: E402

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

_WS = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def flatten(s: str) -> str:
    """Collapse every run of whitespace, including newlines, to one space.

    BPCC's quoted TSV fields preserve line terminators. A newline inside a
    training line would silently split one pair into two misaligned ones,
    because the trainer's loader is line-based.
    """
    return _WS.sub(" ", s).strip()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(out: Path, split: str, rows: list[dict]) -> dict:
    d = out / split / f"{SRC_LANG}-{TGT_LANG}"
    d.mkdir(parents=True, exist_ok=True)
    src_p = d / f"{split}.{SRC_LANG}"
    tgt_p = d / f"{split}.{TGT_LANG}"
    with open(src_p, "w", encoding="utf-8") as sf, open(tgt_p, "w", encoding="utf-8") as tf:
        for r in rows:
            sf.write(r["src"] + "\n")
            tf.write(r["tgt"] + "\n")
    return {
        "split": split,
        "pairs": len(rows),
        "src_file": str(src_p),
        "tgt_file": str(tgt_p),
        "src_sha256": sha256(src_p),
        "tgt_sha256": sha256(tgt_p),
    }


def pair_key(row: dict) -> tuple[str, str]:
    """Identity of a cleaned pair.

    This is the same key the post-normalization dedup uses, so it is unique
    across the cleaned corpus. `build_devsets` records it for every held-out
    pair and this module excludes on it, which is what makes "the training
    corpus shrank by exactly the held-out count" checkable rather than assumed.
    """
    return (row["src"], row["tgt"])


def load_and_clean(
    pairs: Path,
    provenance: str = "all",
    map_punctuation: bool = False,
    verbose: bool = True,
) -> tuple[list[dict], Counter]:
    """Read the pairs JSONL and apply normalization + dedup.

    Factored out so `data.build_devsets` selects its held-out pairs from
    exactly the rows this module would have trained on. If the two ever ran
    different cleaning, an exclusion key built by one would silently miss in
    the other and held-out pairs would stay in the training split.
    """
    rows = read_jsonl(pairs)
    if verbose:
        print(f"  input {len(rows):,} pairs  ({pairs})")

    if provenance != "all":
        rows = [r for r in rows if r.get("provenance") == provenance]
        if verbose:
            print(f"  provenance={provenance}: {len(rows):,} pairs")

    # Targets get the scorer's normalizer; sources get whitespace flattening
    # only, since KashmiriNormalizer has nothing to say about English.
    tgt_cfg = NormConfig(scorer_normalizer=True, map_punctuation=map_punctuation)

    cleaned: list[dict] = []
    dropped: Counter = Counter()
    for r in rows:
        src = flatten(str(r["src"]))
        tgt = flatten(normalize(r["tgt"], tgt_cfg))
        if not src or not tgt:
            dropped["empty"] += 1
            continue
        # PLANNING.md: 25 Devanagari characters survive the >=80% per-sentence
        # script filter. Drop the whole line rather than stripping characters —
        # a half-removed word is worse training signal than one fewer pair.
        if _DEVANAGARI.search(tgt):
            dropped["devanagari"] += 1
            continue
        cleaned.append({**r, "src": src, "tgt": tgt})

    # Normalization can collapse two previously distinct targets onto one.
    seen, deduped = set(), []
    for r in cleaned:
        k = pair_key(r)
        if k in seen:
            dropped["dup_after_norm"] += 1
            continue
        seen.add(k)
        deduped.append(r)

    if verbose:
        print(f"  after normalize + dedup: {len(deduped):,} pairs")
        for k, v in dropped.most_common():
            print(f"    dropped {k:18} {v:>7,}")
    return deduped, dropped


def run_leakage_check(rows: list[dict], dev_kas: Path, dev_en: Path, jaccard: float) -> int:
    """Re-run the R2 detectors. Returns the number of leaked pairs found."""
    from data.leakage import NearDupIndex, load_lines, loose_key, scorer_key

    kas_lines, en_lines = load_lines(dev_kas), load_lines(dev_en)
    kas_exact = {scorer_key(x) for x in kas_lines}
    en_exact = {scorer_key(x) for x in en_lines}
    kas_loose = {loose_key(x) for x in kas_lines} - {""}
    en_loose = {loose_key(x) for x in en_lines} - {""}
    kas_idx, en_idx = NearDupIndex(kas_lines), NearDupIndex(en_lines)

    leaked = 0
    for r in rows:
        hit = (
            scorer_key(r["tgt"]) in kas_exact
            or loose_key(r["tgt"]) in kas_loose
            or scorer_key(r["src"]) in en_exact
            or loose_key(r["src"]) in en_loose
            or kas_idx.best_match(r["tgt"], jaccard) is not None
            or en_idx.best_match(r["src"], jaccard) is not None
        )
        if hit:
            leaked += 1
    return leaked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dev-kas", type=Path, help="FLORES kas reference, for the leakage check")
    ap.add_argument("--dev-en", type=Path, help="FLORES en source, for the leakage check")
    ap.add_argument("--jaccard", type=float, default=0.80)
    ap.add_argument(
        "--provenance",
        default="all",
        choices=["all", "human", "mined"],
        help="'human' builds the stage-2 finishing corpus",
    )
    ap.add_argument("--heldout", type=int, default=1000, help="in-training eval pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="JSONL files of held-out pairs to remove from train (R0, the eval "
             "slice). Excluded by exact pair key; a miscount is fatal.",
    )
    ap.add_argument(
        "--dev-from",
        type=Path,
        help="JSONL to use as the in-training eval split, instead of sampling "
             "at random. Use the length-matched slice from data.build_devsets.",
    )
    ap.add_argument(
        "--map-punctuation",
        action="store_true",
        help="ALSO map Latin punctuation in targets. OFF by default: PROJECT_NOTES.md §3 "
             "assigns punctuation to post-processing, so that the direction can be "
             "verified on FLORES dev instead of baked into the weights.",
    )
    ap.add_argument(
        "--allow-leakage",
        action="store_true",
        help="do not use; present only so the failure is a deliberate act",
    )
    args = ap.parse_args()

    print("\n=== BUILD CORPUS ===")
    deduped, dropped = load_and_clean(
        args.pairs, args.provenance, args.map_punctuation, verbose=True
    )
    pool_size = len(deduped)

    # --- remove the held-out dev sets ----------------------------------------
    # R0 (the register-matched decision set) and the in-training eval slice are
    # both selected by `data.build_devsets` from this same cleaned pool. They
    # are excluded here by exact pair key, and the count is asserted, so a
    # silently-missed exclusion fails the build instead of leaking into train.
    excluded: dict[tuple[str, str], str] = {}
    for spec in args.exclude:
        rows_x = read_jsonl(spec)
        for r in rows_x:
            excluded[pair_key(r)] = spec.name
    if excluded:
        before = len(deduped)
        deduped = [r for r in deduped if pair_key(r) not in excluded]
        removed = before - len(deduped)
        print(f"\n  held-out exclusion: {removed:,} pairs removed "
              f"({before:,} -> {len(deduped):,})")
        if removed != len(excluded):
            print(
                f"\nFATAL: asked to exclude {len(excluded):,} held-out pairs but "
                f"only {removed:,} were found in the cleaned pool. The dev sets "
                f"were built from a different corpus or different normalization "
                f"settings; excluding them here is not reliable."
            )
            return 1

    # --- leakage -------------------------------------------------------------
    if args.dev_kas and args.dev_en:
        print("  running leakage check ...", flush=True)
        leaked = run_leakage_check(deduped, args.dev_kas, args.dev_en, args.jaccard)
        print(f"  leaked pairs: {leaked}")
        if leaked and not args.allow_leakage:
            print(
                "\nFATAL: dev-set leakage detected. Training on this makes every "
                "FLORES number optimistic and the leaderboard correlation "
                "meaningless (PROJECT_NOTES.md §2.3). Re-run data.leakage first."
            )
            return 1
    else:
        print("  !! leakage check SKIPPED — --dev-kas/--dev-en not given. "
              "PROJECT_NOTES.md §2.3 requires it before every training run.")

    # --- split ---------------------------------------------------------------
    rng = random.Random(args.seed)
    if args.dev_from:
        # The in-training eval slice comes from `data.build_devsets`, which
        # length-matches it to the test set. Sampling it at random here gave a
        # 15.7-word mean against a 7.3-word test set, so checkpoint selection
        # optimized the wrong sentence length (PLANNING.md, 2026-08-09).
        dev_rows = read_jsonl(args.dev_from)
        train_rows = list(deduped)
        print(f"  in-training eval slice: {len(dev_rows):,} pairs from {args.dev_from}")
    else:
        # Legacy path. Held-out eval is drawn from human sources so that
        # checkpoint selection is not steered by mined noise.
        print("  !! --dev-from not given: sampling the eval slice at RANDOM. "
              "This is register-mismatched against a 7.3-word test set.")
        human_idx = [i for i, r in enumerate(deduped) if r.get("provenance") == "human"]
        pool = human_idx or list(range(len(deduped)))
        n_dev = min(args.heldout, max(0, len(pool) - 1))
        dev_idx = set(rng.sample(pool, n_dev))
        dev_rows = [r for i, r in enumerate(deduped) if i in dev_idx]
        train_rows = [r for i, r in enumerate(deduped) if i not in dev_idx]
    rng.shuffle(train_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_pairs_file": str(args.pairs),
        "source_pairs_sha256": sha256(args.pairs),
        "provenance_filter": args.provenance,
        "seed": args.seed,
        "cleaned_pool_pairs": pool_size,
        "held_out": {
            "excluded_files": [str(p) for p in args.exclude],
            "excluded_pairs": len(excluded),
            "dev_from": str(args.dev_from) if args.dev_from else None,
        },
        "target_normalization": {
            "scorer_normalizer": True,
            "map_punctuation": args.map_punctuation,
            "note": "KashmiriNormalizer==0.1.0 via data.normalize; one implementation only",
        },
        "dropped": dict(dropped),
        "splits": [write_split(args.out, "train", train_rows),
                   write_split(args.out, "dev", dev_rows)],
        "provenance_counts": dict(Counter(r.get("provenance") for r in train_rows)),
        "config_counts": dict(Counter(r.get("config") for r in train_rows)),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- report --------------------------------------------------------------
    joined = "".join(r["tgt"] for r in train_rows)
    from KashmiriNormalizer.constants import KASHMIRI_DIACRITICS

    dia = sum(1 for c in joined if c in KASHMIRI_DIACRITICS)
    src_words = sum(len(r["src"].split()) for r in train_rows)
    print(f"\n  cleaned pool {pool_size:,}  -  held out {len(excluded):,}  =  train {len(train_rows):,}")
    print(f"  in-training eval slice: {len(dev_rows):,}")
    print(f"  train target diacritic density: {100 * dia / max(1, len(joined)):.2f} per 100 chars "
          f"(FLORES devtest reference: 7.70)")
    print(f"  train mean chars/line: {len(joined) / max(1, len(train_rows)):.1f} "
          f"  mean src words: {src_words / max(1, len(train_rows)):.1f}")
    print(f"    (KATHE test set: 39.0 chars / 7.3 words — R6 tunes toward THIS, "
          f"not FLORES's 124.6; expect over-generation)")
    print(f"  NFC: {joined == ud.normalize('NFC', joined)}")
    print(f"\n  wrote {args.out}/  (manifest.json records hashes and counts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
