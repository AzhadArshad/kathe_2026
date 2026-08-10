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

    rows = read_jsonl(args.pairs)
    print(f"\n=== BUILD CORPUS ===\n  input {len(rows):,} pairs  ({args.pairs})")

    if args.provenance != "all":
        rows = [r for r in rows if r.get("provenance") == args.provenance]
        print(f"  provenance={args.provenance}: {len(rows):,} pairs")

    # --- normalize -----------------------------------------------------------
    # Targets get the scorer's normalizer; sources get whitespace flattening
    # only, since KashmiriNormalizer has nothing to say about English.
    tgt_cfg = NormConfig(scorer_normalizer=True, map_punctuation=args.map_punctuation)

    cleaned: list[dict] = []
    dropped = Counter()
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
        k = (r["src"], r["tgt"])
        if k in seen:
            dropped["dup_after_norm"] += 1
            continue
        seen.add(k)
        deduped.append(r)

    print(f"  after normalize + dedup: {len(deduped):,} pairs")
    for k, v in dropped.most_common():
        print(f"    dropped {k:18} {v:>7,}")

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
    # Held-out eval is drawn from human sources so that checkpoint selection is
    # not steered by mined noise. Falls back to the full pool if none is human.
    rng = random.Random(args.seed)
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
    print(f"\n  train {len(train_rows):,}   dev(held-out) {len(dev_rows):,}")
    print(f"  train target diacritic density: {100 * dia / max(1, len(joined)):.2f} per 100 chars "
          f"(FLORES devtest reference: 7.70)")
    print(f"  train mean chars/line: {len(joined) / max(1, len(train_rows)):.1f} "
          f"(FLORES devtest: 124.6 — expect short-output bias, see R6)")
    print(f"  NFC: {joined == ud.normalize('NFC', joined)}")
    print(f"\n  wrote {args.out}/  (manifest.json records hashes and counts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
