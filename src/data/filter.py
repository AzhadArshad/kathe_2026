#!/usr/bin/env python3
"""
KATHE 2026 — quality filter for extracted English–Kashmiri pairs.

Stages, applied in order, each reported:
  1. empty / whitespace-only drop
  2. exact dedup (on the raw pair)
  3. length-ratio bounds
  4. script sanity — target must be predominantly Perso-Arabic
  5. LaBSE cosine >= threshold

LaBSE is last because it is the only expensive stage; everything cheap runs
first so we embed as few pairs as possible.

Runs on CPU by default so a fresh clone reproduces on any machine. --device mps
is roughly 2x faster on an M2 (measured: ~80 sent/s cpu vs ~166 sent/s mps) and
changes cosines only in the last decimals, well below the 0.80 cutoff.

Usage:
    uv run python -m data.filter \\
        --input data/processed/bpcc_kas_raw.jsonl \\
        --output data/processed/bpcc_kas_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

# Perso-Arabic blocks: Arabic (0600-06FF), Arabic Supplement (0750-077F),
# Arabic Extended-A (08A0-08FF), presentation forms (FB50-FDFF, FE70-FEFF).
_ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]
_DEVANAGARI = (0x0900, 0x097F)


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_profile(text: str) -> dict:
    """Fractions over *letter-ish* characters only; punctuation/digits ignored."""
    arabic = devanagari = latin = other = 0
    for ch in text:
        if ch.isspace() or ud.category(ch).startswith(("P", "N", "Z", "C")):
            continue
        cp = ord(ch)
        if _in_ranges(cp, _ARABIC_RANGES):
            arabic += 1
        elif _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            devanagari += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        else:
            other += 1
    total = arabic + devanagari + latin + other
    if total == 0:
        return {"total": 0, "arabic": 0.0, "devanagari": 0.0, "latin": 0.0}
    return {
        "total": total,
        "arabic": arabic / total,
        "devanagari": devanagari / total,
        "latin": latin / total,
    }


_WS = re.compile(r"\s+")


def stage_nonempty(pairs):
    """
    Also flattens whitespace. BPCC's quoted TSV fields preserve their line
    terminators: 86,526 of 133,247 surviving targets carried a trailing newline
    and 559 had interior ones. Harmless for dedup and leakage (both key on
    stripped text) but they would be trained on verbatim, teaching the model to
    emit stray newlines into submission rows -- and an empty row after
    normalization kills the whole submission (PROJECT_NOTES.md §3).
    """
    out = []
    for p in pairs:
        p["src"] = _WS.sub(" ", p["src"]).strip()
        p["tgt"] = _WS.sub(" ", p["tgt"]).strip()
        if p["src"] and p["tgt"]:
            out.append(p)
    return out


def stage_dedup(pairs):
    seen = set()
    out = []
    for p in pairs:
        key = (p["src"].strip(), p["tgt"].strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def stage_length(pairs, min_words: int, max_words: int, min_ratio: float, max_ratio: float):
    """
    Ratio is target-chars / source-chars. English->Kashmiri is roughly 1:1 in
    characters; the bounds are deliberately loose, they only catch garbage
    (a one-word target against a 40-word source, and the reverse).
    """
    out = []
    for p in pairs:
        s, t = p["src"].strip(), p["tgt"].strip()
        sw = len(s.split())
        if not (min_words <= sw <= max_words):
            continue
        ratio = len(t) / max(1, len(s))
        if not (min_ratio <= ratio <= max_ratio):
            continue
        out.append(p)
    return out


def stage_script(pairs, min_arabic: float):
    out = []
    rejected = Counter()
    for p in pairs:
        prof = script_profile(p["tgt"])
        if prof["total"] == 0:
            rejected["no_letters"] += 1
            continue
        if prof["devanagari"] > 0.10:
            rejected["devanagari"] += 1
            continue
        if prof["arabic"] < min_arabic:
            rejected["not_perso_arabic"] += 1
            continue
        out.append(p)
    return out, rejected


def stage_labse(pairs, threshold: float, batch_size: int, model_name: str,
                device: str = "cpu", chunk_size: int = 8192,
                scope: str = "all", scores_path: Path | None = None):
    """
    scope="mined" scores only bitext-mined pairs and passes human-translated
    seed pairs through untouched.

    Measured on this corpus (2026-08-07), LaBSE cosines for eng-kas are
    compressed downward -- median 0.6077, p95 0.8402 -- and a 0.80 cutoff
    retained the MINED source (10.8%) at a higher rate than three of the five
    HUMAN sources (2.9%-9.9%). A filter that discards 95% of professionally
    translated text is not measuring translation quality; it is measuring
    LaBSE's weak coverage of Kashmiri. Scoping it to mined data confines the
    filter to where alignment errors actually occur.
    """
    if not pairs:
        return [], []
    from sentence_transformers import SentenceTransformer
    import torch

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading {model_name} ({device})...", file=sys.stderr, flush=True)
    model = SentenceTransformer(model_name, device=device)

    # Chunked on purpose. Embedding all source rows into one tensor and all
    # target rows into a second one holds 2 x N x 768 float32 simultaneously --
    # ~1.4 GB at N=233k -- which pushes an 8 GB machine into swap and slows the
    # second pass by ~10x. We only ever need the pairwise diagonal, so score a
    # block at a time and keep nothing but the floats.
    to_score = [p for p in pairs if scope == "all" or p.get("provenance") == "mined"]
    n_skipped = len(pairs) - len(to_score)
    if n_skipped:
        print(f"  scope={scope}: scoring {len(to_score):,} pairs, passing "
              f"{n_skipped:,} human pairs through unscored",
              file=sys.stderr, flush=True)

    scores = []
    total = len(to_score)
    print(f"  embedding {total} pairs x2 in chunks of {chunk_size} ...",
          file=sys.stderr, flush=True)
    for start in range(0, total, chunk_size):
        block = to_score[start:start + chunk_size]
        es = model.encode([p["src"] for p in block], batch_size=batch_size,
                          show_progress_bar=False, convert_to_tensor=True,
                          normalize_embeddings=True)
        et = model.encode([p["tgt"] for p in block], batch_size=batch_size,
                          show_progress_bar=False, convert_to_tensor=True,
                          normalize_embeddings=True)
        scores.extend(torch.sum(es * et, dim=1).cpu().tolist())
        del es, et
        done = min(start + chunk_size, total)
        print(f"    labse {done}/{total}  ({100 * done / total:.1f}%)",
              file=sys.stderr, flush=True)

    for p, sc in zip(to_score, scores):
        p["labse"] = round(float(sc), 4)

    # Cache every score, rejected ones included, so changing the threshold later
    # is a re-read rather than another full embedding pass.
    if scores_path:
        write_jsonl(scores_path, to_score)
        print(f"  cached {len(to_score):,} scored pairs -> {scores_path}",
              file=sys.stderr, flush=True)

    kept = [p for p in pairs
            if "labse" not in p or p["labse"] >= threshold]
    return kept, scores


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def by_config(pairs) -> Counter:
    return Counter(p.get("config", "?") for p in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--labse-threshold", type=float, default=0.80)
    ap.add_argument("--labse-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=8192,
                    help="pairs scored per block; caps peak embedding memory")
    ap.add_argument("--labse-scope", default="mined", choices=["all", "mined"],
                    help="which pairs the LaBSE cutoff applies to. Default 'mined': "
                         "human-translated seed pairs are genuine translations by "
                         "construction, and LaBSE's weak Kashmiri coverage rejects "
                         "them at up to 97%% (see stage_labse docstring)")
    ap.add_argument("--labse-scores", type=Path,
                    default=Path("data/processed/bpcc_kas_labse_scores.jsonl"),
                    help="cache of every LaBSE score, rejected included, so the "
                         "threshold can be changed without re-embedding")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"],
                    help="LaBSE scoring device. cpu is the reproducible default; "
                         "mps uses the Apple GPU (~2x faster on an M2, cosines "
                         "differ only in the last decimals)")
    ap.add_argument("--min-words", type=int, default=1)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--min-ratio", type=float, default=0.5)
    ap.add_argument("--max-ratio", type=float, default=2.0)
    ap.add_argument("--min-arabic", type=float, default=0.80)
    ap.add_argument("--skip-labse", action="store_true",
                    help="run the cheap stages only; useful for a fast dry run")
    args = ap.parse_args()

    pairs = read_jsonl(args.input)
    report = [("input", len(pairs))]
    print(f"\n=== FILTER: {args.input} ===")
    print(f"  loaded {len(pairs)} pairs")

    pairs = stage_nonempty(pairs)
    report.append(("after non-empty", len(pairs)))

    pairs = stage_dedup(pairs)
    report.append(("after exact dedup", len(pairs)))

    pairs = stage_length(pairs, args.min_words, args.max_words,
                         args.min_ratio, args.max_ratio)
    report.append((f"after length ({args.min_ratio}-{args.max_ratio} char ratio, "
                   f"{args.min_words}-{args.max_words} src words)", len(pairs)))

    pairs, script_rej = stage_script(pairs, args.min_arabic)
    report.append((f"after script (>={args.min_arabic:.0%} Perso-Arabic)", len(pairs)))

    if args.skip_labse:
        report.append(("LaBSE", "SKIPPED"))
        scores = []
    else:
        pairs, scores = stage_labse(pairs, args.labse_threshold,
                                    args.batch_size, args.labse_model,
                                    args.device, args.chunk_size,
                                    args.labse_scope, args.labse_scores)
        report.append((f"after LaBSE (cos >= {args.labse_threshold}, "
                       f"scope={args.labse_scope})", len(pairs)))

    print("\n--- survivors per stage ---")
    prev = None
    for label, n in report:
        if isinstance(n, int) and isinstance(prev, int) and prev:
            print(f"  {label:58} {n:>9,}  (-{prev - n:,}, {100 * n / prev:.1f}% kept)")
        else:
            print(f"  {label:58} {n:>9}")
        prev = n

    if script_rej:
        print("\n--- script rejections ---")
        for k, v in script_rej.most_common():
            print(f"  {k:24} {v:>9,}")

    if scores:
        import statistics
        print(f"\n--- LaBSE score distribution (pre-threshold, "
              f"scope={args.labse_scope}, n={len(scores):,}) ---")
        print(f"  mean   {statistics.mean(scores):.4f}")
        print(f"  median {statistics.median(scores):.4f}")
        for q in (0.05, 0.25, 0.50, 0.75, 0.95):
            idx = int(q * (len(scores) - 1))
            print(f"  p{int(q * 100):02d}    {sorted(scores)[idx]:.4f}")

    print("\n--- survivors per config ---")
    for cfg, n in by_config(pairs).most_common():
        print(f"  {cfg:40} {n:>9,}")

    print("\n--- survivors per provenance ---")
    for prov, n in Counter(p.get("provenance", "?") for p in pairs).most_common():
        pct = 100 * n / len(pairs) if pairs else 0
        print(f"  {prov:40} {n:>9,}  ({pct:.1f}%)")

    write_jsonl(args.output, pairs)
    print(f"\n  wrote {len(pairs):,} pairs -> {args.output}")


if __name__ == "__main__":
    main()
