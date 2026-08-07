#!/usr/bin/env python3
"""
KATHE 2026 — extract every English–Kashmiri (kas_Arab) pair from BPCC.

Config names were enumerated with get_dataset_config_names("ai4bharat/BPCC"),
not assumed; the authoritative list lives in src/config.py. Six of the eleven
configs carry a kas_Arab split.

Files are addressed by PATH, not by config, on purpose. BPCC's README defines
config_name "daily" twice -- once over daily/, once over wiki/ -- so loading
that config silently returns only one of the two directories and drops the
other with no error. Path addressing is immune to that.

kas_Deva is deliberately NOT extracted. PROJECT_NOTES.md §3: the target script is
kas_Arab, confirmed from the official scorer, whose normalizer is Perso-Arabic
throughout. Devanagari would pass through unnormalized and score ~0.

Usage:
    set -a; . ./.env; set +a
    uv run python -m data.fetch_bpcc \\
        --output data/processed/bpcc_kas_raw.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

REQUIRED_COLS = {"src_lang", "tgt_lang", "src", "tgt"}


def download(repo: str, path: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset", token=token))


def _scan(local: Path, quoting: int):
    """
    Parse under one quoting dialect and return (header, rows, n_bad).

    BPCC is not internally consistent, so neither is decidable up front:
      - column ORDER varies (bpcc-seed-latest is tgt,src,src_lang,tgt_lang)
      - some files use RFC4180 quoting with newlines inside fields
        (bpcc-seed-v2), others contain bare " that quoting would swallow
    Hence: map columns by header NAME, and pick the dialect empirically.
    """
    with open(local, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=quoting)
        try:
            header = next(reader)
        except StopIteration:
            return None, [], 0
        rows, bad = [], 0
        for parts in reader:
            if len(parts) != len(header):
                bad += 1
                continue
            rows.append(parts)
    return header, rows, bad


def read_tsv(local: Path, file_path: str, config_name: str, provenance: str):
    """
    Nothing goes through pandas here -- PROJECT_NOTES.md §6 records pandas mangling
    Perso-Arabic on read/write.
    """
    stats = Counter()

    # Try both dialects; keep whichever yields more well-formed records.
    best = None
    for name, q in (("QUOTE_MINIMAL", csv.QUOTE_MINIMAL), ("QUOTE_NONE", csv.QUOTE_NONE)):
        try:
            header, rows, bad = _scan(local, q)
        except csv.Error:
            continue
        if header is None:
            continue
        if best is None or len(rows) > len(best[2]):
            best = (name, header, rows, bad)
    if best is None:
        stats["unparseable"] += 1
        return [], stats

    dialect, header, raw_rows, bad = best
    stats["dialect"] = dialect
    stats["malformed_col_count"] = bad

    missing = REQUIRED_COLS - set(header)
    if missing:
        stats[f"missing_columns:{sorted(missing)}"] += 1
        return [], stats
    idx = {c: header.index(c) for c in REQUIRED_COLS}
    if header != ["src_lang", "tgt_lang", "src", "tgt"]:
        stats["nonstandard_column_order"] = 1

    rows = []
    for parts in raw_rows:
        if parts[idx["src_lang"]] != C.SRC_LANG:
            stats["wrong_src_lang"] += 1
            continue
        if parts[idx["tgt_lang"]] != C.TGT_LANG:
            stats["wrong_tgt_lang"] += 1
            continue
        stats["kept"] += 1
        rows.append({
            "src": parts[idx["src"]],
            "tgt": parts[idx["tgt"]],
            "config": config_name,
            "provenance": provenance,
            "source_file": file_path,
        })
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=C.DATA_PROCESSED / "bpcc_kas_raw.jsonl")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if not args.token:
        sys.exit("HF_TOKEN not set. BPCC is gated: put it in .env and "
                 "`set -a; . ./.env; set +a` before running.")

    all_rows = []
    per_file = []
    seen_hashes: dict[str, str] = {}
    skipped_dupes = []

    print(f"\n=== BPCC {C.SRC_LANG} -> {C.TGT_LANG} extraction ===")
    for file_path, config_name, provenance in C.BPCC_KAS_FILES:
        print(f"  downloading {file_path} ...", file=sys.stderr, flush=True)
        local = download(C.BPCC_REPO, file_path, args.token)

        # BPCC ships wiki/kas_Arab.tsv byte-identical to bpcc-seed-v1. Ingesting
        # both would double-count the same human pairs and mislabel half of them
        # as mined, which would skew the human/mined ratio the training mix is
        # chosen from. Catch it structurally rather than hard-coding the pair.
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest in seen_hashes:
            skipped_dupes.append((file_path, seen_hashes[digest]))
            continue
        seen_hashes[digest] = file_path

        rows, stats = read_tsv(local, file_path, config_name, provenance)
        all_rows.extend(rows)
        per_file.append((file_path, config_name, provenance, len(rows), stats))

    if skipped_dupes:
        print("\n--- skipped: byte-identical duplicate files ---")
        for dup, orig in skipped_dupes:
            print(f"  {dup}  ==  {orig}")

    print(f"\n--- pairs per source file ---")
    print(f"  {'file':34} {'config':16} {'type':7} {'pairs':>10}")
    for fp, cfg, prov, n, _ in per_file:
        print(f"  {fp:34} {cfg:16} {prov:7} {n:>10,}")
    print(f"  {'TOTAL':34} {'':16} {'':7} {len(all_rows):>10,}")

    print(f"\n--- by provenance ---")
    prov_counts = Counter(r["provenance"] for r in all_rows)
    for p, n in prov_counts.most_common():
        pct = 100 * n / len(all_rows) if all_rows else 0
        print(f"  {p:10} {n:>10,}  ({pct:.1f}%)")
    print("  NOTE: no back-translated Kashmiri source exists in BPCC. Every "
          "kas_Arab file is\n        either human-translated seed data or "
          "bitext-mined. Synthetic back-translation\n        would be ours to "
          "generate (PLANNING.md R7), not BPCC's to supply.")

    print(f"\n--- parse anomalies ---")
    any_anom = False
    for fp, _, _, _, stats in per_file:
        anom = {k: v for k, v in stats.items()
                if k not in ("kept", "header_skipped")}
        if anom:
            any_anom = True
            print(f"  {fp}: {dict(anom)}")
    if not any_anom:
        print("  none")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(all_rows):,} pairs -> {args.output}")


if __name__ == "__main__":
    main()
