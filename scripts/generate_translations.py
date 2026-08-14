#!/usr/bin/env python3
"""
KATHE 2026 — THE LIVE-ROUND DELIVERABLE. English -> Kashmiri, end to end.

Give it the competition CSV, get back translations ready to score. One command,
no other steps:

    python scripts/generate_translations.py \
        --input  englishdev.csv \
        --output submission.csv

WHY THIS EXISTS SEPARATELY FROM scripts/translate.py
`translate.py` is the decode tool: model in, raw text out. It deliberately stops
after IndicTrans2's own `postprocess_batch`, because the research pipeline needs
un-restored output to build lexicons and train restorers from.

Raw output is NOT the system. IndicTrans2's target vocabulary holds kasra
(U+0650), damma (U+064F) and fatha (U+064E) in exactly one token each, so beam
search never emits them, and no amount of fine-tuning changes a frozen
vocabulary (PROJECT_NOTES.md §3). A learned character-level restorer puts them back.
Measured on the competition input, same model, same decode:

    raw model output                                 10.00
    + diacritic restoration (r11b_dense, this file)  13.81

That +3.81 is larger than every training-side improvement in this project
combined. A batch-translation script that skips it is shipping a system 27%
worse than the one that was built, which is why this file runs the full pipeline
and refuses to write output that looks like restoration silently failed.

WHAT IT DOES
  1. Reads the input (CSV with a `sentence`/`english`/`text`/`src` column, or
     one sentence per line).
  2. Decodes with beam search via `translate.translate`, which applies
     IndicProcessor's `preprocess_batch` AND `postprocess_batch` — skipping
     either produces plausible-looking output that scores badly.
  3. Applies post-processing from `config/postproc_live.yaml`: the scorer's own
     normalizer, then the diacritic restorer.
  4. Writes `ID,kashmiri_text` in the INPUT'S ROW ORDER, never sorted.
  5. Self-checks: row count preserved, no empty rows, restoration actually
     happened.

ROW ORDER IS LOAD-BEARING. The official scorer deletes the ID column from both
frames and zips what remains positionally, so a correctly-ID'd but reordered
file scores near zero while looking perfectly fine (PROJECT_NOTES.md §3).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.normalize import NormConfig, normalize_many  # noqa: E402

# `translate` is imported lazily inside main(): it pulls in torch, transformers
# and IndicTransToolkit at module scope, so importing it here would make
# `--help`, the config check and the post-processing path all require the full
# decode stack. Keeping it late also gives a legible error when the environment
# is short a pin, instead of a bare ModuleNotFoundError before argparse runs.

# The fine-tuned checkpoint behind the best leaderboard score. A local directory
# works too — the live-round operator can point --model at whatever was shipped
# alongside this script.
DEFAULT_MODEL = "Aju360/kathe-r12-200m-selected"
DEFAULT_POSTPROC = ROOT / "config" / "postproc_live.yaml"

# Restorer weights are ~13 MB and live outside git (PROJECT_NOTES.md §2.6). If the path
# in the config is missing, fetch it from the Hub rather than failing — a fresh
# clone must work without a manual download step.
# Must match the --repo default in scripts/push_restorer_hf.py. These two names
# disagreed once and the download would have 404'd against a repo nothing
# publishes to. Apache-licensed checkpoints land here; NonCommercial ones are
# routed to "<repo>-nc" by the push script, and r11b_dense is Apache.
RESTORER_REPO = "Aju360/kathe-r11-restorer"

RESTORABLE = "َُِ"  # fatha, damma, kasra


def load_postproc(path: Path) -> NormConfig:
    """Build a NormConfig from YAML, rejecting unknown keys loudly.

    A typo'd key that silently defaulted would change the output without
    changing the recorded config — the kind of drift that makes a score
    unreproducible.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(NormConfig)}
    unknown = set(raw) - known
    if unknown:
        sys.exit(f"FATAL: unknown post-processing keys in {path}: {sorted(unknown)}")
    return NormConfig(**raw)


def ensure_restorer(cfg: NormConfig) -> NormConfig:
    """Resolve `restore_model` to a file that exists, downloading if needed."""
    if not cfg.restore_model:
        return cfg
    p = Path(cfg.restore_model)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return NormConfig(**{**cfg.__dict__, "restore_model": str(p)})

    print(f"  restorer not at {p} — fetching from {RESTORER_REPO}", file=sys.stderr)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("FATAL: restorer weights missing and huggingface_hub is not "
                 "installed. `pip install huggingface_hub`, or place the "
                 f"checkpoint at {p}.")
    try:
        got = hf_hub_download(repo_id=RESTORER_REPO, filename=p.name)
    except Exception as exc:  # noqa: BLE001 — any failure here is fatal alike
        sys.exit(f"FATAL: could not fetch {p.name} from {RESTORER_REPO}: {exc}\n"
                 f"       Place the checkpoint at {p} and re-run.")
    return NormConfig(**{**cfg.__dict__, "restore_model": got})


def density(texts: list[str], chars: str) -> float:
    joined = "".join(texts)
    if not joined:
        return 0.0
    return 100.0 * sum(joined.count(c) for c in chars) / len(joined)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="English -> Kashmiri, decode + diacritic restoration, "
                    "ready to score.")
    ap.add_argument("--input", required=True, type=Path,
                    help="competition CSV, or one English sentence per line")
    ap.add_argument("--output", required=True, type=Path,
                    help=".csv writes ID,kashmiri_text; any other suffix writes "
                         "one translation per line")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo id or a local checkpoint directory")
    ap.add_argument("--postproc", type=Path, default=DEFAULT_POSTPROC,
                    help="post-processing YAML (default: config/postproc_live.yaml)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--limit", type=int,
                    help="translate only the first N — smoke test, not a run")
    ap.add_argument("--no-restore", action="store_true",
                    help="skip diacritic restoration. Costs ~3.8 points; exists "
                         "only for diagnosing the decode in isolation.")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"FATAL: {args.input} not found.")

    try:
        from translate import pick_device, read_input, translate
    except ImportError as exc:
        sys.exit(f"FATAL: the decode stack is incomplete ({exc}).\n"
                 "       Needs: transformers==4.46.1, indictranstoolkit==1.1.1, "
                 "torch, sentencepiece.\n"
                 "       See requirements-kaggle.txt for the pinned set.")

    sentences, ids = read_input(args.input)
    if args.limit:
        sentences = sentences[: args.limit]
        ids = ids[: args.limit] if ids else None
    n_in = len(sentences)
    device = pick_device(args.device)

    cfg = load_postproc(args.postproc)
    if args.no_restore:
        cfg = NormConfig(**{**cfg.__dict__, "restore_model": None,
                            "diacritic_lexicon": None})
    else:
        cfg = ensure_restorer(cfg)

    print(f"  {n_in:,} sentences  model={args.model}  device={device}  "
          f"beam={args.beam}", file=sys.stderr)

    hyps = translate(sentences, args.model, device, args.batch_size, args.beam,
                     args.max_new_tokens, args.length_penalty)
    if len(hyps) != n_in:
        sys.exit(f"FATAL: {len(hyps)} translations for {n_in} inputs. Positional "
                 "scoring would silently misalign every row.")

    before = density(hyps, RESTORABLE)
    out = normalize_many(hyps, cfg)
    after = density(out, RESTORABLE)

    if len(out) != n_in:
        sys.exit(f"FATAL: post-processing returned {len(out)} rows for {n_in}.")

    blank = [i for i, s in enumerate(out) if not s.strip()]
    if blank:
        sys.exit(f"FATAL: {len(blank)} empty row(s) after post-processing "
                 f"(first: {blank[:5]}). The official scorer raises on an empty "
                 "hypothesis and rejects the whole submission.")

    # Restoration silently doing nothing is the failure mode that would cost the
    # most and show the least — the output would look entirely reasonable.
    print(f"  restorable marks: {before:.3f} -> {after:.3f} per 100 chars",
          file=sys.stderr)
    if not args.no_restore and after <= before:
        sys.exit("FATAL: restoration ran but added no kasra/damma/fatha. The "
                 "checkpoint may have loaded as zeros — verify it generates "
                 "text before trusting any output.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        # Input row order, never sorted (PROJECT_NOTES.md §3).
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ID", "kashmiri_text"])
            for i, h in enumerate(out):
                w.writerow([ids[i] if ids else i + 1, h])
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            for h in out:
                fh.write(h + "\n")

    print(f"  wrote {args.output} — {len(out):,} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
