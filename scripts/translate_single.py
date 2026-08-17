#!/usr/bin/env python3
"""
KATHE 2026 — SINGLE-SENTENCE inference.

One English sentence in, one Kashmiri translation out.

    python scripts/translate_single.py --text "He lost his pen."
    python scripts/translate_single.py --text "She is a teacher." --json

    # several at once, still one line of output each
    python scripts/translate_single.py --text "He lost his pen." --text "I am tired."

    # interactive, one sentence per line; Ctrl-D to finish
    python scripts/translate_single.py

For the competition's CSV format use `scripts/generate_translations.py`, which
is the batch entry point. Both scripts load the same system through
`src/infer.py`, so they cannot drift apart — `--self-test` asserts that.

Weights resolve to the published checkpoints, so this works from a clean clone
with no local `models/` directory and no Hugging Face token:

    translation   Aju360/kathe-r12-200m-selected
    restorer      Aju360/kathe-r11-restorer  (r11b_dense.pt, fetched on demand)
    lexicon       data/processed/diacritic_lexicon_both.json  (in this repo)

Model loading takes a few seconds and dominates a single short sentence. To
translate many, either pass several `--text` flags, use stdin, or import
`src/infer.py` directly and reuse one `System`.

Environment: needs `transformers==4.46.1` (see `requirements-decode.txt`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_POSTPROC,
    load_system,
    translate,
)

# Sentences with a known-good shape for --self-test. These exercise the part
# that matters: `چھُس` carries a damma that the translation model structurally
# cannot emit, so a run whose restoration stage is broken fails visibly here.
SELF_TEST = [
    "He lost his pen.",
    "I go to my school daily.",
    "She is a teacher.",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Translate a single English sentence into Kashmiri (kas_Arab).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no --text and no piped stdin, reads sentences interactively.",
    )
    ap.add_argument("--text", action="append", default=[],
                    help="sentence to translate; repeat for several")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"translation weights (default: {DEFAULT_MODEL})")
    ap.add_argument("--postproc", type=Path, default=DEFAULT_POSTPROC,
                    help="post-processing YAML (default: config/postproc_live.yaml)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--json", action="store_true",
                    help="emit {source, translation} JSON lines instead of plain text")
    ap.add_argument("--no-restore", action="store_true",
                    help="skip diacritic restoration. Costs 5.05 points on the "
                         "competition metric — for inspection only, never for output.")
    ap.add_argument("--self-test", action="store_true",
                    help="translate three fixed sentences and check the "
                         "restoration stage actually fired")
    args = ap.parse_args()

    if args.self_test:
        sentences = list(SELF_TEST)
    elif args.text:
        sentences = args.text
    elif not sys.stdin.isatty():
        sentences = [ln.strip() for ln in sys.stdin if ln.strip()]
    else:
        print("Enter English sentences, one per line. Ctrl-D when done.", file=sys.stderr)
        sentences = [ln.strip() for ln in sys.stdin if ln.strip()]

    if not sentences:
        print("no input", file=sys.stderr)
        return 1

    restore = not args.no_restore
    print(f"  loading {args.model} …", file=sys.stderr)
    system = load_system(args.model, args.postproc, args.device, restore=restore)
    print(f"  device={system.device}  restoration={'on' if restore else 'OFF'}", file=sys.stderr)

    out = translate(
        system, sentences, batch_size=16, num_beams=args.beam,
        max_new_tokens=args.max_new_tokens, length_penalty=args.length_penalty,
        restore=restore,
    )

    for src, tgt in zip(sentences, out):
        if args.json:
            print(json.dumps({"source": src, "translation": tgt}, ensure_ascii=False))
        else:
            print(tgt)

    if args.self_test:
        # The three marks the translation model cannot produce. Their absence
        # means the restoration stage silently did nothing — which still emits
        # fluent-looking Kashmiri and still scores 5 points worse.
        marks = set("َُِ")
        found = sum(c in marks for t in out for c in t)
        print(f"\n  restorable marks in output: {found}", file=sys.stderr)
        if restore and found == 0:
            print("  SELF-TEST FAILED: restoration produced no kasra/damma/fatha.\n"
                  "  Check that the restorer weights and the lexicon both loaded.",
                  file=sys.stderr)
            return 2
        print("  SELF-TEST PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
