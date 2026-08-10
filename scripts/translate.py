#!/usr/bin/env python3
"""
KATHE 2026 — batch English -> Kashmiri translation.

This is the live-round deliverable (PROJECT_NOTES.md §4): the organizers get a
checkpoint and run this script themselves. It must therefore work from a fresh
clone with nothing but `--input` and `--output`, and must not depend on
anything interactive.

Two things here are load-bearing and easy to get wrong:

1. **`IndicProcessor` runs on both sides.** IndicTrans2 expects sources tagged
   with language tokens and transliterated; its output comes back needing
   `postprocess_batch` to be detokenized and mapped back into the target
   script. Skip either and the output looks plausible and scores badly
   (PROJECT_NOTES.md §5).

2. **Row order is preserved exactly.** Scoring is positional — the official
   scorer deletes the ID column and zips what remains, so a reordered output
   scores near zero while looking correct (PROJECT_NOTES.md §3). Sorting by length to
   make batches efficient is a real speedup, so it is done and then undone,
   and the restoration is asserted rather than trusted.

Environment: this needs `transformers==4.46.1`. Newer releases drop the
`PreTrainedTokenizerBase` re-export that IndicTransToolkit imports at package
import time, which breaks inference and not merely training.

Usage:
    python scripts/translate.py --input data/dev/r0/r0.eng_Latn --output out.txt
    python scripts/translate.py --input data/raw/englishdev.csv --output sub.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"
DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-1B"


def read_input(path: Path) -> tuple[list[str], list[str] | None]:
    """Return (sentences, ids). `ids` is None for plain text input."""
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"{path} has no data rows")
        cols = rows[0].keys()
        text_col = next((c for c in ("sentence", "english", "text", "src") if c in cols), None)
        if text_col is None:
            raise SystemExit(f"{path}: no source column among {list(cols)}")
        id_col = "ID" if "ID" in cols else None
        return (
            [str(r[text_col]) for r in rows],
            [str(r[id_col]) for r in rows] if id_col else None,
        )
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh], None


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def translate(
    sentences: list[str],
    model_id: str,
    device: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
) -> list[str]:
    ip = IndicProcessor(inference=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device).eval()

    # Batch by similar length so padding is not the dominant cost, then undo.
    order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i].split()))
    out: list[str | None] = [None] * len(sentences)

    t0 = time.time()
    done = 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = ip.preprocess_batch(
            [sentences[i] for i in idx], src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        enc = tok(batch, truncation=True, padding=True, max_length=256,
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                num_beams=num_beams,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                early_stopping=True,
                use_cache=True,
            )
        with tok.as_target_tokenizer():
            decoded = tok.batch_decode(
                gen.detach().cpu().tolist(), skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        for i, text in zip(idx, ip.postprocess_batch(decoded, lang=TGT_LANG)):
            out[i] = text
        done += len(idx)
        rate = done / max(1e-9, time.time() - t0)
        print(f"  {done}/{len(sentences)}  {rate:.2f} sent/s  "
              f"eta {(len(sentences) - done) / max(1e-9, rate) / 60:.1f} min",
              file=sys.stderr, flush=True)

    if any(o is None for o in out):
        raise SystemExit("internal error: not every sentence was translated")
    return [o for o in out if o is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--limit", type=int, help="translate only the first N (smoke test)")
    args = ap.parse_args()

    sentences, ids = read_input(args.input)
    if args.limit:
        sentences, ids = sentences[: args.limit], ids[: args.limit] if ids else None
    device = pick_device(args.device)
    print(f"  {len(sentences):,} sentences  model={args.model}  device={device}  "
          f"beam={args.beam}", file=sys.stderr)

    hyps = translate(
        sentences, args.model, device, args.batch_size, args.beam,
        args.max_new_tokens, args.length_penalty,
    )
    assert len(hyps) == len(sentences), "row count changed — positional scoring would break"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        # ID,kashmiri_text in the input's row order. Never sort (PROJECT_NOTES.md §3).
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ID", "kashmiri_text"])
            for i, h in enumerate(hyps):
                w.writerow([ids[i] if ids else i + 1, h])
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            for h in hyps:
                fh.write(h + "\n")
    print(f"  wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
