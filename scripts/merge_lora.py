#!/usr/bin/env python3
"""
KATHE 2026 — merge a LoRA adapter into its base model.

Produces a plain checkpoint that loads with an ordinary `from_pretrained`, with
no PEFT dependency and no reference to the GATED base repo. That is a hard
requirement for the live round: the organizers run our script on their machine,
and it must not need a Hub gate accepted or an adapter resolved.

Why this exists as a standalone script rather than only inside `finetune.py`:
R4 trained to completion (4,825/4,825, 3h20m) and then died in
`Trainer._load_best_model()`, because peft's `load_adapter` imports
`transformers.integrations.tensor_parallel`, which does not exist in
transformers 4.46.1 — the version IndicTransToolkit pins us to. Every training
step was already paid for and the adapter checkpoints were safely on the Hub,
so the run needed recovering, not repeating. `finetune.py` no longer sets
`load_best_model_at_end` for LoRA, but this script remains the way to turn any
adapter — from a crashed run, an interrupted one, or the Hub — into a usable
model.

Usage:
    python scripts/merge_lora.py \\
        --base    ai4bharat/indictrans2-en-indic-1B \\
        --adapter Aju360/kathe-r4-1b-lora \\
        --output  /kaggle/working/r4-1b-merged

Then verify by GENERATING TEXT, never by loading without error — a checkpoint
that loads cleanly can still be dead (PROJECT_NOTES.md §5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train.finetune import make_checkpoint_portable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model repo or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter repo or path")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu is fine and avoids VRAM pressure; merging is a "
                         "weight operation, not a forward pass")
    args = ap.parse_args()

    print(f"  base    {args.base}")
    print(f"  adapter {args.adapter}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base, trust_remote_code=True, torch_dtype=torch.float32
    )
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # Snapshot tensors the adapter ACTUALLY targets. Checking an arbitrary
    # tensor is useless: LoRA touches only the projections named in
    # `target_modules`, so an embedding is unchanged by design and reports a
    # meaningless False.
    import json

    from huggingface_hub import hf_hub_download

    try:
        acfg = json.loads(Path(hf_hub_download(args.adapter, "adapter_config.json")).read_text())
    except Exception:
        acfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    targets = list(acfg.get("target_modules") or [])
    print(f"  adapter targets: {targets}")

    watch = [k for k in model.state_dict() if any(f".{t}.weight" in k for t in targets)][:5]
    before = {k: model.state_dict()[k].clone() for k in watch}

    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.float32)
    model = model.merge_and_unload()
    model.eval()

    # A silently-empty adapter would yield a checkpoint identical to the base
    # and score exactly like it — which looks like "LoRA didn't help" rather
    # than "the merge did nothing". Check the targeted tensors specifically.
    after = model.state_dict()
    changed = sum(1 for k in watch if not torch.allclose(before[k], after[k]))
    print(f"  targeted tensors changed by merge: {changed}/{len(watch)}")
    for k in watch[:3]:
        delta = (after[k] - before[k]).abs().max().item()
        print(f"      {k}  max|delta| {delta:.6f}")
    if changed == 0:
        return 1 if not print(
            "\nFATAL: the merge changed nothing in the adapter's own target "
            "modules. The adapter is empty or mismatched with this base."
        ) else 1

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output), safe_serialization=True)
    tok.save_pretrained(str(args.output))
    make_checkpoint_portable(str(args.output))

    print(f"\n  wrote {args.output}")
    print("  NOW VERIFY BY GENERATING TEXT — a checkpoint that loads without "
          "error can still be dead (PROJECT_NOTES.md §5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
