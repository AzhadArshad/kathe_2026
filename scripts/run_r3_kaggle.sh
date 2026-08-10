#!/usr/bin/env bash
# KATHE 2026 — R3 fine-tune on Kaggle, GPU T4 x2.
#
# Run this from a Kaggle Notebook cell as:  !bash scripts/run_r3_kaggle.sh
# It is a script rather than notebook cells so that the live-round package and
# this training run share one reproducible entry point (PROJECT_NOTES.md §1).
#
# PREREQUISITES on the Kaggle side
#   1. Accelerator = GPU T4 x2. P100 was rejected: no bitsandbytes, one card.
#   2. Internet ON (model download from the Hub).
#   3. Kaggle Secret `HF_TOKEN`. indictrans2-en-indic-1B, facebook/flores and
#      ai4bharat/BPCC are all GATED — accept the terms on the Hub first, or the
#      download fails with a misleading error (PROJECT_NOTES.md §5).
#   4. The corpus attached as a Kaggle Dataset. Build it locally first
#      (`python -m data.build_corpus`, ~31 MB output) and upload
#      data/processed/r3_corpus. Rebuilding it inside the session would mean
#      re-downloading BPCC and re-running the LaBSE pass for no benefit.
#
# In the notebook, before this script:
#   from kaggle_secrets import UserSecretsClient
#   import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

set -euo pipefail

REPO="${REPO:-/kaggle/working/kathe_2026}"
CORPUS="${CORPUS:-/kaggle/input/kathe-r3-corpus/r3_corpus}"
CONFIG="${CONFIG:-config/r3_200m_full.yaml}"
OUTPUT="${OUTPUT:-/kaggle/working/r3-200m-full}"

cd "$REPO"

# --- environment -------------------------------------------------------------
# transformers MUST stay at 4.46.1: newer releases drop the
# PreTrainedTokenizerBase re-export that IndicTransToolkit's collator imports
# (PROJECT_NOTES.md §5, verified against the 1.1.1 sdist).
# torch is left at whatever the Kaggle image ships — the pinned 2.10.0+cu128
# does not resolve from plain PyPI and the image already provides a CUDA build.
pip install -q \
    "transformers==4.46.1" \
    "indictranstoolkit==1.1.1" \
    "sacrebleu==2.6.0" \
    "KashmiriNormalizer==0.1.0" \
    "peft" "accelerate" "datasets" "sentencepiece" "pyyaml"

pip install -q -e .   # pyproject maps src/ -> top-level, so `train.finetune` imports

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
print(f"transformers {transformers.__version__}")
assert torch.cuda.device_count() >= 1, "no GPU visible — check the accelerator setting"
from IndicTransToolkit import IndicProcessor, IndicDataCollator  # fails fast if the pin is wrong
print("IndicTransToolkit import OK")
PY

# --- smoke test --------------------------------------------------------------
# Builds both datasets and exits. Catches a bad corpus path or a tokenizer
# mismatch in ~2 minutes instead of after the model download and first epoch.
echo "=== DRY RUN ==="
python -m train.finetune --config "$CONFIG" --data-dir "$CORPUS" --output-dir "$OUTPUT" --dry-run

# --- train -------------------------------------------------------------------
# torchrun, not plain python: with two visible GPUs the Trainer would otherwise
# fall back to DataParallel, which is slower and misbehaves with
# predict_with_generate. DDP gives one process per card.
echo "=== TRAIN (DDP, $(python -c 'import torch;print(torch.cuda.device_count())') GPU) ==="
NPROC="$(python -c 'import torch;print(torch.cuda.device_count())')"
torchrun --nproc_per_node="$NPROC" -m train.finetune \
    --config "$CONFIG" \
    --data-dir "$CORPUS" \
    --output-dir "$OUTPUT"

# Add --resume when a dead session left checkpoints in $OUTPUT.
# Set push_to_hub + hub_model_id in the config to survive session death;
# /kaggle/working is not persistent across a restart.

echo "=== DONE — checkpoint in $OUTPUT ==="
ls -la "$OUTPUT"
