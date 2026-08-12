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

# The T4 OOM'd at step 0 with a 2.69 GiB request while 2.56 GiB was free —
# i.e. fragmentation was part of it, not only total demand. expandable_segments
# lets the allocator grow a segment instead of needing one contiguous block.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

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
# --no-deps is LOAD-BEARING. pyproject.toml declares transformers>=5.14.1 for
# the data/scoring environment; a plain `pip install -e .` upgrades transformers
# back to 5.x, which breaks IndicTransToolkit's import, and drags torch past the
# Kaggle image's build, desynchronizing torchvision/torchaudio and the CUDA
# libs. The editable install exists only to put src/ on the path — every runtime
# dependency is named explicitly below. (Cost a Q9 session on 2026-08-10.)
pip install -q --no-deps -e .

# huggingface-hub pinned explicitly: transformers 4.46.1 requires <1.0 while
# pyproject wants >=1.26.1, so a stale newer hub leaves an incompatible pair.
pip install -q \
    "transformers==4.46.1" \
    "huggingface-hub<1.0" \
    "indictranstoolkit==1.1.1" \
    "sacrebleu==2.6.0" \
    "KashmiriNormalizer==0.1.0" \
    "peft" "accelerate" "datasets" "sentencepiece" "pyyaml"

# REMOVE torchao. peft's LoRA dispatcher calls is_torchao_available(), which
# RAISES ImportError when torchao is installed but older than 0.16.0 — it only
# returns False when the package is absent entirely. The Kaggle image ships
# torchao 0.10.0, so any `peft: lora` run dies at get_peft_model() before step
# 1 (cost one session, 2026-08-11). Nothing here uses torchao — it is a
# quantization backend — so uninstalling is the clean fix, and safer than
# upgrading it, which would drag torch with it.
pip uninstall -q -y torchao 2>/dev/null || true

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
print(f"transformers {transformers.__version__}")
assert torch.cuda.device_count() >= 1, "no GPU visible — check the accelerator setting"
assert transformers.__version__ == "4.46.1", (
    f"transformers is {transformers.__version__}, must be 4.46.1. Something "
    f"re-resolved it — check for a `pip install` without --no-deps."
)
from IndicTransToolkit import IndicProcessor, IndicDataCollator  # fails fast if the pin is wrong
print("IndicTransToolkit import OK")

# Prove the LoRA path can actually be constructed BEFORE the 4.5 GB download and
# the dataset build. get_peft_model() failing after all that costs a session.
import importlib.util
if importlib.util.find_spec("torchao") is not None:
    import sys as _s
    _s.exit("FATAL: torchao is still installed; peft's LoRA dispatcher will "
            "raise on it. The uninstall above did not take effect.")
from peft import LoraConfig, get_peft_model
print("peft LoRA path OK")
PY

# --- gated-repo preflight ------------------------------------------------------
# Gate acceptance on the Hub is PER-REPO and does not carry across a publisher's
# other repos — a token authorized for indictrans2-en-indic-1B still 403s on
# indictrans2-en-indic-dist-200M. Checking one small file costs a second here and
# saves discovering it after the dataset build and a partial model download.
CONFIG="$CONFIG" python - <<'PY'
import os, sys, yaml
from huggingface_hub import hf_hub_download
model = yaml.safe_load(open(os.environ["CONFIG"]))["model"]
try:
    hf_hub_download(model, "config.json", token=os.environ.get("HF_TOKEN"))
    print(f"gated-repo access OK: {model}")
except Exception as e:
    sys.exit(
        f"\nFATAL: cannot access {model}\n  {type(e).__name__}: {e}\n\n"
        f"Accept the terms at https://huggingface.co/{model} using the SAME "
        f"account as HF_TOKEN, then re-run. Acceptance is per-repo.\n"
    )
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
