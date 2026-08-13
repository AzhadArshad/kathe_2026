#!/usr/bin/env bash
# KATHE 2026 — R5: MBR decoding on Kaggle, GPU T4.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_r5_kaggle.sh
#
# What it does, in order:
#   1. R0 pools from both systems (the R3 200M fine-tune, the 1B zero-shot)
#   2. single-system MBR and cross-system MBR, scored on R0 raw
#   3. the same two, scored again AFTER diacritic restoration
#   4. the test-set decode for whichever configuration won
#
# The pools are written to /kaggle/working and are the expensive artifact —
# pool size, sampling method and utility settings can all be re-swept offline
# from them with `decode.mbr select`, with no second GPU pass. Download them.
#
# PREREQUISITES on the Kaggle side
#   1. Accelerator = GPU T4 x2 (only one card is used; MBR is not distributed).
#   2. Internet ON.
#   3. Kaggle Secret HF_TOKEN. indictrans2-en-indic-1B is GATED — accept the
#      terms on the Hub with the SAME account first. Acceptance is per-repo.
#   4. The R3 checkpoint attached, either as a Kaggle Dataset or pulled from
#      Aju360/kathe-r3-200m-full (private — needs the token).
#
# In the notebook, before this script:
#   from kaggle_secrets import UserSecretsClient
#   import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

set -euo pipefail

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

REPO="${REPO:-/kaggle/working/kathe_2026}"
R3="${R3:-Aju360/kathe-r3-200m-full}"
OUT="${OUT:-/kaggle/working/r5}"
LEXICON="${LEXICON:-data/processed/diacritic_lexicon_both.json}"
POSTPROC="${POSTPROC:-config/postproc_r3_both.yaml}"

cd "$REPO"
mkdir -p "$OUT" experiments/r5-mbr

# --- environment -------------------------------------------------------------
# Identical pins to run_r3_kaggle.sh, and for the same reasons. --no-deps is
# load-bearing: pyproject declares transformers>=5.14.1 for the data
# environment, and a plain editable install silently upgrades past 4.46.1,
# breaking IndicTransToolkit's import (cost a session, 2026-08-10).
pip install -q --no-deps -e .
pip install -q \
    "transformers==4.46.1" \
    "huggingface-hub<1.0" \
    "indictranstoolkit==1.1.1" \
    "sacrebleu==2.6.0" \
    "KashmiriNormalizer==0.1.0" \
    "accelerate" "sentencepiece" "pyyaml"

python - <<'PY'
import torch, transformers, inspect
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "no GPU visible — check the accelerator setting"
assert transformers.__version__ == "4.46.1", (
    f"transformers is {transformers.__version__}, must be 4.46.1 — check for a "
    f"pip install without --no-deps.")
from IndicTransToolkit import IndicProcessor
print("IndicTransToolkit import OK")

# epsilon sampling is the configured default. Fail here, in seconds, rather
# than after a 40-minute decode that silently ignored the kwarg.
from transformers import GenerationConfig
assert hasattr(GenerationConfig(), "epsilon_cutoff"), (
    "this transformers build has no epsilon_cutoff — set `sampling: diverse_beam` "
    "in the MBR config (it is the documented fallback) and re-run.")
print("epsilon sampling supported OK")
PY

# --- gated-repo preflight ----------------------------------------------------
python - <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
for repo in ("ai4bharat/indictrans2-en-indic-1B", os.environ.get("R3", "Aju360/kathe-r3-200m-full")):
    try:
        hf_hub_download(repo, "config.json", token=os.environ.get("HF_TOKEN"))
        print(f"repo access OK: {repo}")
    except Exception as e:
        sys.exit(f"\nFATAL: cannot access {repo}\n  {type(e).__name__}: {e}\n"
                 f"Accept the terms at https://huggingface.co/{repo} with the SAME "
                 f"account as HF_TOKEN. Acceptance is per-repo.\n")
PY

# --- 1. candidate pools on R0 -------------------------------------------------
echo "=== POOL: R3 200M on R0 ==="
python -m decode.mbr generate \
    --config config/mbr_r5_200m.yaml --model "$R3" \
    --input data/dev/r0/r0.eng_Latn \
    --output "$OUT/pool.r0.200m.jsonl"

echo "=== POOL: 1B zero-shot on R0 ==="
python -m decode.mbr generate \
    --config config/mbr_r5_1b.yaml \
    --input data/dev/r0/r0.eng_Latn \
    --output "$OUT/pool.r0.1b.jsonl"

# --- 2. selection -------------------------------------------------------------
echo "=== MBR SELECT: single system ==="
python -m decode.mbr select \
    --config config/mbr_r5_200m.yaml \
    --candidates "$OUT/pool.r0.200m.jsonl" \
    --output "$OUT/r0.hyp.mbr-200m" \
    --stats experiments/r5-mbr/select_r0_200m.json

echo "=== MBR SELECT: cross-system ==="
python -m decode.mbr select \
    --config config/mbr_r5_cross.yaml \
    --candidates "$OUT/pool.r0.200m.jsonl" "$OUT/pool.r0.1b.jsonl" \
    --output "$OUT/r0.hyp.mbr-cross" \
    --stats experiments/r5-mbr/select_r0_cross.json

# --- 3. score, raw and restored ----------------------------------------------
# Restoration is applied AFTER selection, never before: it is a fixed transform
# that would otherwise compress the very differences the utility measures.
for NAME in mbr-200m mbr-cross; do
    python -m data.diacritize apply \
        --lexicon "$LEXICON" \
        --input "$OUT/r0.hyp.$NAME" --output "$OUT/r0.hyp.$NAME.diac"

    python -m eval.score --hyp "$OUT/r0.hyp.$NAME" \
        --ref data/dev/r0/r0.kas_Arab --src data/dev/r0/r0.eng_Latn \
        --name "R0 — $NAME, RAW" --json "experiments/r5-mbr/r0_${NAME}_raw.json"
    python -m eval.score --hyp "$OUT/r0.hyp.$NAME.diac" \
        --ref data/dev/r0/r0.kas_Arab --src data/dev/r0/r0.eng_Latn \
        --name "R0 — $NAME + diacritics" --json "experiments/r5-mbr/r0_${NAME}_diac.json"
done

# Baseline for comparison: sub 007's beam-5 decode, same restoration.
python -m data.diacritize apply --lexicon "$LEXICON" \
    --input data/dev/r0/r0.hyp.r3-200m --output "$OUT/r0.hyp.beam.diac"
python -m eval.score --hyp "$OUT/r0.hyp.beam.diac" \
    --ref data/dev/r0/r0.kas_Arab --src data/dev/r0/r0.eng_Latn \
    --name "R0 — sub 007 baseline (beam 5 + diacritics)" \
    --json experiments/r5-mbr/r0_beam_diac.json

# --- 4. diacritic density -----------------------------------------------------
# A sharp move AWAY from the R0 reference density of 9.63 per 100 chars is the
# warning sign that was missed on submission 005 (PLANNING.md 2026-08-12).
for NAME in beam mbr-200m mbr-cross; do
    echo "--- density: $NAME ---"
    python scripts/orthography_diagnostic.py \
        --refs data/dev/r0/r0.kas_Arab --hyps "$OUT/r0.hyp.$NAME.diac"
done

echo "=== DONE — pools and scores in $OUT and experiments/r5-mbr/ ==="
echo "Download $OUT/pool.r0.*.jsonl before the session ends: they are the"
echo "expensive artifact and every later sweep is a CPU-only re-select over them."
ls -la "$OUT"
