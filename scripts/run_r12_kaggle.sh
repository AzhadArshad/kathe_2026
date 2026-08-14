#!/usr/bin/env bash
# KATHE 2026 — R12: 200M full fine-tune on a semantically selected corpus.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_r12_kaggle.sh
#
# TWO ARMS, and the second is what makes the first interpretable:
#
#   selected   upweight (x7) the 20,000 pairs nearest englishdev.csv
#   control    upweight (x7) a RANDOM 20,000 from the same pool
#
# Everything else is identical — same pool, same 3,000-pair dev set, same size
# (228,400 pairs), same repeat factor, same seed, same hyperparameters. So the
# difference between the arms is the retrieval and nothing else. Run `selected`
# alone and a score change cannot be attributed to semantic selection rather
# than to the corpus rebuild (raw BPCC instead of LaBSE-cleaned, mined data
# dropped, 228,400 pairs instead of R3's 123,538) or to upweighting per se.
# Attributing a change to the wrong cause has already cost this project twice.
#
# push_to_hub is ON. R4 trained for 3h20m and then crashed in the save path;
# every checkpoint survived only because it was already on the Hub
# (PLANNING.md 2026-08-12). At ~2h45m per arm this run is firmly in that
# category. Needs HF_TOKEN_WRITE — never the read token.
#
# PREREQUISITES on the Kaggle side
#   1. Accelerator = GPU T4 x2 (torchrun DDP over both cards).
#   2. Internet ON.
#   3. Kaggle Secrets: HF_TOKEN (read, for the gated base model) and
#      HF_TOKEN_WRITE (write, for pushing checkpoints). indictrans2-en-indic-
#      dist-200M is GATED — accept its terms with the SAME account first;
#      acceptance is per-repo and does not carry across AI4Bharat's other repos.
#   4. The corpus dataset attached (built locally by data.build_r12 +
#      data.select_r12; see the bundle).
#
# In the notebook, before this script:
#   from kaggle_secrets import UserSecretsClient
#   import os
#   os.environ["HF_TOKEN"]       = UserSecretsClient().get_secret("HF_TOKEN")
#   os.environ["HF_TOKEN_WRITE"] = UserSecretsClient().get_secret("HF_TOKEN_WRITE")

set -euo pipefail

# The T4 OOM'd at step 0 on R3 with a 2.69 GiB request while 2.56 GiB was free,
# so fragmentation was part of it. expandable_segments lets the allocator grow a
# segment rather than needing one contiguous block.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

REPO="${REPO:-/kaggle/working/kathe_2026}"
BUNDLE="${BUNDLE:-/kaggle/input/kathe-r12-corpus}"
OUT="${OUT:-/kaggle/working/r12}"
ARMS="${ARMS:-selected control}"
# NOTE: finetune.py takes --config/--data-dir/--output-dir/--init-from/--resume
# /--dry-run and nothing else. Epochs live in the YAML; change them there, not
# with a flag that does not exist.

# Locate the bundle by its contents. Kaggle mounts a dataset at
# /kaggle/input/<slug>, but the path picks up an owner prefix in some views and
# moves again if the dataset is copied by hand — both happened on R11.
if [ ! -d "$BUNDLE/r12_corpus" ]; then
  FOUND="$(find /kaggle/input /kaggle/working -maxdepth 5 -type d -name r12_corpus -print -quit 2>/dev/null || true)"
  if [ -n "$FOUND" ]; then
    BUNDLE="$(dirname "$FOUND")"
    echo "  bundle not at the default path; found it at $BUNDLE"
  else
    echo "FATAL: no r12_corpus/ found. Attach the kathe-r12-corpus dataset." >&2
    exit 1
  fi
fi
if [ ! -d "$REPO/src" ] && [ -d "$REPO/code/src" ]; then REPO="$REPO/code"; fi
if [ ! -d "$REPO/src" ]; then
  echo "  no clone at $REPO — using the copy shipped in $BUNDLE/code"
  mkdir -p "$REPO"; cp -r "$BUNDLE/code/." "$REPO/"
fi

cd "$REPO"
mkdir -p "$OUT" data/processed
cp -r "$BUNDLE/r12_corpus" "$BUNDLE/r12_control" data/processed/

# Validate the arm names BEFORE training. A typo discovered after the first
# 2h45m arm, which then aborts the rest, is the worst of both behaviours.
for A in $ARMS; do
  case "$A" in
    selected|control) ;;
    *) echo "FATAL: unknown arm '$A' (want: selected, control)" >&2; exit 1 ;;
  esac
done

# --- environment -------------------------------------------------------------
# NO editable install. It existed only to put src/ on the path, and it FAILS in
# the bundle: pyproject declares `readme = "README.md"` and
# `license-files = ["LICENSE"]`, neither of which ships here, so hatchling dies
# with "Preparing editable metadata (pyproject.toml) did not run successfully".
# PYTHONPATH does the same job with nothing to break, and it also sidesteps the
# older trap where `pip install -e .` silently upgraded transformers over the
# 4.46.1 pin one line after it was set (PLANNING.md 2026-08-10).
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# EXACTLY the list from run_r3_kaggle.sh — the runner that has actually driven
# this trainer to completion. Trimming it is how this run failed twice: first on
# `pip install -e .` (pyproject wants README.md and LICENSE, neither of which
# ships in the bundle), then on a missing KashmiriNormalizer. Do not shorten it
# again. huggingface-hub is pinned because transformers 4.46.1 requires <1.0
# while pyproject wants >=1.26.1, and a stale newer hub leaves an incompatible
# pair.
pip install -q \
    "transformers==4.46.1" \
    "huggingface-hub<1.0" \
    "indictranstoolkit==1.1.1" \
    "sacrebleu==2.6.0" \
    "KashmiriNormalizer==0.1.0" \
    "peft" "accelerate" "datasets" "sentencepiece" "pyyaml"

# REMOVE torchao. peft's LoRA dispatcher calls is_torchao_available(), which
# RAISES ImportError when torchao is installed but older than 0.16.0 — it only
# returns False when absent entirely. Kaggle ships 0.10.0. Cost one session
# (PLANNING.md 2026-08-11).
pip uninstall -y -q torchao 2>/dev/null || true

# PREFLIGHT 1 — import everything the training path touches, before any GPU time.
python - <<'PYEOF'
import torch, transformers
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}")
assert transformers.__version__ == "4.46.1", (
    f"transformers is {transformers.__version__}, must be 4.46.1 — something "
    f"re-resolved it; check for a pip install without --no-deps.")
assert torch.cuda.device_count() >= 2, "torchrun --nproc_per_node=2 needs T4 x2"
for m in ("KashmiriNormalizer", "IndicTransToolkit", "datasets", "peft",
          "accelerate", "sacrebleu", "sentencepiece", "yaml", "safetensors",
          "pandas", "huggingface_hub"):
    __import__(m)
print("  all third-party imports OK")
PYEOF

if [ -z "${HF_TOKEN_WRITE:-}" ]; then
  echo "FATAL: HF_TOKEN_WRITE unset but push_to_hub is on in the configs." >&2
  echo "       Add the Kaggle Secret, or set push_to_hub: false." >&2
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-}"

# PREFLIGHT 2 — `--dry-run` builds the datasets and exits before training,
# exercising the corpus paths, the tokenizer, IndicProcessor and the collator.
# Every arm is proved to load before ANY of them trains, so a config or data
# fault costs seconds instead of the first arm's two hours.
for ARM in $ARMS; do
  echo "  preflight (dry-run): $ARM"
  if ! python -m train.finetune --config "config/r12_200m_${ARM}.yaml" --dry-run; then
    echo "FATAL: dry-run failed for '$ARM'. Nothing trained, no GPU time spent." >&2
    exit 1
  fi
done
echo "  preflight passed for all arms — starting training"

# --- the arms ----------------------------------------------------------------
FAILED=""
for ARM in $ARMS; do
  echo "=== ARM: $ARM ==="
  if HF_TOKEN="$HF_TOKEN_WRITE" torchrun --nproc_per_node=2 -m train.finetune \
       --config "config/r12_200m_${ARM}.yaml" 2>&1 | tee "$OUT/train_${ARM}.log"; then
    cp -r "output/r12-${ARM}" "$OUT/" 2>/dev/null || true
    echo "  == arm '$ARM' OK"
  else
    FAILED="$FAILED $ARM"
    echo "  == arm '$ARM' FAILED — continuing so the other arm is not lost" >&2
  fi
done
# An `[ -n "$X" ] && echo` here would return 1 when nothing failed and, being
# the last command of an AND-OR list under `set -e`, abort the script right
# before the final messages. Same trap as R11.
if [ -n "$FAILED" ]; then
  echo "WARNING: failed arms:$FAILED" >&2
fi

echo "=== R12 done. Checkpoints are on the Hub AND in ${OUT}/. ==="
echo "    Score them locally — the pinned metric stack lives there:"
echo "      python scripts/translate.py --input data/dev/r0/r0.eng_Latn --model <ckpt> ..."
