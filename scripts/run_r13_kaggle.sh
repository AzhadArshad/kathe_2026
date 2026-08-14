#!/usr/bin/env bash
# KATHE 2026 — R13: 200M full fine-tune, AGGRESSIVE semantic upweighting.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_r13_kaggle.sh
#
# TWO ARMS, both turning the aggression knob that R12 barely touched.
#
#   hard    the SAME 20,000-pair slice as R12-selected, upweighted x7 (60% of
#           the mix) instead of x3 (40%). Single variable against sub 011's
#           13.52 — width, queries, dev set, seed and hyperparameters all
#           identical. Isolates WEIGHT.
#   sharp   a NARROWER 5,000-pair slice (per-query cap 10 instead of 25) at
#           x31, holding the 60% share. Each selected pair is seen 31 times per
#           epoch, 186 times over 6 epochs. Isolates WIDTH at fixed weight-share
#           once `hard` has priced the share change.
#
# WHY THIS IS NOT A REPEAT OF R12. The shipped R12 corpus was x3 / 40.4% share
# / 148,400 pairs — NOT the x7 / 61.3% / 228,400 that run_r12_kaggle.sh's header
# and one build log claim. Those describe a build that was discarded; the
# manifests and the 148,400-line train file are authoritative (PLANNING.md
# 2026-08-14). So x7 has been BUILT before but never TRAINED, and the +1.03 that
# semantic selection earned came from the gentlest setting available.
#
# RUNTIME AND WHY 4 EPOCHS, NOT 6. R12 trained 148,400 pairs x 6 epochs in
# ~2h45m per arm = 890,400 sample-passes. These arms hold more pairs, so 6
# epochs would hand them 1.37M (hard) and 1.55M (sharp) — 54-74% MORE training
# than the run they are compared against, and any gain would be unattributable
# between reweighting and compute. The R11 `dense` arm was confounded exactly
# this way. At 4 epochs the budgets match:
#   hard    228,400 x 4 =   913,600 passes (+2.6% vs R12)  -> ~2h50m
#   sharp   258,400 x 4 = 1,033,600 passes (+16%  vs R12)  -> ~3h11m
# Both fit one Kaggle quota (~6h00m of training). Run hard first — it is the
# clean single-variable test — but both in one session is now feasible:
#   bash scripts/run_r13_kaggle.sh              # both, ~6h
#   ARMS=hard bash scripts/run_r13_kaggle.sh    # one at a time
#
# OVERFITTING IS THE POINT, not a side effect. At x31 the sharp arm sees 5,000
# pairs 186 times. Watch eval_geo_proxy for collapse and check the decode for
# degenerate repetition before trusting any checkpoint — but a dev-set metric
# getting worse while the leaderboard improves is an EXPECTED outcome here, not
# a reason to abort. R0 has been anti-predictive all project.
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
BUNDLE="${BUNDLE:-/kaggle/input/kathe-r13-corpus}"
OUT="${OUT:-/kaggle/working/r13}"
ARMS="${ARMS:-}"   # empty = auto-detect from the corpora present in the bundle
# NOTE: finetune.py takes --config/--data-dir/--output-dir/--init-from/--resume
# /--dry-run and nothing else. Epochs live in the YAML; change them there, not
# with a flag that does not exist.

# Locate the bundle by its contents. Kaggle mounts a dataset at
# /kaggle/input/<slug>, but the path picks up an owner prefix in some views and
# moves again if the dataset is copied by hand — both happened on R11.
# Look for EITHER arm's corpus: a bundle may legitimately ship only the arm
# still to be run. Requiring r13_hard specifically made a sharp-only re-upload
# fail at discovery on 2026-08-14.
if [ ! -d "$BUNDLE/r13_hard" ] && [ ! -d "$BUNDLE/r13_sharp" ]; then
  FOUND="$(find /kaggle/input /kaggle/working -maxdepth 5 -type d \
             \( -name r13_hard -o -name r13_sharp \) -print -quit 2>/dev/null || true)"
  if [ -n "$FOUND" ]; then
    BUNDLE="$(dirname "$FOUND")"
    echo "  bundle not at the default path; found it at $BUNDLE"
  else
    echo "FATAL: no r13_hard/ or r13_sharp/ found. Attach the corpus dataset." >&2
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
# Copy only the corpora that are actually present.
for C in r13_hard r13_sharp; do
  [ -d "$BUNDLE/$C" ] && cp -r "$BUNDLE/$C" data/processed/ || true
done

# Default to whatever corpora the bundle actually ships, so a single-arm
# re-upload needs no ARMS override and cannot fail on a missing sibling.
if [ -z "$ARMS" ]; then
  for C in hard sharp; do
    [ -d "data/processed/r13_$C" ] && ARMS="$ARMS $C"
  done
  ARMS="$(echo $ARMS)"
  echo "  ARMS not set — running what the bundle ships: $ARMS"
fi
if [ -z "$ARMS" ]; then echo "FATAL: no arm corpora in the bundle." >&2; exit 1; fi

# Validate the arm names BEFORE training. A typo discovered after the first
# 2h45m arm, which then aborts the rest, is the worst of both behaviours.
for A in $ARMS; do
  case "$A" in
    hard|sharp)
      if [ ! -d "data/processed/r13_$A" ]; then
        echo "FATAL: arm '$A' requested but data/processed/r13_$A is not in the" >&2
        echo "       bundle. Set ARMS to the arm(s) you actually shipped." >&2
        exit 1
      fi ;;
    *) echo "FATAL: unknown arm '$A' (want: hard, sharp)" >&2; exit 1 ;;
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
  if ! python -m train.finetune --config "config/r13_200m_${ARM}.yaml" --dry-run; then
    echo "FATAL: dry-run failed for '$ARM'. Nothing trained, no GPU time spent." >&2
    exit 1
  fi
done
echo "  preflight passed for all arms — starting training"

# --- the arms ----------------------------------------------------------------
FAILED=""
for ARM in $ARMS; do
  echo "=== ARM: $ARM ==="
  export ARM
  df -h /kaggle/working | tail -1 | awk '{print "  disk before "ENVIRON["ARM"]": "$4" free of "$2}'
  if HF_TOKEN="$HF_TOKEN_WRITE" torchrun --nproc_per_node=2 -m train.finetune \
       --config "config/r13_200m_${ARM}.yaml" 2>&1 | tee "$OUT/train_${ARM}.log"; then
    # DISK, NOT GPU, IS WHAT KILLED THE SECOND ARM ON 2026-08-14. A 200M
    # checkpoint directory is ~2.7 GB (0.9 GB weights + 1.8 GB optimizer +
    # scheduler state). The old line here was `cp -r output/r13-$ARM "$OUT/"`,
    # which DUPLICATED every one of them into /kaggle/working — so arm 1 left
    # ~12 GB behind and arm 2 died at "tried to use more disk space than is
    # available" before its first save.
    #
    # Optimizer state is only needed to RESUME. Every checkpoint is already on
    # the Hub, and what the next stage needs is the portable final model, which
    # is a few hundred MB. So: drop the checkpoint dirs, copy only the model.
    rm -rf "output/r13-${ARM}"/checkpoint-*
    mkdir -p "$OUT/r13-${ARM}"
    find "output/r13-${ARM}" -maxdepth 1 -type f -exec cp {} "$OUT/r13-${ARM}/" \; 2>/dev/null || true
    echo "  == arm '$ARM' OK  (checkpoint dirs purged; model kept in $OUT/r13-${ARM})"
    du -sh "$OUT/r13-${ARM}" 2>/dev/null | awk '{print "     kept: "$1}'
    df -h /kaggle/working | tail -1 | awk '{print "     disk now: "$4" free"}'
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

echo "=== R13 done. Checkpoints are on the Hub AND in ${OUT}/. ==="
echo "    Score them locally — the pinned metric stack lives there:"
echo "      python scripts/translate.py --input data/dev/r0/r0.eng_Latn --model <ckpt> ..."
