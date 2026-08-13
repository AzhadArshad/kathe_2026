#!/usr/bin/env bash
# KATHE 2026 — R11: learned character-level diacritic restoration, Kaggle GPU T4.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_r11_kaggle.sh
#
# What it does, in order:
#   1. trains the char tagger on ALL THREE corpora
#   2. trains a second one on BPCC ONLY — the provenance control
#   3. measures CPU wall-clock restoring the 1,730-row test decode
#
# WHAT IT DELIBERATELY DOES NOT DO: score anything against the competition
# metric. That needs `KashmiriNormalizer==0.1.0` and `sacrebleu==2.6.0`, which
# are not on the Kaggle image, and installing them here would put the metric on
# an unpinned second footing — the one thing PROJECT_NOTES.md §5 says must be
# identical everywhere. Restoration is CPU post-processing, so the whole R0
# comparison table runs locally in seconds off the downloaded .pt:
#
#   uv run python scripts/eval_restore.py --hyp data/dev/r0/r0.hyp.r3-200m ...
#
# The per-mark precision/recall printed by `restore.train fit --refs` needs no
# scorer and is reported here.
#
# The model is 3.3M parameters. On one T4 an epoch is a few minutes, so the
# whole script is well under an hour — it does not need two cards, is not
# distributed, and downloads no pretrained weights. If the session dies,
# restarting costs minutes rather than the hours R3/R4 had at stake, which is
# why there is no push_to_hub wiring here. Download the .pt files (13 MB each)
# from /kaggle/working before closing the session.
#
# THE TRAINING TEXT IS BUILT LOCALLY AND UPLOADED, not built here. Two reasons:
#   * `SMUQamar/Kashmiri-English-Parallel-Corpus` is gated MANUALLY, not by
#     click-through. A gate that has not been granted fails the session rather
#     than the command (PROJECT_NOTES.md §5), and this one cannot be fixed in the
#     moment by accepting terms.
#   * `restore_text.jsonl` is pure CPU string work and takes about a minute on
#     a laptop. Paying T4 time for it is waste.
#
# PREREQUISITES on the Kaggle side
#   1. Accelerator = GPU T4 (x2 is fine; only one card is used).
#   2. A Kaggle Dataset holding data/processed/r11_kaggle_bundle/, which
#      `python -m restore.build_text` produces locally. Attach it and point
#      BUNDLE at it. No HF_TOKEN and no internet are needed by this script.
#
# Local step that produces the bundle (run once, before the session):
#   uv run python -m restore.build_text \
#     --source bpcc     data/processed/r3_corpus/train/eng_Latn-kas_Arab/train.kas_Arab \
#     --source external data/processed/external_nawabhussain.kas_Arab \
#     --source qamar30k "<HuggingFace 30K/Kashmiri.txt>" \
#     --exclude data/dev/r0/r0.kas_Arab \
#               data/processed/r3_corpus/dev/eng_Latn-kas_Arab/dev.kas_Arab \
#     --output  data/processed/restore_text.jsonl

set -euo pipefail

REPO="${REPO:-/kaggle/working/kathe_2026}"
BUNDLE="${BUNDLE:-/kaggle/input/kathe-r11-text}"
OUT="${OUT:-/kaggle/working/r11}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-64}"

# Locate the bundle by its CONTENTS, not by one hard-coded mount point. Kaggle
# mounts a dataset at /kaggle/input/<slug>, but the path picks up an owner
# prefix in some views and moves again if the dataset is copied by hand — both
# of which happened on the first real run. Searching for the one file that must
# be there is cheap and cannot be wrong.
if [ ! -f "$BUNDLE/restore_text.jsonl" ]; then
  # Search beside REPO first — copying the code and the data together is the
  # normal thing to do and is what happened on the first real run.
  FOUND="$(find "$REPO" "$(dirname "$REPO")" /kaggle/input /kaggle/working \
             -maxdepth 5 -name restore_text.jsonl -print -quit 2>/dev/null || true)"
  if [ -n "$FOUND" ]; then
    BUNDLE="$(dirname "$FOUND")"
    echo "  bundle not at the default path; found it at $BUNDLE"
  else
    echo "FATAL: no restore_text.jsonl found. Attach the kathe-r11-text" >&2
    echo "       dataset, or set BUNDLE=<path containing restore_text.jsonl>." >&2
    exit 1
  fi
fi

# Likewise the code may be at $REPO/src or one level down at $REPO/code/src,
# depending on whether the bundle's code/ folder was copied or its contents.
if [ ! -d "$REPO/src" ] && [ -d "$REPO/code/src" ]; then
  REPO="$REPO/code"
  echo "  code is one level down; using REPO=$REPO"
fi

# The repo is PRIVATE, so the bundle ships a copy of src/ and scripts/ as well
# and no GitHub token is needed. If REPO is already a clone this is a no-op; if
# it is not, materialise it from the bundle. Either way the code has to end up
# under /kaggle/working, because /kaggle/input is read-only and this script
# writes models/ and experiments/.
if [ ! -d "$REPO/src" ]; then
  echo "  no clone at $REPO — using the copy shipped in $BUNDLE/code"
  mkdir -p "$REPO"
  cp -r "$BUNDLE/code/." "$REPO/"
fi

cd "$REPO"
mkdir -p "$OUT" models/restore experiments/r11-restore data/dev/r0 data/processed

# NO pip install. R11 imports torch and the standard library and nothing else —
# not transformers, not IndicTransToolkit, not the Hub. `pip install -e .` would
# only put src/ on the path, and PYTHONPATH does that without touching the
# environment. Every previous runner needed the editable install and each one
# needed `--no-deps` to stop pyproject's `transformers>=5` from silently
# upgrading over the 4.46.1 pin (PLANNING.md 2026-08-10). Not installing at all
# removes that failure mode instead of guarding it.
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Fail here rather than 40 minutes into a CPU run that looked like it was
# working. `REQUIRE_GPU=0` is for dry-running the script off Kaggle.
REQUIRE_GPU="${REQUIRE_GPU:-1}" python - <<'PY'
import os, torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if os.environ["REQUIRE_GPU"] == "1":
    assert torch.cuda.is_available(), (
        "no GPU — set the accelerator to T4, or pass REQUIRE_GPU=0 for a dry run")
PY

# data/ is gitignored, so the clone has none of these. Copy from the bundle.
cp "$BUNDLE/restore_text.jsonl" data/processed/
# The bundle ships these ALREADY scorer-normalized, so nothing here needs
# KashmiriNormalizer. Verified idempotent, and R0's references were already
# normalized (0 of 1,003 lines changed).
cp "$BUNDLE/r0.kas_Arab.norm" data/dev/r0/r0.kas_Arab
cp "$BUNDLE/test.hyp.r3-200m.txt" data/processed/

# --------------------------------------------------------- 1/2. two models --
# Same seed, same hyperparameters, same held-out fraction. The ONLY difference
# between the two runs is --sources, so a score difference is provenance.
for VARIANT in all bpcc; do
  EXTRA=""
  [ "$VARIANT" = "bpcc" ] && EXTRA="--sources bpcc"
  python -u -m restore.train fit \
    --corpus data/processed/restore_text.jsonl \
    --out "models/restore/r11_${VARIANT}.pt" \
    --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH" \
    --refs data/dev/r0/r0.kas_Arab --none-bias-sweep 0.0 -0.5 -1.0 -1.5 \
    $EXTRA 2>&1 | tee "$OUT/fit_${VARIANT}.log"
  cp "models/restore/r11_${VARIANT}.pt" "models/restore/r11_${VARIANT}.meta.json" "$OUT/"
done

# ------------------------------------------- 3. live-round wall-clock, 1,730 --
# The organizers run the deliverable on their hardware and no GPU is promised,
# so the number that matters is CPU, single process.
python - <<'PY' 2>&1 | tee "$OUT/wallclock.log"
import sys, time
sys.path.insert(0, "src")
from restore.chartag import Restorer
lines = [l.rstrip("\n") for l in open("data/processed/test.hyp.r3-200m.txt", encoding="utf-8")]
r = Restorer("models/restore/r11_all.pt", device="cpu")
t0 = time.time(); out = r.restore_many(lines); dt = time.time() - t0
print(f"CPU restore of {len(out):,} test rows: {dt:.2f}s ({1000*dt/len(out):.2f} ms/row)")
PY

echo "=== R11 done. Download ${OUT}/ before closing the session. ==="
