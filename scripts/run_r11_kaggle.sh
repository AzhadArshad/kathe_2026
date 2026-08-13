#!/usr/bin/env bash
# KATHE 2026 — R11: learned character-level diacritic restoration, Kaggle GPU T4.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_r11_kaggle.sh
#
# What it does, in order:
#   1. trains the char tagger once per ARM — a data-volume / orthographic-
#      consistency sweep over the same corpus (all / clean / dense; see ARMS)
#   2. after each, per-mark P/R on R0 plus the TAIL PRECISION split
#   3. measures CPU wall-clock restoring the 1,730-row test decode
#   4. optionally publishes the restorer to the HF Hub (PUSH_HF=1, off by
#      default) — the RESTORER only, never the translation weights
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
#   3. ONLY if publishing (PUSH_HF=1): Internet ON and a Kaggle Secret
#      HF_TOKEN_WRITE. Never the read HF_TOKEN — see push_restorer_hf.py.
#
# Local step that produces the bundle (run once, before the session):
#   uv run python -m restore.build_text \
#     --source bpcc     data/processed/bpcc_kas_clean.jsonl \
#     --source external data/processed/external_nawabhussain.kas_Arab \
#     --source qamar30k "<HuggingFace 30K/Kashmiri.txt>" \
#     --exclude data/dev/r0/r0.kas_Arab \
#               data/processed/r3_corpus/dev/eng_Latn-kas_Arab/dev.kas_Arab \
#     --output  data/processed/restore_text.jsonl
#
# BPCC is passed as the .jsonl so each subcorpus keeps its own tag
# (`bpcc:daily`, `bpcc:nllb-seed`, ...). They are not one convention: nllb-seed
# and nllb-filtered mark at ~1.6/100c against 5.8-6.9 for daily and seed-v1.

set -euo pipefail

REPO="${REPO:-/kaggle/working/kathe_2026}"
BUNDLE="${BUNDLE:-/kaggle/input/kathe-r11-text}"
OUT="${OUT:-/kaggle/working/r11}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-20}"
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

# If publishing is on, check the credentials NOW rather than discovering after
# the first 50-minute arm that the secret was never set. Same reason the GPU
# assert is above: a config error should cost seconds, not an arm.
if [ "${PUSH_HF:-0}" = "1" ]; then
  if [ -z "${HF_TOKEN_WRITE:-}" ]; then
    echo "FATAL: PUSH_HF=1 but HF_TOKEN_WRITE is unset. Add the Kaggle Secret," >&2
    echo "       or run without PUSH_HF and download the weights by hand." >&2
    exit 1
  fi
  if [ "${HF_TOKEN_WRITE:-}" = "${HF_TOKEN:-__unset__}" ]; then
    echo "FATAL: HF_TOKEN_WRITE equals HF_TOKEN. They are kept separate so a" >&2
    echo "       leak of the read token cannot overwrite published weights." >&2
    exit 1
  fi
  python -c "import huggingface_hub" 2>/dev/null || {
    echo "FATAL: PUSH_HF=1 but huggingface_hub is not importable." >&2; exit 1; }
  echo "  publishing ENABLED -> ${HF_REPO:-Aju360/kathe-r11-restorer} (+ -nc), private"
fi

# data/ is gitignored, so the clone has none of these. Copy from the bundle.
cp "$BUNDLE/restore_text.jsonl" data/processed/
# The bundle ships these ALREADY scorer-normalized, so nothing here needs
# KashmiriNormalizer. Verified idempotent, and R0's references were already
# normalized (0 of 1,003 lines changed).
cp "$BUNDLE/r0.kas_Arab.norm" data/dev/r0/r0.kas_Arab
cp "$BUNDLE/test.hyp.r3-200m.txt" data/processed/
# The lexicon is needed only for the TAIL PRECISION split — the number this
# experiment turns on. It is a plain JSON dict; no scorer dependency.
cp "$BUNDLE/lexicon_clean_all.json" data/processed/

# ------------------------------------------------------------- the arms --
# Same seed, same hyperparameters, same held-out fraction. The ONLY difference
# between the runs is --sources, so a score difference is provenance.
#
# ARMS — a data-volume / orthographic-consistency sweep. R0 references sit at
# 4.68 restorable per 100 characters:
#
#   arm     tags  lines     marks    density  what it drops
#   all        8  168,689   679,734  4.44     nothing — the control
#   clean      5  145,716   637,261  4.86     nllb-seed, nllb-filtered, qamar30k
#   dense      3   58,467   352,506  6.73     ... and seed-v2, seed-latest
#   bpcc       6  112,236   438,539  4.03     the original 2026-08-13 control
#
# `all` answers "was 6 epochs simply too few?" — it is last run's data with a
# 20-epoch schedule. `clean` and `dense` answer "does a consistent convention
# beat more data?", at two strengths: `clean` is a 3-point shift and `dense` a
# 2.3x one, buying orthographic consistency by giving up 65% of the corpus.
#
# NOTE `dense` OVERSHOOTS the target (6.73 vs 4.68) by more than `all`
# undershoots it, and R0's own 4.68 is about two-thirds an artifact of how R0
# was sampled (PLANNING.md 2026-08-13), so "aim at 4.68" is not a well-founded
# objective. `dense` is here to bound the effect, not because 6.73 is right.
#
# Judge these on TAIL PRECISION — the per-mark precision on words the lexicon
# does not know, printed by the --refs evaluation. Micro-F1, held-out loss and
# output density all improve without the score following (results.md). If all
# three stall near the 6-epoch baseline of 34.7%, the constraint is the model's
# ability to generalise to unseen morphology and no reslicing of this corpus
# addresses it.
ARMS="${ARMS:-all clean dense}"
FAILED=""
PUSH_FAILED=""

# Validate every arm name BEFORE training anything. A typo in ARMS is a config
# error, and discovering it after 50 minutes of arm 1 — then aborting the arms
# that would have followed — is the worst of both behaviours.
for VARIANT in $ARMS; do
  case "$VARIANT" in
    all|bpcc|clean|dense) ;;
    *) echo "FATAL: unknown arm '$VARIANT' (want: all, bpcc, clean, dense)" >&2
       exit 1 ;;
  esac
done
echo "  arms to run:$( for a in $ARMS; do printf ' %s' "$a"; done )  (${EPOCHS} epochs each)"

for VARIANT in $ARMS; do
  # A `case` rather than `[ x = y ] && VAR=...`: under `set -e` the latter is a
  # failing AND-OR list on every non-matching arm, which survives only by a
  # POSIX exemption and is one edit away from aborting the run.
  case "$VARIANT" in
    all)   EXTRA="" ;;
    bpcc)  EXTRA="--sources bpcc" ;;
    clean) EXTRA="--sources bpcc:bpcc-seed-v2 bpcc:bpcc-seed-v1 bpcc:bpcc-seed-latest bpcc:daily external" ;;
    dense) EXTRA="--sources bpcc:daily bpcc:bpcc-seed-v1 external" ;;
    *)     echo "FATAL: unknown arm '$VARIANT' (want: all, bpcc, clean, dense)" >&2; exit 1 ;;
  esac
  # Each arm is INDEPENDENT and its results are copied out the moment it
  # finishes. A three-arm sweep is ~2 hours, which means it is run as a Kaggle
  # commit rather than interactively — and a commit that dies partway may not
  # preserve /kaggle/working at all. Letting `set -e` abort the script on arm 3
  # would therefore discard arms 1 and 2 as well. Failures are recorded and
  # reported at the end instead.
  if python -u -m restore.train fit \
      --corpus data/processed/restore_text.jsonl \
      --out "models/restore/r11_${VARIANT}.pt" \
      --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH" \
      --refs data/dev/r0/r0.kas_Arab --none-bias-sweep 0.0 -0.5 -1.0 -1.5 \
      --lexicon data/processed/lexicon_clean_all.json \
      $EXTRA 2>&1 | tee "$OUT/fit_${VARIANT}.log"; then
    # `*_last.pt` exists only when the best epoch was not the final one.
    cp models/restore/r11_${VARIANT}*.pt models/restore/r11_${VARIANT}*.meta.json "$OUT/"
    echo "  == arm '$VARIANT' OK; artifacts copied to $OUT"
    # Publish THIS arm now, not at the end. A three-arm sweep is ~2 hours and
    # is run as a commit; a commit that dies partway may preserve nothing from
    # /kaggle/working. R4 lost its save path after 3h20m and survived only
    # because checkpoints were already on the Hub (PLANNING.md 2026-08-12) —
    # the same reasoning applies once a run is long enough to be worth losing.
    if [ "${PUSH_HF:-0}" = "1" ]; then
      if python scripts/push_restorer_hf.py --checkpoints "$OUT" \
           --only "$VARIANT" --repo "${HF_REPO:-Aju360/kathe-r11-restorer}" \
           --yes 2>&1 | tee -a "$OUT/hf_push.log"; then
        echo "  == arm '$VARIANT' published"
      else
        PUSH_FAILED="$PUSH_FAILED $VARIANT"
        echo "  == arm '$VARIANT' upload FAILED (weights are safe in $OUT)" >&2
      fi
    fi
  else
    FAILED="$FAILED $VARIANT"
    echo "  == arm '$VARIANT' FAILED — continuing with the remaining arms." >&2
  fi
done

if [ -n "$FAILED" ]; then
  echo "WARNING: these arms failed:$FAILED (see $OUT/fit_*.log)" >&2
fi

# ------------------------------------------- 3. live-round wall-clock, 1,730 --
# The organizers run the deliverable on their hardware and no GPU is promised,
# so the number that matters is CPU, single process.
# Time the FIRST arm, whichever it is — hardcoding r11_all.pt breaks any run
# whose ARMS does not include "all".
# Written as a plain loop, NOT inside $( ). A `case` pattern's closing paren
# terminates a command substitution, so the one-liner form is a syntax error
# that `bash -n` does not catch and that only fires here — at the end of a
# two-hour run.
FIRST_ARM=""
for a in $ARMS; do
  case " $FAILED " in
    *" $a "*) ;;                       # this arm failed; skip it
    *) FIRST_ARM="$a"; break ;;
  esac
done
if [ -z "$FIRST_ARM" ]; then
  echo "FATAL: every arm failed; nothing to time." >&2
  exit 1
fi
CKPT="models/restore/r11_${FIRST_ARM}.pt" python - <<'PY' 2>&1 | tee "$OUT/wallclock.log"
import os, sys, time
sys.path.insert(0, "src")
from restore.chartag import Restorer
lines = [l.rstrip("\n") for l in open("data/processed/test.hyp.r3-200m.txt", encoding="utf-8")]
r = Restorer(os.environ["CKPT"], device="cpu")
print(f"checkpoint: {os.environ['CKPT']}")
t0 = time.time(); out = r.restore_many(lines); dt = time.time() - t0
print(f"CPU restore of {len(out):,} test rows: {dt:.2f}s ({1000*dt/len(out):.2f} ms/row)")
PY

# ------------------------------------------------- optional: publish to the Hub --
# OFF by default. Runs LAST, after every checkpoint is already on disk and
# copied into $OUT, and its failure is swallowed — a network or token problem
# must not be able to look like a training problem, or to cost you the weights.
#
# It publishes the RESTORER only. `scripts/push_restorer_hf.py` refuses to
# target the translation repos, requires HF_TOKEN_WRITE (never the read token),
# routes each checkpoint to an Apache-2.0 or CC-BY-NC-SA-4.0 repo according to
# the corpora that checkpoint records for ITSELF, and creates repos private.
#
# Enable with PUSH_HF=1, and set the Kaggle secret HF_TOKEN_WRITE. Needs
# Internet ON, which the rest of this script does not.
if [ "${PUSH_HF:-0}" = "1" ]; then
  if [ -n "$PUSH_FAILED" ]; then
    echo "=== retrying uploads that failed:$PUSH_FAILED ==="
    for VARIANT in $PUSH_FAILED; do
      python scripts/push_restorer_hf.py --checkpoints "$OUT" --only "$VARIANT" \
        --repo "${HF_REPO:-Aju360/kathe-r11-restorer}" --yes 2>&1 \
        | tee -a "$OUT/hf_push.log" \
        || echo "  STILL FAILING: '$VARIANT' — download $OUT by hand." >&2
    done
  else
    echo "  all arms published as they finished."
  fi
else
  echo "  (hub upload skipped; set PUSH_HF=1 and HF_TOKEN_WRITE to enable)"
fi

echo "=== R11 done. Download ${OUT}/ before closing the session. ==="
