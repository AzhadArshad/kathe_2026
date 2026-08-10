#!/usr/bin/env bash
# KATHE 2026 — Q9: zero-shot IT2-1B decode of the register-matched dev sets.
#
# Run from a Kaggle Notebook cell as:  !bash scripts/run_q9_kaggle.sh
#
# WHAT THIS ANSWERS
#   How much of the 1.98x FLORES->leaderboard gap is BLEU arithmetic on short
#   sentences, and how much is genuine domain mismatch (PLANNING.md, Q9).
#
#   The same system -- IndicTrans2-1B, zero-shot, beam 5 -- already has two
#   numbers: 15.83 on FLORES devtest (21.6 words/line) and 8.00 on the
#   leaderboard (7.3 words/line). R0 is a third point: short like the
#   leaderboard, but BPCC-domain like nothing we have scored yet.
#
#       FLORES 15.83  --[length + domain change]-->  R0 ?  --[domain]-->  LB 8.00
#
#   Reporting BLEU and chrF++ separately matters here. chrF++ is far less
#   sensitive to sentence length than 4-gram BLEU, so a FLORES->R0 drop
#   concentrated in BLEU points at arithmetic, while both falling together
#   points at quality.
#
# WHY THIS RUNS ON KAGGLE
#   Decoding 1B at beam 5 on the 8 GB M2 Air measured 0.08 sent/s on CPU
#   (~3.5 h for R0), and MPS at batch 32 exhausted unified memory and hung the
#   machine. On a T4 this is a couple of minutes.
#
#   It doubles as the first real test of scripts/translate.py -- the live-round
#   deliverable -- on GPU, which is the hardware the organizers will use.
#
# PREREQUISITES on the Kaggle side
#   1. Accelerator = GPU T4 x2 (one card is enough for this; the script uses
#      whichever is visible).
#   2. Internet ON.
#   3. Kaggle Secret `HF_TOKEN`. ai4bharat/indictrans2-en-indic-1B is GATED --
#      accept the terms on the Hub first or the download fails misleadingly.
#   4. The dev sets attached as a Kaggle Dataset, or the repo cloned with
#      data/dev/ present. Build them locally with:
#          python -m data.build_devsets --pairs data/processed/bpcc_kas_clean.jsonl \
#              --test-csv data/raw/englishdev.csv --out data/dev \
#              --dev-kas data/dev/flores_kas.txt --dev-en data/dev/flores_en.txt
#
# In the notebook, before this script:
#   from kaggle_secrets import UserSecretsClient
#   import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

set -euo pipefail

REPO="${REPO:-/kaggle/working/kathe_2026}"
DEVDIR="${DEVDIR:-$REPO/data/dev}"
OUTPUT="${OUTPUT:-/kaggle/working/q9}"
MODEL="${MODEL:-ai4bharat/indictrans2-en-indic-1B}"
BEAM="${BEAM:-5}"
BATCH="${BATCH:-32}"

cd "$REPO"

# --- environment -------------------------------------------------------------
# transformers MUST stay at 4.46.1. Newer releases drop the
# PreTrainedTokenizerBase re-export IndicTransToolkit imports at PACKAGE import
# time, so this breaks inference too, not only training (verified locally
# against transformers 5.14.1: `from IndicTransToolkit.processor import
# IndicProcessor` raises ImportError via __init__.py).
# --no-deps is LOAD-BEARING on both lines.
#
# pyproject.toml declares transformers>=5.14.1 (the data/scoring env). A plain
# `pip install -e .` therefore UPGRADES transformers straight back to 5.x and
# breaks IndicTransToolkit's import, undoing the pin installed one line above.
# It also drags torch to a newer build than the Kaggle image's, desynchronizing
# torchvision/torchaudio and the CUDA libs. We only need the editable install
# to put src/ on the path; every runtime dependency is named explicitly here.
pip install -q --no-deps -e .

# huggingface-hub is pinned explicitly because transformers 4.46.1 requires
# <1.0 while pyproject wants >=1.26.1 — if a previous cell already pulled the
# newer hub, installing transformers alone can leave an incompatible pair.
pip install -q \
    "transformers==4.46.1" \
    "huggingface-hub<1.0" \
    "indictranstoolkit==1.1.1" \
    "sacrebleu==2.6.0" \
    "KashmiriNormalizer==0.1.0" \
    "sentencepiece"

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
assert torch.cuda.is_available(), "no GPU visible — check the accelerator setting"
# Fail here, with a clear reason, rather than 200 lines later inside an import.
assert transformers.__version__ == "4.46.1", (
    f"transformers is {transformers.__version__}, must be 4.46.1. Something "
    f"re-resolved it — check for a `pip install` without --no-deps."
)
from IndicTransToolkit.processor import IndicProcessor
print("IndicProcessor import OK")
PY

mkdir -p "$OUTPUT"

# --- decode + score ----------------------------------------------------------
# Beam 5 to match the R1 baseline exactly. Anything else and the comparison to
# 15.83 / 8.00 is not a comparison.
for SET in r0 r3_eval; do
    echo "=== DECODE $SET ==="
    python scripts/translate.py \
        --input  "$DEVDIR/$SET/$SET.eng_Latn" \
        --output "$OUTPUT/$SET.hyp.zeroshot-1b" \
        --model "$MODEL" --device cuda --beam "$BEAM" --batch-size "$BATCH"

    echo "=== SCORE $SET ==="
    python -m eval.score \
        --hyp "$OUTPUT/$SET.hyp.zeroshot-1b" \
        --ref "$DEVDIR/$SET/$SET.kas_Arab" \
        --src "$DEVDIR/$SET/$SET.eng_Latn" \
        --name "$SET — R1 zero-shot IT2-1B beam $BEAM" \
        --json "$OUTPUT/$SET.score.json"
done

# --- the comparison Q9 actually asks for -------------------------------------
echo
echo "=== Q9 ==="
python - <<'PY'
import json, pathlib
out = pathlib.Path("/kaggle/working/q9")
rows = [("FLORES devtest", 1012, 21.64, 7.15, 35.04, 15.83),
        ("KAGGLE leaderboard", 1730, 7.28, None, None, 8.00)]
for s in ("r0", "r3_eval"):
    p = out / f"{s}.score.json"
    if p.exists():
        d = json.loads(p.read_text())
        rows.insert(1, (s, d["lines"], None, d["bleu"],
                        d["chrf_plus_plus"], d["geometric_mean"]))

print(f"{'dev set':<22}{'n':>7}{'src wd':>8}{'BLEU':>8}{'chrF++':>9}{'GEO':>8}")
for name, n, wd, b, c, g in rows:
    f = lambda v, w, p=2: (f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-")
    print(f"{name:<22}{n:>7,}{f(wd, 8)}{f(b, 8)}{f(c, 9)}{f(g, 8)}")

print("\nRatios against FLORES devtest (15.83):")
for name, n, wd, b, c, g in rows[1:]:
    print(f"  {name:<22} {g / 15.83:.3f}")
print("\nReport these numbers into PLANNING.md §Results and §Q9. Name the dev "
      "set on every line; tokenizer is 13a.")
PY

echo "=== DONE — outputs in $OUTPUT ==="
ls -la "$OUTPUT"
