# KATHE 2026 — English → Kashmiri MT

English (`eng_Latn`) → Kashmiri, Perso-Arabic script (`kas_Arab`), for the
KATHE 2026 shared task (Gaash Lab / NIT Srinagar + Bureau of Indian Standards).

Licence: Apache-2.0 (`LICENSE`). Third-party attribution: `NOTICE`.
Model and dataset disclosure: `MODEL_CARD.md`.

## Gated repositories

These Hugging Face repos require accepting their terms **before** anything will
download, each one separately — authorization does not carry across repos from
the same publisher:

- `ai4bharat/indictrans2-en-indic-1B`
- `ai4bharat/indictrans2-en-indic-dist-200M`
- `ai4bharat/indictrans2-indic-en-1B` (only if back-translation is attempted)
- `ai4bharat/BPCC`
- `facebook/flores`

Accept **all** of them before starting a GPU session. AI4Bharat's gates are
auto-approved, so it takes seconds — but a missed one surfaces as a `403
GatedRepoError` partway into a run, and on `ai4bharat/BPCC` it surfaces even
earlier, as a misleading "gated" error from `get_dataset_config_names` (the
dataset card and file listing stay publicly visible either way, so neither
proves you have file access).

Put a token in `.env` as `HF_TOKEN=...` (never committed), or use Kaggle
Secrets inside a notebook.

## Environments

There are three, and the split is forced by a real dependency conflict rather
than by preference.

| File | Env | Purpose |
| --- | --- | --- |
| `pyproject.toml` | `.venv` (`uv sync`) | Data work + scoring. BPCC extraction, filtering, LaBSE, corpus and dev-set building, metric replication. |
| `requirements-decode.txt` | `.venv-decode` | Local IndicTrans2 **inference** only. |
| `requirements-kaggle.txt` | Kaggle GPU | Training and GPU decoding. **This is the one that must reproduce for the live round.** |

`transformers` must stay at **4.46.1** wherever IndicTransToolkit is used: its
`__init__` eagerly imports a collator that imports `PreTrainedTokenizerBase`
from `transformers.tokenization_utils`, a re-export dropped in transformers 5.
This breaks *inference*, not only training. But 4.46.1 caps
`huggingface-hub<1.0` while `datasets>=5.0.1` requires `>=1.26.1`, so the data
env and the decode env cannot be the same interpreter.

```bash
uv sync                                    # .venv — data work + scoring

uv venv --python 3.11 .venv-decode         # .venv-decode — local inference
uv pip install --python .venv-decode/bin/python -r requirements-decode.txt
```

**Hardware note.** Do not decode the 1B model on an 8 GB machine's MPS backend;
beam 5 at batch 32 exhausts unified memory and hangs the machine. CPU works but
runs ~0.08 sent/s. 1B decoding belongs on a GPU — see `scripts/run_q9_kaggle.sh`.

## Reproducing the pipeline

```bash
set -a; . ./.env; set +a                   # HF_TOKEN; BPCC is gated

# 1. Extract, filter, and leakage-check BPCC
uv run python -m data.fetch_bpcc --output data/processed/bpcc_kas_raw.jsonl
uv run python -m data.filter --input data/processed/bpcc_kas_raw.jsonl \
    --output data/processed/bpcc_kas_filtered.jsonl --labse-scope mined
uv run python -m data.leakage --pairs data/processed/bpcc_kas_filtered.jsonl \
    --dev-kas data/dev/flores_kas.txt --dev-en data/dev/flores_en.txt \
    --output-clean data/processed/bpcc_kas_clean.jsonl

# 2. Cut the register-matched dev sets (R0 + the in-training eval slice)
uv run python -m data.build_devsets \
    --pairs data/processed/bpcc_kas_clean.jsonl \
    --test-csv data/raw/englishdev.csv --out data/dev \
    --dev-kas data/dev/flores_kas.txt --dev-en data/dev/flores_en.txt

# 3. Build the training corpus, with those held out by exact pair key
uv run python -m data.build_corpus \
    --pairs data/processed/bpcc_kas_clean.jsonl --out data/processed/r3_corpus \
    --dev-kas data/dev/flores_kas.txt --dev-en data/dev/flores_en.txt \
    --exclude data/dev/r0/r0.jsonl data/dev/r3_eval/r3_eval.jsonl \
    --dev-from data/dev/r3_eval/r3_eval.jsonl
```

Step 2 must run before step 3. Step 3 asserts that the held-out count it
removes equals the count it was given, and fails the build otherwise.

## Translating — the one command that matters

```bash
.venv-decode/bin/python scripts/generate_translations.py \
    --input englishdev.csv --output submission.csv
```

**`scripts/generate_translations.py` is the deliverable.** Input CSV in,
scoreable CSV out, batch mode, no interactivity, no other steps. It decodes with
the fine-tuned checkpoint and then restores the three short vowels that the
model structurally cannot produce, which is not an optional polish step:

| | leaderboard |
| --- | ---: |
| raw model output | 10.00 |
| + diacritic restoration | **13.81** |

IndicTrans2's target vocabulary holds kasra, damma and fatha in exactly one
token each, so beam search never emits them and fine-tuning cannot fix a frozen
vocabulary. A 3.3M-parameter character tagger puts them back, and it is worth
more than every training-side change in this project combined. Restorer weights
are fetched from the Hub automatically if they are not already on disk.

The script refuses to write output whose row count changed, that contains an
empty row, or where restoration ran without adding any marks — the three
failure modes here that produce plausible-looking files scoring near zero.

`--no-restore` exists only to inspect the decode in isolation. It costs ~3.8
points; do not ship it.

### Decoding and scoring during development

```bash
.venv-decode/bin/python scripts/translate.py \
    --input data/dev/r0/r0.eng_Latn --output r0.hyp --beam 5

uv run python -m eval.score --hyp r0.hyp --ref data/dev/r0/r0.kas_Arab \
    --src data/dev/r0/r0.eng_Latn --name "R0 register-matched"
```

`scripts/translate.py` is the raw decode tool and deliberately stops before
restoration, because building lexicons and training restorers needs
un-restored output. Do not use it to produce a submission.

Scoring replicates the official scorer exactly — `KashmiriNormalizer==0.1.0`
on both sides, BLEU tokenizer `13a`, chrF++ with `word_order=2`, reported as
their geometric mean.

Every submission must pass `scripts/validate_submission.py` before upload.
Row order is never sorted: the official scorer is positional, so a
correct-looking submission ordered by ID scores near zero.
