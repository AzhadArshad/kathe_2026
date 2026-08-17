# KATHE 2026 — English → Kashmiri MT

English (`eng_Latn`) → Kashmiri, Perso-Arabic script (`kas_Arab`), for the
KATHE 2026 shared task (Gaash Lab / NIT Srinagar + Bureau of Indian Standards).

Scored as the **geometric mean of BLEU and chrF++** on a held-out test set of
1,730 short everyday sentences. Final score **15.05**, from a starting baseline
of 8.00.

Licence: Apache-2.0 (`LICENSE`). Third-party attribution: `NOTICE`.
Model and dataset disclosure: `MODEL_CARD.md`.

## The system

Two models and a lookup table, and **all three are required**:

```
English  →  IndicTrans2 200M, fine-tuned      →  Kashmiri without short vowels
         →  diacritic restorer (union)        →  Kashmiri with them
```

| Component | Weights | Size |
| --- | --- | ---: |
| Translation | [`Aju360/kathe-r12-200m-selected`](https://huggingface.co/Aju360/kathe-r12-200m-selected) | 211M params |
| Diacritic restorer | [`Aju360/kathe-r11-restorer`](https://huggingface.co/Aju360/kathe-r11-restorer) (`r11b_dense.pt`) | 3.3M params |
| Diacritic lexicon | `data/processed/diacritic_lexicon_both.json` (in this repo) | 48k forms |

Shipping the translation model alone scores **10.00 instead of 15.05.** The
restoration stage is not polish — see [What we learned](#what-we-learned).

## Results

Every number is the official metric on the competition's hidden test set.

| # | System | Score |
| ---: | --- | ---: |
| 001 | IndicTrans2-1B, zero-shot | 8.00 |
| 002 | 200M distilled, fine-tuned on BPCC | 8.83 |
| 003 | + diacritic lexicon (unigram) | 11.16 |
| 004 | + left-context disambiguation | 11.70 |
| 007 | + external monolingual corpus in the lexicon | 11.81 |
| 016 | corpus rebuilt + semantically selected, **no restoration** | 10.00 |
| 011 | + diacritic lexicon | 13.52 |
| 015 | + learned character-level restorer instead | 13.81 |
| 019 | + restorer left unsuppressed | 13.99 |
| 021 | + **union** of restorer and lexicon | 14.82 |
| **024** | **+ tail suppression tuned** | **15.05** |

Where the +7.05 came from:

| Lever | Gain | GPU needed |
| --- | ---: | --- |
| **Diacritic restoration** | **+5.05** | none |
| Training-mix selection | +1.17 | yes |
| Fine-tuning at all | +0.83 | yes |

The largest lever by a factor of four required no GPU time whatsoever.

---

## Inference scripts

Three entry points. All of them resolve to the **published weights** by default,
so they run from a clean clone with no local `models/` directory and no Hugging
Face token.

| | Script | Use |
| --- | --- | --- |
| **Model loading** | [`src/infer.py`](src/infer.py) | `load_system()` / `translate()`. Both scripts below call it, so there is one definition of the system. |
| **Single inference** | [`scripts/translate_single.py`](scripts/translate_single.py) | One sentence in, one translation out. |
| **Batch inference** | [`scripts/generate_translations.py`](scripts/generate_translations.py) | CSV in, scoreable CSV out. |

```bash
git clone https://github.com/AzhadArshad/kathe_2026 && cd kathe_2026
uv venv --python 3.11 .venv-decode
uv pip install --python .venv-decode/bin/python -r requirements-decode.txt

# check the whole system loads and the restoration stage fires
.venv-decode/bin/python scripts/translate_single.py --self-test

# single
.venv-decode/bin/python scripts/translate_single.py --text "He lost his pen."
# -> أمۍ ہوٗر پنُن پین۔

# batch
.venv-decode/bin/python scripts/generate_translations.py \
    --input englishdev.csv --output submission.csv
```

`--self-test` translates three fixed sentences and **fails with exit code 2 if
restoration produced no kasra, damma or fatha.** That is the failure worth
guarding: if the restorer or the lexicon silently fails to load, the output is
still fluent-looking Kashmiri and still scores five points worse. Run it first.

Reusing one loaded system across many sentences:

```python
import sys; sys.path.insert(0, "src")
from infer import load_system, translate

system = load_system()                     # published weights, ~10s
print(translate(system, ["He lost his pen.", "I am tired."]))
```

**Verified**, not merely intended: an anonymous `git clone` of this repository
with no local weights and no `HF_TOKEN` set runs both scripts end to end, and
the single and batch paths produce byte-identical output on the same input.


## What we learned

Nine days, 26 submissions, one language with almost no public parallel data.
The findings below are the ones that changed what we did, each with the number
that forced the change. Most of them generalise past Kashmiri to any
low-resource pair built on a large pretrained multilingual model.

### 1. Audit the tokenizer against the target orthography *before* training

The single largest result in this project. IndicTrans2's target vocabulary has
122,672 entries. Three Kashmiri short vowels — kasra (U+0650), damma (U+064F),
fatha (U+064E) — appear in **exactly one token each**: the bare standalone mark.
Every other Kashmiri diacritic is baked into whole-word subwords (hamza-below is
in 378 tokens, inverted-damma in 244).

To write `چھُس` the model must emit `[چھ][ُ][س]` — split a word to insert a bare
diacritic that occurs in no natural subword context. Beam search never does,
because the undiacritized whole word is always more probable. The model produced
**exactly zero** of all three marks across 1,730 sentences, while reproducing
the two subword-embedded marks at *above* reference rates. Categorical, not
gradual: the signature of a structural limit, not under-training.

The metric preserves diacritics, so this was expensive. An otherwise-perfect
translation missing only these three marks scores **67.66 out of 100**.

We lost two days to a wrong diagnosis first — assuming the training data lacked
the marks, then that preprocessing stripped them, then that the model was
under-trained. What settled it was counting marks per token in `dict.TGT.json`.

**The lesson:** a pretrained model can be *structurally incapable* of producing
characters your target orthography requires, and no amount of fine-tuning,
reweighting or extra epochs will fix a frozen vocabulary. Before committing GPU
time, count how many of your target script's characters actually appear inside
the subword vocabulary. It is a five-minute check that was worth +5.05 here —
four times more than every training-side change combined.

### 2. Your development set can be actively anti-correlated with the truth

We built a careful dev set: 1,003 pairs, stratified to match the test set's
word-length distribution, human references only, held out by exact pair key with
an assertion that fails the build if the count is off. Every leakage check
passed.

Across the whole project its correlation with the leaderboard was **rho −0.39**.

The reason is provenance. Our dev set was cut from BPCC, and IndicTrans2 was
*trained on BPCC* — it is the corpus AI4Bharat released alongside it. So every
system we tested was being scored partly on its own base model's training data.
Holding pairs out of *our* fine-tuning does not hold them out of the base
model's.

This is not an exotic failure. In a low-resource language there is usually one
public parallel corpus, and the strongest pretrained model was trained on it.
The clean dev set — text no pretrained system has seen — is exactly the thing a
low-resource language does not have.

**The lesson:** treat offline metrics as directional, and let the real
evaluation adjudicate. We eventually adopted a blunt rule: evaluate offline to
rank candidates, then spend submissions to decide. Several times the dev set
ranked the eventual winner *last*.

### 3. Leakage checks catch shared examples, not shared provenance

Submission 005 scored the best dev result of the project and **regressed on the
leaderboard by 5.3%**.

It added a 260,000-entry two-sided context table to the diacritic restoration
step. The table was built from training targets only. The dev set was never
trained on. Every leakage check passed, correctly.

The problem was that the table and the dev set were both BPCC n-grams. A table
that large stops carrying lexical knowledge and starts memorising word
sequences — and the dev set, drawn from the same corpus, rewarded exactly that
while the real test set did not. A 21,000-entry table generalised; 260,000 did
not.

**This is dev-set overfitting through post-processing**, a route that gets far
less scrutiny than training. Nothing in a standard leakage check looks for it.
The usable heuristic we ended on: trust the dev set when a change adds genuinely
new knowledge (a new corpus, a better model); distrust it when a change adds
finer-grained context drawn from the dev set's own provenance.

There was a warning sign we missed. The submission's output moved *away* from
reference diacritic density while its dev score rose. Two independent signals
disagreeing is worth more than either agreeing.

### 4. Check what your base model was already trained on

We fine-tuned the 1B model with LoRA and it scored **below its own zero-shot
baseline** (27.78 vs 28.22 on our dev set). Meanwhile a full fine-tune of the
*distilled 200M* gained +15.9% over its base.

The asymmetry has a clean explanation. IndicTrans2 was trained on BPCC. Training
the 1B on BPCC re-teaches it what it already knows, so there is nothing to
recover and the update only perturbs a well-tuned model. The 200M is a
*distilled* checkpoint — distillation discarded information that BPCC training
puts back.

**The lesson:** "use the biggest model" is wrong when the big model has already
consumed your corpus. A smaller distilled checkpoint can have more headroom on
exactly the data that leaves the large one unmoved. This closed our training
line entirely; after it we spent no further GPU time on the translation model.

### 5. Read the metric's parameters, then exploit the asymmetry

chrF++ runs at **beta=2**, weighting recall four times precision. Consequence: a
diacritic the reference has and you omit costs more than one you add that it
lacks.

For eleven submissions we tuned the restorer's output density to match reference
density (~9.6 marks per 100 characters), because matching the reference is the
obvious target. Removing that anchor and letting the model emit its natural 11.5
was worth **+0.18** — and the curve kept paying until 12.5, where BLEU's n-gram
precision finally pulled back.

The nuance that took three more submissions: this holds *globally*, not
everywhere. When we let the model mark more freely on words the lexicon didn't
know, the score fell monotonically (14.82 → 14.53 → 14.25). Extra marks are
cheap where the model is confident and pure noise in the tail.

**The lesson:** the scoring function is a specification, not a black box. Its
parameters tell you which errors are cheap.

### 6. Register mismatch beats data volume

The test set averages **7.3 words per sentence**. BPCC averages 15.4. FLORES —
the standard dev set everyone reaches for — averages 21.6.

We built the whole plan around FLORES, then measured the actual test input and
found it was a different task: `He lost his pen.` · `She plays a viola.` Tuning
on encyclopedic prose would have optimised for sentences we are never scored on.

Rebuilding the corpus and upweighting pairs semantically near the test
distribution was worth **+1.03** on top of **+0.68** for the rebuild alone. But
the aggression curve is monotone downward: upweighting by 7x scored below 3x,
and 31x below that. Sharpening the slice helps; distorting it does not.

**The lesson:** measure your test input directly before choosing a dev set. It
took ten minutes and invalidated a week of planned work.

### 7. MBR needs a quality-aware utility, which low-resource languages lack

We implemented Minimum Bayes Risk decoding with chrF++ as the utility: sample 32
candidates, keep the one with the highest mean similarity to the rest. It scored
**32.67 against beam search's 33.36** on our dev set, at roughly 30x the decode
cost, and lost at every pool size from 2 to 32.

The diagnostic is the useful part. Oracle selection over the *same* 32
candidates would have scored **44.95** — a sample beats beam in 853 of 1,003
sentences. The candidates are excellent; the utility cannot find them. MBR's
pick ranked #9.3 of 32 on average and agreed with beam outright on 58% of
sentences.

chrF++ measures **typicality, not quality**. Consensus finds the mode, and the
mode is what beam already returns. Published MBR gains come from *neural*
utilities (COMET, BLEURT) that correlate with human judgment — and there is no
COMET model for Kashmiri.

**The lesson:** similarity-based MBR is not a free win. The +11.6 of reachable
headroom it revealed is real and remains the largest unexploited lever here, but
claiming it needs a reranker that knows quality, and building one for a
low-resource language is its own project.

### 8. The failures that cost the most were silent

None of these threw an exception:

- **A checkpoint that loaded perfectly and translated everything to the empty
  string.** IndicTrans2 ties `lm_head` to the decoder embeddings;
  `save_pretrained` drops the tied duplicate and the load path then zeroes both.
  766 tensors, no error, a clean "all weights initialized" log, and nothing but
  empty output. *Verify a checkpoint by generating text, never by loading it.*
- **A preprocessing call that hangs forever at 0% CPU.** `IndicProcessor`
  pops one placeholder map per input from a queue; call it with mismatched input
  and output counts and it blocks on `Queue.get()` with no timeout and no log
  line. Hit twice, in two different places.
- **Positional scoring.** The official scorer deletes the ID column and zips
  what remains. A submission sorted by ID against an unsorted input looks
  perfectly correct and scores near zero.
- **A stale config.** Our live-deployment post-processing file still pointed at
  a system from five submissions earlier — everything ran, it was just 1.06
  points worse. Caught by a fresh-clone rehearsal the day before the deadline.

**The lesson:** in a short competition, budget for verification that a thing
produced the *right* output, not that it ran. Every one of these would have
passed a smoke test.

### 9. What we would do differently

- **Audit the tokenizer on day one.** Two days were spent diagnosing an
  impossibility that a vocabulary count would have revealed immediately.
- **Submit earlier and more often.** The first leaderboard reading showed our
  offline proxy overestimated by 1.98x. Everything measured before that had been
  interpreted against the wrong scale.
- **Distrust a dev set drawn from the base model's training corpus** from the
  start, rather than discovering it through a regression.
- **Rehearse deployment before the last day.** The fresh-clone run found three
  blockers, one of which — the shipped restorer had never been uploaded anywhere
  — would have made the system impossible for anyone else to run.

---

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

## Producing a competition submission

`scripts/generate_translations.py` (see [Inference scripts](#inference-scripts))
writes the exact `ID,kashmiri_text` format the task requires.

The script refuses to write output whose row count changed, that contains an
empty row, or where restoration ran without adding any marks — the three
failure modes here that produce plausible-looking files scoring near zero.

`--no-restore` exists only to inspect the decode in isolation. It costs 5.05
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
