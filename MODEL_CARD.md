# Model Card — KATHE 2026 English → Kashmiri

**Status: STUB.** The licence chain and the dataset disclosure below are
complete and current as of 2026-08-10. Everything describing a *trained model*
is a placeholder: no fine-tune has been run yet. Sections marked `TBD` must be
filled before weights are published (due 2026-08-17).

- **Task:** English (`eng_Latn`) → Kashmiri, Perso-Arabic script (`kas_Arab`)
- **Competition:** KATHE 2026, Gaash Lab / NIT Srinagar + Bureau of Indian Standards
- **Author:** Azhad Arshad (solo entry, team *Noore*)
- **Repository:** `<TBD — public URL, due 2026-08-16>`
- **Weights:** `<TBD — HF Hub repo>`

---

## 1. Licence chain

Weights are a derivative of the base checkpoint and are trained on the datasets
below. Both constrain what may be claimed over the released artifact.

| Artifact | Licence | Why |
| --- | --- | --- |
| This repository's **code** | Apache-2.0 | Original work. See `LICENSE`. |
| **Released weights** | Apache-2.0, with the MIT notice retained | Derived from IndicTrans2, which is MIT; MIT permits redistribution under Apache-2.0 provided attribution is preserved. |
| Training data — BPCC human seed | CC-BY-4.0 | **Attribution required.** Discharged by `NOTICE` and §2 below. |
| Training data — BPCC NLLB subsets | CC0 1.0 | Public domain dedication; no obligations. |
| Dev data — FLORES-200 | CC-BY-SA-4.0 | Evaluation only, never trained on, never redistributed. ShareAlike does not attach to the weights. |

Open-sourcing both code and weights by the deadline is an eligibility condition
for every KATHE 2026 participant, not only for winners.

**On relicensing.** Apache-2.0 is claimed here only because every input permits
it. If a base model or corpus with a non-relicensable term is ever introduced —
NLLB-200's CC-BY-NC-4.0 is the live example — the derived weights must ship
under *that* licence instead, and this table must say so. A non-commercial term
cannot be relicensed away by fine-tuning.

### Base model

`ai4bharat/indictrans2-en-indic-1B` — MIT, AI4Bharat / IIT Madras.
Also used: `ai4bharat/indictrans2-en-indic-dist-200M` (MIT) for fast iteration.
`ai4bharat/indictrans2-indic-en-1B` (MIT) is reserved for back-translation and
has not been used at the time of writing.

`facebook/nllb-200-distilled-1.3B` is **not used**. The organizers' 2026-08-08
ruling means its non-commercial licence would not disqualify it, but a second
full fine-tune is not affordable within the remaining GPU budget.

---

## 2. Dataset disclosure

Required by the organizers' 2026-08-07 ruling: *"You may use any publicly
available dataset but you need to disclose it."* Every corpus that reaches the
training mix is listed here **when it is added**, with source, licence and pair
count. A dataset absent from this table is a dataset not used.

### In the training mix

| Corpus | Source | Licence | Pairs used | Added |
| --- | --- | --- | ---: | --- |
| BPCC — `bpcc-seed-v2` | `ai4bharat/BPCC` | CC-BY-4.0 | 77,044 | 2026-08-07 |
| BPCC — `bpcc-seed-v1` | `ai4bharat/BPCC` | CC-BY-4.0 | 15,503 | 2026-08-07 |
| BPCC — `bpcc-seed-latest` | `ai4bharat/BPCC` | CC-BY-4.0 | 10,592 | 2026-08-07 |
| BPCC — `daily` | `ai4bharat/BPCC` | CC-BY-4.0 | 4,279 | 2026-08-07 |
| BPCC — `nllb_seed` | `ai4bharat/BPCC` | CC0 1.0 | 6,081 | 2026-08-07 |
| BPCC — `nllb_filtered` (web-mined) | `ai4bharat/BPCC` | CC0 1.0 | 12,083 | 2026-08-07 |
| **Total after filtering** | | | **125,582** | |

Counts are post-filter pair counts entering the corpus build (extraction and
filter stages are documented in `PLANNING.md` §"Data pipeline"). Of these,
113,499 (90.4%) are human-translated. 2,000 pairs are then held out as dev sets
(§4), leaving **123,538** for training.

`wiki/kas_Arab.tsv` is byte-identical to `bpcc-seed-v1` and is skipped
structurally rather than counted twice.

### Evaluation only, never trained on

| Corpus | Source | Licence | Size | Role |
| --- | --- | --- | ---: | --- |
| FLORES-200 `kas_Arab` devtest | `facebook/flores` | CC-BY-SA-4.0 | 1,012 | Regression check only |
| FLORES-200 `kas_Arab` dev | `facebook/flores` | CC-BY-SA-4.0 | 997 | Largely superseded |
| KATHE `englishdev.csv` | Competition-supplied | Competition terms | 1,730 | Test input. Used as a *length reference* for dev-set construction; no reference translations exist for it. |

### Considered, not yet added

| Corpus | Licence | Status |
| --- | --- | --- |
| `SMUQamar/Kashmiri-English-Dataset-270K` | `<TBD — verify on the Hub before use>` | Not added. Register is a closer match to the test set than BPCC's. **If added, this table must be updated in the same commit.** |

### Tooling that touches data but not weights

`sentence-transformers/LaBSE` (Apache-2.0) was used for parallel-corpus quality
filtering of mined pairs only. It contributes no parameters to the released
weights. Its cosines are a poor quality signal for Kashmiri and are scoped to
mined pairs for that reason.

---

## 3. Intended use and limitations

**Intended use.** English → Kashmiri (Perso-Arabic) sentence-level translation,
specifically short everyday declarative sentences, which is what the KATHE 2026
test set consists of.

**Limitations.**

- Trained overwhelmingly on material longer than its intended input: the
  training corpus averages 15.7 English words per sentence against the test
  set's 7.3. Over-generation on short inputs is the expected failure mode.
- Perso-Arabic output only. Devanagari `kas_Deva` is not supported; the two
  scripts are not interchangeable for this model.
- Sentence-level. No document context, no discourse or pronoun resolution
  across sentences.
- Roughly 10% of the training data is web-mined and unverified.
- Not evaluated for, and not suitable for, any high-stakes use — medical, legal
  or safety-critical translation.

**Diacritics.** Output is expected to carry Kashmiri vowel diacritics (kasra,
damma and others). The evaluation metric preserves them, and stripping them
from otherwise-perfect text costs roughly 87% of the score.

---

## 4. Evaluation

**Metric.** Geometric mean of BLEU and chrF++, computed after running the
official `KashmiriNormalizer` (pinned `==0.1.0`) over both hypotheses and
references. BLEU tokenizer `13a`; chrF++ `char_order=6, word_order=2, beta=2`.
Implemented in `src/data/normalize.py`.

**Development sets.**

| Set | Size | Mean src words | Role |
| --- | ---: | ---: | --- |
| **R0, register-matched** | 1,003 | 6.88 | Primary. All tuning decisions. |
| In-training eval slice | 997 | 6.89 | Checkpoint selection during training. Disjoint from R0. |
| FLORES-200 devtest | 1,012 | 21.6 | Regression check only. |

R0 and the eval slice are held out of training, are drawn only from
human-translated pairs, and are stratified by English word count to match the
distribution of the real test input over the same range. FLORES was demoted
after the test set was measured at 7.3 words per sentence against FLORES's 21.6
— tuning on FLORES optimizes a different task.

**Results.** `TBD` — no fine-tuned checkpoint exists yet. The zero-shot
IndicTrans2-1B baseline scores 15.83 on FLORES devtest and 8.00 on the
competition leaderboard.

---

## 5. Training

`TBD`. Planned: full fine-tune of the 200M distilled checkpoint for iteration,
LoRA on the 1B for the final run, in two stages (mined data, then a short
finishing pass on human data). Targets are normalized with `KashmiriNormalizer`
before training. Hyperparameters live in `config/`, not in code.

To be recorded here once run: base checkpoint, hardware, wall-clock, epochs,
learning rate, batch size, and the exact corpus hash from
`data/processed/r3_corpus/manifest.json`.

---

## 6. Reproduction

See `README.md`. Three Hugging Face repos are gated and require accepting terms
before download: `ai4bharat/indictrans2-en-indic-1B`, `ai4bharat/BPCC`, and
`facebook/flores`.

Inference is `scripts/translate.py --input <file> --output <file>`. It requires
`transformers==4.46.1`; newer releases break `IndicTransToolkit` at import.

---

## 7. Citation

```bibtex
@article{gala2023indictrans,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation
             Models for all 22 Scheduled Indian Languages},
  author  = {Jay Gala and Pranjal A. Chitale and Raghavan AK and Sumanth
             Doddapaneni and Varun Gumma and Aswanth Kumar and Janki Nawale and
             Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and
             Pratyush Kumar and Mitesh M. Khapra and Raj Dabre and Anoop
             Kunchukuttan},
  journal = {Transactions on Machine Learning Research},
  year    = {2023}
}
```

Full third-party attribution, including the CC-BY-4.0 notice BPCC requires, is
in `NOTICE`.
