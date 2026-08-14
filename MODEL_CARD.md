# Model Card — KATHE 2026 English → Kashmiri

**Status: current as of 2026-08-14.** Describes the system that scores **13.81**
on the competition leaderboard.

- **Task:** English (`eng_Latn`) → Kashmiri, Perso-Arabic script (`kas_Arab`)
- **Competition:** KATHE 2026, Gaash Lab / NIT Srinagar + Bureau of Indian Standards
- **Author:** Azhad Arshad (solo entry, team *Noore*)
- **Repository:** `<TBD — public URL, due 2026-08-16>`

**The system is two models, and both are required.** Shipping only the
translation model gives 10.00 instead of 13.81 (§4):

| Component | Weights | Licence | Status |
| --- | --- | --- | --- |
| Translation — 200M full fine-tune | `Aju360/kathe-r12-200m-selected` | Apache-2.0 (MIT notice retained) | **private, must be public by 2026-08-17** |
| Diacritic restorer — 3.3M char tagger | `r11b_dense.pt` → `Aju360/kathe-diacritic-restorer` | Apache-2.0 (see §2) | **not yet published** |

Run both with one command: `scripts/generate_translations.py`.

HF account is `Aju360`; GitHub is `AzhadArshad`. The two differ — do not infer
one from the other in any published URL.

**Open-sourcing code and weights by the deadline is an eligibility condition**
for every participant, not only for winners.

---

## 1. Licence chain

Weights are a derivative of the base checkpoint and are trained on the datasets
below. Both constrain what may be claimed over the released artifact.

| Artifact | Licence | Why |
| --- | --- | --- |
| This repository's **code** | Apache-2.0 | Original work. See `LICENSE`. |
| **Translation weights** | Apache-2.0, with the MIT notice retained | Derived from IndicTrans2, which is MIT; MIT permits redistribution under Apache-2.0 provided attribution is preserved. |
| **Diacritic restorer weights** (`r11b_dense`) | Apache-2.0 | Trained from scratch — no base model. Its training text is BPCC (CC-BY-4.0, attributed) and `nawabhussain/Kashmiri-Language-Corpus` (Apache-2.0). It contains **no** CC-BY-NC-SA material; other checkpoints from the same sweep do. See §2. |
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

Every pair reaching the shipped translation model comes from BPCC. No other
parallel corpus is used, and no third-party translation API is used anywhere in
this project.

**Shipped model (R12-selected).** Candidate pool, before dev exclusion:

| Corpus | Source | Licence | Pairs |
| --- | --- | --- | ---: |
| BPCC — `bpcc-seed-v2` | `ai4bharat/BPCC` | CC-BY-4.0 | 76,981 |
| BPCC — `bpcc-seed-v1` | `ai4bharat/BPCC` | CC-BY-4.0 | 15,504 |
| BPCC — `bpcc-seed-latest` | `ai4bharat/BPCC` | CC-BY-4.0 | 10,582 |
| BPCC — `nllb-seed` | `ai4bharat/BPCC` | CC0 1.0 | 6,065 |
| BPCC — `daily` | `ai4bharat/BPCC` | CC-BY-4.0 | 4,276 |
| **Pool total** | | | **113,408** |

Human-translated only; the web-mined `nllb_filtered` subset is excluded, as is
any LaBSE-based quality filter — dropping both *improved* the leaderboard score
(§4). Deduplicated on `(source, diacritic-stripped target)`. R0, the eval slice
and a 3,000-pair training dev set are then removed, leaving **111,400**; the
20,000 nearest the dev distribution are repeated ×3 for a **148,400**-pair
training file (§5.1).

**Earlier model (R3),** which produced submissions 002–010 and remains the
baseline several results are quoted against: the same BPCC subsets plus
`nllb_filtered` (12,083, CC0), 125,582 pairs after filtering, 123,538 after
holding out 2,000 for dev.

`wiki/kas_Arab.tsv` is byte-identical to `bpcc-seed-v1` and is skipped
structurally rather than counted twice.

### Post-processing and the diacritic restorer (not in the translation mix)

| Corpus | Source | Licence | Size | Role |
| --- | --- | --- | ---: | --- |
| Kashmiri-Language-Corpus | `nawabhussain/Kashmiri-Language-Corpus` | Apache-2.0 | 47,344 sentences | Monolingual Kashmiri. Builds the diacritic-restoration lexicon, and part of the R11 restorer's training text. |
| Kashmiri-English Parallel Corpus, `HuggingFace 30K/Kashmiri.txt` | `SMUQamar/Kashmiri-English-Parallel-Corpus` (Qumar et al., DOI 10.57967/hf/3061) | **CC-BY-NC-SA-4.0** | 29,999 sentences, 16,505 after dedup | Kashmiri side only, monolingual. R11 restorer training text. Access is granted **manually** by the owners. |

Disclosed under the organizers' 2026-08-07 ruling. These are listed separately
from the translation training mix because they never reach the *translation*
model. They are not weight-free, however: the character-level diacritic restorer
(§5.2) is trained on this text, so the heading below is about which model, not
about whether weights exist.

**Licence chain for the restorer — depends on the checkpoint, so check before
publishing.** The SMUQamar corpus is CC-BY-NC-SA-4.0. Any restorer trained on it
is a derivative work under ShareAlike and **must be published CC-BY-NC-SA-4.0,
not Apache-2.0**, with attribution and the NonCommercial term intact. Per
PROJECT_NOTES.md §2.7, do not claim Apache-2.0 over something that cannot be
relicensed.

The R11 sweep produced checkpoints on both sides of that line, because the arms
differ in which sources they read:

| Checkpoint | Training sources | Contains SMUQamar? | Publishable as |
| --- | --- | --- | --- |
| **`r11b_dense` — SHIPPED** | `bpcc:daily` (3,059) + `bpcc:bpcc-seed-v1` (15,460) + external/nawabhussain (39,948) = 58,467 lines | **No** | **Apache-2.0** |
| `r11b_clean` | above + `bpcc-seed-v2`, `bpcc-seed-latest` | No | Apache-2.0 |
| `r11_all`, `r11_bpcc` | all eight source tags | **Yes** | CC-BY-NC-SA-4.0 |

**The shipped restorer is `r11b_dense`, and it is clean.** Its inputs are BPCC
(CC-BY-4.0, attributed in `NOTICE`) and `nawabhussain/Kashmiri-Language-Corpus`
(Apache-2.0). It therefore carries no NonCommercial restriction and is released
under Apache-2.0 alongside the rest of this repository.

Translation weights are unaffected either way: they derive from BPCC and
IndicTrans2 (MIT) only.

**Dev-set contamination, disclosed.** 419 of the 1,003 sentences in the
project's R0 development set (and 410 of the 997-line evaluation slice) appear
verbatim in `nawabhussain/Kashmiri-Language-Corpus`. The diacritic lexicon
behind submissions 006 and 007 was built over that corpus unfiltered, so those
submissions' **R0 scores** are inflated by roughly 0.36 geometric-mean points.
Their **leaderboard** scores are unaffected: the KATHE test set is separate from
R0 and shares no sentences with either corpus. Restoration artifacts built after
2026-08-13 exclude the dev references by exact stripped-and-normalized string.

### Evaluation only, never trained on

| Corpus | Source | Licence | Size | Role |
| --- | --- | --- | ---: | --- |
| FLORES-200 `kas_Arab` devtest | `facebook/flores` | CC-BY-SA-4.0 | 1,012 | Regression check only |
| FLORES-200 `kas_Arab` dev | `facebook/flores` | CC-BY-SA-4.0 | 997 | Largely superseded |
| KATHE `englishdev.csv` | Competition-supplied | Competition terms | 1,730 | Test input. Used as a *length reference* for dev-set construction; no reference translations exist for it. |

### Considered, not yet added

| Corpus | Licence | Status |
| --- | --- | --- |
| `SMUQamar/Kashmiri-English-Dataset-270K` | CC-BY-NC-SA-4.0 | Not added — no access granted. The related 30K corpus from the same authors IS used, and is disclosed above. |
| `injilashah/Kashmiri-terms` | undeclared | **Rejected on inspection 2026-08-13.** Despite the name its 396 rows are English ASR prompt text — zero Perso-Arabic, 0.00 diacritics per 100 characters. |

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

**A caution about R0.** R0 has been measured as *anti-predictive* of the
leaderboard across seven post-processing submissions (Spearman ρ = −0.39). It is
cut from BPCC, which IndicTrans2 was trained on, so every IT2-derived system is
partly scored on its own training data, and its `daily`-weighted sampling gives
it a diacritic profile the real test set does not share. Numbers below are
leaderboard readings wherever one exists. Do not infer a leaderboard position
from an R0 score.

**Leaderboard results** (geometric mean of BLEU and chrF++, held-out test set):

| System | Score |
| --- | ---: |
| IndicTrans2-1B zero-shot | 8.00 |
| R3 — 200M full fine-tune, raw | 8.83 |
| R12-selected — 200M, semantically selected corpus, raw | 10.00 |
| R3 + diacritic lexicon | 11.81 |
| R12-selected + diacritic lexicon | 13.52 |
| **R12-selected + learned diacritic restorer** | **13.81** |

Two levers account for the gap from 8.83, and they are unequal:

- **Diacritic restoration: +3.81.** The single largest contribution.
- **Training-mix selection: +1.17.** Rebuilding the corpus from raw BPCC and
  upweighting pairs semantically near the dev distribution.

---

## 5. Training

### 5.1 Translation model

Full fine-tune (no LoRA) of `ai4bharat/indictrans2-en-indic-dist-200M`.

| | |
| --- | --- |
| Corpus | 148,400 pairs — 111,400-pair raw-BPCC pool, human-translated only, with the nearest 20,000 to the dev distribution repeated ×3 (40.4% of the mix) |
| Held out | R0 (1,003) and the in-training eval slice (997), removed by exact pair key with an asserted count; plus a 3,000-pair dev set with verified zero train∩dev overlap |
| Epochs | 6 |
| Learning rate | 5e-5, inverse-sqrt schedule, 500 warmup steps |
| Batch | 16 per device × 2 GPUs × 4 accumulation = 128 effective |
| Precision | fp16 (T4 is Turing — no bf16) |
| Label smoothing | 0.1 (IndicTrans2's own setting) |
| Max sequence length | 128 |
| Hardware | Kaggle, 2 × Tesla T4, DDP via `torchrun` |
| Wall-clock | ~2h45m |

Selection used `sentence-transformers/all-MiniLM-L6-v2` embeddings, mean cosine
to the 5 nearest queries, with a per-query cap of 25 to stop any one query
dominating the slice. A **random-slice control arm** trained identically scored
12.49 against the selected arm's 13.52, which is what makes the +1.03
attributable to semantic selection rather than to the corpus rebuild or to
upweighting in general.

Configuration lives in `config/r12_200m_selected.yaml`; the corpus manifest,
including per-subset counts, in `data/processed/r12_corpus/manifest.json`.

### 5.2 Diacritic restorer

A separate 3.3M-parameter character-level tagger, trained from scratch. It is
part of the system, not an accessory — see §4.

| | |
| --- | --- |
| Architecture | 4-layer bidirectional transformer, d_model 256, 4-way per-character tagging (none / kasra / damma / fatha) |
| Behaviour | **Insertion-only by construction.** It emits labels over an unchanged character sequence, so it cannot delete, reorder or substitute. |
| Training text | 58,467 lines from the three most consistently diacritized sources (`bpcc:daily`, `bpcc:bpcc-seed-v1`, and the external corpus in §2), 6.73 restorable marks per 100 chars |
| Epochs | 20 |
| Held out | R0 references excluded by stripped-and-normalized string, so dev text cannot leak in undiacritized form |
| Hardware | Kaggle T4; minutes, not hours |

**Why it is necessary.** IndicTrans2's target vocabulary
(`dict.TGT.json`, 122,672 entries) contains kasra, damma and fatha in *exactly
one token each* — the bare standalone mark. To write `چھُس` the model would have
to split a word to insert a bare diacritic that occurs in no natural subword
context, and beam search never does. Measured on the fine-tuned model's output:
0.000 of each per 100 characters, against references near 4.7. This is a
property of the frozen pretrained vocabulary and **no amount of fine-tuning
changes it.**

Decoding uses a logit offset on the "no mark" class (`none_bias`), solved by
bisection so output mark density matches a target. The shipped value is
`+1.6836`: this restorer marks freely (11.49/100c unbiased) and is held back to
9.85. The value is specific to this checkpoint and this input.

---

## 6. Reproduction

See `README.md`.

**Inference** — one command, decode plus restoration:

```bash
python scripts/generate_translations.py --input <input.csv> --output <out.csv>
```

`scripts/translate.py` is the raw decode tool and omits restoration by design;
using it to produce a submission costs ~3.8 points.

**Five** Hugging Face repos are gated and require accepting terms before
download. Acceptance is per-repo and does **not** carry across a publisher's
other repositories — a token that had just downloaded the 1B still returned 403
on the 200M:

- `ai4bharat/indictrans2-en-indic-1B`
- `ai4bharat/indictrans2-en-indic-dist-200M`
- `ai4bharat/indictrans2-indic-en-1B`
- `ai4bharat/BPCC`
- `facebook/flores`

**Pinned environment** (`requirements-kaggle.txt`): `transformers==4.46.1`,
`indictranstoolkit==1.1.1`, `sacrebleu==2.6.0`, `KashmiriNormalizer==0.1.0`.
transformers must stay at 4.46.1 — newer releases drop the
`PreTrainedTokenizerBase` re-export that IndicTransToolkit's collator imports.

**Verify any checkpoint by generating text, never by loading without error.**
IndicTrans2 ties `lm_head` to the decoder embeddings; `save_pretrained` drops
the tied duplicate and this architecture's load path then zeroes both. A
checkpoint here once loaded cleanly, logged "All the weights were initialized
from the model checkpoint", and translated every input to the empty string.
`generate_translations.py` guards against the equivalent failure in the
restorer by refusing to write output when restoration added no marks.

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
