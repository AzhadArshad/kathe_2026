# Submission 001 — R1 zero-shot baseline

**Prepared:** 2026-08-09 · **Status:** validated, ready to upload · **Uploaded:** _(pending — Azhad)_

## Purpose

Calibration, not score. This exists to answer one question: how does FLORES
devtest **15.83** map onto the actual KATHE test set? Every subsequent offline
number is interpretable only against that ratio (PLANNING.md §Submission
strategy). A leaderboard result far from 15.83 implies the test set differs from
FLORES in register or orthographic convention — which is itself the most
valuable finding currently available (Q3).

## What produced it

| | |
| --- | --- |
| Model | `ai4bharat/indictrans2-en-indic-1B` (MIT), zero-shot — no fine-tuning |
| Decoding | beam 5, via `IndicProcessor` (`preprocess_batch` **and** `postprocess_batch`) |
| Language pair | `eng_Latn` → `kas_Arab` |
| Hypotheses | `data/dev/baseline_kas.txt` (sha256 `250ebb10…`, gitignored — see §Reproducibility gap) |
| Source | `data/raw/englishdev.csv` (sha256 `2d87840e…`), 1,730 rows |
| Post-processing | `postproc.yaml` in this directory (snapshot of `config/postproc_zeroshot.yaml`) |
| Output | `submission.csv` (sha256 `53cdd79a…`), 1,730 rows |

Pinned versions at build time: KashmiriNormalizer 0.1.0 · sacrebleu 2.6.0 · pandas 3.0.5.

## Rebuild

```bash
python scripts/make_submission.py \
    --source     data/raw/englishdev.csv \
    --hypotheses data/dev/baseline_kas.txt \
    --postproc   config/postproc_zeroshot.yaml \
    --output     submissions/001/submission.csv

python scripts/validate_submission.py \
    --source     data/raw/englishdev.csv \
    --submission submissions/001/submission.csv
```

## Validator result — PASSED (exit 0)

```
  diacritic density: 6.14 per 100 chars

WARN  output is not NFC — the scorer applies no Unicode normalization, so
      composition must match references
```

Both readings are expected, not defects:

- **Diacritic density 6.14/100c** against FLORES references' 7.70. This is the
  known kasra/damma gap — zero-shot IndicTrans2 emits essentially none of
  either. It is the single largest thing R3 fine-tuning is meant to fix, and the
  gap is quantified: otherwise-perfect text missing only kasra and damma scores
  geo 79.58 (PLANNING.md §Decisions, 2026-08-07).
- **Not NFC.** Correct to leave alone. FLORES devtest and BPCC are *both*
  non-NFC, so there is no measured target composition to convert to, and the
  scorer applies no Unicode normalization of its own. `unicode_form: null` until
  a diagnostic says otherwise.

Row alignment was verified independently of the validator: 1,730 hypotheses
against 1,730 source rows, zero blank lines, IDs identical and in source order.
Twenty random rows were eyeballed — Perso-Arabic throughout, no Devanagari, no
untranslated Latin passthrough.

## Post-processing rationale

`map_punctuation: false`, deliberately. Zero-shot output is already 99.7%
Kashmiri-punctuated against 95.7% in the references, so mapping Latin
punctuation would overshoot. **This flips to `true` for the first fine-tuned
submission** — BPCC is only 89.5% Kashmiri-punctuated, so fine-tuning pulls
output the other way (decision 2026-08-07). Verify the direction on FLORES dev
rather than assuming it.

`scorer_normalizer: true` means the submitted bytes are exactly the bytes we
scored. The scorer normalizes both sides anyway, so this changes no score; it
removes a class of "what did we actually upload" ambiguity.

## Expected score

FLORES devtest gives BLEU 7.15 / chrF++ 35.04 / **geo 15.83** — reproduced
exactly at build time from `data/dev/flores_hyp.txt`. The test set is stated to
be separate from BPCC and its domain is unknown (Q3), so treat 15.83 as an
anchor, not a prediction.

Leaderboard as of 2026-08-07: 1st 23.09, 3rd 14.23, 6th 9.12 — set under the old
submission cap and expected to climb.

## Reproducibility gap — must close before Aug 16

`data/dev/baseline_kas.txt` was generated in a Kaggle notebook that is **not in
this repo**, and it is gitignored along with the rest of `data/dev/`. Nothing
here regenerates it from the model.

That is acceptable for a calibration submission and **not** acceptable for the
live round, where the deliverable is a checkpoint plus a batch-translation
script the organizers run themselves. `scripts/translate.py --input --output`
(PROJECT_NOTES.md §4) does not exist yet and is on the critical path for the Aug 16
fresh-clone test. The decode settings above are recorded from PLANNING.md, not
recovered from committed code.
