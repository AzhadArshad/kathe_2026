# Submission 002 — R3 fine-tuned 200M

**Uploaded 2026-08-11. Leaderboard score: 8.83** (previous best 8.00).
Personal best; +10.4%.

## What produced it

| | |
| --- | --- |
| Model | `Aju360/kathe-r3-200m-full` — R3, full fine-tune of `ai4bharat/indictrans2-en-indic-dist-200M` |
| Training | 5 epochs, 4,825 steps, effective batch 128, lr 5e-5, T4 x2 DDP, 1h29m49s |
| Corpus | `data/processed/r3_corpus`, 123,538 pairs (R0 + eval slice excluded by pair key) |
| Decoding | beam 5, `scripts/translate.py`, MPS, batch 8 |
| Post-processing | `config/postproc_r3.yaml` — `scorer_normalizer` only |
| Validator | PASSED. Diacritic density 5.65/100c |

Files here: `submission.csv` (sha256 `9e04b0c0381e21d4…`), `hyp.r3-200m.txt`
(raw decode, sha256 `34e8382e92e69710…`), `postproc.yaml` (exact config used).

Reproduce:

```bash
.venv-decode/bin/python scripts/translate.py \
    --input data/raw/englishdev.csv --output submissions/002/hyp.r3-200m.txt \
    --model models/r3-200m-full --device mps --beam 5 --batch-size 8
.venv/bin/python scripts/make_submission.py \
    --source data/raw/englishdev.csv --hypotheses submissions/002/hyp.r3-200m.txt \
    --postproc config/postproc_r3.yaml --output submissions/002/submission.csv
.venv/bin/python scripts/validate_submission.py \
    --source data/raw/englishdev.csv --submission submissions/002/submission.csv
```

## What it taught us — more valuable than the score

**R0 understated the gain by ~7x in relative terms.**

| | R0 | Leaderboard | ratio |
| --- | ---: | ---: | ---: |
| 1B zero-shot (sub 001) | 28.22 | 8.00 | 0.2835 |
| R3 200M (sub 002) | 28.64 | 8.83 | 0.3083 |
| delta | **+1.5%** | **+10.4%** | |

The R0 -> leaderboard ratio is **not a constant**. A fixed-multiplier prediction
gave 8.12; the real score was 8.83. Use R0 for direction and ranking, never as a
leaderboard predictor.

The earlier concern — that R0 being cut from BPCC would flatter a BPCC-fine-tuned
model — did not materialise. R0 appears to be a *compressed* proxy: its BPCC
references reward exact n-grams (R3 lost 0.33 BLEU on R0) while the real test
set rewards the character-level gains fine-tuning delivered (chrF++ +2.18).

**Practical consequence: small R0 gains are worth chasing.**

## Known defect in this submission

Output carries **zero kasra, damma and fatha** (0.00 per 100 chars against 2.31 /
1.36 / 1.00 in R0 references). Total diacritic density 4.64 vs 9.63. Fine-tuning
closed the hamza-below and inverted-damma gaps but not the three short vowels.

Visible failure: the gender hedge for "I am" is written `چھس/چھس` —
byte-identical on both sides — where references write `چھُس/چھَس`, distinguished
*only* by damma vs fatha. The model reproduces the form of the hedge while
dropping the mark that carries its meaning.

Root cause unresolved as of upload; see PLANNING.md. This is the largest known
lever remaining.
