# Submission 003 — R3 fine-tuned 200M + diacritic restoration

**Uploaded 2026-08-11. Leaderboard score: 11.16** (previous best 8.83). +26.4%.

Identical model and decode to submission 002 (LB **8.83**). The **only** change
is post-processing: `diacritic_lexicon` added. This isolates the effect of
restoration exactly — same weights, same beam, same row order.

## Why

IndicTrans2 **cannot emit kasra, damma or fatha.** In its 122,672-entry target
vocabulary each of the three appears in exactly ONE token — the bare standalone
mark — while every other Kashmiri diacritic is baked into whole-word subwords
(hamza-below: 378 tokens, inv-damma: 244). Writing `چھُس` would require emitting
`[چھ][ُ][س]`, splitting a word to insert a bare mark that occurs in no natural
subword context; beam search never does, because the undiacritized whole-word
token is always more probable.

Measured on R0: **exactly 0** occurrences of all three across 1,003 sentences,
against 270,799 / 105,855 / 61,885 in the training targets. Ruled out first —
the tokenizer preserves them (round-trip 8.90 → 8.83 per 100c) and
`postprocess_batch` preserves them (3.47 → 3.56). **The defect is the frozen
pretrained vocabulary, so no amount of fine-tuning can fix it.**

Cost of the gap on R0: a perfect translation missing only these three marks
scores **67.66**, not 100.

## The lexicon

`data/processed/diacritic_lexicon.json`, built by `data.diacritize` from
**train targets only** (no R0 leakage): 123,538 lines → 175,705 distinct
undiacritized keys → **6,996 entries** at `min_count=3`, `dominance=0.0`.

Both parameters swept on R0. **The dominance result inverted the prior:** the
argument for a guard was that 12.4% of keys are ambiguous — canonically the
gendered `چھُس` (masc.) vs `چھَس` (fem.) — and that a coin-flip should be left
bare rather than assert a gender. Measured, that is wrong:

| dominance | R0 geo |
| ---: | ---: |
| **0.0 (off)** | **31.20** |
| 0.6 | 30.30 |
| 0.9 | 29.14 |

A *wrong* diacritic beats *no* diacritic: the reference always carries a mark
there, so chrF++ gives partial credit either way while omission guarantees a
miss. `min_count` 1/2/3 tie at 31.20, so 3 is used — 5x smaller lexicon, no loss.

## Effect

| | BLEU | chrF++ | Geo |
| --- | ---: | ---: | ---: |
| sub 002 (no restoration) | 17.67 | 46.43 | 28.64 |
| **sub 003 (restored)** | **19.98** | **48.74** | **31.20** |

**+2.56 geo, +8.9% on R0.**

Transfer to the real test input is better than R0, not worse:

| | word coverage | diacritics/100c |
| --- | ---: | ---: |
| R0 (BPCC-derived) | 10.9% | — |
| `englishdev.csv` | **13.47%** | 5.65 → **8.31** |

R0 references sit at 9.63/100c, so output has moved most of the way. 1,030 of
1,730 rows changed (59.5%); ID order verified identical to 002.

Example — the gendered hedge for "I am", which the model wrote with no marks at
all on either side of the slash:

    002: بہٕ چھس/چھس پرٛیتھ دۄہ سکول گژھان۔
    003: بہٕ چھُس/چھَس پرٛیتھ دۄہ سکول گژھان۔
    ref: … چھُس/چھَس …

## Reproduce

```bash
.venv/bin/python -m data.diacritize build \
    --targets data/processed/r3_corpus/train/eng_Latn-kas_Arab/train.kas_Arab \
    --output  data/processed/diacritic_lexicon.json
.venv/bin/python scripts/make_submission.py \
    --source data/raw/englishdev.csv --hypotheses submissions/003/hyp.r3-200m.txt \
    --postproc config/postproc_r3_diacritized.yaml \
    --output submissions/003/submission.csv
.venv/bin/python scripts/validate_submission.py \
    --source data/raw/englishdev.csv --submission submissions/003/submission.csv
```

`hyp.r3-200m.txt` is byte-identical to submission 002's decode.

## Expectation

Predicted ≥9.6 from R0's +8.9%. **Actual: 11.16, +26.4% — R0 understated by
3.0x.** The lexicon transfers out of BPCC comfortably; coverage on the real test
input (13.47%) was in fact higher than on R0 (10.9%).

R0 has now understated three times running, and its ratio to the leaderboard
rises monotonically (0.2835 → 0.3083 → 0.3577). Treat R0 gains as a floor.
