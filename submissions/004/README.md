# Submission 004 — R3 200M + context-aware diacritic restoration

**Uploaded 2026-08-11. Leaderboard score: 11.70** (previous best 11.16). +4.8%.

Identical model and decode to submissions 002 and 003. The only change from 003
is left-context disambiguation in the lexicon.

## The change

The unigram lexicon always returns a key's majority form regardless of context,
which is wrong roughly half the time on gendered verbs — `چھُس` (masc.) vs
`چھَس` (fem.) for "I am", distinguished only by damma vs fatha.

Submission 004 adds a bigram table keyed `previous_word \t word`, both stripped,
kept only where it is frequent enough AND disagrees with the unigram answer.
Unigram is the backoff. Context is read from the **raw** previous token rather
than the restored one, so an early mistake cannot cascade along the sentence.

| | R0 geo |
| --- | ---: |
| no restoration (sub 002) | 28.64 |
| unigram only (sub 003) | 31.20 |
| **+ left context (sub 004)** | **32.56** |

+1.36 over unigram; +13.7% over no restoration. Diacritic density 8.84/100c
against references at 9.63. 380 of 1,730 rows changed vs 003; ID order verified.

`bigram_min_count=2` is a genuine optimum, not a size compromise: 1 scores 32.53
with 5x the entries, 5 scores 32.33.

## What this submission established

**R0 has become a ~1:1 proxy for the leaderboard.**

| step | R0 | LB | amplification |
| --- | ---: | ---: | ---: |
| 001->002 | +1.5% | +10.4% | 7.0x |
| 002->003 | +8.9% | +26.4% | 3.0x |
| 003->004 | +4.4% | +4.8% | **1.1x** |

The R0->LB ratio rose across the first three submissions and has now flattened
(0.3577 -> 0.3593). The amplification WAS the diacritic defect: while the model
carried a systematic flaw touching every sentence, R0 understated fixes for it.
With the flaw corrected, both sets move together.

Iterate offline and believe R0 from here. Re-check the ratio after any change
that alters output systematically — a new base model or a vocabulary change is
exactly the condition under which it moved before.

## Reproduce

```bash
.venv/bin/python -m data.diacritize build \
    --targets data/processed/r3_corpus/train/eng_Latn-kas_Arab/train.kas_Arab \
    --output  data/processed/diacritic_lexicon.json
.venv/bin/python scripts/make_submission.py \
    --source data/raw/englishdev.csv --hypotheses submissions/004/hyp.r3-200m.txt \
    --postproc config/postproc_r3_diacritized_ctx.yaml \
    --output submissions/004/submission.csv
.venv/bin/python scripts/validate_submission.py \
    --source data/raw/englishdev.csv --submission submissions/004/submission.csv
```

`hyp.r3-200m.txt` is byte-identical to submissions 002 and 003.
