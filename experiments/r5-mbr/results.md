# R5 — MBR decoding with chrF++ as utility

Branch `r5-mbr`. Started 2026-08-12.

## What this is testing

The last untried technical lever. The training line closed on 2026-08-12 (R4
1B LoRA scored **below** the 1B zero-shot), and post-processing is in
diminishing returns — 3.0x then 1.1x amplification across subs 003/004, with
submission 005 actively regressing.

MBR is the one remaining change where **R0 should rank honestly.** Every gain
since sub 003 came from a lookup table built on BPCC, and R0 is cut from BPCC;
that shared provenance is exactly what made sub 005 score 33.50 on R0 and 11.08
on the leaderboard. MBR consults no external table at all — it only compares the
model's own candidate outputs to each other — so there is nothing for R0's
provenance to reward. Under the revised trust rule (PLANNING.md 2026-08-12) this
falls squarely in the "trust R0" case.

## Method

Draw `pool_size` candidates per source, keep the one maximizing mean chrF++
against the rest of its own pool:

    y* = argmax_{y in Y} (1/|Y'|) sum_{y' in Y'} chrF++(y, y')

- **Utility is the competition's own metric.** `word_order=2`, `char_order=6`,
  `beta=2`, computed by reusing sacreBLEU's `CHRF._get_match_statistics` and
  `CHRF._compute_f_score` rather than reimplementing them.
- **Utility is computed on scorer-normalized text**, because that is what the
  metric sees. The RAW candidate at the winning index is what gets written, so
  the downstream `make_submission.py` path is byte-identical to sub 007's.
- **The diagonal is excluded** ("against all *other* candidates"). Duplicates
  still count with multiplicity — a string sampled 5 times contributes 4
  self-matches at 100 toward its own score. That is the consensus signal.
- **Order: MBR first, restoration second.** Restoration is a fixed transform
  applied identically to every candidate, so running it first would compress the
  differences the utility is measuring and turn part of the utility into a
  measure of lexicon agreement.

## Correctness checks run before any score was believed

| Check | Result |
| --- | --- |
| Fast utility vs `sacrebleu.sentence_chrf(word_order=2)` | **exact to 1e-9**, 80 brute-force pool comparisons, both `include_self` modes, duplicates injected |
| Argmax vs brute-force argmax | identical on all 80 |
| Checkpoint decodes real text before scoring | smoke probe in `generate_pool`, per PROJECT_NOTES.md §5 |
| Empty candidates | dropped before selection; `empty_pools` reported and fatal in `translate.py` |
| Pool merge alignment | sources asserted equal row-by-row across pool files |

### The bug this run found — `postprocess_batch` hangs, it does not raise

`IndicProcessor.postprocess_batch` pops one placeholder-entity map per *input*
from an internal `Queue`, using its `num_return_sequences` argument to decide
how many outputs share a map. Called with the default `1` on a 32-candidate
pool it drains the queue after the first `n/32` sentences and then **blocks
forever** on `Queue.get()`. It also **clears the queue on return**, so a second
postprocess call in the same batch (the beam candidate) blocks too.

Diagnosed with `sample(1)` on a stalled process — 0% CPU, stack parked in
`lock_PyThread_acquire_lock` under `postprocess_batch`. This is the same trap
already on record for in-training eval (PLANNING.md 2026-08-09), in a second
place. Fixed in `_decode()`: pass the real pool width, and refill the queue with
a `preprocess_batch` call (pure string work, no GPU cost) before every
postprocess.

**Left unfixed this would have burned a Kaggle session with no error message.**

## Results — R0 (1,003 pairs, 6.88 src words). Tokenizer 13a.

Single-system pool: R3 200M, epsilon sampling (cutoff 0.02, T=1.0), 31 samples
plus the beam-5 hypothesis as candidate 0.

| System | Decode | BLEU | chrF++ | Geo | vs sub 007 |
| --- | --- | ---: | ---: | ---: | ---: |
| R3 200M, raw | beam 5 | 17.67 | 46.43 | 28.64 | — |
| R3 200M, raw | **MBR-32** | 17.14 | 46.10 | **28.11** | −0.53 |
| **R3 200M + lexicon (sub 007)** | beam 5 | 22.27 | 49.99 | **33.36** | — |
| R3 200M + lexicon | **MBR-32** | 21.52 | 49.61 | **32.67** | **−0.69** |

**MBR loses, and it loses at every pool size.** Free CPU re-selects over the one
generated pool:

| pool | 2 | 4 | 8 | 16 | 24 | 32 | beam |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| geo +diac | 30.67 | 31.92 | 32.45 | 32.35 | 32.44 | 32.67 | **33.36** |

Flat from 8 candidates onward. A larger pool will not close a 0.69 gap.
`include_self` and `utility_normalize` changed nothing to two decimals — the
argmax is robust to both, so neither is worth another knob.

## Why it loses — diagnosed, not guessed

Per-sentence chrF++ against the true reference, averaged over 1,003 pools:

| | chrF++ |
| --- | ---: |
| pool ORACLE (best possible pick) | **55.25** |
| beam candidate | 44.44 |
| **what MBR actually picked** | **44.19** |
| pool MEAN candidate | 39.60 |
| pool WORST candidate | 23.73 |

**The pool is rich; the utility cannot find the good candidates.** A sample beats
beam in 853 of 1,003 pools, and oracle selection over the very same pool scores
geo **39.08 raw / 44.95 with the lexicon** — against sub 007's 33.36. That is
+11.6 geo sitting in candidates already generated.

But MBR's pick ranks **#9.3 of 32** and averages 44.19 chrF++, statistically
indistinguishable from beam's 44.44. Consensus over a sampled pool finds the
*mode*, and for this model the mode is what beam already returns — MBR agrees
with beam outright on 585 of 1,003 sentences, and when it deviates it does
slightly worse.

This is the expected failure mode for a **similarity-based** utility. chrF++
measures typicality, not quality, so it cannot rank candidates by correctness.
Published MBR gains of this kind come from *neural* utilities (COMET, BLEURT)
that correlate with human judgment. There is no Kashmiri COMET model, and
training one is not a five-day project.

The +11.6 geo oracle gap is real and is the largest measured headroom left in
the project — but it needs a reranker that knows quality, not agreement.

## Wall-clock (measured, CPU — M2 Air, batch 4)

| stage | total | per sentence |
| --- | ---: | ---: |
| generate 32 candidates | 2,498.9 s | **2.491 s** |
| MBR selection (1,003 pools) | 16.9 s | **0.017 s** |
| **total** | | **2.508 s** |

Selection is 0.7% of the cost; generation is everything. On a T4 the generation
term falls by roughly an order of magnitude, but the ratio to beam search is
unchanged — MBR-32 is ~30x a beam-5 decode for a **negative** result, so the
latency question is moot. `config/decode_beam_fallback.yaml` stays the live-round
configuration.

## Diacritic density — the sub-005 warning sign does not fire

| | diacritics /100c | lines with a diacritic |
| --- | ---: | ---: |
| R0 references | **9.63** | 95.51% |
| beam 5 + lexicon (sub 007) | 7.85 | 91.82% |
| MBR-32 + lexicon | **7.94** | 92.82% |

MBR moves very slightly *toward* reference density, not away. Both signals
therefore agree: this is a genuine quality loss, not the provenance-overfitting
pattern that made sub 005 look good on R0 and regress on the leaderboard.

## Verdict

**Negative. Do not submit.** MBR with a chrF++ utility scores 32.67 on R0
against sub 007's 33.36, at ~30x the decode cost, and the loss is consistent
across every pool size and every utility convention tested. The code, configs
and Kaggle runner are kept because the oracle diagnostic they produced is the
most valuable result here: **+11.6 geo of reachable headroom that a quality-aware
reranker could claim.**

## Cross-system pool — implemented, NOT run

`config/mbr_r5_1b.yaml` + `config/mbr_r5_cross.yaml` + `scripts/run_r5_kaggle.sh`
do this end to end. It was not run because the 1B cannot be decoded on this
machine — beam-5 at batch 32 already forced an involuntary shutdown
(PLANNING.md 2026-08-10), and 32 return sequences is heavier again. It needs a
T4 session.

The single-system diagnostic lowers the prior but does not settle it. Consensus
*across two models* is a different signal from consensus among one model's own
samples — it tests whether two independently-trained systems agree, which does
carry quality information that within-model typicality does not. The entity
analysis found concrete cases where the 1B is right and R3 wrong (ID 789
`70 → 71`, ID 1679 `North London → New York`). Against that: the 1B zero-shot
scores 29.49 with the lexicon against R3's 33.36, so its candidates are weaker
on average, and `winners_by_system` in the selection stats is the number that
decides whether the pool bought consensus or just let one model outvote the
other.
