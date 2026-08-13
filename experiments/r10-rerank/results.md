# R10 — Feature-weighted candidate reranking

Branch `r5-mbr` (same pool, new selector). Run 2026-08-12.

## Verdict, up front

**NEGATIVE. The reranker does not beat sub 007 on held-out data, on any of
three splits. Nothing submitted.**

| split seed | beam / sub 007 | reranked | Δ | weights the tuner returned |
| --- | ---: | ---: | ---: | --- |
| 0 | **34.95** | 34.86 | **−0.09** | `w_urdu 1.0`, everything else 0 |
| 1 | **34.43** | 34.43 | **0.00** | all zero — i.e. "do not rerank" |
| 2 | **34.10** | 34.00 | **−0.10** | `w_urdu 1.0`, everything else 0 |

Held-out R0 geo, 301 sentences, tokenizer 13a, production lexicon applied. The
three baselines differ only because the splits contain different sentences; all
three are the same system scoring 33.36 over the full 1,003.

On the full R0 the reranked system scores **33.38 against 33.36** — +0.02, which
is a rounding artifact of 16 changed sentences out of 1,003, not a gain.

## What was tested

Re-selection over the **existing** R5 pool: 1,003 R0 sentences × 32 candidates
(31 epsilon samples + the beam-5 hypothesis as candidate 0), R3 200M full
fine-tune. No new generation, no GPU. Each candidate scored on a weighted sum of
five features, all oriented higher-is-better and all O(1) so the weights are
directly comparable:

| feature | definition |
| --- | --- |
| `consensus` | mean chrF++ against the rest of the pool ÷ 100 — the R5 MBR utility |
| `density` | −\|diacritics/100c − 9.63\| ÷ 9.63, on the **restored** candidate |
| `lexcov` | fraction of tokens the restoration lexicon can act on |
| `length` | −\|out chars ÷ src chars − 1.037\| ÷ 1.037 |
| `urdu` | −(Urdu function-word tokens ÷ tokens) |

Weights fitted by multi-start random search plus coordinate descent (≈1,320
objective evaluations, 126 s) maximizing corpus geo on a **702-sentence tuning
split**, evaluated on the **301 held out**. The tuner never sees the held-out
split. Ablations re-tune the remaining four weights from scratch, so each row
answers "is this term doing work", not "does this weight matter at these other
weights".

## Harness fidelity — checked before any number was believed

| check | result |
| --- | --- |
| beam (pool candidate 0) vs the saved sub 007 R0 decode | **1,003 / 1,003 byte-identical** |
| beam + lexicon | **33.36** — reproduces sub 007 exactly |
| fast corpus scorer vs `data.normalize.score` | agree to <0.01 geo; asserted at runtime, fatal on mismatch |
| consensus-only selection | **32.67**, 585 beam agreements — reproduces R5's MBR exactly |
| reference density / length ratio re-measured from the reference file | 9.63/100c, 1.037 — match the config; >5% drift is fatal |
| oracle on raw text | **39.08 raw / 44.95 +lexicon** — reproduces R5 exactly |

## Results — R0, 13a, geo with the production lexicon (seed 0)

| system | tune (702) | held-out (301) | full (1,003) | raw | density | = beam |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beam / sub 007 | 32.65 | **34.95** | **33.36** | 28.64 | 7.85 | 1003 |
| MBR (consensus only) | 32.30 | 33.49 | 32.67 | 28.11 | 7.95 | 585 |
| **RERANKED** | 32.72 | **34.86** | 33.38 | 28.68 | 7.86 | 987 |
| ORACLE (ceiling) | 45.70 | 46.39 | **45.92** | 38.19 | 8.08 | 142 |

The tuner gained **+0.07 on the split it was fitted to and lost 0.09 on the
split it was not.** That is the signature of a search fitting noise, and it is
exactly what the held-out split was built to catch.

**Oracle ceiling: 45.92 on the full set, 46.39 held out** — +12.56 over sub 007,
slightly larger than R5's +11.6 because this oracle picks the candidate that is
best *after* restoration, which is what the production pipeline actually emits.
Selecting on raw text reproduces R5's 44.95 exactly. **The reranked system
captures 0.2% of that headroom.**

## Ablation — drop one feature, re-tune the other four (held-out, seed 0)

| configuration | held-out geo | Δ vs full model |
| --- | ---: | ---: |
| full model | 34.86 | — |
| without `consensus` | 34.86 | +0.00 |
| without `density` | 34.86 | +0.00 |
| without `lexcov` | 34.86 | +0.00 |
| without `length` | 34.86 | +0.00 |
| without `urdu` | 33.34 | −1.52 |

Four of the five features contribute **exactly nothing** — removing them does
not change the selection, because their learned weight was already zero. The
`urdu` row is not evidence that the Urdu term works: with it clamped off, the
search is forced away from the near-beam solution, finds something better on the
tuning split and generalizes worse. That is more overfitting, not a lost gain.

## Why it fails — the features, not the search

The weight search is not the bottleneck. Measured per feature, with no search
involved: for each sentence, every candidate's **true** chrF++ against the
reference, then how well the feature ranks the candidates inside that pool.

| feature | mean within-pool ρ vs true chrF++ | solo pick rank (of 32) | beats beam | solo geo |
| --- | ---: | ---: | ---: | ---: |
| `consensus` | **0.406** | 9.47 | 19.2% | 32.67 |
| `density` | 0.005 | 18.56 | 24.9% | 23.98 |
| `lexcov` | 0.028 | 17.26 | 24.7% | 25.40 |
| `length` | 0.048 | 17.96 | 25.3% | 24.46 |
| `urdu` | 0.011 | 9.40 | 0.7% | 33.38 |
| *(oracle)* | 1.000 | 1.00 | — | 45.92 |

**Four of the five features have essentially zero rank correlation with quality
inside a pool (ρ 0.005–0.048).** A feature that cannot rank a pool cannot help
select from one, at any weight — which is why the search returns zero for them
and why no cleverer optimizer would do better.

The diagnosis is that **these are corpus-level statistics being asked to do a
per-sentence job.** 9.63/100c and 1.037 are properties of the R0 reference
*corpus*; an individual 7-word reference may legitimately carry a density of 0
or 20. The features do vary across candidates — within-pool raw density spans a
mean min of 1.20 to a mean max of 9.38 per 100c — so the failure is not that the
signal is absent, it is that proximity to a corpus mean is uncorrelated with
being the right translation of *this* sentence.

`urdu` fails for a different and more mundane reason: it is **sparse**. Only 576
of 32,096 candidates (1.79%) contain a flagged token, and only 71 of 1,003 pools
(7.1%) have any variation at all. In 93% of pools the feature is constant zero,
every candidate ties, and the argmax falls through to candidate 0 — beam. So the
`urdu`-only solution the tuner returned *is* beam, on 987 of 1,003 sentences.

`consensus` is the only feature carrying real signal (ρ 0.406), and R5 already
established that its argmax lands at rank 9.4 of 32 and scores 0.69 below beam.
Adding four uninformative terms to one insufficient term does not fix it.

## Provenance — the flagged feature cost nothing

`lexcov` reads the BPCC-derived restoration lexicon, the same provenance that
made submission 005 look good on R0 and regress on the leaderboard. **Its
learned weight is 0.000 on all three splits**, and the `without_lexcov` ablation
is identical to the full model to two decimals. The provenance question is
therefore moot here: the provenance-carrying feature is not used. Reporting with
and without it gives the same number.

`density` reads the lexicon indirectly (restoration is what supplies
kasra/damma/fatha at all); `density_on: raw` exists to remove that coupling, but
with a learned weight of zero there was nothing to decouple.

## Density check

| | diacritics /100c |
| --- | ---: |
| R0 references | **9.63** |
| beam + lexicon (sub 007) | 7.85 |
| reranked + lexicon | 7.86 |
| MBR-32 + lexicon | 7.95 |
| oracle + lexicon | 8.08 |

The sub-005 warning sign does not fire — but there is also nothing to warn
about, since the reranked output is beam on 987 of 1,003 rows.

## Cost

| stage | total | per sentence |
| --- | ---: | ---: |
| featurize 1,003 × 32 (chrF++ matrix + 4 features) | 17.7 s | 0.018 s |
| one weight evaluation (702 sentences) | ~0.10 s | — |
| full tune: 1,320 evaluations + 5 ablations | ~10 min | — |

CPU, M2 Air, main `.venv`. No GPU, no model load. Selection remains 0.7% of the
cost of a decode, so re-sweeping a generated pool stays free — which is what
made a negative result cheap to establish.

## What would be needed to claim the +12.56

Not a better optimizer over these features. The oracle gap is real and the pool
is rich (a sample beats beam in 853 of 1,003 sentences), but claiming it needs a
selector that estimates **translation quality for this source sentence** —
which means conditioning on the source, which none of these five features do.
That is a learned metric (COMET/BLEURT-shaped), and R5 already recorded that
there is no Kashmiri COMET and training one is not a five-day project.

One thing measured here that was not known before: `consensus` at ρ 0.406 is a
genuinely informative ranker that simply picks too conservatively. If anything in
this direction is ever revisited, the honest starting point is that a *source-
conditioned* model is required, not more source-blind heuristics.

## Deliverables kept

- `src/decode/rerank.py` — featurizer, weighted selector, multi-start tuner with
  held-out evaluation, per-feature diagnostics. Reproduces R5's MBR, R5's oracle
  and sub 007's beam exactly, which is why the negative result is trustworthy.
- `config/rerank_r10.yaml` — weights as tunable keys, unknown keys rejected.
  Left at the defaults (consensus-only); no tuned vector is worth recording.
- `scripts/translate.py --rerank CONFIG` — the live round keeps one entry point.
- `data/dev/r0/pools/pool.r0.200m.jsonl` — the R5 pool, **rescued from a session
  scratchpad into the repo tree**. It cost 42 minutes of CPU decode and was one
  cleanup away from being lost.
- `experiments/r10-rerank/tuning_seed{0,1,2}.json` — full reports.
