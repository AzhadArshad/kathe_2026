# R11 — Learned character-level diacritic restoration

Branch `r11-restore`. Run 2026-08-13, Kaggle T4.

## Verdict, up front

**NEGATIVE as trained. Nothing submitted.** The learned restorer scores **30.41
on R0** against the production lexicon's **33.36**, and every hybrid of the two
lands at or below the lexicon alone. But this is a negative result about a
6-epoch run, not about the approach, and the diagnostic below says which.

| System | R0 geo | vs sub 007 |
| --- | ---: | ---: |
| Sub 007, production lexicon (`context_mode: left`) | **33.36** | — |
| Learned restorer alone (best `none_bias`, −0.5) | 30.41 | **−2.95** |
| Lexicon first, learned model on what it leaves unmarked | 33.16 | **−0.20** |
| *(clean lexicon alone, leak-free — see §Leak)* | *33.00* | *−0.36* |
| *(ceiling: perfect restoration)* | *40.63* | *+7.27* |

## Why it fails, measured rather than guessed

The whole argument for a learned model was the lexicon's **tail**: unseen words
return nothing, and Kashmiri's morphological tail is long. That premise is
correct — measured on R0's references against the leak-free lexicon:

| | word tokens | share of all reference marks |
| --- | ---: | ---: |
| lexicon knows the form | 1,178 (15.8%) | 1,079 (**56.8%**) |
| lexicon does **not** | 6,300 (84.2%) | 819 (**43.2%**) |

**43% of the marks are in the tail.** The prize is real and the lexicon
structurally cannot reach it.

The model does not reach it either. Per-mark precision/recall on R0, split by
whether the lexicon already knows the word:

| `none_bias` | KNOWN words P / R / F1 | **TAIL words P / R / F1** |
| ---: | ---: | ---: |
| 0.0 | **84.0** / 20.2 / 32.6 | **34.7** / 4.2 / 7.5 |
| −0.5 | 77.0 / 26.5 / 39.4 | 28.9 / 7.7 / 12.2 |
| −1.0 | 68.8 / 34.8 / 46.2 | 23.9 / 13.6 / 17.4 |
| −1.5 | 60.0 / 43.2 / 50.2 | 19.6 / 21.9 / 20.7 |
| −2.0 | 51.9 / 52.7 / 52.3 | 15.7 / 30.2 / 20.7 |

**The model is accurate exactly where the lexicon already is, and inaccurate
exactly where the lexicon is silent.** 84% precision on known forms against
34.7% on the tail, and tail precision *falls* to 15.7% as the bias buys recall.

So it has learned the lexicon — frequent word forms memorised as character
patterns — and has not learned generalisable morphology. That explains the
pipeline result exactly: on known words it duplicates what the lexicon does
better, and on unknown words it contributes noise. There is nothing for a merge
rule to exploit, which is why `known` and `changed` give identical scores to two
decimals and both sit below the lexicon alone.

## Two things that are NOT the problem, both checked

Recording these because both are plausible, both were believed here for a while,
and neither survives measurement. Anyone picking this up will think of them.

**1. Output density calibration is not the problem — matching it makes things
worse.** The obvious diagnosis is that the model marks too sparsely (0.90/100c
at `none_bias` 0 against references at 4.68), and that `none_bias` should be
tuned until the two agree. It was, and the score moves the wrong way:

| `none_bias` | output density | R0 geo |
| ---: | ---: | ---: |
| −0.5 | 1.40 | **30.41** |
| −1.0 | 2.28 | 29.58 |
| −1.5 | 3.70 | 28.31 |
| −2.0 | 5.65 | 26.26 |

The best score sits where density is **a third** of the reference, and the score
falls monotonically as density approaches it. A mis-calibrated marking prior is
therefore not what costs the points — if it were, this knob would recover them
for free. It does the opposite, because the extra marks it buys are wrong ones.

This also settles the density question for R11 the way PLANNING.md 2026-08-12
settled it for the lexicon: **density is an anomaly detector, not an objective.**

**2. The training/application LENGTH mismatch is real but not load-bearing.**
R11 trains on text averaging 91 characters and is applied to output averaging
37; 74% of its training marks come from sentences over 80 characters, which are
1.5% of the test set. That looked like the defect. It is not, for two reasons:

* Within a single subset, density barely moves with length — `daily` runs
  6.76 / 6.75 / 5.85 / 5.42 / 5.78 across 0-30c to 131c+, a **1.25× drift over
  5× the length**. Between subsets at *fixed* length the spread is **5.6×**
  (`nllb-filtered` 1.28 vs `bpcc-seed-v1` 7.15 at 31-50c). Source dominates
  length roughly four to one.
* Restricting the corpus to 1-10 English words moves density 3.80 → **3.55**,
  i.e. nothing, and in the wrong direction.

The apparent "R0 density falls with sentence length" (6.70 → 4.93 → 3.75) is a
**composition artifact**: R0's short band is 90% `daily` (6.76/100c) and its long
band is 51% `nllb-seed` (1.79/100c). It is the same source effect wearing a
length costume.

## What R0's density actually is, and the caveat that follows

R0 filters on the **English** side — stratified by English word count over 1-10
words to match `englishdev.csv`, with `daily` over-weighted to ~59% and mined
pairs excluded. Kashmiri length is never inspected. Decomposing its 4.68:

| | restorable /100c |
| --- | ---: |
| whole cleaned corpus | 3.80 |
| restricted to 1-10 English words (natural mix) | **3.55** |
| + human-only (drops `nllb-filtered`) | 4.06 |
| + R0's `daily`-weighted mix | 4.53 |
| **R0 actual** | **4.68** |

**About two-thirds of R0's density is a construction choice, not a property of
short Kashmiri.** Length contributes nothing; excluding mined text and
over-weighting `daily` contribute all of it — decisions made on 2026-08-10 for
register reasons, with orthographic density as an unexamined side effect.

**Consequence: the KATHE test set's diacritic convention is unknown.** Tuning
restorer output toward 4.68/100c is tuning toward our own sampling decision. The
only evidence about the real references is the leaderboard, and it holds one
relevant comparison — subs 006 vs 007, where the *lower*-density lexicon scored
11.81 against 11.74. A 0.07 gap is noise. Treat any density target as unverified.

## The corpora do not share one convention

Per-sentence density (marks per 100 characters *of that sentence*), which
separates "skips sentences" from "marks lightly" in a way the corpus-level
average cannot:

| subset | mean | P50 | P75 | P95 | %zero | **mean given >0** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bpcc-seed-v2` | 3.48 | 2.80 | 5.69 | 9.72 | 27.9% | 4.82 |
| `bpcc-seed-v1` | 6.86 | 6.67 | 8.79 | 12.00 | **0.8%** | 6.92 |
| `bpcc-seed-latest` | 3.00 | 1.49 | 5.49 | 10.13 | 44.6% | 5.42 |
| `nllb-seed` | 1.65 | 1.27 | 2.48 | 5.00 | 28.2% | **2.30** |
| `daily` | 6.18 | 5.68 | 8.33 | 13.33 | 8.6% | 6.76 |
| `nllb-filtered` | 1.07 | 0.00 | 1.60 | 5.00 | 56.0% | **2.42** |
| external | 5.77 | 5.42 | 9.46 | 14.06 | 24.3% | 7.62 |
| qamar30k | 2.81 | 0.00 | 4.55 | 11.58 | 50.0% | 5.61 |
| **R0 references** | 5.14 | 4.17 | 7.89 | 13.64 | 22.1% | **6.60** |

Read the last column. There are **three regimes**, not a gradient:

* **matches R0** (6.6-7.6 when marked): `daily`, `bpcc-seed-v1`, external
* **marks lightly** (4.8-5.6): `bpcc-seed-v2`, `bpcc-seed-latest`, qamar30k
* **a different convention** (2.3-2.4): `nllb-seed`, `nllb-filtered` — even
  their *marked* sentences carry a third of R0's density

Composition of the restorer's actual training text, by tag:

| tag | lines | % lines | marks | % marks | /100c | regime |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bpcc:bpcc-seed-v2` | 76,772 | 45.5% | 255,990 | 37.7% | 3.68 | light |
| `external` | 39,948 | 23.7% | 222,030 | 32.7% | 6.72 | **matches R0** |
| `qamar30k` | 16,505 | 9.8% | 29,172 | 4.3% | 2.14 | **sparse** |
| `bpcc:bpcc-seed-v1` | 15,460 | 9.2% | 117,846 | 17.3% | 6.88 | **matches R0** |
| `bpcc:bpcc-seed-latest` | 10,477 | 6.2% | 28,765 | 4.2% | 3.16 | light |
| `bpcc:nllb-seed` | 5,566 | 3.3% | 12,681 | 1.9% | 1.61 | **sparse** |
| `bpcc:daily` | 3,059 | 1.8% | 12,630 | 1.9% | 5.81 | **matches R0** |
| `bpcc:nllb-filtered` | 902 | 0.5% | 620 | 0.1% | 1.14 | **sparse** |

**The sparse-convention share is 13.6% of lines but only 6.2% of marks** — not
the ~24% quoted from the translation corpus's pair counts, which is the wrong
denominator here. Deduplicating on stripped form collapses `nllb-filtered` from
9.8% of training pairs to **902 lines**, because nearly all of its targets
already appear in another subset. It is very nearly a non-entity for this task.

That materially lowers the expected effect of the provenance arm below: dropping
these removes 14% of the lines and 6% of the supervision, and the bulk of what
goes is qamar30k rather than the two genuinely-different-convention subsets.
Worth testing, no longer worth much optimism.

### Incidental: the LaBSE filter selects against diacritics

Cached scores for the 111,826 mined pairs, restorable density by band:

| LaBSE | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | **≥0.8 (kept)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| /100c | 4.48 | 3.32 | 2.21 | ~1.5 | **1.09** |

Monotonic. The gate discards text **1.82× more diacritized** than what it keeps
— LaBSE's weak Kashmiri coverage (on record 2026-08-07) gets weaker still when
diacritics fragment its tokenization, so it scores correctly-marked Kashmiri as
badly aligned. Filtering also lowered density *within* every subset it touched
(`seed-v2` −0.41, `seed-latest` −0.83, `nllb-filtered` −0.73) while raising the
corpus average +0.26 — Simpson's paradox via the change in mix.

**Not worth acting on.** Recovering the LaBSE-rejected text for a monolingual
task, deduplicated on stripped form against what R11 already has, yields
**+13,208 lines (+12%) and +3% more marks, at 1.98/100c** — below the 3.80 it
already trains on. Nearly all of it is already represented. Recorded as a
diagnosis, not a lever. (A first pass said +53,181 lines; that was a
whitespace-keying error.)

## The reason to retrain rather than close this

**The model is under-trained, and not marginally.** Held-out micro-F1 rose at
every single epoch and the loss was still falling when the schedule ran out:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| held-out micro F1 | 3.77 | 4.99 | 8.23 | 11.04 | 13.39 | **14.93** |
| held-out loss | .1864 | .1812 | .1790 | .1744 | .1688 | **.1657** |

No plateau, no overfitting, and the linear-decay LR reached ~0 at epoch 6 — so
the run ended because the schedule said so, not because the model stopped
learning. This is the same signature R3 showed (PLANNING.md 2026-08-10:
"under-trained, not over-trained").

Frequent forms being learned first and the tail last is the expected order for
this kind of model. A 6-epoch checkpoint sitting at 84% precision on frequent
forms and 35% on rare ones is what a model looks like partway through, not
necessarily what it converges to.

**The number to watch on a retrain is TAIL precision, not micro-F1.** Micro-F1
will rise regardless, driven by frequent forms that are already handled. Nor
density, and nor held-out loss — both improve while the thing that decides the
score does not. If tail precision does not clear roughly 50% at useful recall,
the approach is closed on evidence rather than on a first attempt.

### The retrain, as two arms

More epochs alone is the weaker half of the experiment. Pair it with the one
distributional problem that survived scrutiny — the 24% of training text
following a materially sparser convention:

| arm | training text | question it answers |
| --- | --- | --- |
| **A** | all sources, 18-20 epochs | was 6 epochs simply too few? |
| **B** | drop the three sparse-convention tags — 145,716 lines vs A's 168,689 | does a consistent convention beat more data? |

B trains on 86% of A's lines, so if B wins it is not winning on volume. Both
judged on tail precision. Cost: 948 s for 6 epochs on one T4, ~50 min per arm.

`build_text.py` now tags BPCC per subcorpus (`bpcc:daily`, `bpcc:nllb-seed`, …)
and `--sources` matches either the coarse name or the fine tag, so both arms run
off one corpus file:

    # arm A
    python -m restore.train fit --corpus data/processed/restore_text.jsonl \
        --out models/restore/r11_A.pt --epochs 20 --device cuda \
        --refs data/dev/r0/r0.kas_Arab

    # arm B
    python -m restore.train fit --corpus data/processed/restore_text.jsonl \
        --out models/restore/r11_B.pt --epochs 20 --device cuda \
        --refs data/dev/r0/r0.kas_Arab \
        --sources bpcc:bpcc-seed-v2 bpcc:bpcc-seed-v1 bpcc:bpcc-seed-latest \
                  bpcc:daily external

Note the BPCC input is now `bpcc_kas_clean.jsonl` rather than the R3 training
file, so the per-subset tag survives. The dev exclusion is unchanged and does
the same job — `DEV-LEAK 3,094` where it was 1,096, the difference being the
2,000 held-out pairs that the R3 file had already removed. Resulting corpus:
168,689 lines against the 168,679 R11 actually trained on, i.e. the same corpus.

## Provenance — the multi-source model wins, as designed

| model | held-out micro F1 | R0 geo (bias 0) | R0 geo (bias −0.5) |
| --- | ---: | ---: | ---: |
| BPCC only (107,246 examples) | 5.74 | 29.45 | 29.44 |
| **all three sources** (164,937) | **14.93** | **30.33** | **30.41** |

**All-sources beats BPCC-only by 2.6× on held-out F1 and ~0.9 geo on R0.** The
sub-005 failure mode — a BPCC-derived table scoring higher on R0 (cut from BPCC)
and lower on the leaderboard — does **not** appear here: the multi-source model
wins on R0 itself, which is biased *toward* BPCC. Nothing needs discounting.

The lexicons behave the same way: clean-all 33.00 vs clean-BPCC-only 32.47.

## The leak this work uncovered

**419 of R0's 1,003 reference sentences appear verbatim in
`nawabhussain/Kashmiri-Language-Corpus`** (and 410 of the 997-line eval slice).
Submission 007's lexicon was built over all 47,344 of those lines, so 42% of the
dev set was in its training text.

| lexicon | source text | R0 geo |
| --- | --- | ---: |
| sub 007 production | BPCC train + external, **unfiltered** | **33.36** |
| clean | identical corpora, R0 + eval slice removed by stripped string | **33.00** |

**The leak is worth +0.36 geo on R0.** Submission 007's leaderboard score of
11.81 is unaffected — the test set is separate from R0 — but R0 comparisons
against it are inflated by that much, and every restorer benchmarked here is
measured against both numbers for that reason.

The leak passed every existing check because they were the wrong shape: pairs
were held out of training by `(src, tgt)` key, and the lexicon was built from
train targets only. Nobody checked whether an *external monolingual corpus*
contained the dev references as bare sentences. `restore/build_text.py` now
excludes by stripped-and-normalized string across every source.

## Data

168,679 lines / 15.3M chars, scorer-normalized, deduped across sources, R0 and
the eval slice removed:

| source | read | kept | dropped: dup | **dev leak** | restorable /100c |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPCC train targets | 123,538 | 112,229 | 10,124 | **1,096** | 4.03 |
| `nawabhussain/Kashmiri-Language-Corpus` | 47,344 | 39,945 | 5,612 | **831** | 6.72 |
| `SMUQamar` 30K (`HuggingFace 30K/Kashmiri.txt`) | 29,999 | 16,505 | 13,225 | 3 | 2.14 |
| **total** | | **168,679** | | | **4.44** |

R0's references sit at **4.68 restorable/100c**, so BPCC and the external corpus
bracket the target convention and the 30K sits well below it.

### On the 30K, which was checked before being included

Its unique contribution is the thinnest of the three: 13,225 of its 29,999 lines
duplicate BPCC or the external corpus, and what remains carries **2.14
marks/100c with 54.9% of lines carrying none at all**. Measured against a
per-word expectation learned from BPCC + external, corpus-wide
observed/expected: BPCC 0.93, external 1.16, R0 references 1.06, **30K 0.71**.
It is systematically under-diacritized. Kept anyway as a third provenance — it
is 9.8% of the training text and the provenance ablation is the check on
whether that was right.

### `injilashah/Kashmiri-terms` was rejected

396 rows, and they are **English** — ASR prompt text ("In the serene valleys of
Kashmir, Chashm-e-Shahi offers refreshing water…"). Zero Perso-Arabic, **0.00
marks/100c**. Not a Kashmiri corpus despite the name.

### Mark-free lines are kept deliberately

25–55% of lines in every source carry no restorable mark. They are
under-diacritized rather than genuinely mark-free — expected 0.16–0.22
marks/token against 0 observed — **but so are 22.1% of R0's own human
references**, at the same expectation. Filtering them would train the model on a
denser convention than the one it is scored against. `--min-restorable` is the
knob; default 0.

## Design — insertion-only is structural, not checked-for

The brief asked for a sequence model that cannot alter base characters. Rather
than generate text and validate it, the model never emits text: for each
character of the input it predicts one of {none, kasra, damma, fatha}, and the
label sequence is applied back.

    input   چ ھ س          one position per input character
    predict N N N          4-way classification per position
    output  چھُس

The base-letter sequence is not an output, so it cannot change — for *any* label
vector, including a maximally wrong one. Verified:

| check | result |
| --- | --- |
| round-trip `encode`→`apply_labels` over 124,541 corpus lines | 0.159% of marks lost, all multi-mark typo runs and marks on spaces |
| chunking exactly reversible | 8,000 random cases + words longer than `max_len` |
| base preserved for random label vectors | 500/500 |
| `assert_insertion_only` on every restored line | enforced at inference |
| marks after whitespace / line-initial, at `none_bias` −20 | **0 / 0** of 32,202 inserted |

A mark following a space is corpus noise (345 of 679,144) but a negative
`none_bias` will manufacture them, so `encode` refuses the label and `Restorer`
masks the position — both sides, so training and inference cannot disagree.

## Harness fidelity — checked before any new number was believed

| check | result |
| --- | --- |
| raw R3 200M decode of R0 | **28.64** — matches PLANNING.md |
| sub 004 lexicon | **32.56** — matches |
| sub 007 lexicon | **33.36** — matches |
| perfect-restoration ceiling | **40.63** — matches |
| untrained model in hybrid mode | falls back to exactly the lexicon (33.00) |

The `data/diacritize.py` refactor that added `lookup()` is therefore faithful.

## Model and cost

3,331,844 parameters: 4-layer bidirectional transformer encoder, d_model 256,
4 heads, d_ff 1024, learned positions to 384, character vocabulary of 285.

| | |
| --- | ---: |
| training, 6 epochs, 164,937 examples, one T4 | **948 s** |
| BPCC-only control | 654 s |
| **restoring 1,730 test rows, CPU** | **5.63 s (3.25 ms/row)** |
| restoring 1,003 R0 rows, CPU | 0.16 s |

Latency is a non-issue for the live round — 5.6 s against a decode measured in
minutes — and it needs no GPU, no gated repo and no network.

Training depends on **torch and the standard library only**: no transformers, no
IndicTransToolkit, no KashmiriNormalizer. Verified by simulating a bare image
with all four forced to fail on import. The Kaggle runner therefore does no
`pip install`, which structurally removes the failure that cost a session on
2026-08-10.

## Deliverables kept

- `src/restore/chartag.py` — codec, model, `Restorer`. The insertion-only
  guarantee and the whitespace rule live here.
- `src/restore/build_text.py` — corpus assembly with per-source stats and
  leakage exclusion by stripped string.
- `src/restore/train.py` — training, per-mark P/R, `none_bias` sweep.
- `src/restore/combine.py` — the two hybrid merge rules.
- `scripts/eval_restore.py` — scores any set of restoration strategies over one
  decode; reproduces subs 004/007 exactly.
- `scripts/run_r11_kaggle.sh` — the T4 runner. Locates its bundle by content.
- `experiments/r11-restore/r0_comparison_bias{0.0,-0.5}.json`, `logs/`.
- `models/restore/r11_{all,bpcc}.pt` — 13 MB each, kept for the retrain
  comparison.
