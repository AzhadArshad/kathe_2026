# R3 — full fine-tune, indictrans2-en-indic-dist-200M

**Branch:** `r3-200m-full` · **Status:** prepared 2026-08-09, NOT YET RUN
**Baseline to beat:** geo **15.83** on FLORES devtest (R1 zero-shot IT2-1B, beam 5)

## Hypothesis

Fine-tuning on BPCC attacks translation quality and orthography in one run.
The kasra/damma gap is the measured half: zero-shot output carries ~0 of either,
while BPCC targets carry them at 2.34 and 0.92 per 100 chars against FLORES's
1.50 and 1.07. Perfect text missing only those two diacritics scores 79.58, so
orthography alone cannot be the binding constraint — translation quality is.

Note the model swap. The 15.83 baseline is the **1B** model zero-shot; this run
fine-tunes the **200M distilled** model. A result below 15.83 is therefore
ambiguous between "fine-tuning did not help" and "200M is smaller than 1B". The
unambiguous comparison is R3-200M against **200M zero-shot**, which has still
not been measured. **Measure it in the same session, before training** — two
decodes via `scripts/translate.py`, ~2 min on a T4:

**Controls measured 2026-08-10, both on R0, beam 5, same decode path:**

| System | BLEU | chrF++ | Geo |
| --- | ---: | ---: | ---: |
| IT2-**1B** zero-shot | 18.00 | 44.25 | **28.22** |
| IT2-**200M dist** zero-shot | 14.66 | 41.66 | **24.71** |

The 1B also has FLORES devtest **15.83** and leaderboard **8.00**.

**Two thresholds, and they mean different things:**

1. **> 24.71 — did fine-tuning do anything?** Same model, same dev set, only
   fine-tuning changed. This is the only clean read on the method.
2. **> 28.22 — is the fine-tuned 200M worth submitting over what we already
   have?** The current 8.00 leaderboard entry is the 1B zero-shot. The 200M
   starts 3.51 geo (12.4%) behind it, so R3 must gain **+14.2%** over its own
   base merely to draw level. Distillation cost more than fine-tuning may
   recover — if R3 lands between 24.71 and 28.22 that is a *successful
   fine-tune of a weaker model*, and the right response is to move to the 1B
   LoRA run rather than submit it.

Both controls sit at output/reference char ratio 0.964–0.966, so the bases are
well calibrated for length before training. Any drift above ~1.0 after
fine-tuning is the R6 over-generation risk arriving.

**Read R0 numbers as relative only.** IndicTrans2 was trained on BPCC and R0 is
cut from BPCC, so every IT2-derived checkpoint is partly scored on its own
training data. Ranking checkpoints against each other is valid — the
contamination is common to all of them. Treating an R0 score as a leaderboard
prediction is not. See PLANNING.md §Q9.

## Config

`config/r3_200m_full.yaml`, corpus `data/processed/r3_corpus`.

| | |
| --- | --- |
| Base | `ai4bharat/indictrans2-en-indic-dist-200M` (MIT) |
| Adaptation | full fine-tune (all 200M parameters), not LoRA |
| Train pairs | **123,538** (rebuilt 2026-08-10; R0 + eval slice excluded) |
| Held-out eval | **997, length-matched** (6.89 src words) — see below |
| Target normalization | `KashmiriNormalizer` 0.1.0, `map_punctuation: false` |
| LR / schedule | 5e-5, inverse_sqrt, 1000 warmup |
| Effective batch | 128 (32 × 2 GPUs × 2 accum) |
| Epochs | 5 (≈4,900 optimizer steps) |
| Label smoothing | 0.1 |
| Precision | fp16 (T4 = Turing) |
| Selection | `eval_geo_proxy`, patience 5 |

### Corpus provenance

**Rebuilt 2026-08-10.** The corpus shipped on 08-09 is superseded — re-upload
the Kaggle dataset before running, or this trains on R0 and every number it
produces is contaminated.

Built from `data/processed/bpcc_kas_clean.jsonl` (sha256 `0065aee7…`) by
`python -m data.build_corpus --exclude data/dev/r0/r0.jsonl
data/dev/r3_eval/r3_eval.jsonl --dev-from data/dev/r3_eval/r3_eval.jsonl`.
Dropped during the build: 40 duplicates that collapsed onto each other once the
targets were normalized, 4 lines with residual Devanagari. Cleaned pool 125,538
− 2,000 held out = **123,538 train**, asserted by the build rather than assumed.
Leakage re-run against FLORES: **0 leaked pairs**.

**The eval slice changed, and this is the point of the rebuild.** As sampled on
08-09 it averaged 15.7 English words against a 7.3-word test set, so
`eval_geo_proxy` would have selected checkpoints optimized for the wrong
sentence length. It is now length-matched at 6.89 words, drawn stratified to the
test set's own word-count distribution, human sources only, and disjoint from
R0. Any `eval_geo_proxy` value is not comparable to one from the old slice —
nothing was ever run on it, so nothing is lost.

Measured on the built training targets — matches the R2 diagnostic, which is the
check that normalization did not damage the corpus:

| | train targets | FLORES devtest | KATHE test |
| --- | ---: | ---: | ---: |
| diacritics per 100c | 9.03 | 7.70 | — |
| mean chars/line | 93.4 | 124.6 | **39.0** |
| mean src words | 15.7 | 21.6 | **7.3** |

**R6, corrected direction.** The earlier note here said short output and brevity
penalty were the risk. That was backwards. Training on 93.4-char lines to
translate a 39-char test set biases output *long* — over-generation is the risk.
The 1B zero-shot is currently well calibrated (R0 output/ref char ratio 0.966),
so fine-tuning is what could break it. `len_ratio` is logged at every eval
precisely to catch that, and it should stay near 1.0 against the length-matched
slice, not drift above it.

## Results — run 2026-08-10

Never report a score without naming the dev set. Tokenizer is always 13a.
All rows: beam 5, full `IndicProcessor` post-processing, scored by `eval.score`.

| System | Dev set | BLEU | chrF++ | Geo |
| --- | --- | ---: | ---: | ---: |
| 200M distilled base, zero-shot | R0 | 14.66 | 41.66 | 24.71 |
| **R3 — 200M full fine-tune** | **R0** | **17.67** | **46.43** | **28.64** |
| 1B zero-shot (= leaderboard 8.00) | R0 | 18.00 | 44.25 | 28.22 |

**Fine-tuning works: +3.93 geo, +15.9% over its own base** (BLEU +3.01,
chrF++ +4.77). That is the clean, like-for-like read and it is unambiguous.

**Against the 1B it is a near-tie: +0.42 geo, +1.5%** — and the composition is
odd. chrF++ is up 2.18 while **BLEU is DOWN 0.33**. The fine-tune has learned
the character and morphology conventions (which is what the diacritic gap
predicted) without gaining exact n-gram matches. A 211M model has drawn level
with a 1.1B one on this task.

Wall clock: 1h29m49s on T4 x2, 4,825 steps, effective batch 128.
Length ratio 0.951 (38.5 hyp chars vs 40.5 ref) — no over-generation.

### Do not read 28.64 > 28.22 as "beats the current submission"

Both numbers are on R0, which is cut from BPCC — and **IndicTrans2 was trained
on BPCC**, so both systems are partly scored on their own training data.
Critically, the comparison is not symmetric: R3 has just had *five more epochs*
of BPCC, so R0 flatters it more than it flatters the zero-shot 1B. The +1.5%
edge is exactly the size that contamination could manufacture on its own.

Naive extrapolation at the observed 0.505 R0→leaderboard ratio gives **8.12**
against the current 8.00. That is inside noise. The leaderboard is the only
instrument that can settle it.

### Training curve (in-training proxy, greedy, NOT comparable to the above)

| Step | Epoch | eval_loss | `geo_proxy` | `len_ratio` |
| ---: | ---: | ---: | ---: | ---: |
| 1500 | 1.55 | 2.921 | 37.51 | 0.988 |
| 2000 | 2.07 | 2.863 | 38.90 | 1.009 |
| 2500 | 2.59 | 2.824 | 39.29 | 0.986 |
| 4000 | 4.14 | 2.728 | 42.00 | 0.984 |

Monotonic to the final eval, `eval_empty_preds` 0 throughout. **No overfitting
at 5 epochs — the model is under-trained.** The 1B LoRA run should not use
fewer epochs, and more may pay.

## Verdict

Fine-tuning the 200M is a **clear methodological success and a marginal
leaderboard proposition.** The lift over its own base (+15.9%) is the number
that generalizes; the tie with the 1B is the number that decides what to do
next — and it says stop iterating on the 200M and move to the 1B, which starts
3.51 geo ahead of where this one started.

| Checkpoint | Dev set | BLEU | chrF++ | Geo | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 200M zero-shot | FLORES devtest | | | | the honest control — measure first |
| R3 best | FLORES dev | | | | |
| R3 best | FLORES devtest | | | | vs 15.83 |

Wall-clock: _(fill in)_ · GPU-hours: _(fill in)_ · Best step: _(fill in)_

`eval_geo_proxy` at the selected checkpoint: _(fill in)_ — **not comparable to
the table above**; it is computed in IndicProcessor's internal space because
`postprocess_batch` cannot be called during training. See `src/train/finetune.py`
§metrics.

## Verdict

_(one line, after the run)_

## What to check before trusting the number

1. **`empty_preds` stayed 0.** A non-zero count means decoding collapsed on some
   inputs; those rows would be rejected outright by the real scorer.
2. **`len_ratio` near 1.0.** Well below 1.0 confirms the short-output bias and
   sends this to R6 length calibration before anything else.
3. **Punctuation direction.** BPCC is 89.5% Kashmiri-punctuated against FLORES's
   95.7%, so a fine-tune pulls output away from the 99.7% zero-shot reached.
   Re-measure with `scripts/orthography_diagnostic.py` and flip
   `map_punctuation` in post-processing if the output has drifted Latin —
   verify on FLORES **dev**, not devtest.
4. **Diacritic density.** Should move from the zero-shot ~6.14 toward 7.70. This
   is the primary thing the run was supposed to buy.
5. **Twenty random rows, eyeballed.** Cheap, and it catches degenerate repetition
   that corpus-level metrics smear out.

## Next

- Submit as `submissions/002/` regardless of the outcome — submissions are
  unlimited and a second leaderboard reading calibrates the dev proxy at a
  different quality level (PLANNING.md §Submission strategy).
- Stage 2 of the two-stage plan: a short finishing pass on human seed data only.
  `python -m data.build_corpus --provenance human --out data/processed/r3_corpus_human`
  then `--init-from` this checkpoint with a lower LR.
- If the 200M result is directionally positive, run the 1B LoRA (`peft: lora`).
