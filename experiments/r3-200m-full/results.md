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
unambiguous comparison is R3-200M against 200M zero-shot, which has not been
measured. **Measure it before drawing conclusions** — one decode of FLORES
devtest with the 200M base, via `scripts/translate.py`.

## Config

`config/r3_200m_full.yaml`, corpus `data/processed/r3_corpus`.

| | |
| --- | --- |
| Base | `ai4bharat/indictrans2-en-indic-dist-200M` (MIT) |
| Adaptation | full fine-tune (all 200M parameters), not LoRA |
| Train pairs | 124,538 |
| Held-out eval | 1,000 (human sources only, seed 42) |
| Target normalization | `KashmiriNormalizer` 0.1.0, `map_punctuation: false` |
| LR / schedule | 5e-5, inverse_sqrt, 1000 warmup |
| Effective batch | 128 (32 × 2 GPUs × 2 accum) |
| Epochs | 5 (≈4,900 optimizer steps) |
| Label smoothing | 0.1 |
| Precision | fp16 (T4 = Turing) |
| Selection | `eval_geo_proxy`, patience 5 |

### Corpus provenance

Built from `data/processed/bpcc_kas_clean.jsonl` (sha256 `0065aee7…`) by
`python -m data.build_corpus`. Dropped during the build: 40 duplicates that
collapsed onto each other once the targets were normalized, 4 lines with
residual Devanagari. Leakage check re-run against FLORES: **0 leaked pairs**.

Measured on the built training targets — matches the R2 diagnostic exactly,
which is the check that the normalization did not damage the corpus:

| | train targets | FLORES devtest |
| --- | ---: | ---: |
| diacritics per 100c | 9.04 | 7.70 |
| mean chars/line | 92.6 | 124.6 |

The length gap is the R6 problem in advance: fine-tuning on 92.6-char lines
biases output short, and BLEU's brevity penalty bites. `len_ratio` is logged at
every eval for exactly this reason.

## Results — FILL IN AFTER THE RUN

Never report a score without naming the dev set. Tokenizer is always 13a.

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
