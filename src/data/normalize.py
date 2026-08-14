"""
Shared text normalization for KATHE 2026.

ONE implementation, used in three places:
  1. training target preprocessing  (src/data/build_corpus.py)
  2. inference post-processing      (scripts/translate.py)
  3. local scoring                  (src/eval/score.py)

If these ever diverge you will tune against a metric you aren't submitting to.
Import from here; do not re-implement.

The official scorer runs KashmiriNormalizer().normalize(text) on BOTH sides.
Everything that normalizer collapses is free. This module adds ONLY the fixes
for what it does NOT collapse, and every addition is off by default until the
orthography diagnostic proves it matches the reference convention.
"""

from __future__ import annotations

import re
import unicodedata as ud
from dataclasses import dataclass, replace
from pathlib import Path

from KashmiriNormalizer import KashmiriNormalizer

_KN = KashmiriNormalizer()

# The scorer's normalizer handles none of these.
_LATIN_TO_KASHMIRI_PUNCT = {
    ".": "۔",
    ",": "،",
    "?": "؟",
    ";": "؛",
    "%": "٪",
}
_ZWJ = "\u200d"
_MULTISPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class NormConfig:
    """
    Set these from the orthography diagnostic, not from intuition.
    Record the chosen values in PLANNING.md §Decisions.
    """

    # Apply the official scorer's normalizer. Always True; both sides get it.
    scorer_normalizer: bool = True

    # Unicode composition. The scorer applies no NFC/NFD, so precomposed and
    # decomposed sequences score as different characters. Set to whichever form
    # dominates the references ("NFC", "NFD", or None to leave alone).
    unicode_form: str | None = None

    # Map Latin sentence punctuation to Kashmiri. Enable only if the references
    # are predominantly Kashmiri-punctuated.
    map_punctuation: bool = False

    # ZWJ survives the scorer's normalizer; ZWNJ is already turned into a space.
    strip_zwj: bool = False

    # ZWNJ->space can produce double spaces, which perturb the word-bigram half
    # of chrF++. Safe to enable once references are confirmed single-spaced.
    collapse_whitespace: bool = False

    # Path to a diacritic lexicon from `data.diacritize`. Restores kasra, damma
    # and fatha, which IndicTrans2 CANNOT generate — they exist in its target
    # vocabulary only as bare standalone tokens, never inside a subword, so beam
    # search never emits them (see data/diacritize.py). Measured +2.56 geo
    # (+8.9%) on R0. Applied AFTER the scorer normalizer, because the lexicon is
    # keyed on scorer-normalized word forms.
    diacritic_lexicon: str | None = None

    # Path to a LEARNED restorer checkpoint from `restore.train` (R11). Restores
    # the same three marks as `diacritic_lexicon`, but with a character-level
    # tagger instead of a lookup table, so it generalises to word forms the
    # lexicon has never seen. On R0 it scores 34.02 against the lexicon's 33.36.
    #
    # It may be combined with `diacritic_lexicon` via `restore_merge`, though
    # measured 2026-08-13 the model ALONE beats every combination — once trained
    # to convergence it already reproduces what the lexicon knew (86% precision
    # on lexicon-known words) and consulting the table only drags it back.
    restore_model: str | None = None

    # Logit offset on the "no mark" class. NEGATIVE inserts more marks. Best on
    # R0 is 0.0: pushing output density toward the reference density scores
    # monotonically WORSE (see experiments/r11-restore/results.md).
    restore_none_bias: float = 0.0

    # How to combine the two restorers when both are set: `known` gives the
    # lexicon every word it has an entry for, `changed` only the words it
    # actually marks. Ignored unless BOTH are configured.
    restore_merge: str = "known"

    # Decision rule for the learned restorer. `restore_threshold` emits a mark
    # wherever P(any mark) clears it; `restore_mark_thresholds` gives each mark
    # its own bar. Both replace argmax, which is the WRONG rule here: marks are
    # ~4% of positions, so argmax (effectively P>0.5) discarded half the model's
    # expected marks and cost submission 008 1.76 leaderboard points.
    #
    # Per-mark bars exist because the mark PROPORTIONS were wrong, not just the
    # total. Sub 009 emitted kasra/damma/fatha at 48/45/7 percent against a
    # reference 49/29/21 — damma over-produced, fatha starved. Every system to
    # date, the lexicon included, emits fatha at a fifth of the reference rate.
    restore_threshold: float | None = None
    restore_mark_thresholds: dict | None = None


DEFAULT = NormConfig()

# Cached per path: the lexicon is ~7k entries and normalize() is called once per
# row, so re-reading it per call would dominate the cost of a 1,730-row run.
_LEXICON_CACHE: dict[str, tuple[dict[str, str], dict[str, str]]] = {}


def _lexicon(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (unigram, left-context) tables. `context` may be empty."""
    if path not in _LEXICON_CACHE:
        import json

        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        _LEXICON_CACHE[path] = (blob["lexicon"], blob.get("context") or {})
    return _LEXICON_CACHE[path]


def normalize(text: object, cfg: NormConfig = DEFAULT) -> str:
    """Normalize one cell. Mirrors the official scorer, plus configured fixes."""
    if text is None:
        s = ""
    else:
        s = str(text)
        if s.lower() == "nan":  # pandas float NaN coerced upstream
            s = ""

    if cfg.unicode_form:
        s = ud.normalize(cfg.unicode_form, s)

    if cfg.strip_zwj:
        s = s.replace(_ZWJ, "")

    if cfg.map_punctuation:
        for latin, kash in _LATIN_TO_KASHMIRI_PUNCT.items():
            s = s.replace(latin, kash)

    if cfg.scorer_normalizer:
        s = _KN.normalize(s)  # removeDiacritics=False — diacritics are preserved

    if cfg.collapse_whitespace:
        s = _MULTISPACE.sub(" ", s)

    return _restore_batch([s.strip()], cfg)[0]


# Cached per (path, bias): loading a 3.3M-parameter checkpoint once per ROW
# would dominate everything else in a 1,730-row submission.
_RESTORER_CACHE: dict[tuple, object] = {}


def _restorer(cfg: NormConfig):
    mt = cfg.restore_mark_thresholds
    key = (cfg.restore_model, cfg.restore_none_bias, cfg.restore_threshold,
           tuple(sorted(mt.items())) if mt else None)
    if key not in _RESTORER_CACHE:
        from restore.chartag import Restorer

        _RESTORER_CACHE[key] = Restorer(
            cfg.restore_model, device="cpu", none_bias=cfg.restore_none_bias,
            threshold=cfg.restore_threshold, mark_thresholds=mt)
    return _RESTORER_CACHE[key]


def _restore_batch(rows: list[str], cfg: NormConfig) -> list[str]:
    """Apply whichever diacritic restorer(s) are configured, to a whole batch.

    Restoration is the LAST step and is factored out of `normalize` so that
    `normalize_many` can run the neural restorer on all rows at once. Per-row
    inference on 1,730 rows is correct but wasteful; batching is ~20x faster and
    produces identical output, because the tagger conditions only on its own
    sentence.

    Both restorers are insertion-only and both strip their input first, so this
    is idempotent: handing it already-restored text yields the same result as
    handing it raw model output.
    """
    if not (cfg.diacritic_lexicon or cfg.restore_model):
        return rows

    # Delegate to restore.combine so the backoff chain and the merge rules have
    # exactly ONE implementation. The lexicon path was duplicated here once, and
    # when the lexicon gained two more context tables this copy silently kept
    # using only the old flat one — scoring 31.20 instead of 33.50, no error.
    from restore.combine import restore_all

    lut = ctx = None
    if cfg.diacritic_lexicon:
        lut, ctx = _lexicon(cfg.diacritic_lexicon)
    model = _restorer(cfg) if cfg.restore_model else None

    if lut is not None and model is not None:
        mode = cfg.restore_merge
    elif model is not None:
        mode = "model"
    else:
        mode = "lexicon"
    return restore_all(rows, mode, lut, ctx, model)


def normalize_many(values, cfg: NormConfig = DEFAULT) -> list[str]:
    """Batch form. Identical output to `normalize` per row, but restoration runs
    once over the whole list rather than once per row."""
    plain = replace(cfg, diacritic_lexicon=None, restore_model=None)
    return _restore_batch([normalize(v, plain) for v in values], cfg)


def score(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """
    Exact replication of the official KATHE 2026 scorer.
    Inputs are raw (un-normalized); normalization happens here, as it does there.
    """
    import sacrebleu

    # Scoring uses the scorer's normalizer only — NOT the project's extra fixes.
    # Those belong in post-processing, before the text ever reaches this point.
    scorer_cfg = NormConfig(scorer_normalizer=True)
    h = normalize_many(hyps, scorer_cfg)
    r = normalize_many(refs, scorer_cfg)

    if any(not x.strip() for x in h):
        raise ValueError(
            "Empty hypothesis after normalization — the official scorer raises "
            "on this and the entire submission is rejected."
        )

    bleu = sacrebleu.corpus_bleu(h, [r]).score  # tokenizer 13a
    chrfpp = sacrebleu.corpus_chrf(h, [r], word_order=2).score
    geo = 0.0 if (bleu <= 0 or chrfpp <= 0) else (bleu * chrfpp) ** 0.5
    return {"bleu": bleu, "chrf_plus_plus": chrfpp, "geometric_mean": geo}
