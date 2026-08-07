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
from dataclasses import dataclass

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


DEFAULT = NormConfig()


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

    return s.strip()


def normalize_many(values, cfg: NormConfig = DEFAULT) -> list[str]:
    return [normalize(v, cfg) for v in values]


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
