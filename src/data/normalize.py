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

    if cfg.diacritic_lexicon:
        # Left-context first, unigram as backoff. Context is read from the RAW
        # previous token, not the restored one, so an early mistake cannot
        # cascade down the sentence.
        lut, ctx = _lexicon(cfg.diacritic_lexicon)
        words, out, prev = s.split(), [], "<s>"
        for w in words:
            out.append(ctx.get(f"{prev}\t{w}") or lut.get(w, w))
            prev = w
        s = " ".join(out)

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
