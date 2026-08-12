#!/usr/bin/env python3
"""
KATHE 2026 — hybrid restoration: lexicon first, learned model on the tail.

The two restorers fail in opposite places, which is the whole argument for
combining them:

  * The **lexicon** is a table. On a word it has seen often it is close to
    unbeatable — it is literally the corpus majority form. On a word it has
    never seen it returns nothing, and Kashmiri's morphological tail is long.
  * The **char tagger** never returns nothing. It generalises to unseen words
    because it works on characters and sees the whole sentence, but on a common
    word it can be talked out of the right answer by context.

So: take the lexicon's answer where it has one, and the model's everywhere else.

Two ways to draw that line, both implemented because they are not obviously
ordered a priori:

  `known`   — the lexicon "has an answer" if any of its tables holds the key,
              INCLUDING when that answer is the bare form. This trusts the
              lexicon's silence as evidence.
  `changed` — the lexicon only wins where it actually inserted a mark. Where it
              knows the word and says "leave it bare", the model still gets a
              vote.

`known` is the more conservative reading and `changed` the more aggressive one;
which is better is measured on R0, not argued.

Word alignment is safe because both restorers are insertion-only over the same
stripped base, so `base.split()`, `lexicon_out.split()` and `model_out.split()`
have the same length and the same i-th word. That is asserted per line.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.diacritize import _tables, lookup, strip_key  # noqa: E402
from restore.chartag import Restorer  # noqa: E402

MODES = ("lexicon", "model", "known", "changed")


def combine_line(base: str, lex_out: str, model_out: str, mode: str,
                 lut: dict, tables: tuple[dict, dict, dict]) -> str:
    bw, lw, mw = base.split(), lex_out.split(), model_out.split()
    if not (len(bw) == len(lw) == len(mw)):
        raise AssertionError(
            f"word-count mismatch ({len(bw)}/{len(lw)}/{len(mw)}) — the "
            f"restorers are not insertion-only over the same base:\n  {base!r}")
    out = []
    for i, b in enumerate(bw):
        if mode == "known":
            out.append(lw[i] if lookup(bw, i, lut, tables) is not None else mw[i])
        elif mode == "changed":
            out.append(lw[i] if lw[i] != b else mw[i])
        else:
            raise ValueError(f"unknown merge mode {mode!r}")
    return " ".join(out)


def restore_all(texts: list[str], mode: str, lut: dict | None = None,
                ctx: dict | None = None, restorer: Restorer | None = None) -> list[str]:
    """Restore a batch of scorer-normalized lines under one of `MODES`.

    Inputs are stripped first, so this is idempotent and can be handed either
    raw model output or already-restored text without double-marking.
    """
    from data.diacritize import restore as lexicon_restore

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    bases = [strip_key(t) for t in texts]

    if mode == "model":
        if restorer is None:
            raise ValueError("mode 'model' needs a Restorer")
        return restorer.restore_many(bases)
    if mode == "lexicon":
        if lut is None:
            raise ValueError("mode 'lexicon' needs a lexicon")
        return [lexicon_restore(b, lut, ctx) for b in bases]

    if lut is None or restorer is None:
        raise ValueError(f"mode {mode!r} needs both a lexicon and a Restorer")
    tables = _tables(ctx)
    lex = [lexicon_restore(b, lut, ctx) for b in bases]
    mdl = restorer.restore_many(bases)
    return [combine_line(b, l, m, mode, lut, tables) for b, l, m in zip(bases, lex, mdl)]
