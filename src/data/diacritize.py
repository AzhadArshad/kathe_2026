#!/usr/bin/env python3
"""
KATHE 2026 — post-hoc restoration of kasra, damma and fatha.

Why this exists
---------------
IndicTrans2 **cannot emit these three marks**. In its target vocabulary
(`dict.TGT.json`, 122,672 entries) kasra (U+0650), damma (U+064F) and fatha
(U+064E) each appear in exactly ONE token — the bare standalone mark. Every
other Kashmiri diacritic is baked into whole-word subwords: hamza-below appears
in 378 tokens (`▁تہٕ`, `▁کرنہٕ`), inv-damma in 244 (`▁منٛز`, `▁ہنٛد`).

So writing `چھُس` requires emitting `[چھ][ُ][س]` — splitting a word to insert a
bare diacritic that occurs in no natural subword context. Beam search never
does, because the undiacritized whole-word token is always more probable.

Measured on R0 after the R3 fine-tune: **exactly 0** kasra, damma and fatha in
1,003 sentences, against 270,799 / 105,855 / 61,885 occurrences in the training
targets. The tokenizer preserves them (round-trip 8.90 -> 8.83 per 100c) and
`postprocess_batch` preserves them (3.47 -> 3.56), so the defect is the
vocabulary — which is frozen in the pretrained checkpoint.

**Fine-tuning therefore cannot fix this.** Restoration has to be post-hoc, which
is what this module does. The metric preserves diacritics, so the cost is real:
a perfect translation missing only these three marks scores 67.66, not 100.

Method
------
A unigram lexicon: strip the three marks from every word in the training
targets, and map that key to the most frequent diacritized form. Deliberately
simple — it is deterministic, has no runtime dependency, and adds no latency to
the live round.

Two guards against doing harm, both tuned on R0 rather than assumed:

  --min-count   ignore forms seen fewer than N times. A form seen once is as
                likely to be a typo in the corpus as a fact about the language.
  --dominance   only substitute when the top form holds at least this share of
                the key's occurrences.

**Both defaults come from a sweep on R0 (2026-08-11), and the dominance result
is the opposite of what was expected.** The argument for a dominance guard was
that 12.4% of keys are ambiguous, the canonical case being gendered — `چھُس`
(masc.) vs `چھَس` (fem.) for "I am", distinguished ONLY by damma vs fatha — and
that on a coin-flip it is better to leave a word bare than assert a gender.

Measured, that is wrong:

    dominance 0.0 -> geo 31.20 (+2.56)
    dominance 0.6 -> geo 30.30 (+1.66)
    dominance 0.9 -> geo 29.14 (+0.50)

A *wrong* diacritic beats *no* diacritic. The reference always carries a mark at
that position, so chrF++ gives partial credit either way, and omitting it
guarantees a miss. Hence `dominance=0.0` — substitute always.

`min_count` 1, 2 and 3 all score 31.20 identically, so 3 is chosen: it shrinks
the lexicon roughly 5x (6,996 entries vs 32,980) for no loss, which matters for
the live round. Above 3 the score starts falling (5 -> 31.02).

Build the lexicon from TRAIN TARGETS ONLY. Building it from anything containing
R0 would leak the dev set into the post-processing.

Usage:
    uv run python -m data.diacritize build \\
        --targets data/processed/r3_corpus/train/eng_Latn-kas_Arab/train.kas_Arab \\
        --output  data/processed/diacritic_lexicon.json

    uv run python -m data.diacritize apply \\
        --lexicon data/processed/diacritic_lexicon.json \\
        --input hyp.txt --output hyp.diacritized.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The three marks IndicTrans2 cannot generate. Nothing else is touched: the
# other diacritics already match reference density to within 5%, so "restoring"
# them would only add noise.
RESTORABLE = frozenset("َُِ")  # kasra, damma, fatha


def strip_key(word: str) -> str:
    return "".join(c for c in word if c not in RESTORABLE)


BOS = "<s>"  # context marker for the first word of a line
EOS = "</s>"  # ... and the last


def build_lexicon(
    targets: list[Path],
    min_count: int = 3,
    dominance: float = 0.0,
    context_min_count: int = 2,
    context_mode: str = "left",
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict]:
    """Learn undiacritized-form -> diacritized form: unigram plus three context
    tables, consulted most-specific first.

    All context keys use STRIPPED neighbours. That matters: at apply time the
    neighbours come from raw model output, which never carries these three
    marks, so keying on the stripped form is the only way train and apply see
    the same thing.
    """
    forms: dict[str, Counter] = defaultdict(Counter)
    ctx_both: dict[str, Counter] = defaultdict(Counter)
    ctx_left: dict[str, Counter] = defaultdict(Counter)
    ctx_right: dict[str, Counter] = defaultdict(Counter)
    lines = 0
    for path in targets:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                lines += 1
                words = line.split()
                keys = [strip_key(w) for w in words]
                for i, w in enumerate(words):
                    k = keys[i]
                    p = keys[i - 1] if i else BOS
                    n = keys[i + 1] if i + 1 < len(keys) else EOS
                    forms[k][w] += 1
                    ctx_both[f"{p}\t{k}\t{n}"][w] += 1
                    ctx_left[f"{p}\t{k}"][w] += 1
                    ctx_right[f"{k}\t{n}"][w] += 1

    lut: dict[str, str] = {}
    skipped_rare = skipped_ambiguous = identity = 0
    for key, counter in forms.items():
        total = sum(counter.values())
        if total < min_count:
            skipped_rare += 1
            continue
        best, n = counter.most_common(1)[0]
        if n / total < dominance:
            skipped_ambiguous += 1
            continue
        if best == key:
            # The bare spelling is the corpus majority for this word. Leave it
            # alone. Overriding these was measured and is monotonically WORSE —
            # forcing a mark breaks the majority of cases that were correctly
            # bare (2026-08-12 sweep: 32.56 -> 31.10 -> 14.52 as the override
            # threshold loosens).
            identity += 1
            continue
        lut[key] = best

    def build_ctx(table: dict[str, Counter], key_index: int, prune: bool) -> dict[str, str]:
        """Keep EVERY entry that clears the count threshold.

        It is tempting to drop entries that agree with the unigram answer as
        dead weight. That is wrong here, because the tables are chained: if
        `both` agrees with the unigram and is therefore dropped, the lookup
        falls through to `left`, which may DISAGREE — so pruning a no-op
        silently changes precedence. Measured cost of getting this wrong:
        33.23 instead of 33.50 on R0. Entries whose answer is the bare key are
        kept for the same reason; "context says leave this one alone" is real
        information when the unigram would have substituted.
        """
        out: dict[str, str] = {}
        for ckey, counter in table.items():
            if sum(counter.values()) < context_min_count:
                continue
            best = counter.most_common(1)[0][0]
            if prune:
                # Safe ONLY for a single table, where there is nothing to fall
                # through to. With chained tables this silently reorders
                # precedence — see the docstring.
                key = ckey.split("\t")[key_index]
                if best == key or lut.get(key) == best:
                    continue
            out[ckey] = best
        return out

    # `left` is the PRODUCTION mode. See the module docstring: the richer
    # `both` mode scores higher on R0 and LOWER on the leaderboard.
    if context_mode == "both":
        ctx = {"both": build_ctx(ctx_both, 1, prune=False),
               "left": build_ctx(ctx_left, 1, prune=False),
               "right": build_ctx(ctx_right, 0, prune=False)}
    elif context_mode == "left":
        ctx = {"left": build_ctx(ctx_left, 1, prune=True)}
    elif context_mode == "none":
        ctx = {}
    else:
        raise SystemExit(f"unknown context_mode {context_mode!r}")

    stats = {
        "source_lines": lines,
        "distinct_keys": len(forms),
        "entries": len(lut),
        "context_mode": context_mode,
        "context_both": len(ctx.get("both", {})),
        "context_left": len(ctx.get("left", {})),
        "context_right": len(ctx.get("right", {})),
        "skipped_rare": skipped_rare,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_identity": identity,
        "min_count": min_count,
        "dominance": dominance,
        "context_min_count": context_min_count,
    }
    return lut, ctx, stats


def _tables(ctx: dict | None) -> tuple[dict, dict, dict]:
    ctx = ctx or {}
    # Tolerate the older flat format, which held only the left table.
    if ctx and not isinstance(next(iter(ctx.values()), {}), dict):
        ctx = {"left": ctx}
    return ctx.get("both", {}), ctx.get("left", {}), ctx.get("right", {})


def lookup(words: list[str], i: int, lut: dict[str, str],
           tables: tuple[dict, dict, dict]) -> str | None:
    """The lexicon's answer for `words[i]`, or None if it has none.

    Backoff order: both-sided -> left -> right -> unigram. Returning None rather
    than the input distinguishes "the lexicon left this word alone because it
    knows the bare form is right" from "the lexicon has never seen this word" —
    a distinction `restore` does not need but the learned restorer does, since
    it fills exactly the second case (see restore/combine.py).
    """
    B, L, R = tables
    w = words[i]
    p = words[i - 1] if i else BOS
    n = words[i + 1] if i + 1 < len(words) else EOS
    return (B.get(f"{p}\t{w}\t{n}")
            or L.get(f"{p}\t{w}")
            or R.get(f"{w}\t{n}")
            or lut.get(w))


def restore(text: str, lut: dict[str, str], ctx: dict | None = None) -> str:
    """Substitute known words, most specific context first.

    Backoff order: both-sided -> left -> right -> unigram -> unchanged.
    Measured on R0 (2026-08-12): unigram 31.20, +left 32.58, +right 32.88,
    +left+right 32.90, +both-sided 33.50. Right context alone beats left alone.

    Neighbours are read from the RAW text, never from already-restored output,
    so one mistake cannot cascade along the sentence.
    """
    tables = _tables(ctx)
    words = text.split()
    return " ".join(lookup(words, i, lut, tables) or w for i, w in enumerate(words))


def coverage(texts: list[str], lut: dict[str, str], ctx: dict | None = None) -> dict:
    B, L, R = tables = _tables(ctx)
    total = hit = by_ctx = 0
    for t in texts:
        words = t.split()
        for i, w in enumerate(words):
            total += 1
            p = words[i - 1] if i else BOS
            n = words[i + 1] if i + 1 < len(words) else EOS
            if f"{p}\t{w}\t{n}" in B or f"{p}\t{w}" in L or f"{w}\t{n}" in R:
                by_ctx += 1
                hit += 1
            elif lookup(words, i, lut, tables) is not None:
                hit += 1
    return {"words": total, "substituted": hit, "by_context": by_ctx,
            "pct": round(100 * hit / max(1, total), 2)}


def _read(p: Path) -> list[str]:
    with open(p, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--targets", required=True, type=Path, nargs="+",
                   help="TRAIN targets only — never anything containing R0")
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--min-count", type=int, default=3)
    b.add_argument("--dominance", type=float, default=0.0)
    b.add_argument("--context-min-count", type=int, default=2,
                   help="minimum occurrences for a context entry")
    b.add_argument("--context-mode", default="left", choices=["none", "left", "both"],
                   help="PRODUCTION is 'left'. 'both' scores higher on R0 and "
                        "LOWER on the leaderboard — see the module docstring.")

    a = sub.add_parser("apply")
    a.add_argument("--lexicon", required=True, type=Path)
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)

    args = ap.parse_args()

    if args.cmd == "build":
        lut, ctx, stats = build_lexicon(
            args.targets, args.min_count, args.dominance,
            args.context_min_count, args.context_mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"stats": stats, "lexicon": lut, "context": ctx},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        print("\n=== DIACRITIC LEXICON ===")
        for k, v in stats.items():
            print(f"  {k:20} {v}")
        print(f"  wrote {args.output}")
        return 0

    blob = json.loads(args.lexicon.read_text(encoding="utf-8"))
    lut = blob["lexicon"]
    ctx = blob.get("context") or {}
    lines = _read(args.input)
    cov = coverage(lines, lut, ctx)
    out = [restore(x, lut, ctx) for x in lines]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(x + "\n")
    print(f"  lines {len(lines):,}   words {cov['words']:,}   "
          f"substituted {cov['substituted']:,} ({cov['pct']}%)   "
          f"of which by context {cov['by_context']:,}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
