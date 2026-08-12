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


BOS = "<s>"  # left context for the first word of a line


def build_lexicon(
    targets: list[Path],
    min_count: int = 3,
    dominance: float = 0.0,
    bigram_min_count: int = 2,
) -> tuple[dict[str, str], dict[str, str], dict]:
    """Learn undiacritized-form -> diacritized form, unigram and left-context.

    The bigram table is keyed on `previous_key \\t key`, where BOTH are stripped
    forms. That matters: at apply time the previous word comes from raw model
    output, which never carries these three marks, so keying on the stripped
    form is the only way train and apply see the same thing.
    """
    forms: dict[str, Counter] = defaultdict(Counter)
    ctx: dict[str, Counter] = defaultdict(Counter)
    lines = 0
    for path in targets:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                lines += 1
                words = line.split()
                prev = BOS
                for w in words:
                    k = strip_key(w)
                    forms[k][w] += 1
                    ctx[f"{prev}\t{k}"][w] += 1
                    prev = k

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
            # Nothing to add; keeping it out shrinks the lexicon a lot.
            identity += 1
            continue
        lut[key] = best

    # Only keep a context entry when it is BOTH frequent enough to trust AND
    # disagrees with the unigram answer. Agreeing entries are pure bloat.
    big: dict[str, str] = {}
    for ckey, counter in ctx.items():
        if sum(counter.values()) < bigram_min_count:
            continue
        best = counter.most_common(1)[0][0]
        key = ckey.split("\t", 1)[1]
        if best == key:
            continue
        if lut.get(key) != best:
            big[ckey] = best

    stats = {
        "source_lines": lines,
        "distinct_keys": len(forms),
        "entries": len(lut),
        "context_entries": len(big),
        "skipped_rare": skipped_rare,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_identity": identity,
        "min_count": min_count,
        "dominance": dominance,
        "bigram_min_count": bigram_min_count,
    }
    return lut, big, stats


def restore(text: str, lut: dict[str, str], big: dict[str, str] | None = None) -> str:
    """Substitute known words, preferring left-context when available.

    Whitespace layout is preserved. Context is taken from the RAW previous
    token, not the restored one, so an early mistake cannot cascade.
    """
    words = text.split()
    out = []
    prev = BOS
    for w in words:
        chosen = None
        if big:
            chosen = big.get(f"{prev}\t{w}")
        if chosen is None:
            chosen = lut.get(w, w)
        out.append(chosen)
        prev = w
    return " ".join(out)


def coverage(texts: list[str], lut: dict[str, str], big: dict[str, str] | None = None) -> dict:
    total = hit = ctx_hit = 0
    for t in texts:
        prev = BOS
        for w in t.split():
            total += 1
            if big and f"{prev}\t{w}" in big:
                ctx_hit += 1
                hit += 1
            elif w in lut:
                hit += 1
            prev = w
    return {"words": total, "substituted": hit, "by_context": ctx_hit,
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
    b.add_argument("--bigram-min-count", type=int, default=2,
                   help="0 disables left-context disambiguation entirely")

    a = sub.add_parser("apply")
    a.add_argument("--lexicon", required=True, type=Path)
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)

    args = ap.parse_args()

    if args.cmd == "build":
        lut, big, stats = build_lexicon(
            args.targets, args.min_count, args.dominance, args.bigram_min_count)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"stats": stats, "lexicon": lut, "context": big},
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
    big = blob.get("context") or {}
    lines = _read(args.input)
    cov = coverage(lines, lut, big)
    out = [restore(x, lut, big) for x in lines]
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
