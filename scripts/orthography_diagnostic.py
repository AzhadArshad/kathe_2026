#!/usr/bin/env python3
"""
KATHE 2026 — orthography diagnostic.

Run this BEFORE any fine-tuning. It answers the question that decides the
competition: does our model produce Kashmiri orthography that matches the
reference convention, in the specific ways the official scorer does NOT
normalize away?

Usage:
    python orthography_diagnostic.py --refs bpcc_kas.txt --hyps zeroshot_out.txt

Both files: one sentence per line. --hyps is optional; with refs only, it just
characterizes the reference convention (which is what you match against).

Requires: KashmiriNormalizer==0.1.0, sacrebleu, regex
"""

import argparse
import unicodedata as ud
from collections import Counter

from KashmiriNormalizer import KashmiriNormalizer
from KashmiriNormalizer.constants import KASHMIRI_DIACRITICS

NORM = KashmiriNormalizer()

# Not touched by the scorer's normalizer — these are the levers.
LATIN_PUNCT = ".,?!;:"
KASHMIRI_PUNCT = "؛،٫؟۔٪"
ZWJ = "\u200d"
ZWNJ = "\u200c"

# Sequences the normalizer leaves in whichever form it finds them.
COMPOSITION_PAIRS = [
    ("\u0624", "\u0648\u0654", "waw + hamza  (ؤ)"),
    ("\u0622", "\u0627\u0653", "alef + maddah (آ)"),
    ("\u0626", "\u064a\u0654", "yeh + hamza  (ئ)"),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def profile(lines, label):
    norm = [NORM.normalize(x) for x in lines]
    joined = "".join(norm)
    chars = len(joined)
    if chars == 0:
        print(f"[{label}] empty after normalization")
        return None

    diacritics = sum(1 for c in joined if c in KASHMIRI_DIACRITICS)
    latin_p = sum(1 for c in joined if c in LATIN_PUNCT)
    kash_p = sum(1 for c in joined if c in KASHMIRI_PUNCT)
    zwj = joined.count(ZWJ)
    zwnj_before = "".join(lines).count(ZWNJ)  # pre-normalization; it becomes space
    double_space = sum(1 for x in norm if "  " in x)
    arabic_range = sum(1 for c in joined if "\u0600" <= c <= "\u06ff")
    latin_letters = sum(1 for c in joined if c.isascii() and c.isalpha())
    devanagari = sum(1 for c in joined if "\u0900" <= c <= "\u097f")

    stats = {
        "lines": len(norm),
        "chars": chars,
        "diacritics_per_100c": 100 * diacritics / chars,
        "pct_lines_with_diacritic": 100
        * sum(1 for x in norm if any(c in KASHMIRI_DIACRITICS for c in x))
        / len(norm),
        "latin_punct": latin_p,
        "kashmiri_punct": kash_p,
        "pct_punct_kashmiri": 100 * kash_p / max(1, latin_p + kash_p),
        "zwj_remaining": zwj,
        "zwnj_in_raw": zwnj_before,
        "lines_with_double_space": double_space,
        "pct_chars_arabic": 100 * arabic_range / chars,
        "latin_letters": latin_letters,
        "devanagari_chars": devanagari,
        "mean_chars_per_line": chars / len(norm),
    }

    print(f"\n=== {label} ===")
    for k, v in stats.items():
        print(f"  {k:28} {v:>12.2f}" if isinstance(v, float) else f"  {k:28} {v:>12}")

    # Composition convention
    print("  composition (precomposed vs decomposed, as found):")
    for pre, dec, name in COMPOSITION_PAIRS:
        print(f"    {name:22} precomposed={joined.count(pre):>6}  decomposed={joined.count(dec):>6}")

    nfc = ud.normalize("NFC", joined)
    print(f"    text is already NFC: {nfc == joined}")

    # Most common diacritics actually used
    dcount = Counter(c for c in joined if c in KASHMIRI_DIACRITICS)
    if dcount:
        top = ", ".join(f"U+{ord(c):04X}:{n}" for c, n in dcount.most_common(6))
        print(f"  top diacritics: {top}")

    return stats


def compare(ref_stats, hyp_stats):
    print("\n=== VERDICT ===")
    rd, hd = ref_stats["diacritics_per_100c"], hyp_stats["diacritics_per_100c"]
    ratio = hd / rd if rd else float("inf")
    print(f"  diacritic density  ref={rd:.2f}/100c  hyp={hd:.2f}/100c  ratio={ratio:.2f}")
    if ratio < 0.5:
        print("  >> CRITICAL: model badly under-produces diacritics. Expect the")
        print("     geometric mean to collapse regardless of translation quality.")
        print("     Fix this before any other work. Consider a diacritic")
        print("     restoration post-processor if fine-tuning does not close it.")
    elif ratio < 0.85 or ratio > 1.2:
        print("  >> Meaningful mismatch. Worth a targeted fix.")
    else:
        print("  >> Diacritic density is broadly aligned.")

    rp, hp = ref_stats["pct_punct_kashmiri"], hyp_stats["pct_punct_kashmiri"]
    print(f"  Kashmiri-punct share  ref={rp:.1f}%  hyp={hp:.1f}%")
    if abs(rp - hp) > 15:
        print("  >> Add a Latin->Kashmiri punctuation map to post-processing.")

    if hyp_stats["devanagari_chars"] > 0:
        print("  >> WARNING: Devanagari characters in output. Wrong script leaks = near-zero score.")
    if hyp_stats["latin_letters"] > 0.02 * hyp_stats["chars"]:
        print("  >> WARNING: heavy Latin leakage (untranslated source copied through).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="reference Kashmiri, one per line")
    ap.add_argument("--hyps", help="model output, one per line")
    args = ap.parse_args()

    ref_stats = profile(load(args.refs), "REFERENCES")
    if args.hyps:
        hyp_stats = profile(load(args.hyps), "HYPOTHESES")
        if ref_stats and hyp_stats:
            compare(ref_stats, hyp_stats)


if __name__ == "__main__":
    main()
