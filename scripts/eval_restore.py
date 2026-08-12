#!/usr/bin/env python3
"""
KATHE 2026 — R11: score every restoration strategy over one decode.

Restoration is post-processing, so a whole comparison table costs one decode and
some CPU. Every row here reads the SAME hypothesis file (`--hyp`), which is why
any difference between rows is the restorer and nothing else.

Rows are declared as `name:mode:lexicon:model`, where mode is one of
`raw | lexicon | model | known | changed` (see restore/combine.py). `--ceiling`
adds the diacritic-stripped-reference bound: our own output scored against
references with the three marks removed from both sides, i.e. the score if
restoration were perfect. That is the 40.63 quoted in PLANNING.md.

Usage:
    uv run python scripts/eval_restore.py \\
        --hyp  data/dev/r0/r0.hyp.r3-200m --refs data/dev/r0/r0.kas_Arab \\
        --lexicon prod=data/processed/diacritic_lexicon_both.json \\
        --lexicon clean=data/processed/lexicon_clean_all.json \\
        --model   all=models/restore/r11_all.pt \\
        --row "sub007:lexicon:prod" --row "model:model::all" \\
        --row "hybrid:known:clean:all"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata as ud
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.diacritize import RESTORABLE, strip_key  # noqa: E402
from data.normalize import NormConfig, normalize, score  # noqa: E402
from restore.chartag import Restorer  # noqa: E402
from restore.combine import restore_all  # noqa: E402

SCORER_ONLY = NormConfig(scorer_normalizer=True)


def read(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [normalize(ln.rstrip("\n"), SCORER_ONLY) for ln in fh]


def density(lines: list[str]) -> tuple[float, float]:
    """(restorable marks / 100 chars, ALL diacritics / 100 chars).

    "All" is every Unicode combining mark, not a hand-listed set: R0's
    references carry twelve distinct ones and a fixed list silently misses the
    tail (subscript alef, small low meem, shadda). Measured this way R0's
    references are 9.76/100c after scorer normalization — PLANNING.md's 9.63
    predates the scorer-normalized measurement and counts a shorter list.
    """
    chars = sum(len(x) for x in lines) or 1
    rest = sum(1 for x in lines for c in x if c in RESTORABLE)
    allm = sum(1 for x in lines for c in x if ud.category(c) == "Mn")
    return 100 * rest / chars, 100 * allm / chars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--lexicon", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--row", action="append", required=True,
                    metavar="NAME:MODE[:LEXICON[:MODEL]]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--none-bias", type=float, default=0.0)
    ap.add_argument("--ceiling", action="store_true",
                    help="also report the perfect-restoration bound")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--save-dir", type=Path, help="write each row's restored text here")
    args = ap.parse_args()

    hyp, refs = read(args.hyp), read(args.refs)
    if len(hyp) != len(refs):
        raise SystemExit(f"{len(hyp)} hypotheses vs {len(refs)} references")

    lexicons = {}
    for spec in args.lexicon:
        name, path = spec.split("=", 1)
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        lexicons[name] = (blob["lexicon"], blob.get("context") or {}, path)
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = Restorer(path, device=args.device, none_bias=args.none_bias)

    ref_rest, ref_all = density(refs)
    print(f"\n  {len(hyp):,} rows   references: {ref_rest:.2f} restorable/100c, "
          f"{ref_all:.2f} all-diacritics/100c")

    results = []
    for spec in args.row:
        parts = spec.split(":")
        name, mode = parts[0], parts[1]
        lex_name = parts[2] if len(parts) > 2 and parts[2] else None
        mdl_name = parts[3] if len(parts) > 3 and parts[3] else None
        lut, ctx = (lexicons[lex_name][0], lexicons[lex_name][1]) if lex_name else (None, None)
        restorer = models[mdl_name] if mdl_name else None

        t0 = time.time()
        out = hyp if mode == "raw" else restore_all(hyp, mode, lut, ctx, restorer)
        wall = time.time() - t0
        # Restoration may only insert; if a row's base changed, the comparison
        # below would be measuring a different sentence.
        for a, b in zip(hyp, out):
            if strip_key(a) != strip_key(b).replace("  ", " "):
                assert strip_key(a).split() == strip_key(b).split(), (a, b)
        s = score(out, refs)
        rest, allm = density(out)
        results.append({
            "name": name, "mode": mode, "lexicon": lex_name, "model": mdl_name,
            **{k: round(v, 2) for k, v in s.items()},
            "restorable_per_100c": round(rest, 2), "all_diacritics_per_100c": round(allm, 2),
            "wall_seconds": round(wall, 2),
            "ms_per_line": round(1000 * wall / len(hyp), 3),
        })
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            (args.save_dir / f"{name}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    if args.ceiling:
        s = score([strip_key(h) for h in hyp], [strip_key(r) for r in refs])
        results.append({"name": "CEILING (perfect restoration)", "mode": "-",
                        "lexicon": None, "model": None,
                        **{k: round(v, 2) for k, v in s.items()},
                        "restorable_per_100c": 0.0, "all_diacritics_per_100c": 0.0,
                        "wall_seconds": 0.0, "ms_per_line": 0.0})

    w = max(len(r["name"]) for r in results)
    print(f"\n  {'system':{w}}  {'BLEU':>6} {'chrF++':>7} {'GEO':>7} "
          f"{'rest/100c':>10} {'all/100c':>9} {'ms/line':>8}")
    for r in results:
        print(f"  {r['name']:{w}}  {r['bleu']:>6.2f} {r['chrf_plus_plus']:>7.2f} "
              f"{r['geometric_mean']:>7.2f} {r['restorable_per_100c']:>10.2f} "
              f"{r['all_diacritics_per_100c']:>9.2f} {r['ms_per_line']:>8.2f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"hyp": str(args.hyp), "refs": str(args.refs), "rows": results,
             "reference_density": {"restorable_per_100c": round(ref_rest, 2),
                                   "all_diacritics_per_100c": round(ref_all, 2)},
             "none_bias": args.none_bias},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
