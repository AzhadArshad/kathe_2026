#!/usr/bin/env python3
"""
KATHE 2026 — R12 stage 2: semantic selection against the real test input.

THE QUERY-SIDE SPLIT
--------------------
`englishdev.csv` is the test INPUT (1,730 sentences, no references). Using it to
choose training data is legitimate — it is given and unlabelled — but it creates
a measurement trap:

  * a dev set built WITHOUT the semantic criterion does not resemble the test
    set, so it cannot say whether selection helped. That is R0: 67% test-token
    coverage against the corpus's 98%, and rho = -0.39 against the leaderboard.
  * a dev set built WITH the criterion was chosen the same way as the training
    data, so it flatters the selection by construction.

The fix is to split the QUERIES, not the candidates:

    1,730 test inputs
      |-- 80% SELECT  -> rank candidates -> upweight -> TRAINING
      |-- 20% EVAL    -> rank candidates -> DEV SET (removed from training)

The training selection never sees the EVAL queries, so pairs near them were not
preferentially pulled in. The dev set is still test-like — it is retrieved
toward real test sentences — but it was not the target of the optimisation. If
selection generalises, dev improves; if it only memorises the neighbourhoods it
targeted, dev stays flat and we learn that before spending a submission.

WHAT THIS DEV SET CANNOT DO
---------------------------
Its Kashmiri references are BPCC text, not KATHE text. It measures translation
quality on test-like English; it says NOTHING about orthographic convention,
which is what sank submissions 005, 008 and 010. Use it to RANK training
variants, never to predict a leaderboard number.

SCORING
-------
Each candidate is scored by the mean cosine to its `--knn` nearest queries, not
to the single nearest — one odd test sentence should not promote a candidate.
A per-query cap prevents concentration: without it the top 10% of queries
account for 43% of the selection, and one query attracted 194 candidates.

UPWEIGHTING, NOT REPLACEMENT
----------------------------
Training only on the retrieved slice would mean ~6x less data, and R3's +15.9%
came from volume. The whole pool is kept and the selected slice is replicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.diacritize import RESTORABLE, strip_key  # noqa: E402


def embed(texts, model_name, batch=256, cache: Path | None = None):
    """Encode, with an on-disk cache keyed by the text itself.

    Encoding 111k sentences is ~3 minutes of CPU. Re-tuning the selection
    (dev size, N, per-query cap, upweight share) changes none of the vectors,
    so caching turns every subsequent rerun into seconds. The key is a hash of
    the inputs and the model name, so a changed pool invalidates it rather than
    silently reusing stale vectors.
    """
    import hashlib

    if cache is not None:
        h = hashlib.sha256(("\x00".join(texts) + "|" + model_name).encode()).hexdigest()[:16]
        f = cache / f"emb_{h}.npy"
        if f.exists():
            print(f"    cache hit {f.name} ({len(texts):,} texts)")
            return np.load(f)

    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device="cpu")
    V = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                 show_progress_bar=False)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(f, V)
        print(f"    cached {f.name} ({len(texts):,} texts)")
    return V


def score_against(C: np.ndarray, Q: np.ndarray, knn: int, block: int = 4096):
    """Mean cosine to the `knn` nearest queries, plus each candidate's argmax
    query (used for the per-query cap)."""
    n = len(C)
    s = np.empty(n, dtype=np.float32)
    who = np.empty(n, dtype=np.int32)
    k = min(knn, Q.shape[0])
    for i in range(0, n, block):
        S = C[i:i + block] @ Q.T
        part = np.partition(S, -k, axis=1)[:, -k:]
        s[i:i + block] = part.mean(1)
        who[i:i + block] = S.argmax(1)
    return s, who


def take_capped(order: np.ndarray, who: np.ndarray, n_want: int, cap: int):
    """Walk the ranking, skipping candidates whose nearest query is already
    full. Prevents a handful of test sentences dominating the selection."""
    used = Counter()
    out = []
    for i in order:
        q = int(who[i])
        if used[q] >= cap:
            continue
        used[q] += 1
        out.append(int(i))
        if len(out) >= n_want:
            break
    return out


def profile(rows, tag=""):
    ch = sum(len(r["t"]) for r in rows) or 1
    return {
        "pairs": len(rows),
        "src_words": round(sum(len(r["s"].split()) for r in rows) / max(1, len(rows)), 2),
        "restorable_per_100c": round(100 * sum(1 for r in rows for c in r["t"]
                                               if c in RESTORABLE) / ch, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[],
                    help="dev files (English side) to remove from the pool")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--select-n", type=int, default=20000)
    ap.add_argument("--per-query-cap", type=int, default=25)
    ap.add_argument("--dev-n", type=int, default=1000)
    ap.add_argument("--upweight-share", type=float, default=0.60,
                    help="target share of TRAINING PAIRS taken by the selected "
                         "slice after replication")
    ap.add_argument("--max-repeat", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-control", action="store_true",
                    help="CONTROL ARM: upweight a RANDOM slice of the same size "
                         "instead of the retrieved one. Everything else — pool, "
                         "dev set, size, repeat factor — is identical, so the "
                         "only variable is whether selection was semantic. "
                         "Without this arm a score change cannot be attributed "
                         "to retrieval rather than to the corpus rebuild or to "
                         "upweighting in general.")
    ap.add_argument("--cache", type=Path, default=Path("data/processed/.embcache"),
                    help="on-disk embedding cache; reruns become instant")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    print(f"  pool {len(pool):,}")

    # Remove held-out dev pairs by English source, before anything else.
    drop = set()
    for p in args.exclude:
        drop.update(x.strip() for x in open(p, encoding="utf-8"))
    if drop:
        before = len(pool)
        pool = [r for r in pool if r["s"].strip() not in drop]
        print(f"  removed {before - len(pool):,} pairs matching {len(drop):,} "
              f"held-out dev sources ({', '.join(p.name for p in args.exclude)})")

    with open(args.test_csv, encoding="utf-8") as fh:
        test = [r["sentence"] for r in csv.DictReader(fh)]
    idx = list(range(len(test)))
    random.Random(args.seed).shuffle(idx)
    n_eval = int(len(test) * args.eval_frac)
    eval_q = [test[i] for i in idx[:n_eval]]
    sel_q = [test[i] for i in idx[n_eval:]]
    print(f"  queries: {len(sel_q):,} SELECT / {len(eval_q):,} EVAL (seed {args.seed})")

    print(f"  encoding {len(pool):,} candidates + {len(test):,} queries ...", flush=True)
    C = embed([r["s"] for r in pool], args.model, cache=args.cache)
    Qs = embed(sel_q, args.model, cache=args.cache)
    Qe = embed(eval_q, args.model, cache=args.cache)

    s_sel, who_sel = score_against(C, Qs, args.knn)
    s_eval, who_eval = score_against(C, Qe, args.knn)

    # DEV SET first, from the EVAL queries, then removed from everything else.
    dev_idx = take_capped(np.argsort(-s_eval), who_eval, args.dev_n, args.per_query_cap)
    dev = set(dev_idx)
    print(f"  dev set: {len(dev_idx):,} pairs nearest the EVAL queries")

    # TRAINING selection from the SELECT queries, dev excluded.
    if args.random_control:
        pool_idx = [i for i in range(len(pool)) if i not in dev]
        random.Random(args.seed + 1).shuffle(pool_idx)
        sel_idx = pool_idx[:args.select_n]
        print("  CONTROL ARM: slice chosen at RANDOM, not by similarity")
    else:
        order = [i for i in np.argsort(-s_sel) if i not in dev]
        sel_idx = take_capped(np.array(order), who_sel, args.select_n, args.per_query_cap)
    sel = set(sel_idx)
    print(f"  selected: {len(sel_idx):,} pairs (per-query cap {args.per_query_cap})")

    rest = [i for i in range(len(pool)) if i not in dev and i not in sel]
    # r * |sel| / (r * |sel| + |rest|) = share  ->  r = share*|rest| / ((1-share)*|sel|)
    share = args.upweight_share
    repeat = max(1, min(args.max_repeat,
                        round(share * len(rest) / max(1e-9, (1 - share) * len(sel_idx)))))
    train = [pool[i] for i in rest] + [pool[i] for i in sel_idx] * repeat
    got = repeat * len(sel_idx) / len(train)
    print(f"\n  UPWEIGHT x{repeat}  ->  selected slice is {100*got:.1f}% of "
          f"{len(train):,} training pairs (target {100*share:.0f}%)")

    args.output.mkdir(parents=True, exist_ok=True)
    # Layout must match what train/finetune.py expects:
    #   <data_dir>/<split>/<src>-<tgt>/<split>.<lang>
    for name, rows in (("train", train), ("dev", [pool[i] for i in dev_idx])):
        d = args.output / name / "eng_Latn-kas_Arab"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.eng_Latn", "w", encoding="utf-8") as f1, \
             open(d / f"{name}.kas_Arab", "w", encoding="utf-8") as f2:
            for r in rows:
                f1.write(r["s"] + "\n")
                f2.write(r["t"] + "\n")

    # leakage: dev must not appear in train, by exact pair key
    tk = {(r["s"], strip_key(r["t"])) for r in train}
    leak = sum(1 for i in dev_idx if (pool[i]["s"], strip_key(pool[i]["t"])) in tk)
    print(f"  LEAKAGE dev∩train: {leak}  {'OK' if leak == 0 else 'FATAL'}")
    if leak:
        raise SystemExit("dev set leaked into training")

    stats = {
        "pool": profile(pool), "train": profile(train),
        "dev": profile([pool[i] for i in dev_idx]),
        "selected_slice": profile([pool[i] for i in sel_idx]),
        "repeat": repeat, "selected_share": round(got, 4),
        "queries": {"select": len(sel_q), "eval": len(eval_q), "seed": args.seed},
        "by_subset_selected": dict(Counter(pool[i]["config"] for i in sel_idx)),
        "by_subset_dev": dict(Counter(pool[i]["config"] for i in dev_idx)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {'set':18}{'pairs':>10}{'src words':>11}{'rest/100c':>11}")
    for k in ("pool", "selected_slice", "train", "dev"):
        v = stats[k]
        print(f"  {k:18}{v['pairs']:>10,}{v['src_words']:>11.2f}{v['restorable_per_100c']:>11.2f}")
    print(f"\n  selected slice by subset: "
          + ", ".join(f"{k} {v:,}" for k, v in
                      sorted(stats['by_subset_selected'].items(), key=lambda kv: -kv[1])))
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
