#!/usr/bin/env python3
"""
KATHE 2026 — train and evaluate the character-level diacritic restorer (R11).

What is measured, and why not accuracy
--------------------------------------
"None" is ~79% of all label positions, so a model that predicts nothing scores
79% accuracy and restores nothing. Every number here is therefore **per-mark
precision and recall**:

    precision(kasra) = predicted kasra that were kasra
    recall(kasra)    = true kasra that were predicted kasra

Inserting a wrong mark and omitting a mark are different failures with different
costs — chrF++ gives partial credit for a wrong mark at the right position but
none at all for an omission (see `data/diacritize.py`, where dominance 0.0 beat
0.6 and 0.9 for exactly this reason). Aggregating them would hide the trade the
`--none-bias` knob exists to make.

Two evaluations, and R0 is the honest one
-----------------------------------------
`--eval-heldout` is a random slice of the training text: same corpora, same
register, so it measures fitting.

`--eval-r0` takes R0's human references, strips the three marks, restores, and
compares to the originals. R0 is the register we are scored on and its
references were excluded from the training text by exact stripped string, so it
measures what the model does on sentences of the right kind that it has not
seen. Where the two disagree, R0 is the one to believe.

Usage:
    uv run python -m restore.train fit \\
        --corpus data/processed/restore_text.jsonl \\
        --out    models/restore/r11_all.pt --epochs 6

    uv run python -m restore.train eval \\
        --checkpoint models/restore/r11_all.pt --refs data/dev/r0/r0.kas_Arab
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restore.chartag import (  # noqa: E402
    IGNORE, MARK_NAMES, N_LABELS, PAD, CharTagger, ModelConfig, Restorer, Vocab,
    chunk, encode, save_checkpoint, strip_key,
)


def _normalize_refs(lines: list[str]) -> list[str]:
    """Scorer-normalize reference text, if the scorer's normalizer is installed.

    Training deliberately depends on **torch and the standard library only** —
    no transformers, no IndicTransToolkit, no KashmiriNormalizer — so it runs on
    a bare Kaggle image with no pip install and therefore no chance of an
    unpinned metric package appearing next to the pinned one (PROJECT_NOTES.md §5).
    This is the single place that would have wanted one, and it degrades
    cleanly: `normalize()` is idempotent and R0's references are already
    normalized (verified 2026-08-12: 0 of 1,003 lines changed), so passing them
    through unchanged is exact for the file this is actually run on.
    """
    try:
        from data.normalize import NormConfig, normalize
    except ImportError:
        print("  note: KashmiriNormalizer not installed — references used as-is. "
              "They must already be scorer-normalized.")
        return lines
    cfg = NormConfig(scorer_normalizer=True)
    return [normalize(x, cfg) for x in lines]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_corpus(path: Path, sources: list[str] | None, max_len: int) -> list[tuple[str, list[int], str]]:
    """Read the built corpus into (base, labels, source) examples."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if sources and row["source"] not in sources:
                continue
            for piece, _sep in chunk(row["text"], max_len):
                base, labels = encode(piece)
                if base:
                    out.append((base, labels, row["source"]))
    return out


class Batcher:
    """Length-bucketed batches. Padding is the whole cost at this model size."""

    def __init__(self, examples, vocab: Vocab, batch_size: int, shuffle: bool, seed: int = 0):
        self.ex = examples
        self.vocab = vocab
        self.bs = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return (len(self.ex) + self.bs - 1) // self.bs

    def __iter__(self):
        idx = list(range(len(self.ex)))
        if self.shuffle:
            self.rng.shuffle(idx)
            # Sort within a large window so batches are length-homogeneous
            # without becoming deterministic across epochs.
            window = self.bs * 64
            idx = [j for k in range(0, len(idx), window)
                   for j in sorted(idx[k:k + window], key=lambda i: len(self.ex[i][0]))]
        else:
            idx.sort(key=lambda i: len(self.ex[i][0]))
        batches = [idx[k:k + self.bs] for k in range(0, len(idx), self.bs)]
        if self.shuffle:
            self.rng.shuffle(batches)
        for batch in batches:
            width = max(len(self.ex[i][0]) for i in batch)
            ids = torch.full((len(batch), width), PAD, dtype=torch.long)
            ys = torch.full((len(batch), width), IGNORE, dtype=torch.long)
            for r, i in enumerate(batch):
                base, labels, _ = self.ex[i]
                ids[r, : len(base)] = torch.tensor(self.vocab.encode(base))
                ys[r, : len(labels)] = torch.tensor(labels)
            yield ids, ys


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def per_mark_prf(pred: Counter, gold: Counter, hit: Counter) -> dict:
    """`hit[m]` counts positions where prediction and gold are both mark m."""
    rows = {}
    tp = fp = fn = 0
    for m in range(1, N_LABELS):
        p = hit[m] / pred[m] if pred[m] else 0.0
        r = hit[m] / gold[m] if gold[m] else 0.0
        rows[MARK_NAMES[m]] = {
            "gold": gold[m], "predicted": pred[m], "correct": hit[m],
            "precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(100 * 2 * p * r / (p + r), 2) if p + r else 0.0,
        }
        tp += hit[m]
        fp += pred[m] - hit[m]
        fn += gold[m] - hit[m]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    rows["MICRO"] = {
        "gold": tp + fn, "predicted": tp + fp, "correct": tp,
        "precision": round(100 * p, 2), "recall": round(100 * r, 2),
        "f1": round(100 * 2 * p * r / (p + r), 2) if p + r else 0.0,
    }
    return rows


def score_texts(gold_texts: list[str], restored: list[str]) -> dict:
    """Per-mark P/R by comparing two diacritized strings position by position.

    Both are re-encoded against the same stripped base, so position i means the
    same character in both. `assert_insertion_only` has already guaranteed the
    bases match; this asserts it again per line because a silent misalignment
    here would corrupt every number below it.
    """
    pred: Counter = Counter()
    gold: Counter = Counter()
    hit: Counter = Counter()
    exact = 0
    for g, h in zip(gold_texts, restored):
        gb, gl = encode(g)
        hb, hl = encode(h)
        if gb != hb:
            raise AssertionError(f"base mismatch:\n  gold {gb!r}\n  hyp  {hb!r}")
        for a, b in zip(gl, hl):
            if a:
                gold[a] += 1
            if b:
                pred[b] += 1
            if a and a == b:
                hit[a] += 1
        exact += int(gl == hl)
    out = per_mark_prf(pred, gold, hit)
    out["EXACT_LINES"] = {"n": len(gold_texts), "exact": exact,
                          "pct": round(100 * exact / max(1, len(gold_texts)), 2)}
    return out


def print_prf(title: str, rows: dict) -> None:
    print(f"\n  {title}")
    print(f"    {'mark':10} {'gold':>8} {'pred':>8} {'ok':>8} {'P%':>7} {'R%':>7} {'F1':>7}")
    for name in [*MARK_NAMES[1:], "MICRO"]:
        r = rows[name]
        print(f"    {name:10} {r['gold']:>8,} {r['predicted']:>8,} {r['correct']:>8,} "
              f"{r['precision']:>7.2f} {r['recall']:>7.2f} {r['f1']:>7.2f}")
    if "EXACT_LINES" in rows:
        e = rows["EXACT_LINES"]
        print(f"    exact lines: {e['exact']:,}/{e['n']:,} ({e['pct']}%)")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def evaluate_batches(model: CharTagger, batcher: Batcher, device: str,
                     none_bias: float = 0.0) -> tuple[float, dict]:
    model.eval()
    loss_sum = ntok = 0
    pred: Counter = Counter()
    gold: Counter = Counter()
    hit: Counter = Counter()
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
    with torch.inference_mode():
        for ids, ys in batcher:
            ids, ys = ids.to(device), ys.to(device)
            logits = model(ids).float()
            loss_sum += lossf(logits.view(-1, N_LABELS), ys.view(-1)).item()
            ntok += int((ys != IGNORE).sum())
            if none_bias:
                logits[..., 0] += none_bias
            p = logits.argmax(-1)
            mask = ys != IGNORE
            for m in range(1, N_LABELS):
                pm, gm = (p == m) & mask, (ys == m) & mask
                pred[m] += int(pm.sum())
                gold[m] += int(gm.sum())
                hit[m] += int((pm & gm).sum())
    return loss_sum / max(1, ntok), per_mark_prf(pred, gold, hit)


def fit(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    examples = load_corpus(args.corpus, args.sources, args.max_len)
    by_source = Counter(s for _, _, s in examples)
    print(f"  {len(examples):,} examples from {len(by_source)} source(s): "
          + ", ".join(f"{k} {v:,}" for k, v in by_source.most_common()))

    rng = random.Random(args.seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_dev = min(args.heldout, len(idx) // 20)
    dev = [examples[i] for i in idx[:n_dev]]
    train = [examples[i] for i in idx[n_dev:]]
    print(f"  train {len(train):,} / held-out {len(dev):,}")

    vocab = Vocab.build((b for b, _, _ in train), args.vocab_min_count)
    cfg = ModelConfig(vocab_size=len(vocab), d_model=args.d_model, n_heads=args.heads,
                      n_layers=args.layers, d_ff=args.d_ff, dropout=args.dropout,
                      max_len=args.max_len)
    device = args.device
    model = CharTagger(cfg).to(device)
    print(f"  vocab {len(vocab)}  params {model.n_params():,}  device {device}")

    train_b = Batcher(train, vocab, args.batch_size, shuffle=True, seed=args.seed)
    dev_b = Batcher(dev, vocab, args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    steps = len(train_b) * args.epochs
    warmup = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE, label_smoothing=args.label_smoothing)

    history = []
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = n = 0
        for step, (ids, ys) in enumerate(train_b, 1):
            ids, ys = ids.to(device), ys.to(device)
            loss = lossf(model(ids).view(-1, N_LABELS), ys.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            n += 1
            if step % args.log_every == 0:
                print(f"    epoch {epoch} step {step}/{len(train_b)}  "
                      f"loss {run / n:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{time.time() - t0:.0f}s", flush=True)
                run = n = 0
        dev_loss, dev_prf = evaluate_batches(model, dev_b, device)
        f1 = dev_prf["MICRO"]["f1"]
        print(f"  == epoch {epoch}: held-out loss {dev_loss:.4f}  "
              f"micro P {dev_prf['MICRO']['precision']:.2f} "
              f"R {dev_prf['MICRO']['recall']:.2f} F1 {f1:.2f}  "
              f"({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "dev_loss": dev_loss, "dev_prf": dev_prf})
        if f1 >= best:
            best = f1
            save_checkpoint(args.out, model, vocab, {
                "epoch": epoch, "dev_loss": dev_loss, "dev_prf": dev_prf,
                "sources": dict(by_source), "corpus": str(args.corpus),
                "train_examples": len(train), "heldout_examples": len(dev),
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items() if k != "func"},
                "history": history, "wall_seconds": round(time.time() - t0, 1),
            })
            print(f"     saved {args.out} (best micro-F1 {best:.2f})")

    print_prf("held-out (training-text slice), best checkpoint", history[-1]["dev_prf"])
    print(f"\n  total {time.time() - t0:.0f}s")

    if args.refs:
        evaluate_refs(args.out, args.refs, args.device, args.none_bias_sweep)
    return 0


# --------------------------------------------------------------------------- #
# evaluate on real references
# --------------------------------------------------------------------------- #
def evaluate_refs(checkpoint: Path, refs: Path, device: str,
                  bias_sweep: list[float] | None = None) -> dict:
    """Strip R0's references, restore them, and compare. The honest test."""
    with open(refs, encoding="utf-8") as fh:
        gold = _normalize_refs([ln.rstrip("\n") for ln in fh])
    gold = [g for g in gold if g]
    stripped = [strip_key(g) for g in gold]

    results = {}
    for bias in (bias_sweep or [0.0]):
        r = Restorer(checkpoint, device=device, none_bias=bias)
        t0 = time.time()
        out = r.restore_many(stripped)
        wall = time.time() - t0
        rows = score_texts(gold, out)
        # Each density is over its OWN character count. Using the restored
        # text's length for both makes the reference look denser than it is,
        # because a restored line that inserted nothing is shorter.
        marks = sum(1 for x in out for c in x if c in "َُِ")
        gmarks = sum(1 for x in gold for c in x if c in "َُِ")
        rows["DENSITY"] = {
            "restored_per_100c": round(100 * marks / max(1, sum(len(x) for x in out)), 2),
            "reference_per_100c": round(100 * gmarks / max(1, sum(len(x) for x in gold)), 2),
        }
        rows["WALL"] = {"seconds": round(wall, 2), "lines": len(out),
                        "per_line_ms": round(1000 * wall / max(1, len(out)), 3)}
        print_prf(f"{refs.name}, stripped -> restored  (none_bias {bias:+.1f})", rows)
        d = rows["DENSITY"]
        print(f"    restorable density {d['restored_per_100c']}/100c "
              f"vs references {d['reference_per_100c']}/100c   "
              f"[{rows['WALL']['seconds']}s for {len(out):,} lines]")
        results[f"bias_{bias}"] = rows
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--corpus", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--sources", nargs="*", default=None,
                   help="restrict to these source tags (provenance ablation)")
    f.add_argument("--refs", type=Path, help="also evaluate on these references after fitting")
    f.add_argument("--epochs", type=int, default=6)
    f.add_argument("--batch-size", type=int, default=64)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--label-smoothing", type=float, default=0.0)
    f.add_argument("--dropout", type=float, default=0.1)
    f.add_argument("--d-model", type=int, default=256)
    f.add_argument("--heads", type=int, default=4)
    f.add_argument("--layers", type=int, default=4)
    f.add_argument("--d-ff", type=int, default=1024)
    f.add_argument("--max-len", type=int, default=384)
    f.add_argument("--vocab-min-count", type=int, default=5)
    f.add_argument("--heldout", type=int, default=5000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    f.add_argument("--log-every", type=int, default=200)
    f.add_argument("--none-bias-sweep", type=float, nargs="*",
                   default=[0.0, -0.5, -1.0, -1.5])
    f.set_defaults(func=fit)

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True, type=Path)
    e.add_argument("--refs", required=True, type=Path)
    e.add_argument("--device", default="cpu")
    e.add_argument("--none-bias-sweep", type=float, nargs="*", default=[0.0])
    e.add_argument("--json-out", type=Path)
    e.set_defaults(func=lambda a: (
        (a.json_out.write_text(json.dumps(
            evaluate_refs(a.checkpoint, a.refs, a.device, a.none_bias_sweep),
            ensure_ascii=False, indent=2), encoding="utf-8") if a.json_out
         else evaluate_refs(a.checkpoint, a.refs, a.device, a.none_bias_sweep)), 0)[1])

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
