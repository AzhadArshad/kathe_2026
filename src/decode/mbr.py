#!/usr/bin/env python3
"""
KATHE 2026 — R5: Minimum Bayes Risk decoding with chrF++ as the utility.

Why this is the right shape of lever here
-----------------------------------------
Every post-processing gain so far has come from a lookup table built on BPCC,
and R0 is cut from BPCC — which is how submission 005 scored 33.50 on R0 and
REGRESSED to 11.08 on the leaderboard (PLANNING.md 2026-08-12). MBR uses no
external table at all: it only compares the model's own candidate outputs to
each other, so there is nothing for R0's provenance to leak into. R0 should
therefore rank MBR honestly, which is not true of the context tables.

Method
------
Draw `pool_size` candidates per source sentence, then keep the one with the
highest mean chrF++ against the rest of its own pool. That is the Monte-Carlo
MBR estimator with the competition's own metric as the utility function:

    y* = argmax_{y in Y}  (1/|Y'|) sum_{y' in Y'} chrF++(y, y')

The utility is computed with sacreBLEU's OWN CHRF internals at `word_order=2`,
so it is the identical function the official scorer applies — see
`ChrfUtility`, which reuses `CHRF._get_match_statistics` and
`CHRF._compute_f_score` rather than reimplementing them. Verified equal to
`sacrebleu.sentence_chrf(..., word_order=2)` to full float precision.

ORDER MATTERS: MBR first, diacritic restoration second
------------------------------------------------------
Restoration (`data/diacritize.py`) is a deterministic transform applied to every
candidate alike. Running it before selection would push every candidate toward
the same lexicon forms, compressing exactly the differences MBR is trying to
measure, and would make the utility partly a measure of lexicon agreement
rather than of model consensus. So: select on raw decoder output, restore
afterwards, in post-processing, unchanged from submissions 003-007.

Utility is nonetheless computed on SCORER-NORMALIZED text (`utility_normalize:
scorer`), because that is what the metric sees. The RAW candidate at the winning
index is what gets written out, so the downstream `make_submission.py` pipeline
is byte-for-byte the same as it was for sub 007.

Two stages, deliberately separate
---------------------------------
`generate` is GPU-bound and expensive; `select` is CPU-bound and cheap. Keeping
them apart means pool size, sampling method and utility settings can be re-swept
offline from one decode, and means the cross-system pool is a `select` over two
pool files rather than a second decode.

    python -m decode.mbr generate --config config/mbr_r5_200m.yaml \\
        --input data/dev/r0/r0.eng_Latn --output data/dev/r0/pool.200m.jsonl

    python -m decode.mbr select --config config/mbr_r5_200m.yaml \\
        --candidates data/dev/r0/pool.200m.jsonl \\
        --output data/dev/r0/r0.hyp.mbr

    # cross-system: same select, two pools
    python -m decode.mbr select --config config/mbr_r5_cross.yaml \\
        --candidates data/dev/r0/pool.200m.jsonl data/dev/r0/pool.1b.jsonl \\
        --output data/dev/r0/r0.hyp.mbr-cross

Environment: generation needs `transformers==4.46.1` and IndicTransToolkit, i.e.
`.venv-decode` locally or the Kaggle env. Selection needs only sacrebleu, so it
runs in the main `.venv`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MBRConfig:
    """Every knob lives here and comes from YAML, never from a flag (PROJECT_NOTES.md
    §6), so the file recorded in `submissions/<n>/` fully determines the output.
    """

    # --- generation ---------------------------------------------------------
    model: str = "models/r3-200m-full"
    system: str = ""  # label carried into the pool file; defaults to `model`

    pool_size: int = 32
    # epsilon | ancestral | topk | topp | diverse_beam
    #
    # `epsilon` is the default on evidence from the MBR literature: truncating
    # the tail at a probability floor removes the degenerate samples that
    # ancestral sampling draws from a 122k-token vocabulary, without the
    # distribution distortion that a hard top-k imposes. `diverse_beam` is the
    # documented fallback for a transformers build that lacks `epsilon_cutoff`.
    sampling: str = "epsilon"
    epsilon_cutoff: float = 0.02
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    # diverse_beam only. num_beams is forced to pool_size and must be divisible
    # by num_beam_groups.
    num_beam_groups: int = 8
    diversity_penalty: float = 0.5

    # Add the plain beam-search hypothesis to the pool as a safety floor: if the
    # samples are all poor, MBR can still fall back to what beam would have
    # produced, and it is judged by the same utility as everything else.
    include_beam: bool = True
    beam_size: int = 5
    length_penalty: float = 1.0

    max_new_tokens: int = 256
    batch_size: int = 8  # sequences in flight = batch_size * pool_size
    seed: int = 42
    device: str = "auto"

    # --- utility ------------------------------------------------------------
    utility_metric: str = "chrf++"
    char_order: int = 6
    word_order: int = 2  # 2 == chrF++, matching the official scorer
    beta: int = 2
    # Compute the utility on scorer-normalized text, because that is what the
    # metric sees. The RAW candidate is still what gets written out.
    utility_normalize: str = "scorer"  # scorer | none
    # "mean chrF++ against all OTHER candidates" — the diagonal is excluded.
    # Duplicates still count: a candidate sampled 5 times contributes 4 self-
    # matches at 100 to its own score, which is the consensus signal, not a bug.
    include_self: bool = False


def load_config(path: Path) -> MBRConfig:
    """Build an MBRConfig from YAML, rejecting unknown keys loudly.

    Same discipline as `make_submission.load_postproc`: a typo'd key that
    silently defaulted would change the output without changing the recorded
    config, which is how a leaderboard reading becomes unreproducible.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(MBRConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"FATAL: unknown MBR config keys in {path}: {sorted(unknown)}")
    cfg = MBRConfig(**raw)
    if cfg.pool_size < 2:
        raise SystemExit("FATAL: pool_size must be >= 2; MBR needs something to compare against")
    if cfg.sampling == "diverse_beam" and cfg.pool_size % cfg.num_beam_groups:
        raise SystemExit(
            f"FATAL: diverse beam needs pool_size ({cfg.pool_size}) divisible by "
            f"num_beam_groups ({cfg.num_beam_groups})"
        )
    if cfg.sampling not in {"epsilon", "ancestral", "topk", "topp", "diverse_beam"}:
        raise SystemExit(f"FATAL: unknown sampling method {cfg.sampling!r}")
    return cfg


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------


class ChrfUtility:
    """chrF++ between two candidates, sharing sacreBLEU's own implementation.

    n-grams are extracted ONCE per candidate and reused across the whole
    pairwise matrix. That is the only reason a 32x32 pool over 1,003 sentences
    is affordable in Python: the naive route re-tokenizes each candidate 64
    times per sentence.
    """

    def __init__(self, char_order: int = 6, word_order: int = 2, beta: int = 2):
        from sacrebleu.metrics.chrf import CHRF

        self._chrf = CHRF(char_order=char_order, word_order=word_order, beta=beta)
        self.word_order = word_order

    def ngrams(self, text: str) -> list[Counter]:
        from sacrebleu.metrics.helpers import extract_all_char_ngrams, extract_word_ngrams

        c = self._chrf
        stats = extract_all_char_ngrams(text, c.char_order, c.whitespace)
        if self.word_order > 0:
            words = c._remove_punctuation(text)
            stats.extend(extract_word_ngrams(words, n) for n in range(1, self.word_order + 1))
        return stats

    def score(self, hyp: list[Counter], ref: list[Counter]) -> float:
        c = self._chrf
        stats: list[int] = []
        for h, r in zip(hyp, ref):
            stats.extend(c._get_match_statistics(h, r))
        return c._compute_f_score(stats)


def mbr_select(
    candidates: list[str],
    util: ChrfUtility,
    keys: list[str] | None = None,
    include_self: bool = False,
) -> tuple[int, list[float]]:
    """Return (winning index into `candidates`, per-candidate expected utility).

    `keys` is the text the utility is computed on — normally the scorer-
    normalized form of each candidate — while `candidates` is what gets
    returned. They are parallel lists.

    Identical candidates are collapsed before the O(n^2) pass and re-expanded
    with multiplicity, which is exact and typically halves the work: epsilon
    sampling on a 7-word sentence draws the same string many times over.
    """
    keys = keys if keys is not None else candidates
    if not candidates:
        raise ValueError("empty candidate pool")
    if len(candidates) == 1:
        return 0, [0.0]

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    uniq = list(groups)
    counts = [len(groups[u]) for u in uniq]
    ngrams = [util.ngrams(u) for u in uniq]
    n = len(uniq)

    # chrF is NOT symmetric — recall is weighted beta^2 times precision — so
    # both directions are computed. Only the diagonal is free: chrF(y, y) == 100.
    totals: list[float] = []
    for i in range(n):
        ng_i = ngrams[i]
        acc = 100.0 * (counts[i] if include_self else counts[i] - 1)
        for j in range(n):
            if i != j:
                acc += counts[j] * util.score(ng_i, ngrams[j])
        totals.append(acc)

    denom = len(candidates) if include_self else max(1, len(candidates) - 1)
    expected = [t / denom for t in totals]

    best_u = max(range(n), key=lambda i: (expected[i], counts[i], -i))
    # Report per-candidate utilities in the ORIGINAL indexing, so callers can
    # tell which system a winner came from in a cross-system pool.
    per_candidate = [0.0] * len(candidates)
    for u_idx, u in enumerate(uniq):
        for orig in groups[u]:
            per_candidate[orig] = expected[u_idx]
    # Ties inside a duplicate group resolve to its first occurrence, which in a
    # cross-system pool means the first system listed. Deterministic either way.
    return groups[uniq[best_u]][0], per_candidate


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gen_kwargs(cfg: MBRConfig, n: int) -> dict:
    """Decoding kwargs for one sampling method. Kept in one place so the config
    is the only thing that decides how candidates are drawn."""
    common = {"num_return_sequences": n, "max_new_tokens": cfg.max_new_tokens, "use_cache": True}
    if cfg.sampling == "diverse_beam":
        return {
            **common,
            "do_sample": False,
            "num_beams": n,
            "num_beam_groups": cfg.num_beam_groups,
            "diversity_penalty": cfg.diversity_penalty,
            "length_penalty": cfg.length_penalty,
            "early_stopping": True,
        }
    sample = {**common, "do_sample": True, "num_beams": 1, "temperature": cfg.temperature}
    if cfg.sampling == "epsilon":
        sample["epsilon_cutoff"] = cfg.epsilon_cutoff
    elif cfg.sampling == "topk":
        sample["top_k"] = cfg.top_k or 50
    elif cfg.sampling == "topp":
        sample["top_p"] = cfg.top_p if cfg.top_p < 1.0 else 0.9
        sample["top_k"] = 0
    else:  # ancestral — no truncation at all
        sample["top_k"] = 0
        sample["top_p"] = 1.0
    return sample


def generate_pool(
    sentences: list[str],
    cfg: MBRConfig,
    progress: bool = True,
) -> tuple[list[list[str]], dict]:
    """Draw `cfg.pool_size` candidates for every sentence.

    IndicProcessor runs on BOTH sides, exactly as in `scripts/translate.py`:
    sources need language tags and transliteration going in, and output needs
    `postprocess_batch` coming out. Skipping either produces text that looks
    plausible and scores badly (PROJECT_NOTES.md §5).
    """
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = _pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    ip = IndicProcessor(inference=True)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device).eval()

    # A checkpoint that loads without error can still emit nothing but empty
    # strings — the tied-lm_head failure of 2026-08-10 did exactly that, with a
    # clean `from_pretrained` log. Decode before trusting anything (PROJECT_NOTES.md §5).
    _smoke_test(model, tok, ip, device, cfg)

    order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i].split()))
    pools: list[list[str] | None] = [None] * len(sentences)

    n_sample = cfg.pool_size - (1 if cfg.include_beam else 0)
    gen_kwargs = _gen_kwargs(cfg, n_sample)

    iterator = range(0, len(order), cfg.batch_size)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            iterator,
            total=(len(order) + cfg.batch_size - 1) // cfg.batch_size,
            unit="batch",
            desc=f"mbr {cfg.sampling} x{cfg.pool_size} {device}",
            file=sys.stderr,
            dynamic_ncols=True,
        )

    t0 = time.time()
    for start in iterator:
        idx = order[start : start + cfg.batch_size]
        srcs = [sentences[i] for i in idx]
        batch = ip.preprocess_batch(srcs, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)

        with torch.inference_mode():
            sampled = model.generate(**enc, **gen_kwargs)
            beam = (
                model.generate(
                    **enc,
                    num_beams=cfg.beam_size,
                    num_return_sequences=1,
                    max_new_tokens=cfg.max_new_tokens,
                    length_penalty=cfg.length_penalty,
                    early_stopping=True,
                    use_cache=True,
                )
                if cfg.include_beam
                else None
            )

        texts = _decode(tok, ip, sampled, srcs, n_sample)
        beam_texts = _decode(tok, ip, beam, srcs, 1) if beam is not None else None
        for b, i in enumerate(idx):
            cands = texts[b * n_sample : (b + 1) * n_sample]
            if beam_texts is not None:
                cands = [beam_texts[b]] + cands
            pools[i] = cands

    wall = time.time() - t0
    if any(p is None for p in pools):
        raise SystemExit("internal error: not every sentence produced a pool")

    stats = {
        "sentences": len(sentences),
        "pool_size": cfg.pool_size,
        "device": device,
        "wall_seconds": round(wall, 2),
        "seconds_per_sentence": round(wall / max(1, len(sentences)), 4),
        "sentences_per_second": round(len(sentences) / max(1e-9, wall), 3),
    }
    return [p for p in pools if p is not None], stats


def _decode(tok, ip, gen, srcs: list[str], num_return_sequences: int) -> list[str]:
    """Detokenize and postprocess `len(srcs) * num_return_sequences` outputs.

    Two things here HANG rather than raise if you get them wrong, which is the
    worst failure mode on a metered GPU session — this cost a stalled run on
    2026-08-12 and is the same trap already on record for in-training eval
    (PLANNING.md 2026-08-09).

    1. `postprocess_batch` pops ONE placeholder-entity map per *input* from an
       internal `Queue`, using `num_return_sequences` to decide how many outputs
       share a map. Called with the default 1 on a pool of 32 candidates it
       drains the queue after the first 32/32 sentences and then blocks forever
       on `Queue.get()`. Pass the real pool width.

    2. It CLEARS the queue when it returns. So a second postprocess call — the
       beam candidate, here — finds an empty queue and blocks. Hence the
       `preprocess_batch` refill below: it is pure string work, costs nothing on
       the GPU, and the maps it produces are deterministic and identical to the
       ones the encoder pass queued.
    """
    ip.preprocess_batch(srcs, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    with tok.as_target_tokenizer():
        decoded = tok.batch_decode(
            gen.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    if len(decoded) != len(srcs) * num_return_sequences:
        raise SystemExit(
            f"internal error: {len(decoded)} decoded sequences for {len(srcs)} sources "
            f"x {num_return_sequences} — postprocess_batch would misalign the "
            "placeholder maps silently."
        )
    return ip.postprocess_batch(decoded, lang=TGT_LANG,
                                num_return_sequences=num_return_sequences)


def _smoke_test(model, tok, ip, device, cfg) -> None:
    import torch

    probe = ["He lost his pen.", "I go to my school daily.", "She is a teacher."]
    batch = ip.preprocess_batch(probe, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(**enc, num_beams=1, max_new_tokens=64, use_cache=True)
    out = _decode(tok, ip, gen, probe, 1)
    if not all(t.strip() for t in out):
        raise SystemExit(
            f"FATAL: {cfg.model} decoded an empty string on the smoke probe. "
            "This is the tied-lm_head failure (PLANNING.md 2026-08-10) — the "
            "checkpoint loads cleanly and produces nothing. Do not score it."
        )
    print("  smoke test OK:", file=sys.stderr)
    for s, t in zip(probe, out):
        print(f"    {s}  ->  {t}", file=sys.stderr)


# ---------------------------------------------------------------------------
# pool file I/O
# ---------------------------------------------------------------------------


def write_pool(path: Path, sentences: list[str], pools: list[list[str]], cfg: MBRConfig, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    system = cfg.system or cfg.model
    with open(path, "w", encoding="utf-8") as fh:
        for i, (src, cands) in enumerate(zip(sentences, pools)):
            fh.write(json.dumps({"i": i, "src": src, "system": system, "candidates": cands},
                                ensure_ascii=False) + "\n")
    meta = path.with_suffix(path.suffix + ".meta.json")
    meta.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def read_pools(paths: list[Path]) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Merge one or more pool files into (sources, candidates, systems).

    Merging is BY ROW INDEX and the sources are asserted equal across files.
    Scoring is positional (PROJECT_NOTES.md §3), so a silent misalignment between two
    pool files would look correct and score near zero.
    """
    merged_src: list[str] | None = None
    merged_cands: list[list[str]] = []
    merged_sys: list[list[str]] = []
    for p in paths:
        srcs, cands, syss = [], [], []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                srcs.append(rec["src"])
                cands.append(rec["candidates"])
                syss.append([rec.get("system", p.stem)] * len(rec["candidates"]))
        if merged_src is None:
            merged_src = srcs
            merged_cands = [list(c) for c in cands]
            merged_sys = [list(s) for s in syss]
            continue
        if len(srcs) != len(merged_src):
            raise SystemExit(f"FATAL: {p} has {len(srcs)} rows, expected {len(merged_src)}")
        bad = [i for i, (a, b) in enumerate(zip(srcs, merged_src)) if a != b]
        if bad:
            raise SystemExit(
                f"FATAL: {p} row {bad[0]} source differs from the first pool file. "
                "Pools must be generated from the same input in the same order."
            )
        for i in range(len(merged_cands)):
            merged_cands[i].extend(cands[i])
            merged_sys[i].extend(syss[i])
    if merged_src is None:
        raise SystemExit("FATAL: no pool files given")
    return merged_src, merged_cands, merged_sys


# ---------------------------------------------------------------------------
# selection driver
# ---------------------------------------------------------------------------


def select_all(
    pools: list[list[str]],
    cfg: MBRConfig,
    systems: list[list[str]] | None = None,
    progress: bool = True,
) -> tuple[list[str], dict]:
    """Run MBR over every pool. Returns (selected raw candidates, stats)."""
    sys_path = str(Path(__file__).resolve().parents[1])
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from data.normalize import NormConfig, normalize

    util = ChrfUtility(cfg.char_order, cfg.word_order, cfg.beta)
    norm_cfg = NormConfig(scorer_normalizer=True)
    use_norm = cfg.utility_normalize == "scorer"

    chosen: list[str] = []
    winner_systems: Counter = Counter()
    beam_wins = 0
    empty_pools = 0
    dropped_empty = 0
    uniq_total = 0
    cand_total = 0

    iterator = range(len(pools))
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, unit="sent", desc="mbr select", file=sys.stderr, dynamic_ncols=True)

    t0 = time.time()
    for i in iterator:
        raw = pools[i]
        srcs = systems[i] if systems else [""] * len(raw)
        # An empty candidate is fatal to a submission (the scorer raises), and
        # sampling does occasionally produce one. Drop them before selection
        # rather than letting one win on a degenerate utility.
        keep = [j for j, c in enumerate(raw) if c.strip()]
        dropped_empty += len(raw) - len(keep)
        if not keep:
            empty_pools += 1
            chosen.append("")
            continue
        cands = [raw[j] for j in keep]
        cand_sys = [srcs[j] for j in keep]
        keys = [normalize(c, norm_cfg) for c in cands] if use_norm else cands
        # Normalization can itself empty a candidate; fall back to raw keys.
        keys = [k if k.strip() else cands[n] for n, k in enumerate(keys)]

        best, _ = mbr_select(cands, util, keys=keys, include_self=cfg.include_self)
        chosen.append(cands[best])
        winner_systems[cand_sys[best]] += 1
        if cfg.include_beam and keep[best] == 0:
            beam_wins += 1
        uniq_total += len(set(keys))
        cand_total += len(cands)

    wall = time.time() - t0
    stats = {
        "sentences": len(pools),
        "wall_seconds": round(wall, 2),
        "seconds_per_sentence": round(wall / max(1, len(pools)), 4),
        "mean_candidates": round(cand_total / max(1, len(pools)), 2),
        "mean_unique_candidates": round(uniq_total / max(1, len(pools)), 2),
        "dropped_empty_candidates": dropped_empty,
        "empty_pools": empty_pools,
        "beam_candidate_won": beam_wins,
        "winners_by_system": dict(winner_systems),
    }
    return chosen, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        import csv

        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        cols = rows[0].keys()
        col = next((c for c in ("sentence", "english", "text", "src") if c in cols), None)
        if col is None:
            raise SystemExit(f"{path}: no source column among {list(cols)}")
        return [str(r[col]) for r in rows]
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="draw candidate pools (GPU)")
    g.add_argument("--config", required=True, type=Path)
    g.add_argument("--input", required=True, type=Path)
    g.add_argument("--output", required=True, type=Path, help="pool JSONL")
    g.add_argument("--model", help="override config.model")
    g.add_argument("--system", help="override the system label recorded in the pool")
    g.add_argument("--device", help="override config.device")
    g.add_argument("--batch-size", type=int)
    g.add_argument("--limit", type=int, help="first N sentences only (smoke test)")

    s = sub.add_parser("select", help="MBR selection over pools (CPU)")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--candidates", required=True, type=Path, nargs="+")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--pool-size", type=int, help="truncate each file's pool to the first N")
    s.add_argument("--stats", type=Path, help="also write selection stats here")

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "generate":
        overrides = {k: v for k, v in
                     (("model", args.model), ("system", args.system), ("device", args.device),
                      ("batch_size", args.batch_size)) if v}
        if overrides:
            from dataclasses import replace

            cfg = replace(cfg, **overrides)
        sentences = _read_input(args.input)
        if args.limit:
            sentences = sentences[: args.limit]
        print(f"  {len(sentences):,} sentences  model={cfg.model}  pool={cfg.pool_size}  "
              f"sampling={cfg.sampling}", file=sys.stderr)
        pools, stats = generate_pool(sentences, cfg)
        write_pool(args.output, sentences, pools, cfg, stats)
        print(f"\n=== GENERATE ===", file=sys.stderr)
        for k, v in stats.items():
            print(f"  {k:24} {v}", file=sys.stderr)
        print(f"  wrote {args.output}", file=sys.stderr)
        return 0

    srcs, pools, systems = read_pools(args.candidates)
    if args.pool_size:
        # Truncation is per FILE, applied at read time above; here it is a plain
        # prefix on the merged pool, which is only meaningful for one file.
        if len(args.candidates) > 1:
            raise SystemExit("FATAL: --pool-size with multiple pool files is ambiguous; "
                             "regenerate or truncate each file separately")
        pools = [p[: args.pool_size] for p in pools]
        systems = [s[: args.pool_size] for s in systems]
    chosen, stats = select_all(pools, cfg, systems)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for c in chosen:
            fh.write(c + "\n")

    print("\n=== MBR SELECT ===")
    for k, v in stats.items():
        print(f"  {k:26} {v}")
    print(f"  wrote {args.output}")
    if stats["empty_pools"]:
        print(f"  WARNING: {stats['empty_pools']} sentence(s) have no non-empty candidate. "
              "The official scorer REJECTS a submission containing an empty row.")
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2,
                                         ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
