#!/usr/bin/env python3
"""
KATHE 2026 — R10: feature-weighted reranking over an existing candidate pool.

Why this exists
---------------
R5 settled that MBR loses: chrF++ consensus scores 32.67 on R0 against beam's
33.36, at every pool size from 2 to 32 (experiments/r5-mbr/results.md). That is
not re-litigated here. What R5 also measured is the reason this module exists:

    pool ORACLE   44.95 geo (+lexicon)      <- best candidate in the pool
    beam / sub007 33.36
    MBR-32        32.67
    pool MEAN     ~ pool mean chrF++ 39.60 against the reference

A sample beats beam in 853 of 1,003 sentences. The candidates are already good;
**the selector is the problem.** MBR's utility is pure typicality — it finds the
mode of the sampling distribution, and for this model the mode is what beam
already returns (they agree outright on 585 of 1,003). It is blind to everything
the competition actually scores.

This module scores each candidate on a weighted combination of features, of
which chrF++ consensus is only ONE term:

  consensus  mean chrF++ against the rest of the pool — the R5 utility, kept
             because typicality is real information, just not sufficient.
  density    distance from the R0 references' diacritic density (9.63/100c).
             Diacritics are preserved by the official scorer and are the single
             largest lever measured in this project (PLANNING.md 2026-08-11);
             consensus ignores them entirely.
  lexcov     fraction of a candidate's tokens present in the restoration
             lexicon. Higher coverage means restoration works better on that
             candidate. **PROVENANCE-FLAGGED — see below.**
  length     output chars / source chars, distance from the R0 references' ratio
             (1.037). Under-generation is a measured, if small, defect.
  urdu       fraction of tokens that are Urdu function words. 2.7% of sub 007
             rows carry one against 1.2% of references, and those rows are badly
             wrong (PLANNING.md 2026-08-12). Provenance-neutral.

Every feature is oriented HIGHER = BETTER, so a positive learned weight always
means "this term helps"; the distance features are returned already negated.

The provenance flag on `lexcov`, and on `density`
-------------------------------------------------
Submission 005 scored 33.50 on R0 and REGRESSED to 11.08 on the leaderboard,
because a large BPCC-derived context table memorised BPCC word sequences and R0
is cut from BPCC. The revised trust rule (PLANNING.md 2026-08-12) is: distrust
R0 when a change adds finer-grained context drawn from R0's OWN provenance.

`lexcov` reads the restoration lexicon directly, and that lexicon is 72% BPCC by
line count. `density` reads the lexicon indirectly when `density_on: restored`,
because restoration is what puts kasra/damma/fatha into the text at all. Both
weights are therefore reported separately, and `tune` reports the held-out score
with `lexcov` clamped to zero so the provenance-free subset can be read off
directly. `density_on: raw` removes the indirect dependency at the cost of
measuring a density the final output will not have.

Order: rerank first, restoration second — unchanged from R5
------------------------------------------------------------
Restoration is a fixed transform applied identically to every candidate, so
running it before selection would compress the differences the features
measure. The RAW candidate at the winning index is what gets written out, so
`make_submission.py` sees byte-identical input to sub 007's. Features that need
the restored form compute it internally without changing what is emitted.

Usage
-----
    # tune weights on a held-out split of R0 (CPU, no GPU, ~0.02 s/sentence)
    python -m decode.rerank tune --config config/rerank_r10.yaml \\
        --candidates data/dev/r0/pools/pool.r0.200m.jsonl \\
        --refs data/dev/r0/r0.kas_Arab \\
        --report experiments/r10-rerank/tuning.json

    # select with fixed weights from the config
    python -m decode.rerank select --config config/rerank_r10.yaml \\
        --candidates data/dev/r0/pools/pool.r0.200m.jsonl \\
        --output data/dev/r0/r0.hyp.rerank

    # live round: one entry point, same config
    python scripts/translate.py --input data/raw/englishdev.csv --output sub.csv \\
        --rerank config/rerank_r10.yaml

Selection needs only sacrebleu + KashmiriNormalizer, so it runs in the main
`.venv`. Only `generate` (inherited from decode.mbr) needs `.venv-decode`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decode.mbr import (  # noqa: E402
    MBRConfig,
    ChrfUtility,
    generate_pool,
    mbr_select,
    read_pools,
    write_pool,
)

FEATURES = ("consensus", "density", "lexcov", "length", "urdu")


# ---------------------------------------------------------------------------
# Urdu function-word inventory
# ---------------------------------------------------------------------------
#
# RECONSTRUCTED 2026-08-12. The 2.7%-of-sub-007 measurement in PLANNING.md was
# an inline analysis that left no code behind, so this list was rebuilt and then
# validated against the two numbers that analysis recorded:
#
#     sub 007 test output   50 / 1,730 rows (2.89%)   recorded: 46 (2.66%)
#     R0 human references   13 / 1,003 rows (1.30%)   recorded: ~1.2%
#
# Close, not identical — treat the inventory as a reconstruction, not a replay.
#
# Built by frequency, not intuition: a hand-written list of Urdu function words
# was counted against 170,882 lines of real Kashmiri (BPCC targets + the
# nawabhussain external corpus) and every token occurring more than 200 times
# per million was DROPPED as genuinely Kashmiri or too ambiguous to call — that
# removed یہ, کہ, تم, کیا, اور, کی, کے, ہیں, پر, ہو, گی and سے-adjacent forms.
# `ہے` is the one deliberate re-inclusion: it is the Urdu copula, Kashmiri uses
# چھُ, and it is by far the most frequent single marker in our own output (22 of
# the 50 hits). Its 318/M rate in BPCC is Urdu contamination in the corpus, not
# evidence that it is Kashmiri.
URDU_MARKERS = frozenset(
    """ہے سے میں ہوا کا ایک تک دو سب کو میری جو کرنا وہ نے بہت اس لیے ساتھ گے
    گئی جاتا رہا گیا گا میرا بھی کچھ جاتی تھا آپ میرے نہیں گئے تھی لئے تھے
    کرتے اپنی ہوں اپنے جب کرتا ہوتا ہوئی سکتی کرتی لیکن اپنا سکتے چاہیے رہے
    ہوتے جاتے ہمارے ہوتی اسے سکتا ہمارا رہی ہوئے اسکا""".split()
)

_STRIP = "۔،؟!.,?؛:\"'()«»…؍٫٪"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankConfig(MBRConfig):
    """Generation and utility knobs are inherited from MBRConfig so one pool
    file, one generator and one chrF++ implementation serve both R5 and R10.
    Everything below is new.

    Weights multiply features that are all oriented higher-is-better and are all
    O(1) in magnitude, so the numbers are directly comparable to each other. The
    argmax is invariant to positive rescaling of the whole vector, so only the
    RATIOS between weights carry meaning — `w_consensus` is not pinned to 1.0
    because `tune` needs to be able to drive it to zero in the ablation.
    """

    w_consensus: float = 1.0
    w_density: float = 0.0
    w_lexcov: float = 0.0
    w_length: float = 0.0
    w_urdu: float = 0.0

    # Targets are R0 REFERENCE statistics, measured not assumed; `tune --report`
    # re-measures them from the reference file it is given and refuses to run if
    # they disagree by more than 5%.
    target_density: float = 9.63  # KASHMIRI_DIACRITICS per 100 chars
    target_length_ratio: float = 1.037  # target chars / source chars

    # restored | raw. `restored` measures the density the SUBMITTED text will
    # have, which is the quantity 9.63 is a target for; it reads the lexicon
    # indirectly. `raw` is provenance-free but measures a density no submitted
    # row ever has, since restoration is what supplies kasra/damma/fatha.
    density_on: str = "restored"

    # The production lexicon (submission 007). Needed for `lexcov`, for
    # `density_on: restored`, and for scoring the way the pipeline scores.
    diacritic_lexicon: str = "data/processed/diacritic_lexicon_both.json"


def load_config(path: Path) -> RerankConfig:
    """Build a RerankConfig from YAML, rejecting unknown keys loudly — same
    discipline as `make_submission.load_postproc` and `decode.mbr.load_config`.
    A typo'd weight that silently defaulted to 0.0 would change the selection
    without changing the recorded config."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(RerankConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"FATAL: unknown rerank config keys in {path}: {sorted(unknown)}")
    cfg = RerankConfig(**raw)
    if cfg.pool_size < 2:
        raise SystemExit("FATAL: pool_size must be >= 2; there is nothing to rerank otherwise")
    if cfg.density_on not in {"restored", "raw"}:
        raise SystemExit(f"FATAL: density_on must be 'restored' or 'raw', got {cfg.density_on!r}")
    return cfg


def weights_of(cfg: RerankConfig) -> dict[str, float]:
    return {f: getattr(cfg, f"w_{f}") for f in FEATURES}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def _diacritic_density(text: str) -> float:
    """KASHMIRI_DIACRITICS per 100 characters — the same definition
    `scripts/orthography_diagnostic.py` uses, so densities reported by the two
    are the same number."""
    from KashmiriNormalizer.constants import KASHMIRI_DIACRITICS

    if not text:
        return 0.0
    return 100.0 * sum(1 for c in text if c in KASHMIRI_DIACRITICS) / len(text)


def _urdu_fraction(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(_STRIP) in URDU_MARKERS)
    return hits / len(words)


def _lexicon_coverage(text: str, lut: dict, ctx: dict) -> float:
    """Fraction of tokens the restoration lexicon can act on, counting a context
    hit and a unigram hit alike. Mirrors `data.diacritize.coverage`, but per
    sentence rather than per corpus."""
    from data.diacritize import BOS, EOS

    left = ctx.get("left", {})
    both = ctx.get("both", {})
    right = ctx.get("right", {})
    words = text.split()
    if not words:
        return 0.0
    hit = 0
    for i, w in enumerate(words):
        p = words[i - 1] if i else BOS
        n = words[i + 1] if i + 1 < len(words) else EOS
        if f"{p}\t{w}\t{n}" in both or f"{p}\t{w}" in left or f"{w}\t{n}" in right or w in lut:
            hit += 1
    return hit / len(words)


class Featurizer:
    """Computes every feature for every candidate, once.

    Kept as a class with an explicit cache because tuning re-scores the same
    pools several thousand times: features are computed once (~20 s for 1,003 x
    32 on the M2 Air) and every subsequent weight vector is a dot product and an
    argmax. That is the whole reason a five-parameter sweep is affordable
    offline.
    """

    def __init__(self, cfg: RerankConfig):
        from data.diacritize import restore
        from data.normalize import NormConfig, normalize

        self.cfg = cfg
        self._restore = restore
        self._normalize = normalize
        self._scorer_cfg = NormConfig(scorer_normalizer=True)

        blob = json.loads(Path(cfg.diacritic_lexicon).read_text(encoding="utf-8"))
        self.lut: dict = blob["lexicon"]
        self.ctx: dict = blob.get("context") or {}
        self.util = ChrfUtility(cfg.char_order, cfg.word_order, cfg.beta)

    def run(self, srcs: list[str], pools: list[list[str]], progress: bool = True) -> dict:
        """Returns a dict of parallel per-sentence lists:

            raw[i][j]     the candidate as generated — what gets written out
            scored[i][j]  that candidate after PRODUCTION post-processing, i.e.
                          what the official scorer would see if it won
            feats[i][j]   {feature: value}, all higher-is-better
        """
        cfg = self.cfg
        iterator = range(len(pools))
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, unit="sent", desc="featurize", file=sys.stderr,
                            dynamic_ncols=True)

        raw_all: list[list[str]] = []
        scored_all: list[list[str]] = []
        feats_all: list[list[dict[str, float]]] = []
        dropped_empty = 0
        empty_pools = 0

        t0 = time.time()
        for i in iterator:
            # An empty candidate is fatal to a submission (the scorer raises on
            # it), and epsilon sampling does occasionally produce one. Drop them
            # before selection rather than let one win on degenerate features.
            raw = [c for c in pools[i] if c.strip()]
            dropped_empty += len(pools[i]) - len(raw)
            if not raw:
                empty_pools += 1
                raw_all.append([""])
                scored_all.append([""])
                feats_all.append([{f: 0.0 for f in FEATURES}])
                continue

            # Scorer-normalized form: what the metric sees, and what the utility
            # is computed on (utility_normalize: scorer).
            norm = [self._normalize(c, self._scorer_cfg) for c in raw]
            norm = [n if n.strip() else raw[j] for j, n in enumerate(norm)]
            # Production post-processing = scorer normalizer + restoration.
            restored = [self._restore(n, self.lut, self.ctx) for n in norm]

            keys = norm if cfg.utility_normalize == "scorer" else raw
            _, consensus = mbr_select(raw, self.util, keys=keys, include_self=cfg.include_self)

            src_chars = max(1, len(srcs[i]))
            density_text = restored if cfg.density_on == "restored" else norm

            row = []
            for j in range(len(raw)):
                d = _diacritic_density(density_text[j])
                ratio = len(restored[j]) / src_chars
                row.append({
                    # /100 puts consensus on the same O(1) scale as the others,
                    # so the learned weights are directly comparable.
                    "consensus": consensus[j] / 100.0,
                    "density": -abs(d - cfg.target_density) / cfg.target_density,
                    "lexcov": _lexicon_coverage(norm[j], self.lut, self.ctx),
                    "length": -abs(ratio - cfg.target_length_ratio) / cfg.target_length_ratio,
                    "urdu": -_urdu_fraction(norm[j]),
                })

            raw_all.append(raw)
            scored_all.append(restored)
            feats_all.append(row)

        return {
            "raw": raw_all,
            "scored": scored_all,
            "feats": feats_all,
            "stats": {
                "sentences": len(pools),
                "wall_seconds": round(time.time() - t0, 2),
                "dropped_empty_candidates": dropped_empty,
                "empty_pools": empty_pools,
            },
        }


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def argmax_indices(feats: list[list[dict[str, float]]], w: dict[str, float]) -> list[int]:
    """Winning candidate index per sentence under weight vector `w`.

    Ties resolve to the LOWEST index, which is candidate 0, which is the beam
    hypothesis (`include_beam: true`). That makes beam the tie-break floor: a
    weight vector that cannot separate candidates degenerates to sub 007 rather
    than to an arbitrary sample.
    """
    out = []
    for row in feats:
        best_i, best_s = 0, None
        for j, f in enumerate(row):
            s = 0.0
            for k, wk in w.items():
                if wk:
                    s += wk * f[k]
            if best_s is None or s > best_s:
                best_i, best_s = j, s
        out.append(best_i)
    return out


def select_all(pools, cfg: RerankConfig, systems=None, progress: bool = True, srcs=None):
    """Rerank every pool. Returns (selected RAW candidates, stats).

    Signature mirrors `decode.mbr.select_all` so `scripts/translate.py` wires
    both the same way and the live round has one code path. `srcs` is the extra
    argument the reranker needs and MBR does not: the length feature is a ratio
    against the source, so it is not computable from the pool alone.
    """
    if srcs is None:
        if cfg.w_length:
            raise SystemExit(
                "FATAL: w_length is nonzero but no sources were passed to select_all. "
                "The length feature is output chars / SOURCE chars; without sources it "
                "would silently score every candidate against a denominator of 1.")
        srcs = [""] * len(pools)
    return select_with_srcs(srcs, pools, cfg, systems, progress)


def select_with_srcs(srcs, pools, cfg: RerankConfig, systems=None, progress: bool = True):
    fz = Featurizer(cfg)
    data = fz.run(srcs, pools, progress=progress)
    w = weights_of(cfg)
    idx = argmax_indices(data["feats"], w)

    chosen = [data["raw"][i][j] for i, j in enumerate(idx)]
    beam_wins = sum(1 for j in idx if j == 0) if cfg.include_beam else 0
    winners_by_system: dict[str, int] = {}
    if systems:
        for i, j in enumerate(idx):
            # `systems` indexes the ORIGINAL pool; empties were dropped, so map
            # back by matching the chosen string rather than by position.
            try:
                orig = pools[i].index(chosen[i])
            except ValueError:  # pragma: no cover - chosen always came from pools[i]
                orig = 0
            s = systems[i][orig]
            winners_by_system[s] = winners_by_system.get(s, 0) + 1

    stats = {
        **data["stats"],
        "weights": w,
        "beam_candidate_won": beam_wins,
        "mean_winning_rank": round(sum(idx) / max(1, len(idx)), 2),
        "winners_by_system": winners_by_system,
    }
    return chosen, stats


# ---------------------------------------------------------------------------
# scoring helpers (tuning only)
# ---------------------------------------------------------------------------


def _corpus_geo(hyps: list[str], refs_norm: list[str]) -> dict[str, float]:
    """BLEU 13a + chrF++ + geometric mean over pre-normalized text.

    `data.normalize.score` is the authority and normalizes internally; this is
    the same computation with the normalization hoisted out of the inner loop,
    because tuning calls it thousands of times. `tune` asserts the two agree on
    the beam baseline before trusting a single number from here.
    """
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs_norm]).score
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs_norm], word_order=2).score
    geo = 0.0 if (bleu <= 0 or chrfpp <= 0) else (bleu * chrfpp) ** 0.5
    return {"bleu": bleu, "chrf_plus_plus": chrfpp, "geometric_mean": geo}


def _oracle_indices(scored: list[list[str]], refs_norm: list[str], util: ChrfUtility) -> list[int]:
    """Per-sentence best candidate by chrF++ against the TRUE reference.

    This is a ceiling, not a selectable system — it reads the reference. It is
    reported so every configuration can be quoted as a fraction of the +11.6 geo
    the R5 diagnostic found.
    """
    out = []
    for cands, ref in zip(scored, refs_norm):
        rg = util.ngrams(ref)
        out.append(max(range(len(cands)), key=lambda j: util.score(util.ngrams(cands[j]), rg)))
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, average ranks for ties. Written out rather than pulled
    from scipy because scipy is not in the decode environment and this is
    fifteen lines."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def diagnose_features(feats, scored, refs_norm, util: ChrfUtility) -> dict:
    """Per-feature signal, measured WITHOUT any weight search.

    The question a five-parameter sweep cannot answer cleanly is whether a
    feature carries quality information at all. So for every sentence, compute
    each candidate's TRUE chrF++ against the reference and ask, within that
    pool, how well the feature ranks the candidates:

      spearman   mean within-pool rank correlation between the feature and true
                 chrF++. This is the number that matters — a feature that does
                 not correlate with quality inside a pool cannot help select
                 from it, whatever weight it is given.
      pick_rank  mean true-quality rank of the candidate this feature alone
                 would pick (1 = the oracle's choice). R5's comparable number
                 for chrF++ consensus was 9.3 of 32.
      beats_beam fraction of sentences where the feature's solo pick has a
                 higher true chrF++ than the beam candidate.
    """
    n_feat = {f: {"rho": [], "rank": [], "beats": 0, "n": 0} for f in FEATURES}
    for i, (cands, ref) in enumerate(zip(scored, refs_norm)):
        if len(cands) < 2:
            continue
        rg = util.ngrams(ref)
        true = [util.score(util.ngrams(c), rg) for c in cands]
        order = sorted(range(len(cands)), key=lambda j: -true[j])
        rank_of = {j: r + 1 for r, j in enumerate(order)}
        for f in FEATURES:
            vals = [feats[i][j][f] for j in range(len(cands))]
            acc = n_feat[f]
            acc["rho"].append(_spearman(vals, true))
            pick = max(range(len(cands)), key=lambda j: (vals[j], -j))
            acc["rank"].append(rank_of[pick])
            acc["beats"] += 1 if true[pick] > true[0] else 0
            acc["n"] += 1
    out = {}
    for f, acc in n_feat.items():
        n = max(1, acc["n"])
        out[f] = {
            "mean_spearman_vs_true_chrf": round(sum(acc["rho"]) / n, 4),
            "mean_pick_rank": round(sum(acc["rank"]) / n, 2),
            "beats_beam_pct": round(100 * acc["beats"] / n, 1),
        }
    return out


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------


def tune_weights(
    feats,
    scored,
    refs_norm,
    idx_tune: list[int],
    fixed_zero: tuple[str, ...] = (),
    n_random: int = 300,
    passes: int = 4,
    seed: int = 0,
    span: float = 4.0,
    grid: int = 17,
    n_starts: int = 5,
) -> tuple[dict[str, float], float, int]:
    """Random search then coordinate descent, maximizing corpus geo on the TUNE
    split only. Returns (weights, tune geo, evaluations).

    Five free parameters against 700 sentences will fit noise if it is allowed
    to — which is exactly why the held-out split exists and why this function
    never sees it. Coordinate descent rather than anything cleverer because the
    objective is a step function of the weights (it changes only when an argmax
    flips), so gradients do not exist and a local grid scan is both honest and
    cheap.
    """
    rng = random.Random(seed)
    free = [f for f in FEATURES if f not in fixed_zero]
    sub_feats = [feats[i] for i in idx_tune]
    sub_scored = [scored[i] for i in idx_tune]
    sub_refs = [refs_norm[i] for i in idx_tune]
    evals = 0

    def objective(w: dict[str, float]) -> float:
        nonlocal evals
        evals += 1
        idx = argmax_indices(sub_feats, w)
        hyps = [sub_scored[i][j] for i, j in enumerate(idx)]
        return _corpus_geo(hyps, sub_refs)["geometric_mean"]

    def zeros() -> dict[str, float]:
        return {f: 0.0 for f in FEATURES}

    # MULTI-START, and the reason is a measured failure: seeded only at
    # consensus-only (MBR) with a thin random sample, coordinate descent sat in
    # the MBR basin and returned it unchanged, while the same search with
    # consensus clamped to zero found a strictly better point. One start plus
    # one-at-a-time descent is not enough on a step-function objective.
    starts = [
        zeros(),  # all-zero: every candidate ties, argmax falls to beam
        {**zeros(), **{f: 1.0 for f in free if f == "consensus"}},  # MBR
    ]
    for f in free:  # each feature alone, both signs
        starts.append({**zeros(), f: 1.0})
    scored_starts = [(objective(w), w) for w in starts]

    for _ in range(n_random):
        w = zeros()
        for f in free:
            w[f] = round(rng.uniform(-span, span), 3)
        scored_starts.append((objective(w), w))

    scored_starts.sort(key=lambda t: -t[0])
    step = 2 * span / (grid - 1)
    best, best_w = scored_starts[0]

    for start_score, start_w in scored_starts[:n_starts]:
        cur, cur_w = start_score, dict(start_w)
        for _ in range(passes):
            improved = False
            for f in free:
                base = dict(cur_w)
                for k in range(grid):
                    v = round(-span + k * step, 4)
                    if v == base[f]:
                        continue
                    w = dict(base)
                    w[f] = v
                    s = objective(w)
                    if s > cur + 1e-9:
                        cur, cur_w, improved = s, w, True
            # Halving refinement: the coarse grid is 0.5 wide, enough to flip
            # argmaxes but not to sit at the optimum.
            for f in free:
                base = dict(cur_w)
                for delta in (-step / 2, step / 2, -step / 4, step / 4):
                    w = dict(base)
                    w[f] = round(base[f] + delta, 4)
                    s = objective(w)
                    if s > cur + 1e-9:
                        cur, cur_w, improved = s, w, True
            if not improved:
                break
        if cur > best:
            best, best_w = cur, cur_w
    return best_w, best, evals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_lines(p: Path) -> list[str]:
    with open(p, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def _cmd_generate(args, cfg: RerankConfig) -> int:
    from decode.mbr import _read_input

    overrides = {k: v for k, v in (("model", args.model), ("device", args.device),
                                   ("batch_size", args.batch_size)) if v}
    if overrides:
        cfg = replace(cfg, **overrides)
    sentences = _read_input(args.input)
    if args.limit:
        sentences = sentences[: args.limit]
    pools, stats = generate_pool(sentences, cfg)
    write_pool(args.output, sentences, pools, cfg, stats)
    print(f"  wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_select(args, cfg: RerankConfig) -> int:
    srcs, pools, systems = read_pools(args.candidates)
    if args.pool_size:
        if len(args.candidates) > 1:
            raise SystemExit("FATAL: --pool-size with multiple pool files is ambiguous")
        pools = [p[: args.pool_size] for p in pools]
        systems = [s[: args.pool_size] for s in systems]
    chosen, stats = select_with_srcs(srcs, pools, cfg, systems)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for c in chosen:
            fh.write(c + "\n")

    print("\n=== RERANK SELECT ===")
    for k, v in stats.items():
        print(f"  {k:26} {v}")
    print(f"  wrote {args.output}")
    if stats["empty_pools"]:
        print(f"  WARNING: {stats['empty_pools']} sentence(s) have no non-empty candidate. "
              "The official scorer REJECTS a submission containing an empty row.")
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(
            json.dumps({"config": asdict(cfg), "stats": stats}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return 0


def _cmd_tune(args, cfg: RerankConfig) -> int:
    from data.normalize import NormConfig, normalize, score as official_score

    srcs, pools, _ = read_pools(args.candidates)
    refs = _read_lines(args.refs)
    if len(refs) != len(pools):
        raise SystemExit(f"FATAL: {len(refs)} references vs {len(pools)} pools — "
                         "scoring is positional, these files are not aligned.")

    scorer_cfg = NormConfig(scorer_normalizer=True)
    refs_norm = [normalize(r, scorer_cfg) for r in refs]

    # The two targets are REFERENCE statistics. Re-measure them here rather than
    # trust the config, because a stale target silently turns two of the five
    # features into noise.
    ref_density = _diacritic_density("".join(refs_norm)) if refs_norm else 0.0
    # MEAN OF PER-SENTENCE RATIOS, not the ratio of corpus totals. The feature is
    # a per-sentence distance, so the target has to be the per-sentence average.
    # The two differ materially here — 1.037 against 1.019 — and 1.037 is the
    # number PLANNING.md 2026-08-12 records for the R0 references.
    ref_ratio = sum(len(r) / max(1, len(s)) for r, s in zip(refs_norm, srcs)) / max(1, len(srcs))
    for name, measured, configured in (("target_density", ref_density, cfg.target_density),
                                       ("target_length_ratio", ref_ratio, cfg.target_length_ratio)):
        if abs(measured - configured) / max(1e-9, configured) > 0.05:
            raise SystemExit(
                f"FATAL: {name} in the config is {configured:.3f} but this reference file "
                f"measures {measured:.3f} (>5% apart). Fix the config; a stale target makes "
                "the feature noise.")
    print(f"  reference density {ref_density:.2f}/100c (config {cfg.target_density})   "
          f"length ratio {ref_ratio:.3f} (config {cfg.target_length_ratio})", file=sys.stderr)

    fz = Featurizer(cfg)
    data = fz.run(srcs, pools)
    feats, scored, raw = data["feats"], data["scored"], data["raw"]

    # Fast path vs the authority: `_corpus_geo` hoists normalization out of the
    # loop, so prove it reproduces `data.normalize.score` on the beam baseline
    # before a single tuned number is believed. Beam is candidate 0.
    beam_raw = [r[0] for r in raw]
    beam_scored = [s[0] for s in scored]
    fast = _corpus_geo(beam_scored, refs_norm)
    lex_cfg = NormConfig(scorer_normalizer=True, diacritic_lexicon=cfg.diacritic_lexicon)
    slow = official_score([normalize(h, lex_cfg) for h in beam_raw], refs)
    if abs(fast["geometric_mean"] - slow["geometric_mean"]) > 0.01:
        raise SystemExit(
            f"FATAL: fast scorer {fast['geometric_mean']:.4f} != data.normalize.score "
            f"{slow['geometric_mean']:.4f} on the beam baseline. Do not trust any tuned number.")
    print(f"  scorer agreement OK: beam+lexicon geo {fast['geometric_mean']:.2f} "
          f"(authority {slow['geometric_mean']:.2f})", file=sys.stderr)

    # --- split ------------------------------------------------------------
    n = len(pools)
    order = list(range(n))
    random.Random(args.split_seed).shuffle(order)
    n_tune = args.tune_size if args.tune_size else int(round(0.7 * n))
    idx_tune, idx_held = sorted(order[:n_tune]), sorted(order[n_tune:])
    print(f"  split: tune {len(idx_tune)}  held-out {len(idx_held)}  "
          f"(seed {args.split_seed})", file=sys.stderr)

    def geo_on(idx_sel: list[int], split: list[int], use_lexicon: bool = True) -> dict:
        hyps = []
        for i in split:
            j = idx_sel[i]
            hyps.append(scored[i][j] if use_lexicon else
                        normalize(raw[i][j], scorer_cfg) or raw[i][j])
        return _corpus_geo(hyps, [refs_norm[i] for i in split])

    def report(idx_sel: list[int]) -> dict:
        return {
            "tune": {k: round(v, 2) for k, v in geo_on(idx_sel, idx_tune).items()},
            "held_out": {k: round(v, 2) for k, v in geo_on(idx_sel, idx_held).items()},
            "full": {k: round(v, 2) for k, v in geo_on(idx_sel, list(range(n))).items()},
            "tune_raw": round(geo_on(idx_sel, idx_tune, False)["geometric_mean"], 2),
            "held_out_raw": round(geo_on(idx_sel, idx_held, False)["geometric_mean"], 2),
            "full_raw": round(geo_on(idx_sel, list(range(n)), False)["geometric_mean"], 2),
            "density_per_100c": round(
                _diacritic_density("".join(scored[i][idx_sel[i]] for i in range(n))), 2),
            "beam_agreement": sum(1 for j in idx_sel if j == 0),
            "mean_rank": round(sum(idx_sel) / max(1, n), 2),
        }

    results: dict = {"reference": {"density_per_100c": round(ref_density, 2),
                                   "length_ratio": round(ref_ratio, 4)}}

    baseline_idx = [0] * n
    results["baseline_beam_sub007"] = report(baseline_idx)

    mbr_w = {f: (1.0 if f == "consensus" else 0.0) for f in FEATURES}
    results["mbr_consensus_only"] = {"weights": mbr_w, **report(argmax_indices(feats, mbr_w))}

    oracle_idx = _oracle_indices(scored, refs_norm, fz.util)
    results["oracle"] = report(oracle_idx)

    # Solo-feature selection and per-feature rank correlation. Run BEFORE the
    # weight search, because if no feature ranks candidates by quality inside a
    # pool then no weighting of them can either, and that is the finding.
    results["feature_diagnostics"] = diagnose_features(feats, scored, refs_norm, fz.util)
    results["solo"] = {}
    for f in FEATURES:
        sw = {k: (1.0 if k == f else 0.0) for k in FEATURES}
        results["solo"][f] = {"weights": sw, **report(argmax_indices(feats, sw))}

    # --- full model -------------------------------------------------------
    t0 = time.time()
    w, tune_geo, ev = tune_weights(feats, scored, refs_norm, idx_tune,
                                   n_random=args.n_random, seed=args.seed)
    full_idx = argmax_indices(feats, w)
    results["reranked"] = {"weights": {k: round(v, 4) for k, v in w.items()},
                           "evaluations": ev, "tune_seconds": round(time.time() - t0, 1),
                           **report(full_idx)}
    print(f"  full model tuned in {ev} evaluations, {time.time() - t0:.0f}s  "
          f"tune geo {tune_geo:.2f}", file=sys.stderr)

    # --- ablations: drop one feature and RE-TUNE the rest -----------------
    # Clamping a weight to zero without re-tuning would measure "how much does
    # this weight matter at these other weights", which is not the same question
    # as "is this term doing work".
    results["ablations"] = {}
    for f in FEATURES:
        aw, _, aev = tune_weights(feats, scored, refs_norm, idx_tune, fixed_zero=(f,),
                                  n_random=args.n_random, seed=args.seed)
        rep = report(argmax_indices(feats, aw))
        results["ablations"][f"without_{f}"] = {
            "weights": {k: round(v, 4) for k, v in aw.items()}, "evaluations": aev, **rep}
        print(f"  ablation -{f:10} held-out {rep['held_out']['geometric_mean']:.2f}",
              file=sys.stderr)

    results["config"] = asdict(cfg)
    results["split"] = {"seed": args.split_seed, "tune": len(idx_tune), "held_out": len(idx_held)}
    results["featurize_stats"] = data["stats"]

    _print_report(results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  wrote {args.report}")
    if args.write_weights:
        tuned = replace(cfg, **{f"w_{k}": round(v, 4) for k, v in w.items()})
        _write_weight_config(args.write_weights, tuned, results)
        print(f"  wrote {args.write_weights}")
    return 0


def _print_report(r: dict) -> None:
    def row(label, d):
        return (f"  {label:26} {d['tune']['geometric_mean']:>7.2f} "
                f"{d['held_out']['geometric_mean']:>9.2f} {d['full']['geometric_mean']:>7.2f} "
                f"{d['full_raw']:>7.2f} {d['density_per_100c']:>8.2f} {d['beam_agreement']:>6}")

    print("\n=== R10 RERANK — R0, tokenizer 13a, geo with the production lexicon ===")
    print(f"  {'system':26} {'tune':>7} {'held-out':>9} {'full':>7} {'raw':>7} "
          f"{'dens':>8} {'=beam':>6}")
    print("  " + "-" * 72)
    print(row("beam / sub 007", r["baseline_beam_sub007"]))
    print(row("MBR (consensus only)", r["mbr_consensus_only"]))
    print(row("RERANKED", r["reranked"]))
    print(row("ORACLE (ceiling)", r["oracle"]))
    print(f"\n  reference density {r['reference']['density_per_100c']}/100c")

    print("\n  learned weights:")
    for k, v in r["reranked"]["weights"].items():
        flag = "   <- BPCC-provenance" if k == "lexcov" else ""
        print(f"    w_{k:12} {v:+8.3f}{flag}")

    base = r["reranked"]["held_out"]["geometric_mean"]
    print(f"\n  ablation (drop one, re-tune the rest) — held-out geo, full model {base:.2f}:")
    for name, d in r["ablations"].items():
        h = d["held_out"]["geometric_mean"]
        print(f"    {name:22} {h:7.2f}   {h - base:+6.2f}")

    print("\n  per-feature signal, no weight search — can this feature rank a pool?")
    print(f"    {'feature':12} {'rho vs true chrF++':>19} {'solo pick rank':>15} "
          f"{'beats beam':>11} {'solo geo (full)':>16}")
    for f in FEATURES:
        d = r["feature_diagnostics"][f]
        print(f"    {f:12} {d['mean_spearman_vs_true_chrf']:>19.3f} "
              f"{d['mean_pick_rank']:>15.2f} {d['beats_beam_pct']:>10.1f}% "
              f"{r['solo'][f]['full']['geometric_mean']:>16.2f}")
    print(f"    {'(oracle)':12} {1.0:>19.3f} {1.0:>15.2f} "
          f"{'—':>11} {r['oracle']['full']['geometric_mean']:>16.2f}")


def _write_weight_config(path: Path, cfg: RerankConfig, results: dict) -> None:
    """Emit a config carrying the tuned weights, with the numbers that produced
    them in the header — a weight vector with no provenance is unreproducible."""
    r = results["reranked"]
    lines = [
        "# R10 — tuned reranking weights. GENERATED by `python -m decode.rerank tune`.",
        f"# Tuned on {results['split']['tune']} R0 sentences (split seed "
        f"{results['split']['seed']}), evaluated on {results['split']['held_out']} held out.",
        "#",
        f"#   beam / sub 007   held-out geo "
        f"{results['baseline_beam_sub007']['held_out']['geometric_mean']:.2f}",
        f"#   MBR consensus    held-out geo "
        f"{results['mbr_consensus_only']['held_out']['geometric_mean']:.2f}",
        f"#   RERANKED         held-out geo {r['held_out']['geometric_mean']:.2f}",
        f"#   oracle ceiling   held-out geo "
        f"{results['oracle']['held_out']['geometric_mean']:.2f}",
        "#",
        "# `w_lexcov` reads the BPCC-derived restoration lexicon — the same provenance",
        "# that made submission 005 score high on R0 and regress on the leaderboard.",
        "",
    ]
    d = asdict(cfg)
    for k in sorted(d):
        v = d[k]
        lines.append(f"{k}: {'null' if v is None else v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="draw candidate pools (GPU) — same as decode.mbr")
    g.add_argument("--config", required=True, type=Path)
    g.add_argument("--input", required=True, type=Path)
    g.add_argument("--output", required=True, type=Path)
    g.add_argument("--model")
    g.add_argument("--device")
    g.add_argument("--batch-size", type=int)
    g.add_argument("--limit", type=int)

    s = sub.add_parser("select", help="rerank with the config's weights (CPU)")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--candidates", required=True, type=Path, nargs="+")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--pool-size", type=int)
    s.add_argument("--stats", type=Path)

    t = sub.add_parser("tune", help="fit weights on a held-out split (CPU)")
    t.add_argument("--config", required=True, type=Path)
    t.add_argument("--candidates", required=True, type=Path, nargs="+")
    t.add_argument("--refs", required=True, type=Path)
    t.add_argument("--tune-size", type=int, help="sentences in the tuning split (default 70%%)")
    t.add_argument("--split-seed", type=int, default=0)
    t.add_argument("--seed", type=int, default=0, help="search seed")
    t.add_argument("--n-random", type=int, default=300)
    t.add_argument("--report", type=Path)
    t.add_argument("--write-weights", type=Path, help="emit a config with the tuned weights")

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "generate":
        return _cmd_generate(args, cfg)
    if args.cmd == "select":
        return _cmd_select(args, cfg)
    return _cmd_tune(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
