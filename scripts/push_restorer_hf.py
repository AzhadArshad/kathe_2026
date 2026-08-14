#!/usr/bin/env python3
"""
KATHE 2026 — publish R11 diacritic-restorer checkpoints to the HF Hub.

This uploads ONLY the character-level restorer (R11). It must never touch the
translation weights, and it must never put a checkpoint under a licence its
training data does not permit. Both are enforced here rather than trusted.

THE LICENCE SPLIT IS THE WHOLE POINT
------------------------------------
The restorer's training text comes from three corpora with different terms:

  ai4bharat/BPCC                        CC-BY-4.0 / CC0   attribution
  nawabhussain/Kashmiri-Language-Corpus Apache-2.0        attribution
  SMUQamar/...-Parallel-Corpus (30K)    CC-BY-NC-SA-4.0   ShareAlike + NC

ShareAlike is viral: a checkpoint trained on the 30K is a derivative work and
**must** ship CC-BY-NC-SA-4.0. A checkpoint trained without it need not be.

The R11 arms differ in exactly this respect — `all` includes qamar30k, `clean`
and `dense` do not — so publishing them into one repo under one licence would
misstate at least one of them. This script therefore reads each checkpoint's
OWN recorded `sources` and routes it to the matching repo. The licence follows
the data the weights actually saw, not the arm's name, so renaming or
redefining an arm cannot silently relicense anything.

SAFETY PROPERTIES, all asserted before a single byte is uploaded
----------------------------------------------------------------
1. Opt-in. Does nothing unless `--yes` is passed (or PUSH_HF=1 in the runner).
2. Write token only. Uses HF_TOKEN_WRITE and REFUSES to fall back to HF_TOKEN,
   which is the widely-copied read token for gated downloads (PLANNING.md
   2026-08-10 keeps them apart precisely so a read-token leak cannot overwrite
   published weights).
3. A denylist of the translation repos. `kathe-r3-200m-full` and
   `kathe-r4-1b-lora` are the real deliverable; a typo in --repo must not be
   able to overwrite them, so the name is checked against them explicitly.
4. Only `r11_*.pt` / `r11_*.meta.json` are uploaded. Never a folder, never a
   wildcard that could sweep in a translation checkpoint sitting nearby.
5. Private by default. Publishing is a deliberate second step.
6. A checkpoint whose meta does not record `sources` is REFUSED — unknown
   provenance means unknown licence.
7. Failure is non-fatal to the caller. The weights already exist on disk; a
   network problem must not look like a training problem.

Usage:
    python scripts/push_restorer_hf.py --checkpoints /kaggle/working/r11 --yes
    python scripts/push_restorer_hf.py --checkpoints models/restore --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repos holding the TRANSLATION deliverable. Never a target of this script.
PROTECTED = {"kathe-r3-200m-full", "kathe-r4-1b-lora"}

# Corpora whose terms force the licence of anything trained on them.
SHAREALIKE_SOURCES = {"qamar30k"}

APACHE = "apache-2.0"
NCSA = "cc-by-nc-sa-4.0"


def licence_for(sources: dict) -> tuple[str, str]:
    """(licence id, why) from the sources the checkpoint records for itself."""
    tainted = sorted(s for s in sources if s.split(":")[0] in SHAREALIKE_SOURCES)
    if tainted:
        return NCSA, (
            f"trained on {', '.join(tainted)} (SMUQamar 30K, CC-BY-NC-SA-4.0); "
            "ShareAlike makes this checkpoint a derivative work that must carry "
            "the same licence"
        )
    return APACHE, (
        "trained only on BPCC (CC-BY-4.0 / CC0) and "
        "nawabhussain/Kashmiri-Language-Corpus (Apache-2.0), both of which "
        "permit relicensing of derivatives with attribution"
    )


def card(name: str, meta: dict, licence: str, why: str) -> str:
    prf = meta.get("dev_prf", {}).get("MICRO", {})
    srcs = "\n".join(f"| `{k}` | {v:,} |" for k, v in
                     sorted(meta.get("sources", {}).items(), key=lambda kv: -kv[1]))
    return f"""---
license: {licence}
language: [ks]
tags: [kashmiri, diacritics, diacritic-restoration, token-classification, kathe-2026]
---

# KATHE 2026 — character-level Kashmiri diacritic restorer (`{name}`)

Restores the three short-vowel marks **kasra (U+0650), damma (U+064F) and
fatha (U+064E)** to Perso-Arabic Kashmiri text.

It exists because IndicTrans2 **cannot emit them**: each appears in exactly one
token of its 122,672-entry target vocabulary, so beam search never produces
them. Restoration therefore has to be post-hoc. This is a companion to a
translation model, not a translation model.

## What it is

A 3.3M-parameter bidirectional transformer encoder used as a **per-character
tagger**: for every input character it predicts one of
`{{none, kasra, damma, fatha}}`. The base-letter sequence is not an output, so
restoration is **insertion-only by construction** — it cannot alter, reorder or
drop a character, for any label vector.

## Training data

Monolingual Kashmiri, self-supervised: strip the three marks from any sentence
and `(stripped, original)` is a training pair.

| source | examples |
| --- | ---: |
{srcs}

Held-out micro-F1 {prf.get('f1', '?')} (precision {prf.get('precision', '?')},
recall {prf.get('recall', '?')}), best at epoch {meta.get('epoch', '?')}.

The development sets of the parent project (R0 and its evaluation slice) were
excluded from this text by exact stripped-and-normalized string.

## Licence

**{licence}** — {why}.

Attribution is required for the corpora above regardless of this licence; see
the parent repository's `NOTICE`.

## Status

Published as an experimental artifact. On the parent project's development set
this restorer scored **below** a much simpler lexicon-lookup baseline: it is
accurate on words the lexicon already knows and inaccurate on the ones it does
not, which is the opposite of what would make it useful. Read the numbers in
the repository's `experiments/r11-restore/results.md` before relying on it.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, type=Path,
                    help="directory holding r11_*.pt")
    ap.add_argument("--only", default="",
                    help="publish just one arm, e.g. --only clean. Used by the "
                         "runner to upload each arm the moment it finishes, so "
                         "a later crash cannot cost the arms already trained.")
    ap.add_argument("--repo", default="Aju360/kathe-r11-restorer",
                    help="target for Apache-2.0-clean checkpoints. ShareAlike "
                         "ones go to <repo>-nc automatically.")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public. Default is PRIVATE.")
    ap.add_argument("--yes", action="store_true", help="actually upload")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, including licence routing, and stop")
    args = ap.parse_args()

    # `r11*_...` and not `r11_...`: the longer-trained rerun of each arm is named
    # r11b_clean.pt / r11b_dense.pt, and r11b_dense IS THE SHIPPED RESTORER
    # (leaderboard 13.81). A `r11_*.pt` glob silently skips every r11b file, so
    # the one checkpoint that must be published would never have been uploaded
    # and the run would have exited "nothing to publish" looking successful.
    pattern = f"r11*_{args.only}*.pt" if args.only else "r11*_*.pt"
    ckpts = sorted(args.checkpoints.glob(pattern))
    if not ckpts:
        print(f"  no {pattern} under {args.checkpoints} — nothing to publish")
        return 0
    print(f"  {len(ckpts)} checkpoint(s) match {pattern}: "
          + ", ".join(c.name for c in ckpts))

    owner = args.repo.split("/")[0] if "/" in args.repo else ""
    plan = []
    for ck in ckpts:
        meta_path = ck.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"  REFUSED {ck.name}: no {meta_path.name}, provenance unknown", file=sys.stderr)
            return 2
        meta = json.loads(meta_path.read_text(encoding="utf-8"))["meta"]
        sources = meta.get("sources")
        if not sources:
            print(f"  REFUSED {ck.name}: checkpoint records no training sources, "
                  "so its licence cannot be determined", file=sys.stderr)
            return 2
        lic, why = licence_for(sources)
        repo = args.repo if lic == APACHE else f"{args.repo}-nc"
        name = repo.split("/")[-1]
        if name in PROTECTED or any(p in repo for p in PROTECTED):
            print(f"  REFUSED: {repo!r} is a TRANSLATION weights repo. This "
                  "script publishes the R11 restorer only.", file=sys.stderr)
            return 2
        plan.append((ck, meta_path, meta, repo, lic, why))

    print(f"\n  {len(plan)} checkpoint(s), routed by the data each one actually saw:\n")
    for ck, _, meta, repo, lic, why in plan:
        print(f"    {ck.name:24} -> {repo}")
        print(f"      licence {lic:16} because {why[:96]}")
        print(f"      sources {', '.join(sorted(meta['sources']))}")
    targets = sorted({p[3] for p in plan})
    print(f"\n  repos: {', '.join(targets)}   visibility: "
          f"{'PUBLIC' if args.public else 'private'}")

    if args.dry_run or not args.yes:
        print("\n  dry run — nothing uploaded. Pass --yes to publish.")
        return 0

    # Write token ONLY. The read token is copied into notebooks and shared for
    # gated downloads; letting it write here would defeat the reason the two
    # are separate (PLANNING.md 2026-08-10).
    token = os.environ.get("HF_TOKEN_WRITE")
    if not token:
        print("  FATAL: HF_TOKEN_WRITE is not set. This script deliberately does "
              "NOT fall back to HF_TOKEN, which is the read token used for gated "
              "downloads.", file=sys.stderr)
        return 2
    if token == os.environ.get("HF_TOKEN"):
        print("  FATAL: HF_TOKEN_WRITE equals HF_TOKEN. They must be different "
              "tokens with different scopes.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for repo in targets:
        api.create_repo(repo, private=not args.public, exist_ok=True)
        print(f"  repo ready: {repo}")

    for ck, meta_path, meta, repo, lic, why in plan:
        api.upload_file(path_or_fileobj=str(ck), path_in_repo=ck.name, repo_id=repo)
        api.upload_file(path_or_fileobj=str(meta_path), path_in_repo=meta_path.name,
                        repo_id=repo)
        api.upload_file(
            path_or_fileobj=card(ck.stem, meta, lic, why).encode("utf-8"),
            path_in_repo=f"README_{ck.stem}.md", repo_id=repo)
        print(f"  uploaded {ck.name} + meta + card -> {repo}")

    # One top-level card per repo, from the first checkpoint that landed there.
    for repo in targets:
        first = next(p for p in plan if p[3] == repo)
        api.upload_file(
            path_or_fileobj=card(repo.split("/")[-1], first[2], first[4], first[5]).encode("utf-8"),
            path_in_repo="README.md", repo_id=repo)
    print(f"\n  done. {len(plan)} checkpoint(s) to {len(targets)} repo(s), "
          f"{'public' if args.public else 'PRIVATE'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
