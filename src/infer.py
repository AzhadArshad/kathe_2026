#!/usr/bin/env python3
"""
KATHE 2026 — model loading and inference, in one reusable place.

This is the module the two inference scripts share:

    scripts/translate_single.py   one sentence in, one translation out
    scripts/generate_translations.py   CSV in, CSV out (the competition format)

Both call `load_system()` and then `translate()`, so there is exactly one
definition of "the system" and a fix in either script cannot drift from the
other.

WHAT "THE SYSTEM" IS
--------------------
Three components, and all three are required:

    English --> IndicTrans2 200M fine-tune --> Kashmiri WITHOUT short vowels
            --> diacritic restorer (union)  --> Kashmiri WITH them

Dropping the restoration stage costs 5.05 points of the competition metric
(10.00 against 15.05), and it is not a quality issue that more training would
fix. IndicTrans2's target vocabulary holds 122,672 tokens, and kasra (U+0650),
damma (U+064F) and fatha (U+064E) each appear in exactly ONE of them -- the bare
standalone mark. Writing `چھُس` would require splitting a word to insert a bare
diacritic that occurs in no natural subword context, and beam search never does.
The translation model therefore emits exactly zero of these three marks however
it is trained; the vocabulary is frozen in the pretrained checkpoint. See
README.md "What we learned" §1.

Defaults resolve to the PUBLISHED weights, so this runs from a clean clone with
no local checkpoint directory and no Hugging Face token:

    translation   Aju360/kathe-r12-200m-selected     (Hub)
    restorer      Aju360/kathe-r11-restorer          (Hub, r11b_dense.pt)
    lexicon       data/processed/diacritic_lexicon_both.json   (this repo)

ENVIRONMENT
-----------
Requires `transformers==4.46.1`. Later releases drop the
`PreTrainedTokenizerBase` re-export that IndicTransToolkit imports at package
import time, which breaks inference, not merely training. See
`requirements-decode.txt`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

DEFAULT_MODEL = "Aju360/kathe-r12-200m-selected"
DEFAULT_POSTPROC = ROOT / "config" / "postproc_live.yaml"
RESTORER_REPO = "Aju360/kathe-r11-restorer"


@dataclass
class System:
    """A loaded translation system. Hold one and reuse it across calls."""

    model: object
    tokenizer: object
    processor: object
    device: str
    postproc: object  # data.normalize.NormConfig


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_restorer(cfg):
    """Point `restore_model` at a file that exists, downloading it if needed.

    The published checkpoint is fetched by NAME from the Hub, so a clone with no
    `models/` directory works unchanged. Kept identical to the resolution in
    `scripts/generate_translations.py` -- both now call this one copy.
    """
    from data.normalize import NormConfig

    if not cfg.restore_model:
        return cfg
    p = Path(cfg.restore_model)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return NormConfig(**{**cfg.__dict__, "restore_model": str(p)})

    from huggingface_hub import hf_hub_download

    got = hf_hub_download(repo_id=RESTORER_REPO, filename=Path(cfg.restore_model).name)
    return NormConfig(**{**cfg.__dict__, "restore_model": got})


def _resolve_lexicon(cfg):
    """Make `diacritic_lexicon` absolute against the repo root.

    Without this, running an inference script from any directory other than the
    repo root silently loses the lexicon half of the restorer -- which is a
    1.06-point regression that raises no error, because the model half still
    works and the output still looks correct.
    """
    from data.normalize import NormConfig

    if not cfg.diacritic_lexicon:
        return cfg
    p = Path(cfg.diacritic_lexicon)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(
            f"FATAL: diacritic lexicon not found at {p}\n"
            f"It ships in this repository. Without it the system silently "
            f"degrades from 15.05 to 13.99 on the competition metric."
        )
    return NormConfig(**{**cfg.__dict__, "diacritic_lexicon": str(p)})


def load_postproc(path: Path | str = DEFAULT_POSTPROC):
    """Read the post-processing config, rejecting unknown keys loudly."""
    from dataclasses import fields

    import yaml

    from data.normalize import NormConfig

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(NormConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"FATAL: unknown post-processing keys in {path}: {sorted(unknown)}")
    return _resolve_lexicon(_resolve_restorer(NormConfig(**raw)))


def load_system(
    model_id: str = DEFAULT_MODEL,
    postproc: Path | str = DEFAULT_POSTPROC,
    device: str = "auto",
    restore: bool = True,
) -> System:
    """Load the translation model, its processor, and the restoration config.

    `restore=False` disables the diacritic stage. It exists only to inspect the
    decode in isolation and costs 5.05 points -- do not use it in production.
    """
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from data.normalize import NormConfig

    dev = pick_device(device)
    cfg = load_postproc(postproc)
    if not restore:
        cfg = NormConfig(**{**cfg.__dict__, "restore_model": None, "diacritic_lexicon": None})

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    )
    model.to(dev).eval()
    return System(model=model, tokenizer=tok, processor=IndicProcessor(inference=True),
                  device=dev, postproc=cfg)


def translate(
    system: System,
    sentences: list[str],
    batch_size: int = 16,
    num_beams: int = 5,
    max_new_tokens: int = 256,
    length_penalty: float = 1.0,
    restore: bool = True,
) -> list[str]:
    """Translate English sentences into Kashmiri, in the order given.

    ORDER IS PRESERVED EXACTLY. Sentences are sorted by length internally so
    padding is not the dominant cost, then restored to input order; the
    competition scorer is positional, so a reordered output looks correct and
    scores near zero.
    """
    import torch

    from data.normalize import normalize_many

    if not sentences:
        return []

    order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i].split()))
    raw: list[str | None] = [None] * len(sentences)

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = system.processor.preprocess_batch(
            [sentences[i] for i in idx], src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        enc = system.tokenizer(batch, truncation=True, padding=True, max_length=256,
                               return_tensors="pt").to(system.device)
        with torch.inference_mode():
            gen = system.model.generate(
                **enc, num_beams=num_beams, num_return_sequences=1,
                max_new_tokens=max_new_tokens, length_penalty=length_penalty,
                early_stopping=True, use_cache=True,
            )
        with system.tokenizer.as_target_tokenizer():
            decoded = system.tokenizer.batch_decode(
                gen.detach().cpu().tolist(), skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # postprocess_batch pops one placeholder map per INPUT from an internal
        # queue. Passing a count that disagrees with the inputs makes it block
        # on Queue.get() forever -- no exception, no log line, 0% CPU.
        for i, text in zip(idx, system.processor.postprocess_batch(decoded, lang=TGT_LANG)):
            raw[i] = text

    if any(r is None for r in raw):
        raise RuntimeError("internal error: not every sentence was translated")
    out = [r for r in raw if r is not None]

    if not restore:
        return out
    # Restoration runs over the whole list at once: the tagger conditions only
    # on its own sentence, so this is identical to per-row and ~20x faster.
    return normalize_many(out, system.postproc)


def translate_one(system: System, sentence: str, **kw) -> str:
    """Single-sentence convenience wrapper around `translate`."""
    return translate(system, [sentence], **kw)[0]
