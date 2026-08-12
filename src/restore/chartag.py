#!/usr/bin/env python3
"""
KATHE 2026 — character-level diacritic restoration, model and codec.

Why a TAGGER and not a seq2seq model
------------------------------------
The brief calls for "undiacritized Kashmiri in, diacritized Kashmiri out". A
free-running seq2seq model can do that, but it can also drop a word, reorder a
clause or hallucinate a letter — a translation bug wearing a spelling hat. The
requirement is INSERTION-ONLY, so it is designed in rather than checked for:

    for every character of the input, predict which of {nothing, kasra, damma,
    fatha} follows it.

The base-letter sequence is not an output of the model at all, so it cannot
change. `apply_labels(strip(x), labels)` stripped back down is `strip(x)` for
ANY label vector — the guarantee is structural, and `assert_insertion_only`
re-checks it at runtime anyway because a cheap assert on a silent failure mode
is worth keeping.

This is also why per-mark precision/recall is well defined here: input and
output positions correspond one to one, so an inserted-but-wrong mark and an
omitted mark are separate, countable events.

Labelling
---------
The label at position i is the first restorable mark following character i in
the original text. Multi-mark runs (`ِِ`) exist but are corpus typos: 410 of
440,000 marks in BPCC train, 71 of 240,000 in the external corpus. They are
truncated to their first mark. A mark at position 0 has no character to attach
to (15 lines in BPCC, 17 in the external corpus) and is dropped.

Chunking
--------
Sentences longer than `max_len` characters are split at spaces. Production rows
are ~40 characters, so this only affects training text; the split is
non-overlapping and loses one word of left context at each boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

# Import rather than redefine: one definition of "restorable" across the lexicon
# restorer and this one. `RESTORABLE` is frozenset("َُِ") — fatha, damma, kasra.
try:
    from data.diacritize import RESTORABLE, strip_key  # noqa: F401
except ImportError:  # pragma: no cover - allows `python src/restore/chartag.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.diacritize import RESTORABLE, strip_key  # noqa: F401

# Index 0 is "no mark". Order is fixed forever: it is baked into checkpoints.
MARKS = ["", "ِ", "ُ", "َ"]  # none, kasra, damma, fatha
MARK_NAMES = ["none", "kasra", "damma", "fatha"]
MARK_INDEX = {m: i for i, m in enumerate(MARKS) if m}
N_LABELS = len(MARKS)

PAD, UNK = 0, 1  # reserved character ids
IGNORE = -100  # CrossEntropyLoss ignore_index


# --------------------------------------------------------------------------- #
# codec
# --------------------------------------------------------------------------- #
def encode(text: str) -> tuple[str, list[int]]:
    """Diacritized text -> (stripped text, one label per stripped character)."""
    base: list[str] = []
    labels: list[int] = []
    for ch in text:
        if ch in RESTORABLE:
            # Attach to the preceding character unless it is whitespace. A mark
            # after a space is corpus noise — 345 occurrences in 679,144 across
            # all three corpora — and allowing the label would let a negative
            # `none_bias` manufacture them at inference, where they are
            # guaranteed errors. Dropping it here and masking the same positions
            # in `Restorer` keeps training and inference agreeing.
            if base and labels[-1] == 0 and not base[-1].isspace():
                labels[-1] = MARK_INDEX[ch]
            continue  # leading mark, mark on a space, or a second mark: drop
        base.append(ch)
        labels.append(0)
    return "".join(base), labels


def apply_labels(base: str, labels: list[int] | torch.Tensor) -> str:
    """(stripped text, labels) -> diacritized text. Insertion-only by construction."""
    if len(labels) != len(base):
        raise ValueError(f"label/base length mismatch: {len(labels)} vs {len(base)}")
    out: list[str] = []
    for ch, y in zip(base, labels):
        out.append(ch)
        y = int(y)
        if y:
            out.append(MARKS[y])
    return "".join(out)


def assert_insertion_only(src: str, out: str) -> None:
    """The restorer may only insert the three marks. Nothing else may move.

    Checked on every restored line in `Restorer.restore_many`. A failure here
    means a spelling model has started rewriting words, which is exactly the
    failure the tagger design exists to prevent.
    """
    if strip_key(out) != strip_key(src):
        raise AssertionError(
            "restoration changed the base-character sequence:\n"
            f"  in : {src!r}\n  out: {out!r}"
        )


def chunk(text: str, max_len: int) -> list[tuple[str, str]]:
    """Split into pieces of at most `max_len` characters, preferring spaces.

    Returns `[(piece, separator_before), ...]` such that

        "".join(sep + piece for piece, sep in parts) == text

    exactly — the separator is recorded rather than assumed, because a word
    longer than `max_len` has to be cut mid-word and rejoining THAT with a space
    would insert a character the restorer is not allowed to insert.
    """
    if len(text) <= max_len:
        return [(text, "")]
    parts: list[tuple[str, str]] = []
    cur, sep = "", ""
    for i, word in enumerate(text.split(" ")):
        if i and cur and len(cur) + 1 + len(word) > max_len:
            parts.append((cur, sep))
            cur, sep = word, " "
        else:
            cur = f"{cur} {word}" if i else word
        while len(cur) > max_len:  # single word longer than max_len
            parts.append((cur[:max_len], sep))
            cur, sep = cur[max_len:], ""
    parts.append((cur, sep))
    joined = "".join(s + p for p, s in parts)
    assert joined == text, "chunking is not reversible"
    return parts


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    """Character vocabulary over STRIPPED text (marks are labels, not inputs)."""

    def __init__(self, chars: list[str]):
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    @classmethod
    def build(cls, texts, min_count: int = 5) -> "Vocab":
        from collections import Counter

        counts: Counter = Counter()
        for t in texts:
            counts.update(t)
        chars = sorted(c for c, n in counts.items() if n >= min_count)
        return cls(chars)

    def to_json(self) -> list[str]:
        return self.itos[2:]

    @classmethod
    def from_json(cls, chars: list[str]) -> "Vocab":
        return cls(list(chars))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 384


class CharTagger(nn.Module):
    """Bidirectional transformer encoder, one 4-way decision per character.

    Bidirectional matters and is the point of departure from the lexicon: the
    lexicon sees one word of LEFT context, this sees the whole sentence on both
    sides. Kashmiri short vowels mark person/gender agreement carried by
    material that can sit either side of the verb.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor is a no-op under norm_first and only emits a
        # warning on every construction; turn it off rather than filter it.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, N_LABELS)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.size(1)
        pos = torch.arange(n, device=ids.device).unsqueeze(0)
        h = self.drop(self.embed(ids) * math.sqrt(self.cfg.d_model) + self.pos(pos))
        h = self.encoder(h, src_key_padding_mask=ids.eq(PAD))
        return self.head(self.norm(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model: CharTagger, vocab: Vocab, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.cfg),
            "vocab": vocab.to_json(),
            "meta": meta,
        },
        path,
    )
    (path.with_suffix(".meta.json")).write_text(
        json.dumps({"config": asdict(model.cfg), "meta": meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class Restorer:
    """Load a checkpoint and restore text. This is what the live round runs.

    `none_bias` is added to the logit of the "no mark" class before the argmax.
    NEGATIVE values make the model insert more marks. It exists because the
    lexicon sweep established that a WRONG diacritic beats NO diacritic under
    chrF++ (data/diacritize.py: dominance 0.0 > 0.6 > 0.9), so the
    accuracy-optimal threshold is not the score-optimal one.
    """

    def __init__(self, path: str | Path, device: str = "cpu", none_bias: float = 0.0,
                 batch_size: int = 128):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig(**blob["config"])
        self.vocab = Vocab.from_json(blob["vocab"])
        self.model = CharTagger(self.cfg)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.none_bias = none_bias
        self.batch_size = batch_size
        self.meta = blob.get("meta", {})

    def _tag(self, pieces: list[str]) -> list[list[int]]:
        out: list[list[int]] = []
        for start in range(0, len(pieces), self.batch_size):
            batch = pieces[start:start + self.batch_size]
            width = max(len(p) for p in batch)
            ids = torch.full((len(batch), width), PAD, dtype=torch.long)
            for i, p in enumerate(batch):
                ids[i, : len(p)] = torch.tensor(self.vocab.encode(p), dtype=torch.long)
            with torch.inference_mode():
                logits = self.model(ids.to(self.device)).float()
                if self.none_bias:
                    logits[..., 0] += self.none_bias
                pred = logits.argmax(-1).cpu()
            for i, p in enumerate(batch):
                labels = pred[i, : len(p)].tolist()
                # No mark may follow whitespace — see `encode`. This is the
                # inference half of that rule; without it `none_bias` buys
                # recall by inventing marks in the one place they cannot go.
                out.append([0 if c.isspace() else y for c, y in zip(p, labels)])
        return out

    def restore_many(self, texts: list[str]) -> list[str]:
        """Restore a batch. Input may already carry marks; they are stripped
        first so the model always sees the distribution it was trained on."""
        pieces: list[str] = []
        seps: list[str] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            for piece, sep in chunk(strip_key(t), self.cfg.max_len):
                pieces.append(piece)
                seps.append(sep)
                owner.append(i)
        if not pieces:
            return list(texts)

        # Long pieces first: batches are padded to their longest member.
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
        tagged: list[list[int]] = [[]] * len(pieces)
        for idx, labels in zip(order, self._tag([pieces[i] for i in order])):
            tagged[idx] = labels

        out: list[list[str]] = [[] for _ in texts]
        for piece, sep, labels, i in zip(pieces, seps, tagged, owner):
            out[i].append(sep + apply_labels(piece, labels))
        result = ["".join(parts) for parts in out]
        for src, res in zip(texts, result):
            assert_insertion_only(src, res)
        return result

    def restore(self, text: str) -> str:
        return self.restore_many([text])[0]
