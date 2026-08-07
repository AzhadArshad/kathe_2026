"""
KATHE 2026 — single source of truth for identifiers.

PROJECT_NOTES.md §5: verify repo IDs on the HF Hub before use; do not assume from
memory. Anything that names a model, a language code, or a dataset path belongs
here, so a change is one edit rather than a grep.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_DEV = REPO_ROOT / "data" / "dev"

# --- language codes ----------------------------------------------------------
# PROJECT_NOTES.md §3: kas_Arab is CONFIRMED from the official scorer. Devanagari
# output passes through the normalizer unchanged and scores ~0.
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"
TGT_LANG_WRONG_SCRIPT = "kas_Deva"  # exists in BPCC; must never be trained on

# --- models ------------------------------------------------------------------
MODEL_PRIMARY = "ai4bharat/indictrans2-en-indic-1B"
MODEL_FAST = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_BACKTRANSLATE = "ai4bharat/indictrans2-indic-en-1B"
MODEL_SECOND_SYSTEM = "facebook/nllb-200-distilled-1.3B"
MODEL_LABSE = "sentence-transformers/LaBSE"

# --- BPCC --------------------------------------------------------------------
BPCC_REPO = "ai4bharat/BPCC"

# Enumerated from get_dataset_config_names(BPCC_REPO) on 2026-08-07:
#   bpcc-seed-latest, bpcc-seed-v2, bpcc-seed-v1, daily, comparable, ilci,
#   nllb-seed, massive, nllb-filtered, samanantar-filtered, samanantar++
# Only six carry a kas_Arab split; comparable/ilci/massive/samanantar-* have no
# Kashmiri at all.
#
# We address files by PATH rather than by config because BPCC's README defines
# config_name "daily" TWICE -- once over daily/, once over wiki/. Loading the
# config silently yields only one of them, so wiki/ would be lost with no error.
#
# provenance: "human"  = human-translated seed data
#             "mined"  = bitext-mined from web/comparable corpora
BPCC_KAS_FILES = [
    ("bpcc-seed-v1/kas_Arab.tsv", "bpcc-seed-v1", "human"),
    ("bpcc-seed-v2/kas_Arab.tsv", "bpcc-seed-v2", "human"),
    ("bpcc-seed-latest/kas_Arab.tsv", "bpcc-seed-latest", "human"),
    ("nllb_seed/kas_Arab.tsv", "nllb-seed", "human"),
    ("daily/kas_Arab.tsv", "daily", "human"),
    ("wiki/kas_Arab.tsv", "daily(wiki)", "mined"),
    ("nllb_filtered/kas_Arab.tsv", "nllb-filtered", "mined"),
]

# --- dev sets ----------------------------------------------------------------
DEV_FLORES_KAS = DATA_DEV / "flores_kas.txt"
DEV_FLORES_EN = DATA_DEV / "flores_en.txt"
