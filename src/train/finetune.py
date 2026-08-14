#!/usr/bin/env python3
"""
KATHE 2026 — R3 fine-tuning.

ADAPTED FROM AI4Bharat's `IndicTrans2/huggingface_interface/train_lora.py`
(MIT). PROJECT_NOTES.md §2.8 forbids a hand-rolled training loop, so the data loading,
the tokenization, the collator wiring and the Trainer construction are kept in
the shape AI4Bharat shipped them. What changed, and why:

  1. **Full fine-tune is now a first-class mode** (`peft: none`). Their script
     always wraps the model with LoRA and saves only the adapter. The 200M
     distilled model is fine-tuned whole; LoRA stays available for the 1B.
  2. **Config comes from YAML, not thirty CLI flags** (PROJECT_NOTES.md §6), and the
     resolved config is written next to the checkpoint so a run is reproducible
     from its own output directory.
  3. **The eval metric is the competition metric**, not raw BLEU/chrF. It is a
     geometric mean of BLEU and chrF++ computed through `data.normalize`, the
     same module the scorer path uses — see §metrics below for the one caveat.
  4. **`evaluation_strategy` -> `eval_strategy`**, required by transformers
     4.46.1, the version this repo is pinned to.
  5. **Hub checkpointing every save**, because Kaggle sessions die mid-run.

§metrics — READ THIS BEFORE TRUSTING eval_geo_proxy
---------------------------------------------------
The number logged during training is `eval_geo_proxy`, and it is NOT comparable
to the 15.83 zero-shot baseline.

`IndicProcessor.preprocess_batch` leaves training data in an internal space:
Indic-tokenized, punctuation-spaced, entity placeholders unresolved. Turning it
back into real Kashmiri is `postprocess_batch`'s job — and that method cannot be
called here. It unconditionally does a **blocking** `Queue.get()` per sentence
against a placeholder-map queue that `IndicProcessor(inference=False)` never
fills. Calling it during in-training eval does not raise; it hangs the run
forever, which on a 9-hour Kaggle session is the worst available failure.

So predictions and references are both scored in that internal space. They are
mutually consistent, which makes the number a sound *relative* signal for
ranking checkpoints, and that is all it is used for. Report real scores with
`scripts/translate.py` over FLORES devtest, through the full
preprocess -> generate -> postprocess -> normalize path.

Usage (Kaggle T4 x2):
    torchrun --nproc_per_node=2 -m train.finetune --config config/r3_200m_full.yaml

Single GPU, or a data-only smoke test:
    python -m train.finetune --config config/r3_200m_full.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
import yaml
from datasets import Dataset
from IndicTransToolkit import IndicDataCollator, IndicProcessor
import IndicTransToolkit.collator as _itt_collator
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# --- upstream bugfix: IndicTransToolkit 1.1.1 IndicDataCollator -------------
# collator.py puts its imports INSIDE the class body:
#
#     @dataclass
#     class IndicDataCollator:
#         from transformers.data.data_collator import pad_without_fast_tokenizer_warning
#         ...
#         def __call__(self, features, ...):
#             features = pad_without_fast_tokenizer_warning(...)   # line 40
#
# Names bound in a class body are class attributes and are NOT in scope inside
# that class's methods, so the bare call raises
# `NameError: name 'pad_without_fast_tokenizer_warning' is not defined` on the
# very first non-empty batch. The collator is unusable as shipped — this is not
# a version-skew problem and pinning differently does not avoid it.
#
# Injecting the name into the module's globals makes the bare reference resolve
# through the normal local -> enclosing -> global lookup, with upstream's
# behaviour otherwise untouched. Preferred over subclassing or swapping in
# transformers' DataCollatorForSeq2Seq, because IndicDataCollator also forces
# left-padding and pads labels itself, and IndicTrans2 depends on both.
#
# The annotations at collator.py lines 12/14 are fine: they are evaluated in the
# class body, where the names *are* in scope. Only the line-40 call is broken.
if not hasattr(_itt_collator, "pad_without_fast_tokenizer_warning"):
    from transformers.data.data_collator import pad_without_fast_tokenizer_warning

    _itt_collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import NormConfig, normalize_many  # noqa: E402

# Scorer-normalizer only. The project's extra orthography fixes belong in
# post-processing, not in a training-time metric (data/normalize.py).
_SCORER = NormConfig(scorer_normalizer=True)


# --- data (structure preserved from AI4Bharat) --------------------------------
def load_and_process_translation_dataset(
    data_dir,
    split="train",
    tokenizer=None,
    processor=None,
    src_lang_list=None,
    tgt_lang_list=None,
    num_proc=8,
    max_length=256,
    seed=42,
):
    complete_dataset = {"sentence_SRC": [], "sentence_TGT": []}

    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
            tgt_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}")
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                raise FileNotFoundError(
                    f"Source ({split}.{src_lang}) or Target ({split}.{tgt_lang}) file "
                    f"not found in {data_dir}. Run `python -m data.build_corpus` first."
                )
            with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
                src_lines = src_file.readlines()
                tgt_lines = tgt_file.readlines()

            assert len(src_lines) == len(tgt_lines), (
                f"Source and Target files have different number of lines for "
                f"{split}.{src_lang} and {split}.{tgt_lang}"
            )

            complete_dataset["sentence_SRC"] += processor.preprocess_batch(
                src_lines, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False
            )
            complete_dataset["sentence_TGT"] += processor.preprocess_batch(
                tgt_lines, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
            )

    complete_dataset = Dataset.from_dict(complete_dataset).shuffle(seed=seed)
    return complete_dataset.map(
        lambda example: preprocess_fn(example, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        num_proc=num_proc,
    )


def preprocess_fn(example, tokenizer, max_length=256, **kwargs):
    model_inputs = tokenizer(
        example["sentence_SRC"], truncation=True, padding=False, max_length=max_length
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["sentence_TGT"], truncation=True, padding=False, max_length=max_length
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# --- metrics ------------------------------------------------------------------
def compute_metrics_factory(tokenizer, print_samples=False, n_samples=5):
    """Competition metric: geometric mean of BLEU(13a) and chrF++.

    See the module docstring, §metrics — this is computed in IndicProcessor's
    internal space and ranks checkpoints; it does not report scores.
    """

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id

        with tokenizer.as_target_tokenizer():
            preds = [
                x.strip()
                for x in tokenizer.batch_decode(
                    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]
            labels = [
                x.strip()
                for x in tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]

        assert len(preds) == len(labels), "Predictions and Labels have different lengths"

        # Same normalizer the scorer uses, so the metric moves for the same
        # reasons the leaderboard would.
        hyps = normalize_many(preds, _SCORER)
        refs = normalize_many(labels, _SCORER)

        # An empty hypothesis makes the REAL scorer raise and reject the whole
        # submission. Here it must not kill a training run, so it is counted and
        # substituted — a rising empty_preds is the signal that decoding has
        # collapsed.
        empty = sum(1 for h in hyps if not h.strip())
        hyps = [h if h.strip() else "۔" for h in hyps]

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        geo = 0.0 if (bleu <= 0 or chrfpp <= 0) else (bleu * chrfpp) ** 0.5

        if print_samples:
            df = pd.DataFrame({"Predictions": preds, "References": labels}).sample(
                n=min(n_samples, len(preds))
            )
            for pred, label in zip(df["Predictions"].values, df["References"].values):
                print(f" | > Prediction: {pred}")
                print(f" | > Reference : {label}\n")

        # Output/reference length ratio — R6 length calibration watches this.
        # BPCC lines average 92.6 chars against FLORES's 124.6, so a fine-tune
        # biases short and BLEU's brevity penalty bites.
        ref_chars = sum(len(r) for r in refs) or 1
        return {
            "bleu": bleu,
            "chrf_pp": chrfpp,
            "geo_proxy": geo,
            "len_ratio": sum(len(h) for h in hyps) / ref_chars,
            "empty_preds": empty,
        }

    return compute_metrics


# --- config -------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_checkpoint_portable(output_dir: str) -> None:
    """Make a saved checkpoint loadable on a machine that is not this one.

    Two defects in what `tokenizer.save_pretrained` writes for IndicTrans2, both
    of which make the checkpoint load ONLY on the box that produced it. Found
    2026-08-10, after the R3 checkpoint failed to load off the training host:

    1. `src_vocab_file` / `tgt_vocab_file` are serialized as ABSOLUTE paths into
       the training machine's HF cache (`/root/.cache/...` on Kaggle). Elsewhere
       those paths do not exist, and they also collide with the positional
       arguments `from_pretrained` supplies -> `TypeError: got multiple values
       for keyword argument 'src_vocab_file'`. The base repo ships neither key;
       dropping them lets the tokenizer resolve `dict.SRC.json` / `dict.TGT.json`
       from the checkpoint directory, where they already are.

    2. `auto_map` points back at `ai4bharat/indictrans2-en-indic-dist-200M`,
       which is a GATED repo. Loading would then require Hub access and an
       accepted gate — exactly the dependency the live-round package must not
       have. The tokenizer source is copied in and `auto_map` made local.

    3. **`lm_head.weight` is absent, and its absence silently ZEROES the model.**
       IndicTrans2 sets `share_decoder_input_output_embed: True`, so `lm_head`
       ties to `model.decoder.embed_tokens`. `save_pretrained` drops the tied
       duplicate, which is normally fine — but this architecture's load path
       resolves the tie the wrong way and overwrites the good embedding with the
       missing head, leaving both all-zero. `from_pretrained` reports "All the
       weights ... were initialized from the model checkpoint" while handing
       back a model that emits EOS immediately and translates everything to "".
       The only visible hint at save time is a `There were missing keys in the
       checkpoint model loaded: ['lm_head.weight']` line that reads as benign.
       Writing the tensor explicitly costs ~250 MB and makes a plain
       `from_pretrained` correct — which is what the live round will run.

    Safe to call more than once.
    """
    import glob
    import shutil

    ck = Path(output_dir)

    st = ck / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file, save_file

        sd = load_file(str(st))
        tied = "model.decoder.embed_tokens.weight"
        if "lm_head.weight" not in sd and tied in sd:
            # .clone(): safetensors rejects tensors that share storage.
            sd["lm_head.weight"] = sd[tied].clone()
            save_file(sd, str(st), metadata={"format": "pt"})
            print(" | > wrote lm_head.weight into model.safetensors "
                  "(tied-weight save would otherwise load as all-zero)")
    cfg_path = ck / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    removed = [k for k in ("src_vocab_file", "tgt_vocab_file", "name_or_path") if k in cfg]
    for k in removed:
        cfg.pop(k)

    if not (ck / "tokenization_indictrans.py").exists():
        hits = glob.glob(str(
            Path.home() / ".cache/huggingface/modules/transformers_modules"
            / "**/tokenization_indictrans.py"
        ), recursive=True)
        if hits:
            shutil.copy(hits[0], ck / "tokenization_indictrans.py")
    if (ck / "tokenization_indictrans.py").exists():
        cfg["auto_map"] = {
            "AutoTokenizer": ["tokenization_indictrans.IndicTransTokenizer", None]
        }
    else:
        print(" | > WARNING: tokenization_indictrans.py not vendored — the "
              "checkpoint will need the GATED base repo to load.")

    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" | > checkpoint made portable (dropped {removed or 'nothing'}, "
          f"tokenizer code vendored)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-dir", help="override config data_dir (stage 2 uses this)")
    ap.add_argument("--output-dir", help="override config output_dir")
    ap.add_argument("--init-from", help="start from a previous checkpoint instead of the base model")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    ap.add_argument("--dry-run", action="store_true", help="build datasets, print shapes, exit before training")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = args.data_dir or cfg["data_dir"]
    output_dir = args.output_dir or cfg["output_dir"]
    model_id = args.init_from or cfg["model"]

    set_seed(cfg["seed"])
    is_main = int(os.environ.get("RANK", "0")) == 0

    if is_main:
        print(f" | > config     {args.config}")
        print(f" | > model      {model_id}")
        print(f" | > data_dir   {data_dir}")
        print(f" | > output_dir {output_dir}")
        print(f" | > git commit {git_commit()}")

    print(f" | > Loading {model_id} and tokenizer ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,  # IndicTrans2 ships custom modeling code
        attn_implementation="eager",
        dropout=cfg["dropout"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # inference=False is what training wants, and is also why postprocess_batch
    # must never be called on this instance. See the module docstring, §metrics.
    processor = IndicProcessor(inference=False)

    data_collator = IndicDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=8,  # fp16 tensor cores want multiples of 8
        label_pad_token_id=-100,
    )

    common = dict(
        tokenizer=tokenizer,
        processor=processor,
        src_lang_list=[cfg["src_lang"]],
        tgt_lang_list=[cfg["tgt_lang"]],
        num_proc=cfg["num_proc"],
        max_length=cfg["max_seq_length"],
        seed=cfg["seed"],
    )
    train_dataset = load_and_process_translation_dataset(data_dir, split="train", **common)
    print(f" | > Loaded train dataset from {data_dir}. Size: {len(train_dataset)} ...")
    eval_dataset = load_and_process_translation_dataset(data_dir, split="dev", **common)
    print(f" | > Loaded eval dataset from {data_dir}. Size: {len(eval_dataset)} ...")

    if args.dry_run:
        ex = train_dataset[0]
        print("\n | > DRY RUN — first training example")
        print(f"   SRC : {ex['sentence_SRC'][:160]}")
        print(f"   TGT : {ex['sentence_TGT'][:160]}")
        print(f"   input_ids {len(ex['input_ids'])}  labels {len(ex['labels'])}")
        print(" | > datasets built, exiting before training.")
        return 0

    # Label smoothing lives on IndicTrans2's custom model class, and must be set
    # BEFORE any PEFT wrapper — the wrapper does not forward the setter.
    if hasattr(model, "set_label_smoothing"):
        model.set_label_smoothing(cfg["label_smoothing"])
    elif cfg["label_smoothing"]:
        print(" | > WARNING: model has no set_label_smoothing; label smoothing NOT applied")

    if cfg["peft"] == "lora":
        from peft import LoraConfig, get_peft_model

        lora = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["r"],
                bias="none",
                inference_mode=False,
                task_type="SEQ_2_SEQ_LM",
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"].split(","),
            ),
        )
        model.print_trainable_parameters()
    else:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" | > FULL fine-tune — {n:,} trainable parameters")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        seed=cfg["seed"],
        fp16=cfg["fp16"],
        logging_strategy="steps",
        eval_strategy="steps",  # renamed from evaluation_strategy in transformers 4.41+
        save_strategy="steps",
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        predict_with_generate=True,
        # OFF for LoRA, and this is not a preference. `_load_best_model()` calls
        # peft's `load_adapter`, which reaches
        # `transformers.integrations.tensor_parallel` — a module that does not
        # exist in transformers 4.46.1, the version IndicTransToolkit forces.
        # The result is a ModuleNotFoundError AFTER training completes, which
        # destroys the final save while every training step is already paid for
        # (2026-08-11: R4 crashed at 4825/4825 after 3h20m).
        #
        # The cost is small: `eval_geo_proxy` rose monotonically across R3 and
        # R4, so the last checkpoint IS the best one. `trainer_state.json`
        # records `best_model_checkpoint` either way, so a non-monotonic run can
        # still be recovered by hand.
        load_best_model_at_end=(cfg["peft"] != "lora"),
        max_steps=cfg["max_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        # DDP walks the whole autograd graph every micro-batch when this is
        # True, and HF defaults it to True. Its own warning reports finding no
        # unused parameters — with LoRA the frozen base has requires_grad=False,
        # so DDP never considers it. At grad_accum 16 that traversal is paid 16
        # times per optimizer step, which is why batch 4 measured SLOWER per
        # step (4.57 s/it) than batch 8 (2.48 s/it) despite identical samples
        # per step. Default False; set true in config only if a run reports
        # genuinely unused parameters.
        ddp_find_unused_parameters=cfg.get("ddp_find_unused_parameters", False),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        eval_accumulation_steps=cfg["grad_accum_steps"],
        weight_decay=cfg["weight_decay"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optimizer"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        dataloader_num_workers=cfg["num_workers"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        report_to=cfg["report_to"],
        generation_max_length=cfg["max_seq_length"],
        generation_num_beams=cfg["eval_num_beams"],
        group_by_length=cfg["group_by_length"],
        sortish_sampler=cfg["sortish_sampler"],
        # Kaggle sessions die (PROJECT_NOTES.md §5); every save is a Hub push.
        push_to_hub=cfg["push_to_hub"],
        hub_model_id=cfg["hub_model_id"],
        hub_private_repo=cfg["hub_private_repo"],
        hub_strategy="every_save" if cfg["push_to_hub"] else "end",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_factory(
            tokenizer=tokenizer, print_samples=cfg["print_samples"]
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"],
                early_stopping_threshold=cfg["early_stopping_threshold"],
            )
        ],
    )

    # Report VRAM before the first step. A CUBLAS_STATUS_INTERNAL_ERROR in
    # backward is nearly always OOM in disguise, and without this line there is
    # no way to tell a genuinely tight card from a real cuBLAS fault. An
    # interactive session and a committed run do not always leave the same
    # amount free (2026-08-11: batch 8 survived one and died at step 0 in the
    # other).
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = torch.cuda.memory_allocated()
        print(f" | > GPU{torch.cuda.current_device()} "
              f"free {free / 2**30:.2f} GiB / total {total / 2**30:.2f} GiB "
              f"| already allocated by this process {used / 2**30:.2f} GiB")

    print(" | > Starting training ...")
    try:
        trainer.train(resume_from_checkpoint=args.resume or None)
    except KeyboardInterrupt:
        print(" | > Training interrupted — saving what exists ...")

    if is_main:
        # MERGED weights, never a bare adapter — the live round must not need
        # the gated base repo to load anything (PLANNING.md, 2026-08-07).
        # `trainer.save_model` on a PEFT-wrapped model writes only the adapter,
        # so LoRA runs are merged first. merge_and_unload() folds B·A back into
        # W and returns a plain model that loads with a normal from_pretrained.
        if cfg["peft"] == "lora":
            merged = trainer.model
            merged = getattr(merged, "module", merged)  # unwrap DDP
            merged = merged.merge_and_unload()
            merged.save_pretrained(output_dir, safe_serialization=True)
            print(" | > LoRA adapter merged into base weights before saving")
        else:
            trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        make_checkpoint_portable(output_dir)
        Path(output_dir, "train_config.resolved.yaml").write_text(
            yaml.safe_dump(
                {**cfg, "data_dir": data_dir, "output_dir": output_dir,
                 "model": model_id, "git_commit": git_commit()},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest = Path(data_dir, "manifest.json")
        if manifest.exists():
            Path(output_dir, "corpus_manifest.json").write_text(
                manifest.read_text(encoding="utf-8"), encoding="utf-8"
            )
        print(f" | > saved to {output_dir}")

        # RE-UPLOAD the portable copy. `hub_strategy="every_save"` has been
        # pushing raw Trainer checkpoints throughout training, and NONE of them
        # contain `lm_head.weight` — save_pretrained drops it as a tied
        # duplicate, and this architecture's load path then zeroes both it and
        # the decoder embedding. Every one of those Hub checkpoints therefore
        # loads without error and translates every input to the empty string
        # (PLANNING.md 2026-08-10). Without this step push_to_hub is not crash
        # insurance at all: it stores unusable weights.
        #
        # An explicit upload_folder is also what rescued R3 when the session
        # nearly died (PLANNING.md 2026-08-10), so it is the recovery path with
        # a track record here.
        if cfg["push_to_hub"] and cfg.get("hub_model_id"):
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=os.environ.get("HF_TOKEN"))
                api.create_repo(cfg["hub_model_id"], exist_ok=True,
                                private=cfg.get("hub_private_repo", True))
                api.upload_folder(folder_path=output_dir,
                                  repo_id=cfg["hub_model_id"],
                                  commit_message="portable checkpoint "
                                                 "(lm_head.weight written, paths stripped)")
                print(f" | > re-uploaded PORTABLE checkpoint to {cfg['hub_model_id']}")
            except Exception as e:  # never lose a finished run to a push failure
                print(f" | > WARNING: portable re-upload failed ({type(e).__name__}: {e}). "
                      f"The local copy at {output_dir} IS correct — upload it by hand, "
                      f"and note that anything already on the Hub is NOT loadable "
                      f"until make_checkpoint_portable() is applied to it.")

        print(json.dumps(trainer.evaluate(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
