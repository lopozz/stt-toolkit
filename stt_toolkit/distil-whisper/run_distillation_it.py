#!/usr/bin/env python
"""
Thin wrapper around run_distillation_patch.py (a locally-patched copy of
upstream's run_distillation.py - see the PATCH notes in that file) that
redirects one sentinel dataset name to our range-request streaming loader
over all seven Italian pseudo-labelled sources
(stream_pseudo_labeled_dataset.py), instead of a real `datasets.load_dataset`
call - which can't load this repo directly (it uses an unsupported loading
script; see that module's docstring).

Requires the same extra deps as upstream distil-whisper's training scripts
(accelerate, evaluate, wandb, ...) on top of stt-toolkit's own requirements -
these are NOT part of stt-toolkit's pyproject.toml, install them separately.

Usage: identical to run_distillation.py's own CLI, just pass this sentinel
name as both the train and eval dataset:

  accelerate launch run_distillation_it.py \\
    --model_name_or_path ./models/whisper-small-it-student-init \\
    --teacher_model_name_or_path openai/whisper-large-v3 \\
    --train_dataset_name it-multilingual-pseudo-labeled \\
    --train_split_name train \\
    --eval_dataset_name it-multilingual-pseudo-labeled \\
    --eval_split_name validation \\
    --use_pseudo_labels True \\
    --text_column_name text \\
    --language it --task transcribe \\
    ... (see run_distillation.py --help for the rest)
"""

import os

SENTINEL_NAME = "it-multilingual-pseudo-labeled"

# Number of interleaved examples held out as "validation" (first N, deterministic
# given the fixed seed in stream_it_multilingual) - the rest go to "train" via skip().
# NOTE: skip() on a streaming IterableDataset actually fetches and discards that
# many real examples (one network round-trip each) before the first training
# example is yielded - keep this small for smoke tests, override via env var
# (e.g. IT_EVAL_HOLDOUT_SIZE=2000) once you're doing a real training run.
EVAL_HOLDOUT_SIZE = int(os.environ.get("IT_EVAL_HOLDOUT_SIZE", "20"))

# Concurrent audio fetches PER SOURCE (7 sources interleaved -> peak total
# concurrency is roughly 7x this). Override via env var, e.g. IT_MAX_IN_FLIGHT=8.
MAX_IN_FLIGHT = int(os.environ.get("IT_MAX_IN_FLIGHT", "4"))

WER_THRESHOLD = 20.0

import run_distillation_patch as rd  # noqa: E402  (same directory, not a package import)
from stream_pseudo_labeled_dataset import stream_it_multilingual  # noqa: E402

_original_load_dataset = rd.load_dataset


def _patched_load_dataset(path, name=None, split="train", streaming=True, **kwargs):
    if path != SENTINEL_NAME:
        return _original_load_dataset(path, name, split=split, streaming=streaming, **kwargs)

    if not streaming:
        raise ValueError(
            f"{SENTINEL_NAME!r} is only available in streaming mode (pass --streaming True); "
            "the underlying sources are far too large (~784GB for Italian alone) to load non-streaming."
        )

    print(f"[_patched_load_dataset] building streaming dataset for split={split!r} (wer_threshold={WER_THRESHOLD})")

    # Rebuilding this per call keeps train/eval disjoint: both start from the
    # same deterministic interleaving order (fixed seed), so skip/take never overlap.
    full = stream_it_multilingual(wer_threshold=WER_THRESHOLD, max_in_flight=MAX_IN_FLIGHT)

    if split == "train":
        print(f"[_patched_load_dataset] split=train -> skipping first {EVAL_HOLDOUT_SIZE} interleaved examples")
        return full.skip(EVAL_HOLDOUT_SIZE)
    print(f"[_patched_load_dataset] split={split} -> taking first {EVAL_HOLDOUT_SIZE} interleaved examples")
    return full.take(EVAL_HOLDOUT_SIZE)


rd.load_dataset = _patched_load_dataset


if __name__ == "__main__":
    rd.main()
