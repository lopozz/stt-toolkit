import re
import io
import math

import numpy as np
import soundfile as sf

from pathlib import Path
from datasets import load_dataset_builder


def safe_filename(text):
    text = text.split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def waveform_to_in_memory_wav(waveform, sr, name="audio.wav"):
    """
    Writes a waveform array into an in-memory WAV file, making possible
    to send it to an API without saving anything to disk.
    """
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(waveform, dtype=np.float32), sr, format="WAV")
    buffer.name = name
    buffer.seek(0)
    return buffer


def estimate_training_steps(
    dataset_ids,
    desired_epochs,
    per_device_train_batch_size,
    gradient_accumulation_steps=1,
    num_gpus=1,
    config_name="default",
    split="train",
    max_train_samples=None,
):
    """Return approximate optimizer steps without decoding or downloading audio.

    dataset_ids accepts a list of Hub IDs/local directories, or a single ID.
    config_name and split accept shared strings or one value per dataset.
    Counts precede trainer filtering; interleaving may repeat or omit examples.
    num_gpus is the number of training processes (use 1 for a CPU run).
    """
    dataset_ids = [dataset_ids] if isinstance(dataset_ids, str) else list(dataset_ids)
    configs = (
        [config_name] * len(dataset_ids)
        if isinstance(config_name, str)
        else list(config_name)
    )
    splits = [split] * len(dataset_ids) if isinstance(split, str) else list(split)
    if (
        not dataset_ids
        or len(configs) != len(dataset_ids)
        or len(splits) != len(dataset_ids)
    ):
        raise ValueError(
            "Provide at least one dataset and one config and split per dataset"
        )
    if not math.isfinite(desired_epochs) or desired_epochs <= 0:
        raise ValueError("desired_epochs must be positive and finite")
    for value in (per_device_train_batch_size, gradient_accumulation_steps, num_gpus):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                "Batch size, gradient accumulation and process count must be positive integers"
            )
    if max_train_samples is not None and (
        not isinstance(max_train_samples, int)
        or isinstance(max_train_samples, bool)
        or max_train_samples <= 0
    ):
        raise ValueError("max_train_samples must be a positive integer")

    counts = []
    for dataset_id, config, split_name in zip(dataset_ids, configs, splits):
        builder = load_dataset_builder(dataset_id, name=config)
        info = (builder.info.splits or {}).get(split_name)
        count = info.num_examples if info is not None else None
        if count is None and Path(dataset_id).is_dir():
            import pyarrow.parquet as pq

            files = (builder.config.data_files or {}).get(split_name, [])
            if files and all(str(path).endswith(".parquet") for path in files):
                count = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        if count is None or count <= 0:
            raise ValueError(
                f"No positive example count available for {dataset_id!r}, split {split_name!r}"
            )
        counts.append(count)

    total_examples = sum(counts)
    if max_train_samples is not None:
        total_examples = min(total_examples, max_train_samples)
    effective_batch_size = (
        per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    )
    return {
        "dataset_counts": counts,
        "total_examples": total_examples,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": math.ceil(total_examples / effective_batch_size),
        "max_steps": math.ceil(desired_epochs * total_examples / effective_batch_size),
        "train_dataset_samples": "+".join(map(str, counts)),
    }
