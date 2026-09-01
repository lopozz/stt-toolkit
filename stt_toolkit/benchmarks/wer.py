import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime, timezone

from jiwer import (
    Compose,
    ReduceToListOfListOfWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
    wer,
)

from stt_toolkit.cache import ResultCache
from stt_toolkit.backends import STTBackend
from stt_toolkit.utils import waveform_to_in_memory_wav


def change_audio_speed(waveform, speed: float):
    if speed <= 0:
        raise ValueError(f"Invalid speed: {speed}")
    if speed == 1.0:
        return np.asarray(waveform, dtype=np.float32)

    waveform = np.asarray(waveform, dtype=np.float32)
    new_length = max(1, int(len(waveform) / speed))
    src_positions = np.arange(len(waveform), dtype=np.float32)
    dst_positions = np.linspace(0, len(waveform) - 1, new_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, waveform).astype(np.float32)


def task_name(task_config: dict) -> str:
    dataset_name = task_config["dataset"]
    split = task_config["split"]
    subset = task_config.get("subset")
    max_samples = task_config["max_samples"]

    name = f"{dataset_name}[{split}]"
    if subset:
        name = f"{dataset_name}/{subset}[{split}]"
    if max_samples is not None:
        name = f"{name}[:{max_samples}]"
    return name


def load_task_dataset(task_config: dict):
    dataset_name = task_config["dataset"]
    subset = task_config["subset"]
    split = task_config["split"]

    if subset:
        return load_dataset(dataset_name, subset, split=split, streaming=True)
    return load_dataset(dataset_name, split=split, streaming=True)


def evaluate_wer(
    model: str,
    tasks: list[dict],
    cache: ResultCache,
    backend: STTBackend,
    speeds: list[float] | None = None,
    overwrite: bool = False,
):
    speeds = speeds or [1.0]
    task_names = []

    norm = Compose(
        [
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToListOfListOfWords(),
        ]
    )

    for task_config in tasks:
        dataset_name = task_config["dataset"]
        split = task_config["split"]
        task_speeds = task_config["speeds"]
        subset = task_config["subset"]
        audio_column = task_config["audio_column"]
        text_column = task_config["text_column"]
        source_column = task_config["source_column"]
        max_samples = task_config["max_samples"]
        task = task_name(task_config)
        task_names.append(task)

        if cache.has_result(model, task) and not overwrite:
            print(f"Skipping cached result: model={model}, task={task}")
            print(f"Cached file: {cache.result_path(model, task)}")
            continue

        dataset_label = f"{dataset_name}/{subset}" if subset else dataset_name
        print(f"Streaming dataset: {dataset_label} [{split}]")

        results = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "task": task,
                "dataset": dataset_name,
                "subset": subset,
                "split": split,
                "audio_column": audio_column,
                "text_column": text_column,
                "source_column": source_column,
                "max_samples": max_samples,
                "speeds": task_speeds,
                "benchmark": "wer_bench",
            },
            "results": {},
        }

        print("Sending audio data...")
        for speed in task_speeds:
            refs, preds = [], []
            speed_key = f"{speed:g}x"
            results["results"][speed_key] = {"wer": None, "samples": []}

            print(f"\nEvaluating speed {speed_key}...")

            dataset = load_task_dataset(task_config)
            total = dataset.info.splits[split].num_examples
            if max_samples is not None:
                total = min(total, max_samples)
            progress = tqdm(
                dataset,
                total=total,
                desc=f"{task} @ {speed_key}",
                unit="sample",
            )
            for i, example in enumerate(progress):
                if max_samples is not None and i >= max_samples:
                    break

                ref_text = example[text_column].replace("\n", " ").strip()
                audio = example[audio_column]
                sped_up_audio = change_audio_speed(audio["array"], speed)
                buffer = waveform_to_in_memory_wav(
                    sped_up_audio, audio["sampling_rate"]
                )
                pred_text = backend.transcribe(buffer)

                sample_wer = wer(
                    ref_text,
                    pred_text,
                    reference_transform=norm,
                    hypothesis_transform=norm,
                )

                refs.append(ref_text)
                preds.append(pred_text)

                sample_result = {
                    "source": example.get(source_column, f"sample_{i}")
                    if source_column
                    else f"sample_{i}",
                    "wer": sample_wer,
                    "pred": pred_text,
                }
                results["results"][speed_key]["samples"].append(sample_result)

                progress.set_postfix(wer=f"{sample_wer:.3f}")
            progress.close()

            overall_wer = wer(
                refs, preds, reference_transform=norm, hypothesis_transform=norm
            )
            results["results"][speed_key]["wer"] = overall_wer

            print("\n" + "=" * 100)
            print("RESULTS")
            print("=" * 100)
            print(f"Dataset     : {dataset_label} [{split}]")
            print(f"Model       : {model}")
            print(f"Speed       : {speed_key}")
            print(f"Overall WER : {overall_wer:.4f}")
            print("=" * 100)

        output_path = cache.save_result(results)
        print(f"Saved results to: {output_path}")

    return cache.load_results(models=[model], tasks=task_names)
