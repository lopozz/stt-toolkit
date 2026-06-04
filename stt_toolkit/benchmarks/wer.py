from datetime import datetime, timezone

import numpy as np
from datasets import load_dataset
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


def evaluate_wer(
    model: str,
    tasks: list[dict],
    cache: ResultCache,
    backend,
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
        split = task_config.get("split", "train")
        task_speeds = task_config.get("speeds", speeds)
        task = f"{dataset_name}[{split}]"
        task_names.append(task)

        if cache.has_result(model, task) and not overwrite:
            print(f"Skipping cached result: model={model}, task={task}")
            print(f"Cached file: {cache.result_path(model, task)}")
            continue

        print(f"Loading dataset: {dataset_name} [{split}]")
        dataset = load_dataset(dataset_name, split=split)

        results = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "task": task,
                "dataset": dataset_name,
                "split": split,
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

            for i, example in enumerate(dataset):
                ref_text = example["text"].replace("\n", " ").strip()
                audio = example["audio"]
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
                    "source": example.get("source", f"sample_{i}"),
                    "wer": sample_wer,
                    "pred": pred_text,
                }
                results["results"][speed_key]["samples"].append(sample_result)

                print(
                    f"[{i + 1}/{len(dataset)}] {sample_result['source']}  "
                    f"speed={speed_key}  WER={sample_wer:.3f}"
                )

            overall_wer = wer(
                refs, preds, reference_transform=norm, hypothesis_transform=norm
            )
            results["results"][speed_key]["wer"] = overall_wer

            print("\n" + "=" * 100)
            print("RESULTS")
            print("=" * 100)
            print(f"Dataset     : {dataset_name} [{split}]")
            print(f"Model       : {model}")
            print(f"Speed       : {speed_key}")
            print(f"Overall WER : {overall_wer:.4f}")
            print("=" * 100)

        output_path = cache.save_result(results)
        print(f"Saved results to: {output_path}")

    return cache.load_results(models=[model], tasks=task_names)
