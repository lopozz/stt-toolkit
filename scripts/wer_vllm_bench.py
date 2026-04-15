import os
import sys
import time
import json
import yaml
import httpx
import argparse
import subprocess
import numpy as np

from openai import OpenAI
from datasets import load_dataset
from utils import waveform_to_in_memory_wav, safe_filename
from datetime import datetime, timezone

from jiwer import (
    wer,
    Compose,
    ToLowerCase,
    RemovePunctuation,
    RemoveMultipleSpaces,
    Strip,
    ReduceToListOfListOfWords,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM endpoint",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of YAML configs, one per model/server",
    )
    parser.add_argument(
        "--dataset",
        default="lopozz/accenti_italiani",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where JSON results will be saved",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[1.0],
        help="Audio speed factors to evaluate, e.g. --speeds 1.0 2.0",
    )

    return parser.parse_args()


def model_is_ready(base_url, model):
    try:
        r = httpx.get(f"{base_url}/models", timeout=2.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return model in models
    except Exception:
        return False


def change_audio_speed(waveform, speed):
    if speed <= 0:
        raise ValueError(f"Invalid speed: {speed}")
    if speed == 1.0:
        return np.asarray(waveform, dtype=np.float32)

    waveform = np.asarray(waveform, dtype=np.float32)
    new_length = max(1, int(len(waveform) / speed))
    src_positions = np.arange(len(waveform), dtype=np.float32)
    dst_positions = np.linspace(0, len(waveform) - 1, new_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, waveform).astype(np.float32)


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, "wer_bench")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)

    norm = Compose(
        [
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToListOfListOfWords(),
        ]
    )
    for config_path in args.configs:
        started_here = False
        if not os.path.exists(config_path):
            print(f"Skipping: {config_path} (File not found)")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = cfg["model"]

        results = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "dataset": f"{args.dataset}[{args.split}]",
                "speeds": args.speeds,
            },
            "results": {},
        }

        try:
            if not model_is_ready(args.base_url, model):
                print("Model not ready, starting vLLM server...")

                subprocess.run(
                    [
                        sys.executable,
                        os.path.join("scripts", "start_vllm.py"),
                        config_path,
                    ],
                    check=True,
                )

                started_here = True

                for _ in range(60 // 2):
                    if model_is_ready(args.base_url, model):
                        break
                    print("vLLM is loading...")
                    time.sleep(10)
                else:
                    raise RuntimeError("vLLM server did not become ready")

            print("Sending audio data...")

            client = OpenAI(
                base_url=args.base_url,
                api_key="EMPTY",
            )

            for speed in args.speeds:
                refs, preds = [], []
                speed_key = f"{speed:g}x"
                results["results"][speed_key] = {"wer": None, "samples": []}

                print(f"\nEvaluating speed {speed_key}...")

                for i, example in enumerate(ds):
                    ref_text = example["text"].replace("\n", " ").strip()
                    audio = example["audio"]
                    sped_up_audio = change_audio_speed(audio["array"], speed)

                    buffer = waveform_to_in_memory_wav(
                        sped_up_audio, audio["sampling_rate"]
                    )
                    response = client.audio.transcriptions.create(
                        model=model,
                        file=buffer,
                    )
                    pred_text = response.text.strip()

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
                        f"[{i + 1}/{len(ds)}] {sample_result['source']}  "
                        f"speed={speed_key}  WER={sample_wer:.3f}"
                    )

                overall_wer = wer(
                    refs, preds, reference_transform=norm, hypothesis_transform=norm
                )
                results["results"][speed_key]["wer"] = overall_wer

                print("\n" + "=" * 100)
                print("RESULTS")
                print("=" * 100)
                print(f"Dataset     : {args.dataset} [{args.split}]")
                print(f"Model       : {model}")
                print(f"Speed       : {speed_key}")
                print(f"Overall WER : {overall_wer:.4f}")
                print("=" * 100)

        except Exception as e:
            print(f"Failed to process {config_path}: {type(e).__name__} - {e}")

        finally:
            if started_here:
                subprocess.run(
                    [
                        sys.executable,
                        os.path.join("scripts", "stop_vllm.py"),
                        config_path,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                time.sleep(5)

            dataset_name = safe_filename(args.dataset)
            model_name = safe_filename(model)
            output_path = os.path.join(output_dir, f"{dataset_name}--{model_name}.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved results to: {output_path}")

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
