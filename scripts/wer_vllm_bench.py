import argparse
import io
import os
import re
import sys
import time
import json
import yaml
import subprocess
from datetime import datetime, timezone

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI
from datasets import load_dataset
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

    return parser.parse_args()


def model_is_ready(base_url, model):
    try:
        r = httpx.get(f"{base_url}/models", timeout=2.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return model in models
    except Exception:
        return False


def safe_filename(text):
    text = text.split("/")[1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
        results = {
            "metadata": {
                "dataset": args.dataset,
                "split": args.split,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "results": {"wer": None, "samples": []},
        }

        with open(config_path, "r", encoding="utf-8") as f:
            model = yaml.safe_load(f)["model"]

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

        refs, preds = [], []

        for i, example in enumerate(ds):
            ref_text = example["text"].replace("\n", " ").strip()
            audio = example["audio"]

            buffer = io.BytesIO()
            sf.write(
                buffer,
                np.asarray(audio["array"], dtype=np.float32),
                audio["sampling_rate"],
                format="WAV",
            )
            buffer.name = "audio.wav"
            buffer.seek(0)

            response = client.audio.transcriptions.create(
                model=model,
                file=buffer,
            )
            pred_text = response.text.strip()

            sample_wer = wer(
                ref_text, pred_text, reference_transform=norm, hypothesis_transform=norm
            )

            refs.append(ref_text)
            preds.append(pred_text)

            sample_result = {
                "source": example.get("source", f"sample_{i}"),
                "wer": sample_wer,
                "pred": pred_text,
            }
            results["results"]["samples"].append(sample_result)

            print(
                f"[{i + 1}/{len(ds)}] {sample_result['source']}  WER={sample_wer:.3f}"
            )

        overall_wer = wer(
            refs, preds, reference_transform=norm, hypothesis_transform=norm
        )
        results["results"]["wer"] = overall_wer

        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)
        print(f"Dataset     : {args.dataset} [{args.split}]")
        print(f"Model       : {model}")
        print(f"Overall WER : {overall_wer:.4f}")
        print("=" * 100)

        subprocess.run(
            [sys.executable, os.path.join("scripts", "stop_vllm.py"), config_path],
            check=False,
            capture_output=True,
            text=True,
        )
        time.sleep(5)

        dataset_name = safe_filename(args.dataset)
        model_name = safe_filename(model)
        output_path = os.path.join(
            args.output_dir, f"{dataset_name}--{model_name}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved results to: {output_path}")

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
