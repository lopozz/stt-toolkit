"""
Benchmark transcription accuracy for vLLM-hosted STT models.

This CLI starts a vLLM backend when needed, then runs the generic
`stt_toolkit.evaluate(..., benchmark="wer")` entrypoint.
"""

import argparse
import os
import subprocess
import sys
import time

import stt_toolkit
import yaml

from stt_toolkit import ResultCache
from stt_toolkit.backends.vllm import VllmBackend, model_is_ready
from stt_toolkit.config import Config


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun benchmarks even when a cached result already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache = ResultCache(args.output_dir)
    task = {
        "dataset": args.dataset,
        "split": args.split,
        "speeds": args.speeds,
    }
    task_name = f"{args.dataset}[{args.split}]"

    for config_path in args.configs:
        started_here = False
        if not os.path.exists(config_path):
            print(f"Skipping: {config_path} (File not found)")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.model_validate(yaml.safe_load(f) or {})

        model = cfg.model
        if cache.has_result(model, task_name) and not args.overwrite:
            print(f"Skipping cached result: model={model}, task={task_name}")
            print(f"Cached file: {cache.result_path(model, task_name)}")
            continue

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

            backend = VllmBackend(model=model, base_url=args.base_url)
            stt_toolkit.evaluate(
                model=model,
                tasks=[task],
                cache=cache,
                backend=backend,
                benchmark="wer",
                kwargs={"overwrite": args.overwrite},
            )

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

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
