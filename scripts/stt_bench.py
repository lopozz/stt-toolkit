"""
Benchmark a local STT backend with single-user latency and RTF metrics.

This CLI is a thin wrapper around `stt_toolkit.benchmarks.speed.run_speed_benchmark`.
"""

import os
import yaml
import argparse

from stt_toolkit.config import Config
from stt_toolkit.cache import SpeedResultCache
from stt_toolkit.backends import WhisperCppBackend
from stt_toolkit.benchmarks.speed import run_speed_benchmark

LOCAL_AUDIO_FILE = "./data/network_1976/mad_as_hell.wav"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a local STT backend with one user request at a time"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="YAML configs, one per model/backend",
    )
    parser.add_argument(
        "--audio-file",
        default=LOCAL_AUDIO_FILE,
        help="Path to a local WAV file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of sequential transcription runs",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        help="Thread counts to benchmark. Defaults to the config value.",
    )
    parser.add_argument(
        "--processors",
        type=int,
        nargs="+",
        help="Processor counts to benchmark. Defaults to the config value.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where JSON result files are saved",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun benchmarks even when a cached result already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache = SpeedResultCache(args.output_dir)

    for config_path in args.configs:
        if not os.path.exists(config_path):
            print(f"Skipping: {config_path} (File not found)")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.model_validate(yaml.safe_load(f) or {})

        try:
            if cfg.backend == "whispercpp":
                tasks = [
                    {
                        "audio_file": args.audio_file,
                        "runs": args.runs,
                        "overwrite": args.overwrite,
                    }
                ]
            else:
                raise ValueError(
                    f"Unsupported backend for local speed benchmark: {cfg.backend}"
                )

            thread_values = args.threads
            processor_values = args.processors

            for threads in thread_values:
                for processors in processor_values:
                    print(
                        "\nRunning config: "
                        f"{config_path} | threads={threads} | processors={processors}"
                    )
                    backend = WhisperCppBackend(
                        model_path=cfg.whispercpp_model_path,
                        language=cfg.language,
                        threads=threads,
                        processors=processors,
                        extra_args=cfg.extra_whispercpp_args,
                    )
                    run_speed_benchmark(
                        model=cfg.model,
                        tasks=tasks,
                        cache=cache,
                        backend=backend,
                        runs=args.runs,
                        overwrite=args.overwrite,
                    )
        except Exception as e:
            print(f"Failed to process {config_path}: {type(e).__name__} - {e}")

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
