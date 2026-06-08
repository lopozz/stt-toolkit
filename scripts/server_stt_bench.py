"""
Benchmark a vLLM OpenAI-compatible audio transcription endpoint.

This CLI is a thin wrapper around `stt_toolkit.benchmarks.speed`.
"""

import argparse
import asyncio

from stt_toolkit.benchmarks.speed import run_server_speed_benchmark
from stt_toolkit.backends import VllmBackend
from stt_toolkit.cache import BatchSpeedResultCache

LOCAL_AUDIO_FILE = "./data/network_1976/mad_as_hell.wav"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM transcription endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as registered in vLLM",
    )
    parser.add_argument(
        "--requests",
        type=int,
        nargs="+",
        default=[50],
        help="Total number of transcription requests to send (one or more values)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[8],
        help="Max concurrent requests (one or more values)",
    )
    parser.add_argument(
        "--audio-file",
        default=LOCAL_AUDIO_FILE,
        help="Path to a local WAV file",
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


async def main():
    args = parse_args()
    cache = BatchSpeedResultCache(args.output_dir)
    backend = VllmBackend(model=args.model, base_url=args.base_url)
    tasks = [
        {
            "audio_file": args.audio_file,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "overwrite": args.overwrite,
        }
    ]

    await run_server_speed_benchmark(
        model=args.model,
        tasks=tasks,
        cache=cache,
        backend=backend,
        request_counts=args.requests,
        concurrency_values=args.concurrency,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    asyncio.run(main())
