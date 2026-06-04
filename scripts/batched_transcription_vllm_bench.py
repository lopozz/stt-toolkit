"""
Benchmark a vLLM OpenAI-compatible audio transcription endpoint.

This CLI is a thin wrapper around `stt_toolkit.benchmarks.speed`.
"""

import argparse
import asyncio

from stt_toolkit.benchmarks.speed import run_speed_benchmark

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
    return parser.parse_args()


async def main():
    args = parse_args()
    await run_speed_benchmark(
        base_url=args.base_url,
        model=args.model,
        request_counts=args.requests,
        concurrency_values=args.concurrency,
        audio_file=args.audio_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
