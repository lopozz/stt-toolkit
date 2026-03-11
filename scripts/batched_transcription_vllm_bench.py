"""
Benchmark a vLLM OpenAI-compatible audio transcription endpoint in an
offline batched setting.

What the script does
--------------------
1. Reads a local WAV file once.
2. Extracts its duration and sample rate from the WAV header.
3. Sends `N` transcription requests to the endpoint.
4. Limits the number of in-flight requests to `concurrency`.
5. Records latency and failures for each request.
6. Aggregates summary metrics.
7. Saves the results as a JSON file, one run per file.

Metrics
-------
- "Mean Latency (s)":
    Average end-to-end latency across successful requests only.
- "RPS":
    Requests per second, computed as:
        successful_requests / total_wall_time
- "RTF":
    In this script, RTF is defined as:
        total_audio_processed_seconds / total_wall_time
    where:
        total_audio_processed_seconds = recording_length_seconds * successful_requests

    This tells you how many audio-seconds are processed per wall-clock second.
    Higher is better.
"""

import os
import json
import time
import wave
import asyncio
import argparse

from pathlib import Path
from openai import AsyncOpenAI
from datetime import datetime, timezone

LOCAL_AUDIO_FILE = "./data/network_1976/mad_as_hell.wav"


async def single_request(
    client: AsyncOpenAI,
    model: str,
    audio_bytes: bytes,
) -> dict:
    """Send one transcription request and return timing/error info."""
    start = time.perf_counter()
    error = None

    try:
        response = await client.audio.transcriptions.create(
            model=model,
            file=("audio.wav", audio_bytes),
        )
        _ = response.text
    except Exception as exc:
        error = str(exc)
        print(error)

    latency = time.perf_counter() - start
    return {"latency": latency, "error": error}


async def run_benchmark(
    client: AsyncOpenAI,
    model: str,
    audio_bytes: bytes,
    total_requests: int,
    concurrency: int,
) -> list[dict]:
    """
    Simulate offline batched transcription:
    - Fill a semaphore-limited pool up to `concurrency` concurrent requests.
    - Fire all `total_requests` as fast as the pool allows.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request():
        async with semaphore:
            return await single_request(client, model, audio_bytes)

    tasks = [asyncio.create_task(bounded_request()) for _ in range(total_requests)]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % max(1, total_requests // 10) == 0 or completed == total_requests:
            print(f"  Progress: {completed}/{total_requests}", flush=True)

    return results


def compute_metrics(
    results: list[dict],
    model: str,
    recording_length_s: float,
    total_requests: int,
    concurrency: int,
    wall_time_s: float,
) -> dict:
    latencies = [r["latency"] for r in results if r["error"] is None]
    errors = sum(1 for r in results if r["error"] is not None)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    rps = len(latencies) / wall_time_s if wall_time_s > 0 else 0.0

    total_audio_s = recording_length_s * len(latencies)
    print(wall_time_s)
    rtf = total_audio_s / wall_time_s if wall_time_s > 0 else 0.0

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "audio_length_s": round(recording_length_s, 3),
            "requests": total_requests,
            "concurrency": concurrency,
        },
        "results": {
            "avg_latency_s": round(mean_latency, 4),
            "rps": round(rps, 4),
            "rtf": round(rtf, 4),
        },
        "errors": errors,
    }


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

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(str(audio_path), "rb") as f:
        audio_bytes = f.read()

    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        recording_length_s = n_frames / sample_rate

    print(f"Audio file: {audio_path}")
    print(
        f"Audio duration: {recording_length_s:.2f}s  |  Sample rate: {sample_rate} Hz"
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    for requests in args.requests:
        for concurrency in args.concurrency:
            print(
                f"\nRunning benchmark: "
                f"model={args.model}, requests={requests}, concurrency={concurrency}"
            )
            wall_start = time.perf_counter()
            results = await run_benchmark(
                client=client,
                model=args.model,
                audio_bytes=audio_bytes,
                total_requests=requests,
                concurrency=concurrency,
            )
            wall_time = time.perf_counter() - wall_start

            metrics = compute_metrics(
                results,
                model=args.model,
                recording_length_s=recording_length_s,
                total_requests=requests,
                concurrency=concurrency,
                wall_time_s=wall_time,
            )

            print("\n--- Results ---")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

            safe_model_name = args.model.replace("/", "_").replace(":", "_")
            output_dir = os.path.join(args.output_dir, "batched_transcription_bench")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"{safe_model_name}_{requests}reqs_{concurrency}concs_{recording_length_s:.0f}s.json",
            )

            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
