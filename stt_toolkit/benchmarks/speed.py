import time
import wave
import asyncio

from pathlib import Path
from datetime import datetime, timezone


from stt_toolkit.backends.vllm import VllmBackend
from stt_toolkit.cache import SpeedResultCache


async def single_request(
    backend: VllmBackend,
    audio_bytes: bytes,
) -> dict:
    start = time.perf_counter()
    error = None

    try:
        _ = await backend.transcribe_bytes(audio_bytes)
    except Exception as exc:
        error = str(exc)
        print(error)

    latency = time.perf_counter() - start
    return {"latency": latency, "error": error}


async def run_benchmark(
    backend: VllmBackend,
    audio_bytes: bytes,
    total_requests: int,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request():
        async with semaphore:
            return await single_request(backend, audio_bytes)

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
    audio_file: str,
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
    rtf = total_audio_s / wall_time_s if wall_time_s > 0 else 0.0

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "audio_file": audio_file,
            "audio_length_s": round(recording_length_s, 3),
            "requests": total_requests,
            "concurrency": concurrency,
            "benchmark": "batched_transcription_bench",
        },
        "results": {
            "avg_latency_s": round(mean_latency, 4),
            "rps": round(rps, 4),
            "rtf": round(rtf, 4),
        },
        "errors": errors,
    }


async def run_speed_benchmark(
    base_url: str,
    model: str,
    request_counts: list[int],
    concurrency_values: list[int],
    audio_file: str,
    output_dir: str,
    overwrite: bool = False,
) -> None:
    cache = SpeedResultCache(output_dir)
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()

    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        recording_length_s = n_frames / sample_rate

    print(f"Audio file: {audio_path}")
    print(
        f"Audio duration: {recording_length_s:.2f}s  |  Sample rate: {sample_rate} Hz"
    )

    backend = VllmBackend(model=model, base_url=base_url)

    for requests in request_counts:
        for concurrency in concurrency_values:
            if (
                cache.has_result(
                    model=model,
                    audio_file=str(audio_path),
                    requests=requests,
                    concurrency=concurrency,
                    audio_length_s=recording_length_s,
                )
                and not overwrite
            ):
                print(
                    f"Skipping cached result: model={model}, "
                    f"requests={requests}, concurrency={concurrency}"
                )
                print(
                    "Cached file: "
                    f"{cache.result_path(model, str(audio_path), requests, concurrency, recording_length_s)}"
                )
                continue

            print(
                f"\nRunning benchmark: "
                f"model={model}, requests={requests}, concurrency={concurrency}"
            )
            wall_start = time.perf_counter()
            results = await run_benchmark(
                backend=backend,
                audio_bytes=audio_bytes,
                total_requests=requests,
                concurrency=concurrency,
            )
            wall_time = time.perf_counter() - wall_start

            metrics = compute_metrics(
                results,
                model=model,
                audio_file=str(audio_path),
                recording_length_s=recording_length_s,
                total_requests=requests,
                concurrency=concurrency,
                wall_time_s=wall_time,
            )

            print("\n--- Results ---")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

            output_path = cache.save_result(metrics)

            print(f"\nResults saved to {output_path}")
