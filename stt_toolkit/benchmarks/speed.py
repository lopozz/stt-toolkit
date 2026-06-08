import io
import time
import wave
import asyncio

from pathlib import Path
from datetime import datetime, timezone

from stt_toolkit.backends import STTBackend, VllmBackend, WhisperCppBackend
from stt_toolkit.cache import SpeedResultCache, BatchSpeedResultCache


async def single_request(
    backend: STTBackend,
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
    backend: STTBackend,
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


def run_speed_benchmark(
    model: str,
    tasks: list[dict],
    cache: SpeedResultCache,
    backend: STTBackend,
    runs: int = 1,
    overwrite: bool = False,
):
    assert isinstance(backend, WhisperCppBackend), (
        "run_speed_benchmark only supports WhisperCppBackend"
    )

    for task in tasks:
        audio_path = Path(task["audio_file"])

        if runs < 1:
            raise ValueError("runs must be >= 1")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            recording_length_s = n_frames / sample_rate

        if (
            cache.has_result(
                model=model,
                audio_file=str(audio_path),
                threads=backend.threads,
            )
            and not overwrite
        ):
            print(f"Skipping cached result: model={model}, audio_file={audio_path}")
            print(
                "Cached file: "
                f"{cache.result_path(model, str(audio_path), backend.threads)}"
            )
            continue

        audio_bytes = audio_path.read_bytes()
        transcription_times_s: list[float] = []
        total_times_s: list[float] = []
        load_times_s: list[float] = []
        timing_runs: list[dict] = []
        errors = 0

        print(f"Audio file: {audio_path}")
        print(
            f"Audio duration: {recording_length_s:.2f}s  |  Sample rate: {sample_rate} Hz"
        )

        for run in range(runs):
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = audio_path.name

            try:
                transcription = backend.transcribe_with_timings(audio_buffer)
            except Exception as exc:
                errors += 1
                print(f"Run {run + 1}/{runs} failed: {exc}")
                continue

            timings = transcription["timings"]
            timing_runs.append(timings)
            transcription_time_ms = timings["transcription_time_ms"]
            total_time_ms = timings["total_time_ms"]
            load_time_ms = timings["load_time_ms"]

            transcription_time_s = transcription_time_ms / 1000
            transcription_times_s.append(transcription_time_s)
            total_times_s.append(total_time_ms / 1000)
            load_times_s.append(load_time_ms / 1000)

            load_display = load_time_ms / 1000
            total_display = total_time_ms / 1000
            print(
                f"Run {run + 1}/{runs}: "
                f"transcription={transcription_time_s:.4f}s, "
                f"load={load_display:.4f}s, "
                f"total={total_display:.4f}s"
            )

        if not transcription_times_s:
            raise RuntimeError(
                "No successful speed benchmark runs with parsed whisper.cpp timings"
            )

        avg_transcription_time_s = sum(transcription_times_s) / len(
            transcription_times_s
        )
        avg_total_time_s = sum(total_times_s) / len(total_times_s)
        avg_load_time_s = sum(load_times_s) / len(load_times_s)
        rtf = recording_length_s / avg_transcription_time_s

        metrics = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "audio_file": str(audio_path),
                "threads": backend.threads,
                "benchmark": "transcription_bench",
            },
            "results": {
                "audio_length_s": round(recording_length_s, 3),
                "avg_latency_s": round(avg_transcription_time_s, 4),
                "avg_load_time_s": round(avg_load_time_s, 4),
                "avg_total_time_s": round(avg_total_time_s, 4),
                "rtf": round(rtf, 4),
            },
            "timings": timing_runs,
            "errors": errors,
        }

        output_path = cache.save_result(metrics)
        print(f"\nResults saved to {output_path}")

    return cache.load_results(models=[model])


async def run_server_speed_benchmark(
    model: str,
    tasks: list[dict],
    cache: BatchSpeedResultCache,
    backend: STTBackend,
    request_counts: list[int] = [50],
    concurrency_values: list[int] = [8],
    overwrite: bool = False,
) -> None:
    assert isinstance(backend, VllmBackend), (
        "run_server_speed_benchmark only supports VllmBackend"
    )

    for task in tasks:
        audio_path = Path(task["audio_file"])

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

    return cache.load_results(models=[model])
