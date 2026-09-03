#!/usr/bin/env python
"""
Stream individual clips out of bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual
without downloading the dataset.

The manifests are tiny JSONL files; each record points at its audio with
`audio_zip_filepath: "<cluster-path>/<name>.zip:<offset>:<length>"`. The zip
itself lives next to the manifest in the same repo directory, so we can fetch
just the bytes we need with an HTTP Range request instead of downloading the
whole (often multi-GB) zip shard.
"""

import argparse
import io
import json
import os
import posixpath
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import datasets
import requests
import soundfile as sf
from huggingface_hub import get_token, hf_hub_download, hf_hub_url

REPO_ID = "bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual"

# Verified by hand (see conversation history): each is the final, fully-filtered
# manifest for that source's Italian config. NOTE: yodas/it000 also has a
# `..._translated_processed.json` sibling whose `whisper_transcript` was
# overwritten by an EN translation pass - do not use it for ASR training.
IT_MANIFESTS = {
    "mcv": (
        "mozilla-foundation/common_voice_17_0/it/train_concatenated/"
        "train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc_l31_processed.json"
    ),
    "mls": (
        "facebook/multilingual_librispeech/italian/train_concatenated/"
        "train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
    "voxpopuli": (
        "facebook/voxpopuli/it/train_concatenated/"
        "train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
    "mtedx": (
        "multilingual-tedx/it-it/train_concatenated/"
        "train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
    "yodas-it000": (
        "espnet/yodas/it000/train_concatenated/"
        "train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
    "yodas-it100": (
        "espnet/yodas/it100/train_concatenated/"
        "train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
    "yodas-it101": (
        "espnet/yodas/it101/train_concatenated/"
        "train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json"
    ),
}

FEATURES = datasets.Features(
    {
        "audio": datasets.Audio(sampling_rate=16000),
        "text": datasets.Value("string"),
        "whisper_transcript": datasets.Value("string"),
        # NOTE: upstream run_distillation.py has a bug consuming these two
        # columns (see the PATCH notes in run_distillation_patch.py) - use
        # that patched script, not the vendored original, when training on
        # data exposing this column, otherwise it will crash.
        "condition_on_prev": datasets.Value("bool"),
        "prev_text": datasets.Value("string"),
        "prev_whisper_transcript": datasets.Value("string"),
        "speaker_id": datasets.Value("string"),
        "duration": datasets.Value("float64"),
        "wer": datasets.Value("float64"),
        "id": datasets.Value("string"),
        "source": datasets.Value("string"),
    }
)


def resolve_zip_path(manifest_path: str, audio_zip_filepath: str) -> tuple[str, int, int]:
    """Map a record's `audio_zip_filepath` to (repo-relative zip path, offset, length).

    The zip lives under the same directory as the manifest, but not always
    directly in it - larger sources (e.g. yodas-it100/it101) shard it one
    level deeper (`train_concatenated/00000008/00000004.zip` instead of
    `train_concatenated/00000009.zip`). So instead of just taking the zip's
    basename, keep everything from the manifest's own directory name onward.
    """
    cluster_path, offset, length = audio_zip_filepath.rsplit(":", 2)
    manifest_dir = posixpath.dirname(manifest_path)
    anchor = posixpath.basename(manifest_dir)

    marker = f"/{anchor}/"
    idx = cluster_path.rfind(marker)
    if idx == -1:
        raise ValueError(
            f"Could not find manifest dir {anchor!r} in audio_zip_filepath {audio_zip_filepath!r}"
        )
    relative_suffix = cluster_path[idx + 1 :]  # e.g. "train_concatenated/00000008/00000004.zip"
    zip_path = posixpath.join(posixpath.dirname(manifest_dir), relative_suffix)
    return zip_path, int(offset), int(length)


# Shared across all threads/sources so concurrent fetches reuse warm
# (keep-alive) connections instead of each paying a fresh TLS handshake -
# measured ~1.2s/request cold vs ~0.7-0.9s reused. pool_maxsize sized generously
# for interleaving many sources at once, each with its own thread pool.
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def fetch_audio_bytes(
    repo_id: str,
    zip_path: str,
    offset: int,
    length: int,
    max_retries: int = 5,
    timeout: float = 30.0,
) -> bytes:
    url = hf_hub_url(repo_id, filename=zip_path, repo_type="dataset")
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error = None
    for attempt in range(max_retries):
        try:
            response = _session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            last_error = e
            wait = min(2**attempt, 30)
            print(
                f"fetch_audio_bytes: attempt {attempt + 1}/{max_retries} failed for "
                f"{zip_path} ({e}); retrying in {wait}s"
            )
            time.sleep(wait)

    raise last_error


def iter_manifest(repo_id: str, manifest_path: str):
    print(f"[{manifest_path}] downloading/loading manifest...")
    local_path = hf_hub_download(repo_id, filename=manifest_path, repo_type="dataset")
    print(f"[{manifest_path}] manifest ready at {local_path}, starting to read records")
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _bounded_prefetch(jobs, fetch_fn, max_in_flight=4):
    """
    Run `fetch_fn(job)` for each job in `jobs` using a pool of up to
    `max_in_flight` concurrent threads, yielding (job, result) pairs in the
    SAME order as `jobs` (a sliding window, not a first-done-first-out queue) -
    so callers relying on deterministic ordering (e.g. skip()/take() on the
    resulting stream) are unaffected. Threads are the right tool here since
    each job is just waiting on network I/O (HTTP range requests), not doing
    CPU-bound work, so the GIL isn't a bottleneck.
    """
    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        pending = deque()
        job_iter = iter(jobs)

        def submit_next():
            job = next(job_iter, None)
            if job is None:
                return False
            pending.append((job, executor.submit(fetch_fn, job)))
            return True

        for _ in range(max_in_flight):
            if not submit_next():
                break

        while pending:
            job, future = pending.popleft()
            result = future.result()
            submit_next()
            yield job, result


def stream_pseudo_labeled(
    repo_id: str,
    manifest_path: str,
    wer_threshold: float | None = None,
    max_samples: int | None = None,
    source: str | None = None,
    max_in_flight: int = 4,
):
    """
    Yields dicts with a decoded `audio` (array + sampling_rate) alongside the
    record's original fields (text, whisper_transcript, condition_on_prev, ...).

    Audio bytes for up to `max_in_flight` records are fetched concurrently
    (see `_bounded_prefetch`) instead of one at a time - each fetch is a
    blocking network round-trip, so doing them sequentially means the whole
    pipeline (and eventually the GPU, waiting for batches) sits idle for the
    full latency of every single request, one after another.
    """
    tag = source or manifest_path

    def filtered_jobs():
        seen = 0
        for record in iter_manifest(repo_id, manifest_path):
            seen += 1
            if wer_threshold is not None and record.get("wer", float("inf")) >= wer_threshold:
                if seen % 20 == 0:
                    print(f"[{tag}] scanned {seen} records so far (filtering by wer < {wer_threshold})")
                continue
            zip_path, offset, length = resolve_zip_path(manifest_path, record["audio_zip_filepath"])
            yield record, zip_path, offset, length

    def fetch_job(job):
        _record, zip_path, offset, length = job
        return fetch_audio_bytes(repo_id, zip_path, offset, length)

    n = 0
    for (record, zip_path, offset, length), audio_bytes in _bounded_prefetch(
        filtered_jobs(), fetch_job, max_in_flight=max_in_flight
    ):
        array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        print(f"[{tag}] yielded #{n + 1}: decoded audio ({len(array)} samples @ {sampling_rate}Hz) from {zip_path}")

        yield {
            "audio": {"array": array, "sampling_rate": sampling_rate},
            "text": record.get("text"),
            "whisper_transcript": record.get("whisper_transcript"),
            "condition_on_prev": bool(record.get("condition_on_prev", False)),
            "prev_text": record.get("prev_text") or "",
            "prev_whisper_transcript": record.get("prev_whisper_transcript") or "",
            "speaker_id": str(record.get("speaker_id")),
            "duration": float(record.get("duration", 0.0)),
            "wer": float(record.get("wer", 0.0)),
            "id": str(record.get("id")),
            "source": source or "",
        }

        n += 1
        if max_samples is not None and n >= max_samples:
            return


def to_iterable_dataset(
    repo_id: str,
    manifest_path: str,
    wer_threshold: float | None = None,
    max_samples: int | None = None,
    source: str | None = None,
    max_in_flight: int = 4,
) -> datasets.IterableDataset:
    """Wrap `stream_pseudo_labeled` as an IterableDataset with a fixed schema,
    so multiple sources can be interleaved via `datasets.interleave_datasets`."""
    return datasets.IterableDataset.from_generator(
        stream_pseudo_labeled,
        features=FEATURES,
        gen_kwargs={
            "repo_id": repo_id,
            "manifest_path": manifest_path,
            "wer_threshold": wer_threshold,
            "max_samples": max_samples,
            "source": source,
            "max_in_flight": max_in_flight,
        },
    )


def stream_it_multilingual(
    repo_id: str = REPO_ID,
    manifests: dict[str, str] | None = None,
    wer_threshold: float | None = 20.0,
    probabilities: dict[str, float] | None = None,
    seed: int | None = 42,
    stopping_strategy: str = "first_exhausted",
    max_in_flight: int = 4,
) -> datasets.IterableDataset:
    """
    Interleave all (or a chosen subset of) the Italian sources into a single
    streaming IterableDataset, matching how `run_distillation.py`'s
    `load_multiple_datasets` combines multiple HF datasets - just backed by
    our range-request loader instead of `datasets.load_dataset`.

    `probabilities` (per source name, must sum to 1) controls sampling weight;
    left as None, `interleave_datasets` samples each source with equal
    probability regardless of its size (same default as upstream when no
    `--train_dataset_samples` is given).

    `max_in_flight` is the number of concurrent audio fetches PER SOURCE - with
    7 sources all active during interleaving, actual peak concurrency across
    the whole pipeline is roughly `7 * max_in_flight` connections.
    """
    manifests = manifests or IT_MANIFESTS
    sources = list(manifests.keys())
    print(f"[stream_it_multilingual] interleaving sources: {sources} (stopping_strategy={stopping_strategy})")

    per_source_datasets = [
        to_iterable_dataset(
            repo_id, manifests[name], wer_threshold=wer_threshold, source=name, max_in_flight=max_in_flight
        )
        for name in sources
    ]

    probs = [probabilities[name] for name in sources] if probabilities else None

    return datasets.interleave_datasets(
        per_source_datasets,
        probabilities=probs,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Repo-relative path to a manifest JSON, e.g. "
        "multilingual-tedx/it-it/train_concatenated/"
        "train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc_l31_processed.json",
    )
    parser.add_argument("--wer-threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--save-audio-dir",
        default=None,
        help="If set, also write each streamed clip's audio as a .wav file here.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_audio_dir:
        os.makedirs(args.save_audio_dir, exist_ok=True)

    for i, example in enumerate(
        stream_pseudo_labeled(
            args.repo_id,
            args.manifest,
            wer_threshold=args.wer_threshold,
            max_samples=args.max_samples,
        )
    ):
        audio = example["audio"]
        print(f"[{i}] id={example.get('id')} duration={example.get('duration')}s wer={example.get('wer')}")
        print(f"    text:               {example.get('text')}")
        print(f"    whisper_transcript: {example.get('whisper_transcript')}")
        print(f"    condition_on_prev:  {example.get('condition_on_prev')}  prev_text: {example.get('prev_text')!r}")

        if args.save_audio_dir:
            out_path = f"{args.save_audio_dir}/{i:03d}_{example.get('id')}.wav"
            sf.write(out_path, audio["array"], audio["sampling_rate"])
            print(f"    saved audio -> {out_path}")


if __name__ == "__main__":
    main()
