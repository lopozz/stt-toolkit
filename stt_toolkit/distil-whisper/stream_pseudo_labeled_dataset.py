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

import requests
import soundfile as sf
from huggingface_hub import get_token, hf_hub_download, hf_hub_url

REPO_ID = "bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual"


def resolve_zip_path(manifest_path: str, audio_zip_filepath: str) -> tuple[str, int, int]:
    """Map a record's `audio_zip_filepath` to (repo-relative zip path, offset, length)."""
    cluster_path, offset, length = audio_zip_filepath.rsplit(":", 2)
    zip_name = posixpath.basename(cluster_path)
    zip_path = posixpath.join(posixpath.dirname(manifest_path), zip_name)
    return zip_path, int(offset), int(length)


def fetch_audio_bytes(repo_id: str, zip_path: str, offset: int, length: int) -> bytes:
    url = hf_hub_url(repo_id, filename=zip_path, repo_type="dataset")
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


def iter_manifest(repo_id: str, manifest_path: str):
    local_path = hf_hub_download(repo_id, filename=manifest_path, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_pseudo_labeled(
    repo_id: str,
    manifest_path: str,
    wer_threshold: float | None = None,
    max_samples: int | None = None,
):
    """
    Yields dicts with a decoded `audio` (array + sampling_rate) alongside the
    record's original fields (text, whisper_transcript, condition_on_prev, ...).
    """
    n = 0
    for record in iter_manifest(repo_id, manifest_path):
        if wer_threshold is not None and record.get("wer", float("inf")) >= wer_threshold:
            continue

        zip_path, offset, length = resolve_zip_path(manifest_path, record["audio_zip_filepath"])
        audio_bytes = fetch_audio_bytes(repo_id, zip_path, offset, length)
        array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

        yield {**record, "audio": {"array": array, "sampling_rate": sampling_rate}}

        n += 1
        if max_samples is not None and n >= max_samples:
            return


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
