import argparse
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

LOCAL_AUDIO_DIR = "./data/network_1976"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch transcribe all WAV files in a directory using a vLLM OpenAI-compatible endpoint"
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
        "--src-lang",
        default=None,
        help="Language of the input audio, e.g. en, it, fr",
    )
    parser.add_argument(
        "--tgt-lang",
        default=None,
        help="Desired output language. If omitted or equal to source, plain transcription is used.",
    )
    parser.add_argument(
        "--input-dir",
        default=LOCAL_AUDIO_DIR,
        help="Directory containing WAV files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/transcriptions",
        help="Directory where .txt transcripts are saved",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent transcription requests",
    )
    return parser.parse_args()


async def transcribe_audio_file(
    client: AsyncOpenAI,
    model: str,
    audio_path: Path,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
) -> str:
    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    file_obj = (audio_path.name, audio_bytes)
    if tgt_lang is None or tgt_lang == src_lang:
        response = await client.audio.transcriptions.create(
            model=model,
            file=file_obj,
            language=src_lang,
        )
    else:
        response = await client.audio.translations.create(
            model=model,
            file=file_obj,
            extra_body=dict(
                language=src_lang,
                to_language=tgt_lang,
            ),
        )

    return response.text


async def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(wav_files)} WAV files in {input_dir}")
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(src: Path):
        async with sem:
            rel = src.relative_to(input_dir)
            dst = output_dir / rel.with_suffix(".txt")
            dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                text = await transcribe_audio_file(
                    client,
                    args.model,
                    src,
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang,
                )
                dst.write_text(text, encoding="utf-8")
                print(f"[ok] {rel} -> {dst.relative_to(output_dir)}")
                return True
            except Exception as exc:
                print(f"[err] {rel}: {exc}")
                return False

    results = await asyncio.gather(*(worker(src) for src in wav_files))
    ok = sum(results)
    err = len(results) - ok
    print(f"Done. OK={ok} ERR={err} Total={len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
