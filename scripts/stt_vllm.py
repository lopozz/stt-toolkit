import json
import asyncio
import argparse

from pathlib import Path
from openai import AsyncOpenAI
from openai.types.audio.translation import Translation
from openai.types.audio.transcription import Transcription
from openai.types.audio.transcription_verbose import TranscriptionVerbose


LOCAL_AUDIO_DIR = "./data/network_1976"

OUTPUT_SUFFIX_BY_FORMAT = {
    "json": ".json",
    "text": ".txt",
    "verbose_json": ".json",
}

SUPPORTED_MODELS_FOR_VERBOSE_JSON = "openai/whisper-large-v3-turbo"


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
        "--response-format",
        choices=["json", "text", "verbose_json"],
        default="json",
        help="Response format",
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
    response_format: str = "json",
) -> str | Transcription | Translation | TranscriptionVerbose:
    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    file_obj = (audio_path.name, audio_bytes)
    if tgt_lang is None or tgt_lang == src_lang:
        response = await client.audio.transcriptions.create(
            model=model,
            file=file_obj,
            language=src_lang,
            response_format=response_format,
        )
    else:
        response = await client.audio.translations.create(
            model=model,
            file=file_obj,
            response_format=response_format,
            extra_body=dict(
                language=src_lang,
                to_language=tgt_lang,
            ),
        )
    return response


def serialize_response(
    response: str | Transcription | Translation | TranscriptionVerbose,
    response_format: str,
) -> str:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json", warnings=False)
        if isinstance(payload.get("duration"), str):
            payload["duration"] = float(payload["duration"])
        if response_format in {"json"}:
            payload = {
                "text": payload["text"]
            }  # Transcription and Translation had different formats
        if response_format in {"json", "verbose_json"}:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    else:
        return json.loads(response)["text"]


async def main():
    args = parse_args()

    if (
        args.response_format == "verbose_json"
        and args.model not in SUPPORTED_MODELS_FOR_VERBOSE_JSON
    ):
        print(
            f"[warn] verbose_json is not available for {args.model}; "
            "defaulting to json."
        )
        args.response_format = "json"

    if (
        args.tgt_lang is not None
        and args.tgt_lang != args.src_lang
        and args.response_format == "verbose_json"
    ):
        print(
            "[warn] verbose_json is not available for translation endpoint; "
            "defaulting to json."
        )
        args.response_format = "json"

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
            suffix = OUTPUT_SUFFIX_BY_FORMAT[args.response_format]
            dst = output_dir / rel.with_suffix(suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                response = await transcribe_audio_file(
                    client,
                    args.model,
                    src,
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang,
                    response_format=args.response_format,
                )
                output_content = serialize_response(response, args.response_format)
                dst.write_text(output_content, encoding="utf-8")
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
