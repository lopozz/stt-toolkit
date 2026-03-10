import argparse
import time
import subprocess

import httpx
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM endpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model name to use",
    )
    parser.add_argument(
        "--audio-path",
        default="/home/lpozzi/Git/stt-benchmark/data/accenti_italiani/audio/abruzzo.wav",
        help="Path to audio file",
    )

    return parser.parse_args()


def model_is_ready(base_url, model):
    try:
        r = httpx.get(f"{base_url}/models", timeout=2.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return model in models
    except Exception:
        return False


def main():
    args = parse_args()

    if not model_is_ready(args.base_url, args.model):
        subprocess.run(
            ["scripts/start-vllm.sh", args.model],
            check=True,
        )

        for _ in range(args.timeout // 2):
            if model_is_ready(args.base_url, args.model):
                break
            print("vLLM is loading...")
            time.sleep(2)
        else:
            raise RuntimeError("vLLM server did not become ready")

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    with open(args.audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=args.model,
            file=f,
        )

    print(transcription.text)


if __name__ == "__main__":
    main()
