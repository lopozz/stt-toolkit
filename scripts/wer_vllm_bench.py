import argparse
import io
import time
import subprocess

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI
from datasets import load_dataset
from jiwer import wer


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
        "--dataset",
        default="lopozz/accenti_italiani",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
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

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)
    # ds = ds.cast_column("audio", Audio(decode=False))

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
    else:
        print("Sending audio data...")

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    refs = []
    preds = []
    rows = []

    for i, example in enumerate(ds):
        ref_text = example["text"].strip()
        audio = example["audio"]

        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.asarray(audio["array"]),
            audio["sampling_rate"],
            format="WAV",
        )
        buffer.name = "audio.wav"
        buffer.seek(0)

        response = client.audio.transcriptions.create(
            model=args.model,
            file=buffer,
        )
        pred_text = response.text.strip()

        sample_wer = wer(ref_text, pred_text)

        refs.append(ref_text)
        preds.append(pred_text)

        rows.append(
            {
                "idx": i,
                "source": example.get("source", f"sample_{i}"),
                "wer": sample_wer,
                "ref": ref_text,
                "pred": pred_text,
            }
        )

        print(f"[{i + 1}/{len(ds)}] {rows[-1]['source']}  WER={sample_wer:.3f}")

    overall_wer = wer(refs, preds)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"Dataset     : {args.dataset}")
    print(f"Split       : {args.split}")
    print(f"Model       : {args.model}")
    print(f"Num samples : {len(rows)}")
    print(f"Overall WER : {overall_wer:.4f}")
    print("=" * 100)

    print("\nPer-sample results:\n")
    for row in rows:
        print("-" * 100)
        print(f"Index : {row['idx']}")
        print(f"Source: {row['source']}")
        print(f"WER   : {row['wer']:.4f}")
        print(f"REF   : {row['ref']}")
        print(f"PRED  : {row['pred']}")
    print("-" * 100)


if __name__ == "__main__":
    main()
