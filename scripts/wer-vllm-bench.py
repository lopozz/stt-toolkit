import time
import subprocess
from openai import OpenAI
import httpx

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_PATH = "/home/lpozzi/Git/stt-benchmark/data/accenti_italiani/audio/abruzzo.wav"


def model_is_ready():
    try:
        r = httpx.get(f"{BASE_URL}/models", timeout=2.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return MODEL in models
    except Exception:
        return False


if not model_is_ready():
    subprocess.run(
        ["scripts/start-vllm.sh", MODEL],
        check=True,
    )

    for _ in range(60):
        if model_is_ready():
            break
        print("vLLM is loading...")
        time.sleep(2)
    else:
        raise RuntimeError("vLLM server did not become ready")

client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
)

with open(AUDIO_PATH, "rb") as f:
    transcription = client.audio.transcriptions.create(
        model=MODEL,
        file=f,
    )

print(transcription.text)
