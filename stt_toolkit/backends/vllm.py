from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI


def model_is_ready(base_url: str, model: str, api_key: str | None = None) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(f"{base_url}/models", headers=headers, timeout=2.0)
    except httpx.RequestError as e:
        print(f"Could not reach {base_url}/models: {e}")
        return False

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"{base_url}/models returned {response.status_code} {response.reason_phrase}: "
            f"{response.text}. Check that VLLM_API_KEY is set in this shell and matches the "
            "key the container was started with."
        )
    if response.status_code != 200:
        print(f"{base_url}/models returned {response.status_code}: {response.text}")
        return False

    models = [item["id"] for item in response.json().get("data", [])]
    return model in models


class VllmBackend:
    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def transcribe(self, audio_file: Any) -> str:
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
        )
        return response.text.strip()

    async def transcribe_bytes(
        self, audio_bytes: bytes, filename: str = "audio.wav"
    ) -> str:
        response = await self.async_client.audio.transcriptions.create(
            model=self.model,
            file=(filename, audio_bytes),
        )
        return response.text.strip()
