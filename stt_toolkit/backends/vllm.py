from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI


def model_is_ready(base_url: str, model: str) -> bool:
    try:
        response = httpx.get(f"{base_url}/models", timeout=2.0)
        response.raise_for_status()
        models = [item["id"] for item in response.json().get("data", [])]
        return model in models
    except Exception:
        return False


class VllmBackend:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.async_client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

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
