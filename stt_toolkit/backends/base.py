from typing import Protocol


class STTBackend(Protocol):
    def transcribe(self, audio_file) -> str:
        pass
