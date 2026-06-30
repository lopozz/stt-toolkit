from typing import Any

import numpy as np
import soundfile as sf
from transformers import pipeline


class TransformersBackend:
    def __init__(
        self,
        model: str,
        device: str = "cuda:0",
        dtype: str = "auto",
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            device=device,
            dtype=dtype,
        )

    def transcribe(self, audio_file: Any) -> str:
        audio_file.seek(0)
        waveform, sampling_rate = sf.read(audio_file, dtype="float32")

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        result = self.pipeline(
            {
                "raw": np.asarray(waveform, dtype=np.float32),
                "sampling_rate": sampling_rate,
            }
        )
        return result["text"].strip()
