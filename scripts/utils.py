import re
import io

import numpy as np
import soundfile as sf


def safe_filename(text):
    text = text.split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def waveform_to_in_memory_wav(waveform, sr, name="audio.wav"):
    """
    Writes a waveform array into an in-memory WAV file, making possible
    to send it to an API without saving anything to disk.
    """
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(waveform, dtype=np.float32), sr, format="WAV")
    buffer.name = name
    buffer.seek(0)
    return buffer
