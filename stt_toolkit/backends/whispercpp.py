import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class WhisperCppBackend:
    def __init__(
        self,
        model_path: str,
        language: str | None = None,
        threads: int | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_path = model_path
        self.language = language
        self.threads = threads
        self.extra_args = extra_args or []

    def transcribe(self, audio_file) -> str:
        return self.transcribe_with_timings(audio_file)["text"]

    def transcribe_with_timings(self, audio_file) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "audio.wav"
            output_prefix = tmp_path / "transcription"

            audio_file.seek(0)
            audio_path.write_bytes(audio_file.read())

            cmd = [
                "whisper.cpp/build/bin/whisper-cli",
                "-m",
                self.model_path,
                "-f",
                str(audio_path),
                "--output-json",
                "--output-file",
                str(output_prefix),
            ]

            if self.language:
                cmd.extend(["-l", self.language])
            if self.threads:
                cmd.extend(["-t", str(self.threads)])
            cmd.extend(self.extra_args)

            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    cmd,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )

            return {
                "text": self._parse_json(output_prefix.with_suffix(".json")),
                "timings": self._parse_timings(output),
                "output": output,
            }

    def _parse_json(self, json_path: Path) -> str:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        transcription = payload.get("transcription", [])
        return " ".join(
            item["text"].strip() for item in transcription if item["text"].strip()
        ).strip()

    def _parse_timings(self, output: str) -> dict[str, Any]:
        timings: dict[str, Any] = {}

        for key, label in {
            "load_time_ms": "load time",
            "mel_time_ms": "mel time",
            "sample_time_ms": "sample time",
            "encode_time_ms": "encode time",
            "decode_time_ms": "decode time",
            "batchd_time_ms": "batchd time",
            "prompt_time_ms": "prompt time",
            "total_time_ms": "total time",
        }.items():
            timings[key] = self._extract_time_ms(output, label)

        required_keys = ("load_time_ms", "total_time_ms")
        missing_keys = [key for key in required_keys if timings[key] is None]
        if missing_keys:
            raise ValueError(
                "Could not extract required whisper.cpp timings: "
                f"{', '.join(missing_keys)}\n"
                f"Captured output:\n{output[-4000:]}"
            )

        load_time_ms = timings["load_time_ms"]
        total_time_ms = timings["total_time_ms"]
        timings["transcription_time_ms"] = total_time_ms - load_time_ms

        return timings

    @staticmethod
    def _extract_time_ms(output: str, label: str) -> float | None:
        match = re.search(
            rf"{re.escape(label)}\s*=\s*(\d+(?:\.\d+)?)\s*ms",
            output,
        )
        return float(match.group(1)) if match else None
