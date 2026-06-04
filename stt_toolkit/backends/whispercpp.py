import json
import subprocess
import tempfile
from pathlib import Path


class WhisperCppBackend:
    def __init__(
        self,
        model_path: str,
        executable: str = "whisper-cli",
        language: str | None = None,
        threads: int | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_path = model_path
        self.executable = executable
        self.language = language
        self.threads = threads
        self.extra_args = extra_args or []

    def transcribe(self, audio_file) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "audio.wav"
            output_prefix = tmp_path / "transcription"

            audio_file.seek(0)
            audio_path.write_bytes(audio_file.read())

            cmd = [
                self.executable,
                "-m",
                self.model_path,
                "-f",
                str(audio_path),
                "--output-json",
                "--output-file",
                str(output_prefix),
                "--no-prints",
            ]

            if self.language:
                cmd.extend(["-l", self.language])
            if self.threads:
                cmd.extend(["-t", str(self.threads)])
            cmd.extend(self.extra_args)

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            return self._parse_json(output_prefix.with_suffix(".json"))

    def _parse_json(self, json_path: Path) -> str:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        transcription = payload.get("transcription", [])
        return " ".join(
            item.get("text", "").strip() for item in transcription if item.get("text")
        ).strip()
