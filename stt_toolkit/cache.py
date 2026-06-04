import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ResultCollection:
    results: list[dict[str, Any]]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.to_rows())

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.results:
            metadata = item.get("metadata", {})
            result_data = item.get("results", {})
            base_row = {
                "model": metadata.get("model"),
            }

            if self._is_speed_result(result_data):
                for speed, speed_result in result_data.items():
                    rows.append(
                        {
                            "task_name": metadata.get("task")
                            or metadata.get("dataset"),
                            **base_row,
                            "speed": speed,
                            "wer": speed_result.get("wer"),
                        }
                    )
            elif self._is_latency_result(result_data):
                rows.append(
                    {
                        **base_row,
                        "audio_file": metadata.get("audio_file"),
                        "audio_length_s": metadata.get("audio_length_s"),
                        "requests": metadata.get("requests"),
                        "concurrency": metadata.get("concurrency"),
                        "avg_latency_s": result_data.get("avg_latency_s"),
                        "rps": result_data.get("rps"),
                        "rtf": result_data.get("rtf"),
                        "errors": item.get("errors"),
                    }
                )
            else:
                rows.append(
                    {
                        **base_row,
                        "speed": None,
                        "wer": result_data.get("wer"),
                    }
                )
        return rows

    @staticmethod
    def _is_speed_result(result_data: dict[str, Any]) -> bool:
        return any(
            isinstance(value, dict) and "wer" in value for value in result_data.values()
        )

    @staticmethod
    def _is_latency_result(result_data: dict[str, Any]) -> bool:
        return {"avg_latency_s", "rps", "rtf"}.issubset(result_data)


class ResultCache:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root).expanduser()

    def has_result(self, model: str, task: str) -> bool:
        return self.result_path(model, task).exists()

    def result_path(self, model: str, task: str) -> Path:
        return (
            self.root
            / "wer_bench"
            / _model_dir(model)
            / f"{_safe_filename(str(task))}.json"
        )

    def save_result(self, result: dict[str, Any]) -> Path:
        metadata = result.setdefault("metadata", {})
        metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("benchmark", "wer_bench")

        model = metadata["model"]
        task = metadata.get("task") or metadata.get("dataset") or "unknown_task"
        output_path = self.result_path(model, task)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def load_results(
        self,
        models: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> ResultCollection:
        model_filter = set(models or [])
        task_filter = set(tasks or [])
        loaded: list[dict[str, Any]] = []

        for path in sorted(self.root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            model = metadata.get("model")
            task = metadata.get("task") or metadata.get("dataset")
            if model_filter and model not in model_filter:
                continue
            if task_filter and task not in task_filter:
                continue

            payload["_path"] = str(path)
            loaded.append(payload)

        return ResultCollection(loaded)


class SpeedResultCache:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root).expanduser()

    def result_path(
        self,
        model: str,
        audio_file: str,
        requests: int,
        concurrency: int,
        audio_length_s: float,
    ) -> Path:
        audio_name = Path(audio_file).stem
        filename = (
            f"{_safe_filename(audio_name)}_"
            f"{requests}reqs_"
            f"{concurrency}concs_"
            f"{audio_length_s:.0f}s.json"
        )
        return self.root / "batched_transcription_bench" / _model_dir(model) / filename

    def has_result(
        self,
        model: str,
        audio_file: str,
        requests: int,
        concurrency: int,
        audio_length_s: float,
    ) -> bool:
        return self.result_path(
            model=model,
            audio_file=audio_file,
            requests=requests,
            concurrency=concurrency,
            audio_length_s=audio_length_s,
        ).exists()

    def save_result(self, result: dict[str, Any]) -> Path:
        metadata = result.setdefault("metadata", {})
        metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("benchmark", "batched_transcription_bench")

        output_path = self.result_path(
            model=metadata["model"],
            audio_file=metadata["audio_file"],
            requests=metadata["requests"],
            concurrency=metadata["concurrency"],
            audio_length_s=metadata["audio_length_s"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def load_results(
        self,
        models: list[str] | None = None,
    ) -> ResultCollection:
        model_filter = set(models or [])
        loaded: list[dict[str, Any]] = []

        for path in sorted((self.root / "batched_transcription_bench").rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            model = metadata.get("model")
            if model_filter and model not in model_filter:
                continue

            payload["_path"] = str(path)
            loaded.append(payload)

        return ResultCollection(loaded)


def _safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _model_dir(model: str) -> str:
    model_parts = str(model).split("/", maxsplit=1)
    if len(model_parts) == 2:
        return f"{_safe_filename(model_parts[0])}__{_safe_filename(model_parts[1])}"
    return _safe_filename(str(model))
