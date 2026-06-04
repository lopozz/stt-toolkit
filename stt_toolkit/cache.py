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
                "task_name": metadata.get("task") or metadata.get("dataset"),
                "model": metadata.get("model"),
            }

            if self._is_speed_result(result_data):
                for speed, speed_result in result_data.items():
                    rows.append(
                        {
                            **base_row,
                            "speed": speed,
                            "wer": speed_result.get("wer"),
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
        return any(isinstance(value, dict) and "wer" in value for value in result_data.values())


class ResultCache:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root).expanduser()

    def save_result(self, result: dict[str, Any]) -> Path:
        metadata = result.setdefault("metadata", {})
        metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("benchmark", "wer_bench")

        model = metadata["model"]
        task = metadata.get("task") or metadata.get("dataset") or "unknown_task"
        model_parts = str(model).split("/", maxsplit=1)
        if len(model_parts) == 2:
            model_dir = f"{_safe_filename(model_parts[0])}__{_safe_filename(model_parts[1])}"
        else:
            model_dir = _safe_filename(str(model))

        output_dir = self.root / "wer_bench" / model_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{_safe_filename(str(task))}.json"
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
        model_filter = {_normalize_model_name(model) for model in models or []}
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
            if model_filter and not (model_filter & _model_aliases(str(model))):
                continue
            if task_filter and task not in task_filter:
                continue

            payload["_path"] = str(path)
            loaded.append(payload)

        return ResultCollection(loaded)


def _safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _model_aliases(model: str) -> set[str]:
    aliases = {model}
    if "/" in model:
        aliases.add(model.rsplit("/", maxsplit=1)[-1])
    aliases.add(_safe_filename(model))
    return {_normalize_model_name(alias) for alias in aliases}


def _normalize_model_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.lower())
