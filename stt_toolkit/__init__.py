"""Utilities for running and evaluating STT models."""

from stt_toolkit.cache import (
    BatchSpeedResultCache,
    ResultCache,
    ResultCollection,
    SpeedResultCache,
)

__all__ = [
    "BatchSpeedResultCache",
    "ResultCache",
    "ResultCollection",
    "SpeedResultCache",
    "evaluate",
]


def __getattr__(name: str):
    if name == "evaluate":
        from stt_toolkit.evaluate import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
