from stt_toolkit.benchmarks.wer import evaluate_wer


def evaluate(
    model: str,
    tasks: list[dict],
    cache,
    backend,
    benchmark: str,
    kwargs: dict | None = None,
):
    kwargs = kwargs or {}

    if benchmark == "wer":
        return evaluate_wer(
            model=model,
            tasks=tasks,
            cache=cache,
            backend=backend,
            **kwargs,
        )

    raise ValueError(f"Unsupported benchmark: {benchmark}")
