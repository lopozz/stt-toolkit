import os
import yaml
import argparse
import stt_toolkit

from stt_toolkit import ResultCache
from stt_toolkit.backends import VllmBackend, WhisperCppBackend, model_is_ready
from stt_toolkit.config import Config, VllmConfig, WhisperCppConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="YAML configs, one per model/backend",
    )
    parser.add_argument(
        "--dataset",
        default="lopozz/accenti_italiani",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where JSON results will be saved",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[1.0],
        help="Audio speed factors to evaluate, e.g. --speeds 1.0 2.0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun benchmarks even when a cached result already exists",
    )
    return parser.parse_args()


def run_vllm(cfg: VllmConfig, config_path: str, args, cache: ResultCache):
    task_name = f"{args.dataset}[{args.split}]"
    if cache.has_result(cfg.model, task_name) and not args.overwrite:
        print(f"Skipping cached result: model={cfg.model}, task={task_name}")
        print(f"Cached file: {cache.result_path(cfg.model, task_name)}")
        return

    if not model_is_ready(cfg.base_url, cfg.model):
        raise RuntimeError(
            f"Model is not ready on {cfg.base_url}: {cfg.model}. "
            f"Start it with: .venv/bin/python scripts/start_vllm.py {config_path}"
        )

    backend = VllmBackend(model=cfg.model, base_url=cfg.base_url)
    task = {
        "dataset": args.dataset,
        "split": args.split,
        "speeds": args.speeds,
    }

    stt_toolkit.evaluate(
        model=cfg.model,
        tasks=[task],
        cache=cache,
        backend=backend,
        benchmark="wer",
        kwargs={"overwrite": args.overwrite},
    )


def run_whispercpp(cfg: WhisperCppConfig, args, cache: ResultCache):
    task_name = f"{args.dataset}[{args.split}]"
    if cache.has_result(cfg.model, task_name) and not args.overwrite:
        print(f"Skipping cached result: model={cfg.model}, task={task_name}")
        print(f"Cached file: {cache.result_path(cfg.model, task_name)}")
        return

    backend = WhisperCppBackend(
        model_path=cfg.whispercpp_model_path,
        executable=cfg.whispercpp_executable,
        language=cfg.language,
        threads=cfg.threads,
        extra_args=cfg.extra_whispercpp_args,
    )

    task = {
        "dataset": args.dataset,
        "split": args.split,
        "speeds": args.speeds,
    }

    stt_toolkit.evaluate(
        model=cfg.model,
        tasks=[task],
        cache=cache,
        backend=backend,
        benchmark="wer",
        kwargs={"overwrite": args.overwrite},
    )


def main():
    args = parse_args()
    cache = ResultCache(args.output_dir)

    for config_path in args.configs:
        if not os.path.exists(config_path):
            print(f"Skipping: {config_path} (File not found)")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.model_validate(yaml.safe_load(f) or {})

        try:
            if cfg.backend == "vllm":
                run_vllm(cfg=cfg, config_path=config_path, args=args, cache=cache)
            elif cfg.backend == "whispercpp":
                run_whispercpp(cfg=cfg, args=args, cache=cache)
            else:
                raise ValueError(f"Unsupported backend: {cfg.backend}")
        except Exception as e:
            print(f"Failed to process {config_path}: {type(e).__name__} - {e}")

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
