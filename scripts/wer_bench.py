import os
import yaml
import argparse
import stt_toolkit

from stt_toolkit import ResultCache
from stt_toolkit.benchmarks.wer import task_name
from stt_toolkit.backends import VllmBackend, WhisperCppBackend, model_is_ready
from stt_toolkit.config import Config, DatasetConfig, VllmConfig, WhisperCppConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-configs",
        nargs="+",
        required=True,
        help="YAML configs, one per model/backend",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="YAML config for the Hugging Face dataset",
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


def build_backend(cfg: VllmConfig | WhisperCppConfig, config_path: str):
    if cfg.backend == "vllm":
        if not model_is_ready(cfg.base_url, cfg.model):
            raise RuntimeError(
                f"Model is not ready on {cfg.base_url}: {cfg.model}. "
                f"Start it with: .venv/bin/python scripts/start_vllm.py {config_path}"
            )

        return VllmBackend(model=cfg.model, base_url=cfg.base_url)

    if cfg.backend == "whispercpp":
        return WhisperCppBackend(
            model_path=cfg.whispercpp_model_path,
            language=cfg.language,
            threads=cfg.threads,
            processors=cfg.processors,
            extra_args=cfg.extra_whispercpp_args,
        )
    raise ValueError(f"Unsupported backend: {cfg.backend}")


def run_benchmark(
    cfg: VllmConfig | WhisperCppConfig,
    config_path: str,
    dataset_cfg: DatasetConfig,
    args,
    cache: ResultCache,
):
    task = task_name(dataset_cfg.model_dump())
    if cache.has_result(cfg.model, task) and not args.overwrite:
        print(f"Skipping cached result: model={cfg.model}, task={task}")
        print(f"Cached file: {cache.result_path(cfg.model, task)}")
        return

    backend = build_backend(cfg, config_path)
    task_config = dataset_cfg.model_dump()
    task_config["speeds"] = args.speeds

    stt_toolkit.evaluate(
        model=cfg.model,
        tasks=[task_config],
        cache=cache,
        backend=backend,
        benchmark="wer",
        kwargs={"overwrite": args.overwrite},
    )


def main():
    args = parse_args()
    cache = ResultCache(args.output_dir)

    with open(args.dataset_config, "r", encoding="utf-8") as f:
        dataset_cfg = DatasetConfig.model_validate(yaml.safe_load(f) or {})

    for config_path in args.model_configs:
        if not os.path.exists(config_path):
            print(f"Skipping: {config_path} (File not found)")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.model_validate(yaml.safe_load(f) or {})

        try:
            run_benchmark(
                cfg=cfg,
                config_path=config_path,
                dataset_cfg=dataset_cfg,
                args=args,
                cache=cache,
            )
        except Exception as e:
            print(f"Failed to process {config_path}: {type(e).__name__} - {e}")

    print("BENCHMARK FINISHED")


if __name__ == "__main__":
    main()
