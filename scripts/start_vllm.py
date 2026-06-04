import argparse
import os
import shlex
import subprocess

import yaml

from stt_toolkit.config import Config


def append_cli_arg(cmd, flag, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            cmd.extend([flag, str(item)])
        return
    cmd.extend([flag, str(value)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = Config.model_validate(yaml.safe_load(f) or {})

    subprocess.run(
        ["docker", "rm", "-f", cfg.container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        cfg.container_name,
        "--gpus",
        "all",
        "--ipc=host",
        "-p",
        f"{cfg.port}:8000",
        "-v",
        f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
    ]

    vllm_args = [
        cfg.model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-model-len",
        str(cfg.max_model_len),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
    ]

    if cfg.max_num_batched_tokens is not None:
        vllm_args.extend(["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)])
    if cfg.max_tokens_per_mm_item is not None:
        vllm_args.extend(["--max-tokens-per-mm-item", str(cfg.max_tokens_per_mm_item)])

    optional_vllm_args = {
        "--quantization": cfg.quantization,
        "--load-format": cfg.load_format,
        "--dtype": cfg.dtype,
        "--kv-cache-dtype": cfg.kv_cache_dtype,
        "--tensor-parallel-size": cfg.tensor_parallel_size,
        "--pipeline-parallel-size": cfg.pipeline_parallel_size,
        "--max-num-partial-prefills": cfg.max_num_partial_prefills,
        "--limit-mm-per-prompt": cfg.limit_mm_per_prompt,
        "--trust-remote-code": cfg.trust_remote_code,
        "--enforce-eager": cfg.enforce_eager,
    }
    for flag, value in optional_vllm_args.items():
        append_cli_arg(vllm_args, flag, value)

    for extra_arg in cfg.extra_vllm_args:
        vllm_args.append(str(extra_arg))

    if cfg.extra_deps:
        install_cmd = shlex.join(["pip", "install", "--no-cache-dir", *cfg.extra_deps])
        serve_cmd = shlex.join(["vllm", "serve", *vllm_args])
        cmd = docker_cmd + [
            "--entrypoint",
            "bash",
            cfg.image,
            "-lc",
            f"{install_cmd} && {serve_cmd}",
        ]
    else:
        cmd = docker_cmd + [cfg.image, *vllm_args]

    print("Running:")
    print(shlex.join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
