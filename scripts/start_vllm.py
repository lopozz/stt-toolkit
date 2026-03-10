import argparse
import subprocess
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/qwen3-asr.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model", "Qwen/Qwen3-ASR-1.7B")
    port = str(cfg.get("port", 8000))
    image = cfg.get("image", "vllm-openai-audio:latest")
    container_name = cfg.get("container_name", "vllm-audio-server")
    gpu_memory_utilization = str(cfg.get("gpu_memory_utilization", 0.95))
    max_model_len = str(cfg.get("max_model_len", 448))
    max_num_seqs = str(cfg.get("max_num_seqs", 1))
    max_num_batched_tokens = cfg.get("max_num_batched_tokens")
    max_tokens_per_mm_item = cfg.get("max_num_batched_tokens")

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--ipc=host",
        "-p",
        f"{port}:8000",
        "-v",
        f"{subprocess.os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        image,
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-model-len",
        max_model_len,
        "--max-num-seqs",
        max_num_seqs,
    ]

    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    if max_tokens_per_mm_item is not None:
        cmd.extend(["--max-tokens-per-mm-item", str(max_tokens_per_mm_item)])

    print("Running:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
