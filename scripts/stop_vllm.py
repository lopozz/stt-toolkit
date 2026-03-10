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

    container_name = cfg.get("container_name", "vllm-audio-server")

    result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"Stopped and removed container: {container_name}")
    else:
        print(f"Container not found or could not be removed: {container_name}")
        if result.stderr.strip():
            print(result.stderr.strip())


if __name__ == "__main__":
    main()
