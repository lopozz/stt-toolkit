import argparse
import subprocess

import yaml

from stt_toolkit.config import Config


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

    result = subprocess.run(
        ["docker", "rm", "-f", cfg.container_name],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"Stopped and removed container: {cfg.container_name}")
    else:
        print(f"Container not found or could not be removed: {cfg.container_name}")
        if result.stderr.strip():
            print(result.stderr.strip())


if __name__ == "__main__":
    main()
