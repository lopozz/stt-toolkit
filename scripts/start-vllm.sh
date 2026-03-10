#!/usr/bin/env bash
set -e

MODEL="${1:-Qwen/Qwen3-ASR-1.7B}"
PORT="${2:-8000}"
IMAGE="${3:-vllm-openai-audio:latest}"
CONTAINER_NAME="${4:-vllm-audio-server}"

# docker run --rm -d \
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${IMAGE}" \
  --model "${MODEL}" \
  --task transcription \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128 \
  --max-num-seqs 2