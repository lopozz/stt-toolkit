docker run --rm -it --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-openai-audio:0.10.1 \
  --model openai/whisper-large-v3-turbo \
  --task transcription \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128 \
  --max-num-seqs 2