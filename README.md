# 💬 stt-toolkit
Handy scripts to run and evaluate STT models.

## 🛠 Installation & Setup
Install the Python Library
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

Build the vllm docker image with audio features enabled:
```
docker compose build
```

## 🐳 Usage
Add your audio data to `data` and serve a model with vLLM:

```
docker run --rm -it --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-openai-audio:latest \
  --model openai/whisper-large-v3-turbo \
  --task transcription \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128 \
  --max-num-seqs 2
```

Check the server is working with a simple call:
```
curl http://localhost:8000/v1/models
```