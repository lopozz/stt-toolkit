# stt-toolkit

Utilities to run and evaluate speech-to-text models.

The toolkit currently focuses on vLLM-hosted STT models exposed through an
OpenAI-compatible audio endpoint. The code is organized so additional backends,
such as whisper.cpp or Transformers, can be added behind the same benchmark
interfaces.

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you install in editable mode, the library is imported as:

```python
import stt_toolkit
```

## Benchmarks

### WER benchmark

Measures transcription quality with Word Error Rate (WER) over a Hugging Face
dataset described by a dataset config.

```bash
.venv/bin/python scripts/wer_bench.py \
  --model-configs configs/models/whisper-large-v3-turbo.yaml \
  --dataset-config configs/datasets/lopozz-accenti-italiani.yaml \
  --speeds 1.0
```

Load cached WER results:

```python
from stt_toolkit import ResultCache

cache = ResultCache("results")
results = cache.load_results(
    models=["openai/whisper-large-v3-turbo"],
    tasks=["lopozz/accenti_italiani[train]"],
)

df = results.to_dataframe()
```

Run the same WER benchmark with whisper.cpp:

```bash
.venv/bin/python scripts/wer_bench.py \
  --model-configs configs/models/whispercpp-large-v3-turbo.yaml \
  --dataset-config configs/datasets/lopozz-accenti-italiani.yaml
```

### Speed benchmark

Measures endpoint throughput and latency for repeated transcription requests.

```bash
.venv/bin/python scripts/batched_transcription_vllm_bench.py \
  --model openai/whisper-large-v3-turbo \
  --audio-file data/network_1976/mad_as_hell.wav \
  --requests 5 50 \
  --concurrency 1 4 8
```

Metrics:

- `avg_latency_s`: average successful request latency
- `rps`: successful requests per wall-clock second
- `rtf`: audio seconds processed per wall-clock second


Load cached speed results:

```python
from stt_toolkit import BatchSpeedResultCache

cache = BatchSpeedResultCache("results")
results = cache.load_results(
    models=["openai/whisper-large-v3-turbo"],
)

df = results.to_dataframe()
```


## whisper.cpp backend setup

If you want to use [whisper.cpp](https://github.com/ggml-org/whisper.cpp), build it from source:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
```

Download a converted model:

```bash
sh ./models/download-ggml-model.sh large-v3-turbo
```

Smoke test the CLI:

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f samples/jfk.wav
```
