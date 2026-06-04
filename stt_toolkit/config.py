from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter


class BaseConfig(BaseModel):
    model: str
    backend: str


class VllmConfig(BaseConfig):
    backend: Literal["vllm"] = "vllm"
    image: str = "vllm/vllm-openai:v0.17.1"
    port: int = 8000
    container_name: str = "vllm-stt-server"
    base_url: str = "http://localhost:8000/v1"

    gpu_memory_utilization: float = 0.95
    max_model_len: int = 448
    max_num_seqs: int = 1
    max_num_batched_tokens: int | None = None
    max_tokens_per_mm_item: int | None = None

    quantization: str | None = None
    load_format: str | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    max_num_partial_prefills: int | None = None
    limit_mm_per_prompt: str | None = None
    trust_remote_code: bool | None = None
    enforce_eager: bool | None = None

    extra_deps: list[str] = Field(default_factory=list)
    extra_vllm_args: list[str] = Field(default_factory=list)


class WhisperCppConfig(BaseConfig):
    backend: Literal["whispercpp"] = "whispercpp"
    whispercpp_model_path: str
    whispercpp_executable: str = "whisper-cli"
    language: str | None = None
    threads: int | None = None
    extra_whispercpp_args: list[str] = Field(default_factory=list)


ConfigAdapter = TypeAdapter(VllmConfig | WhisperCppConfig)


class Config:
    @classmethod
    def model_validate(cls, data):
        return ConfigAdapter.validate_python(data)
