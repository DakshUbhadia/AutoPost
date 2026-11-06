"""Model semantics and execution settings have separate identities."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from .base import DurableModel
from .serialization import content_hash

PositiveInt = Annotated[int, Field(gt=0)]


class SamplingConfig(DurableModel):
    temperature: Annotated[float, Field(ge=0)]
    top_p: Annotated[float, Field(gt=0, le=1)]
    top_k: Annotated[int, Field(ge=-1)]
    min_p: Annotated[float, Field(ge=0, le=1)]
    max_new_tokens: PositiveInt | None


class ModelSpec(DurableModel):
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_args: dict[str, JsonValue]
    max_context_tokens: PositiveInt | None
    lora_targets: tuple[str, ...]
    mode: Literal["non-thinking"]

    @property
    def spec_id(self) -> str:
        return content_hash(self)


class ExecutionProfile(DurableModel):
    validation_mode: Literal["smoke", "dev", "research"]
    backend: str
    dtype: Literal["bfloat16", "float16", "float32"]
    quantization: str | None
    generation_backend: str
    training_backend: str
    max_concurrency: PositiveInt
    tensor_parallel_size: PositiveInt
    gradient_checkpointing: bool
    cpu_offload: bool
    rollout_sampling: SamplingConfig
    eval_sampling: SamplingConfig
    lora: dict[str, JsonValue]
    importance_sampling: dict[str, JsonValue]
    backend_options: dict[str, JsonValue]

    @property
    def profile_id(self) -> str:
        return content_hash(self)
