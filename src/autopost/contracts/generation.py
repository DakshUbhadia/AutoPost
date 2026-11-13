"""Generation values contain no tokenizer, tensor or backend runtime types."""

from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from .base import DurableModel
from .models import PositiveInt, SamplingConfig
from .refs import EvalReadId, JobId, PromptId, StateRef


class GenerationPrompt(DurableModel):
    opaque_text_fields = frozenset({"text"})
    prompt_id: PromptId
    template_id: str
    text: str


class RolloutRequest(DurableModel):
    job_id: JobId
    state: StateRef
    prompts: tuple[GenerationPrompt, ...]
    sampling: SamplingConfig
    seed: int
    samples_per_prompt: PositiveInt


class EvalGenerationRequest(RolloutRequest):
    eval_read_id: EvalReadId
    fresh_reference: bool


class GeneratedCompletion(DurableModel):
    opaque_text_fields = frozenset({"text"})
    prompt_id: PromptId
    template_id: str
    sample_idx: Annotated[int, Field(ge=0)]
    text: str
    token_ids: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    truncated: bool

    @model_validator(mode="after")
    def aligned_tokens(self):
        if len(self.token_ids) != len(self.sampled_logprobs):
            raise ValueError("token IDs and sampled log probabilities must align")
        if any(token < 0 for token in self.token_ids) or any(
            value > 0 for value in self.sampled_logprobs
        ):
            raise ValueError(
                "token IDs must be nonnegative and log probabilities nonpositive"
            )
        return self


class RolloutBatch(DurableModel):
    job_id: JobId
    state: StateRef
    completions: tuple[GeneratedCompletion, ...]
    metadata: dict[str, JsonValue]


class EvalGenerationBatch(RolloutBatch):
    eval_read_id: EvalReadId
