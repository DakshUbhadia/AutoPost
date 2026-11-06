"""Generation values contain no tokenizer, tensor or backend runtime types."""

from pydantic import JsonValue, model_validator

from .base import DurableModel
from .models import PositiveInt, SamplingConfig
from .refs import EvalReadId, JobId, PromptId, StateRef


class GenerationPrompt(DurableModel):
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
    prompt_id: PromptId
    template_id: str
    sample_idx: int
    text: str
    token_ids: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    truncated: bool

    @model_validator(mode="after")
    def aligned_tokens(self):
        if len(self.token_ids) != len(self.sampled_logprobs):
            raise ValueError("token IDs and sampled log probabilities must align")
        return self


class RolloutBatch(DurableModel):
    job_id: JobId
    state: StateRef
    completions: tuple[GeneratedCompletion, ...]
    metadata: dict[str, JsonValue]


class EvalGenerationBatch(RolloutBatch):
    eval_read_id: EvalReadId
