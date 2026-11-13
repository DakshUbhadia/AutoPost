"""Project-owned generation providers; optional runtime objects stop here."""

import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

from autopost.contracts import (
    CompatibilityError,
    EvalGenerationBatch,
    EvalGenerationRequest,
    GeneratedCompletion,
    RolloutBatch,
    RolloutRequest,
    content_hash,
)

from .capabilities import discover, require_compatible


class ArtifactAdapterResolver:
    """Materialize immutable adapter bytes only inside the backend.

    An explicit base StateRef uses the pinned base model. Other states resolve
    through metadata; the probe may supply a stable StateRef -> ArtifactRef map.
    """

    def __init__(
        self, artifacts, base_state, *, metadata=None, stack_id=None, adapters=None
    ):
        self.artifacts, self.base = artifacts, base_state
        self.metadata, self.stack_id = metadata, stack_id
        self.adapters = dict(adapters or {})
        self._temporary = TemporaryDirectory(prefix="autopost-adapters-")

    def resolve(self, state):
        state.to_bytes()
        if state == self.base:
            return None
        ref = self.adapters.get(state.state_id)
        if ref is None:
            if self.metadata is None:
                raise CompatibilityError("state has no registered adapter artifact")
            manifest = self.metadata.get_state(state)
            if self.stack_id is None or manifest.stack_id != self.stack_id:
                raise CompatibilityError(
                    "adapter state belongs to another execution stack"
                )
            ref = manifest.adapter
        manifest = self.artifacts.resolve(ref)
        if manifest.kind != "state":
            raise CompatibilityError("adapter must be a state artifact")
        components = {component.name for component in manifest.components}
        required = {"adapter_config.json", "adapter_model.safetensors"}
        if not required <= components:
            raise CompatibilityError(
                "adapter artifact lacks config or safetensors weights"
            )
        directory = Path(self._temporary.name) / ref.sha256
        directory.mkdir(exist_ok=True)
        for name in required:
            blob = self.artifacts.read(ref, name)
            target = directory / name
            if target.exists() and target.read_bytes() != blob:
                raise CompatibilityError("materialized adapter bytes changed")
            target.write_bytes(blob)
        return directory

    def close(self):
        self._temporary.cleanup()


class VLLMBackend:
    """Lazy, serialized colocated engine. Test doubles are injected privately."""

    def __init__(self, model, execution, resolver, stack_id, *, runtime=None):
        model.to_bytes()
        execution.to_bytes()
        self.model, self.execution, self.resolver, self.stack_id = (
            model,
            execution,
            resolver,
            stack_id,
        )
        self._runtime = runtime
        self._engine = self._tokenizer = None
        self._lock = RLock()

    def _load(self):
        if self._engine is not None:
            return
        if self._runtime is None:
            require_compatible(discover(self.model, self.execution, self.stack_id))
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            self._runtime = (AutoTokenizer, LLM, SamplingParams, LoRARequest)
        tokenizer_cls, engine_cls, _, _ = self._runtime
        self._tokenizer = tokenizer_cls.from_pretrained(
            self.model.tokenizer_id,
            revision=self.model.tokenizer_revision,
            trust_remote_code=False,
        )
        self._engine = engine_cls(
            model=self.model.model_id,
            revision=self.model.revision,
            tokenizer=self.model.tokenizer_id,
            tokenizer_revision=self.model.tokenizer_revision,
            trust_remote_code=False,
            dtype=self.execution.dtype,
            tensor_parallel_size=self.execution.tensor_parallel_size,
            max_model_len=self.model.max_context_tokens,
            enable_lora=True,
            max_lora_rank=self.execution.lora["r"],
            gpu_memory_utilization=self.execution.backend_options[
                "vllm_gpu_memory_utilization"
            ],
            seed=0,
            generation_config="vllm",
        )

    def generate(self, request):
        request = type(request).from_bytes(request.to_bytes())
        if not request.prompts or len({p.prompt_id for p in request.prompts}) != len(
            request.prompts
        ):
            raise CompatibilityError(
                "generation requires nonempty, uniquely identified prompts"
            )
        if request.sampling.max_new_tokens is None or request.sampling.top_k == 0:
            raise CompatibilityError(
                "generation token cap and supported top_k must be explicit"
            )
        with self._lock:
            # Resolve states before initializing/downloading a model.
            adapter = self.resolver.resolve(request.state)
            self._load()
            _, _, sampling_cls, lora_cls = self._runtime
            prompts, sampling, identities = [], [], []
            for prompt in request.prompts:
                rendered = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.model.chat_template_args,
                )
                token_count = len(
                    self._tokenizer.encode(rendered, add_special_tokens=False)
                )
                if (
                    token_count + request.sampling.max_new_tokens
                    > self.model.max_context_tokens
                ):
                    raise CompatibilityError(
                        "rendered prompt plus completion exceeds context limit"
                    )
                for index in range(request.samples_per_prompt):
                    identity = {
                        "seed": request.seed,
                        "job": request.job_id,
                        "state": request.state.state_id,
                        "prompt": prompt.prompt_id,
                        "template": prompt.template_id,
                        "sample": index,
                        "read": getattr(request, "eval_read_id", None),
                    }
                    seed = int(content_hash(identity)[:8], 16)
                    prompts.append(rendered)
                    sampling.append(
                        sampling_cls(
                            n=1,
                            temperature=request.sampling.temperature,
                            top_p=request.sampling.top_p,
                            top_k=request.sampling.top_k,
                            min_p=request.sampling.min_p,
                            max_tokens=request.sampling.max_new_tokens,
                            seed=seed,
                            logprobs=1,
                        )
                    )
                    identities.append((prompt, index))
            lora = (
                None
                if adapter is None
                else lora_cls(
                    request.state.state_id,
                    int(
                        hashlib.sha256(request.state.state_id.encode()).hexdigest()[
                            :12
                        ],
                        16,
                    )
                    + 1,
                    str(adapter),
                )
            )
            outputs = self._engine.generate(
                prompts, sampling, lora_request=lora, use_tqdm=False
            )
            if len(outputs) != len(identities):
                raise CompatibilityError("generation backend omitted request outputs")
            completions = []
            for output, (prompt, index) in zip(outputs, identities):
                if len(output.outputs) != 1:
                    raise CompatibilityError(
                        "backend returned an unexpected sample count"
                    )
                generated = output.outputs[0]
                tokens = tuple(int(token) for token in generated.token_ids)
                if len(
                    tokens
                ) > request.sampling.max_new_tokens or generated.finish_reason not in {
                    "stop",
                    "length",
                }:
                    raise CompatibilityError(
                        "backend exceeded the cap or returned an incomplete sample"
                    )
                if generated.logprobs is None or len(generated.logprobs) != len(tokens):
                    raise CompatibilityError(
                        "sampled-token log probabilities are missing"
                    )
                logprobs = []
                for token, scores in zip(tokens, generated.logprobs):
                    if (
                        scores is None
                        or token not in scores
                        or not math.isfinite(scores[token].logprob)
                    ):
                        raise CompatibilityError(
                            "sampled token has no finite log probability"
                        )
                    logprobs.append(float(scores[token].logprob))
                completions.append(
                    GeneratedCompletion(
                        prompt_id=prompt.prompt_id,
                        template_id=prompt.template_id,
                        sample_idx=index,
                        text=generated.text,
                        token_ids=tokens,
                        sampled_logprobs=tuple(logprobs),
                        truncated=generated.finish_reason == "length",
                    )
                )
            common = {
                "job_id": request.job_id,
                "state": request.state,
                "completions": tuple(completions),
                "metadata": {
                    "backend": "local-vllm@1",
                    "model_spec_id": self.model.spec_id,
                    "execution_profile_id": self.execution.profile_id,
                    "stack_id": self.stack_id,
                    "seed_policy": "semantic-per-sample@1",
                    "raw_logprobs": True,
                },
            }
            if isinstance(request, EvalGenerationRequest):
                return EvalGenerationBatch(**common, eval_read_id=request.eval_read_id)
            return RolloutBatch(**common)

    def close(self):
        # vLLM engines/allocators live within the owning job process. Ending that
        # process is the reliable resource boundary; no runtime object escapes.
        self._engine = self._tokenizer = None
        self.resolver.close()


class LocalVLLMRolloutProvider:
    def __init__(self, backend):
        self.backend = backend

    def generate(self, request: RolloutRequest) -> RolloutBatch:
        if type(request) is not RolloutRequest:
            raise TypeError("rollout provider requires a RolloutRequest")
        return self.backend.generate(request)


class LocalEvalGenerationProvider:
    def __init__(self, backend):
        self.backend = backend

    def generate(self, request: EvalGenerationRequest) -> EvalGenerationBatch:
        if type(request) is not EvalGenerationRequest:
            raise TypeError("evaluation provider requires an EvalGenerationRequest")
        # No reference-output cache: every invocation reaches the engine.
        return self.backend.generate(request)
