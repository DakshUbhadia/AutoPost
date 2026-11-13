"""Backend boundary tests use runtime doubles and never download a model."""

from types import SimpleNamespace as NS

import pytest

from autopost.config import load_config
from autopost.contracts import (
    CompatibilityError,
    EvalGenerationBatch,
    EvalGenerationRequest,
    ExecutionProfile,
    GeneratedCompletion,
    GenerationPrompt,
    ModelSpec,
    RolloutRequest,
    SamplingConfig,
    StateRef,
)
from autopost.execution.capabilities import (
    GPU_PINS,
    algorithm_controls,
    discover,
    require_compatible,
)
from autopost.execution.generation import (
    LocalEvalGenerationProvider,
    LocalVLLMRolloutProvider,
    VLLMBackend,
)


@pytest.fixture
def config():
    cfg = load_config()
    return (
        cfg,
        ModelSpec.model_validate_json(__import__("json").dumps(cfg["model"])),
        ExecutionProfile.model_validate_json(
            __import__("json").dumps(cfg["execution"])
        ),
    )


def snapshot(capability=(8, 0), **overrides):
    return {
        "platform": "Linux",
        "versions": dict(GPU_PINS),
        "cuda_available": True,
        "devices": [
            {
                "capability": capability,
                "name": "synthetic GPU",
                "memory_bytes": 24 * 1024**3,
            }
        ],
        **overrides,
    }


def test_capabilities_explicit_dtype_and_algorithm_assertions(config):
    cfg, model, profile = config
    assert discover(model, profile, "a" * 64, snapshot=snapshot()).compatible
    report = discover(model, profile, "a" * 64, snapshot=snapshot((7, 5)))
    with pytest.raises(CompatibilityError, match="dtype"):
        require_compatible(report)
    assert discover(
        model,
        profile.model_copy(update={"dtype": "float16"}),
        "a" * 64,
        snapshot=snapshot((7, 5)),
    ).compatible
    controls = algorithm_controls(cfg["experiment"], profile)
    assert controls["vllm_importance_sampling_mode"] == "token_truncate"
    assert controls["vllm_importance_sampling_clip_max"] == 2.0
    with pytest.raises(CompatibilityError):
        algorithm_controls(
            cfg["experiment"],
            profile.model_copy(update={"importance_sampling": {"mode": "sequence"}}),
        )
    assert not discover(
        model, profile, "a" * 64, snapshot=snapshot(cuda_available=False, devices=[])
    ).compatible


class Tokenizer:
    @classmethod
    def from_pretrained(cls, *_, **__):
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "chat:" + messages[0]["content"]

    def encode(self, text, **_):
        return list(range(len(text)))


class Engine:
    def __init__(self, **kwargs):
        assert kwargs["generation_config"] == "vllm"
        self.calls = []
        self.missing_logprobs = False

    def generate(self, prompts, sampling, **kwargs):
        self.calls.append((prompts, sampling, kwargs))
        return [
            NS(
                outputs=[
                    NS(
                        text=r"\boxed{2}",
                        token_ids=[5],
                        logprobs=None
                        if self.missing_logprobs
                        else [{5: NS(logprob=-0.5)}],
                        finish_reason="stop",
                    )
                ]
            )
            for _ in prompts
        ]


class Resolver:
    def resolve(self, state):
        if state.state_id != "base":
            raise CompatibilityError("unknown state")

    def close(self):
        pass


def request(**overrides):
    return RolloutRequest(
        **{
            "job_id": "test-job",
            "state": StateRef(state_id="base"),
            "prompts": (
                GenerationPrompt(
                    prompt_id="p1", template_id="T_train", text="Compute 1+1"
                ),
            ),
            "sampling": SamplingConfig(
                temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, max_new_tokens=16
            ),
            "seed": 42,
            "samples_per_prompt": 2,
            **overrides,
        }
    )


def backend(config):
    _, model, profile = config
    return VLLMBackend(
        model,
        profile,
        Resolver(),
        "a" * 64,
        runtime=(Tokenizer, Engine, lambda **values: NS(**values), lambda *args: args),
    )


def test_provider_mapping_seeds_logprobs_and_fresh_eval_calls(config):
    engine = backend(config)
    provider = LocalVLLMRolloutProvider(engine)
    output = provider.generate(request())
    assert [c.sample_idx for c in output.completions] == [0, 1]
    assert output.completions[0].text == r"\boxed{2}"
    assert output.completions[0].sampled_logprobs == (-0.5,)
    assert type(output).from_bytes(output.to_bytes()) == output
    assert provider.generate(request()) == output
    seeds = [p.seed for p in engine._engine.calls[0][1]]
    assert len(set(seeds)) == 2
    assert seeds == [p.seed for p in engine._engine.calls[1][1]]
    values = request().model_dump(exclude={"prompts", "sampling", "state"})
    evaluation = EvalGenerationRequest(
        **values,
        prompts=request().prompts,
        sampling=request().sampling,
        state=request().state,
        eval_read_id="read-one",
        fresh_reference=True,
    )
    evaluator = LocalEvalGenerationProvider(engine)
    batch = evaluator.generate(evaluation)
    assert isinstance(batch, EvalGenerationBatch)
    evaluator.generate(evaluation)
    assert len(engine._engine.calls) == 4
    assert batch.eval_read_id == "read-one"
    assert type(batch).from_bytes(batch.to_bytes()) == batch
    with pytest.raises(TypeError):
        provider.generate(evaluation)


def test_context_unknown_state_and_missing_logprobs_fail_closed(config):
    engine = backend(config)
    with pytest.raises(CompatibilityError, match="unknown"):
        engine.generate(request(state=StateRef(state_id="unknown")))
    assert engine._engine is None
    engine.generate(request())
    engine._engine.missing_logprobs = True
    with pytest.raises(CompatibilityError, match="probabilities"):
        engine.generate(request())
    with pytest.raises(CompatibilityError, match="context"):
        engine.generate(
            request(
                prompts=(
                    GenerationPrompt(
                        prompt_id="huge", template_id="T", text="x" * 5000
                    ),
                )
            )
        )


def test_opaque_math_text_does_not_relax_locator_validation():
    value = GeneratedCompletion(
        prompt_id="p",
        template_id="T",
        sample_idx=0,
        text=r"\boxed{2}",
        token_ids=(1,),
        sampled_logprobs=(-0.1,),
        truncated=False,
    )
    assert GeneratedCompletion.from_bytes(value.to_bytes()) == value
    with pytest.raises(ValueError):
        value.model_copy(update={"template_id": "C:/physical/path"}).to_bytes()
    with pytest.raises(ValueError):
        value.model_copy(update={"sampled_logprobs": (0.1,)}).to_bytes()


def test_adapter_resolver_materializes_only_immutable_state_artifacts(storage):
    import hashlib

    from autopost.contracts import ArtifactComponent, ArtifactManifest
    from autopost.execution.generation import ArtifactAdapterResolver

    payload = {
        "adapter_config.json": b'{"peft_type":"LORA"}',
        "adapter_model.safetensors": b"synthetic weights",
    }
    manifest = ArtifactManifest.create(
        artifact_id="test-adapter",
        kind="state",
        producer_job_id="fixture-job",
        components=tuple(
            ArtifactComponent(
                name=name,
                sha256=hashlib.sha256(blob).hexdigest(),
                size_bytes=len(blob),
                media_type="application/octet-stream",
            )
            for name, blob in payload.items()
        ),
    )
    staged = storage.artifacts.stage("test-adapter")
    for name, blob in payload.items():
        storage.artifacts.write(staged, name, blob)
    storage.artifacts.finalize(staged, manifest)
    resolver = ArtifactAdapterResolver(
        storage.artifacts,
        StateRef(state_id="base"),
        adapters={"adapter": manifest.artifact},
    )
    try:
        assert resolver.resolve(StateRef(state_id="base")) is None
        directory = resolver.resolve(StateRef(state_id="adapter"))
        assert (directory / "adapter_model.safetensors").read_bytes() == payload[
            "adapter_model.safetensors"
        ]
        assert resolver.resolve(StateRef(state_id="adapter")) == directory
        with pytest.raises(CompatibilityError):
            resolver.resolve(StateRef(state_id="unregistered"))
    finally:
        resolver.close()
