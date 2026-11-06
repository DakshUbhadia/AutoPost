"""Sprint 1 exit gates; no backend, GPU, model or database is needed."""

import inspect
import json
from pathlib import Path

import pytest

from autopost import contracts as c

DIGEST = "a" * 64
WHEN = "2026-11-06T00:00:00Z"


def job_inputs():
    return {
        "run_id": "run-1",
        "config_id": DIGEST,
        "stack_id": DIGEST,
        "model_spec_id": DIGEST,
        "execution_profile_id": DIGEST,
        "code_revision": "abc123",
        "created_at": WHEN,
        "expected_outputs": ("output-1",),
        "seed": 7,
    }


def training_job(**overrides):
    inputs = dict(
        **job_inputs(),
        start=c.StateRef(state_id="base"),
        training_slice=c.DatasetRef(dataset_id="train", sha256=DIGEST),
        expected_state=c.StateRef(state_id="next"),
        steps=50,
        verifier_version="verify-1",
    )
    inputs.update(overrides)
    return c.TrainingJobSpec.create(**inputs)


def evaluation_job(**overrides):
    inputs = dict(
        **job_inputs(),
        states=(c.StateRef(state_id="base"), c.StateRef(state_id="next")),
        dataset=c.DatasetRef(dataset_id="gate-b", sha256=DIGEST),
        eval_read_id="read-1",
        instrument_id="instrument-1",
        template_ids=("eval-1",),
        k=8,
        k_ref=16,
        fresh_reference=True,
        split="gate_b",
    )
    inputs.update(overrides)
    return c.EvaluationJobSpec.create(**inputs)


def examples():
    state, parent = c.StateRef(state_id="next"), c.StateRef(state_id="base")
    artifact = c.ArtifactRef(artifact_id="artifact", sha256=DIGEST)
    dataset = c.DatasetRef(dataset_id="train", sha256=DIGEST)
    sampling = c.SamplingConfig(
        temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, max_new_tokens=64
    )
    model = c.ModelSpec(
        model_id="example/model",
        revision="a" * 40,
        tokenizer_id="example/model",
        tokenizer_revision="a" * 40,
        chat_template_args={"enable_thinking": False},
        max_context_tokens=2048,
        lora_targets=("q_proj",),
        mode="non-thinking",
    )
    execution = c.ExecutionProfile(
        validation_mode="dev",
        backend="fixture",
        dtype="float32",
        quantization=None,
        generation_backend="fixture",
        training_backend="fixture",
        max_concurrency=1,
        tensor_parallel_size=1,
        gradient_checkpointing=False,
        cpu_offload=False,
        rollout_sampling=sampling,
        eval_sampling=sampling,
        lora={"r": 32},
        importance_sampling={"mode": "token_truncate", "clip_max": 2.0},
        backend_options={},
    )
    spec, evaluation = training_job(), evaluation_job()
    component = c.ArtifactComponent(
        name="adapter",
        sha256=DIGEST,
        size_bytes=3,
        media_type="application/octet-stream",
    )
    manifest = c.ArtifactManifest(
        artifact=artifact,
        kind="state",
        components=(component,),
        producer_job_id=spec.job_id,
    )
    full_state = c.StateManifest(
        state=state,
        parent=parent,
        job_id=spec.job_id,
        global_step=50,
        config_id=DIGEST,
        stack_id=DIGEST,
        training_slice=dataset,
        verifier_version="verify-1",
        **{
            name: artifact
            for name in (
                "adapter",
                "optimizer",
                "scheduler",
                "rng",
                "trainer",
                "sampler_cursor",
            )
        },
    )
    evidence = c.EvalEvidence(
        eval_read_id=evaluation.eval_read_id,
        job_id=evaluation.job_id,
        instrument_id=evaluation.instrument_id,
        config_id=DIGEST,
        stack_id=DIGEST,
        dataset=evaluation.dataset,
        states=evaluation.states,
        completions=artifact,
        scorer_version="exact-1",
        template_ids=evaluation.template_ids,
        seeds=(7,),
        fresh_reference=True,
        aggregates={"accuracy": 0.5},
    )
    common = {
        key: getattr(spec, key)
        for key in (
            "job_id",
            "run_id",
            "config_id",
            "stack_id",
            "model_spec_id",
            "execution_profile_id",
            "code_revision",
            "attempt",
        )
    }
    common.update(
        started_at=WHEN, finished_at=WHEN, status="succeeded", artifacts=(artifact,)
    )
    train_result = c.TrainingJobResult(
        **common,
        start=parent,
        training_slice=dataset,
        state=state,
        global_step=50,
        rollouts=artifact,
        metrics={"kl": 0.0},
    )
    eval_result = c.EvaluationJobResult(
        **{**common, "job_id": evaluation.job_id},
        states=evaluation.states,
        dataset=evaluation.dataset,
        eval_read_id=evaluation.eval_read_id,
        evidence=evidence,
    )
    error = c.JobError(code="executor_failure", message="worker exited", retryable=True)
    failure = c.TrainingJobResult(
        **{**common, "status": "failed", "artifacts": (), "error": error},
        start=parent,
        training_slice=dataset,
    )
    prompt = c.GenerationPrompt(
        prompt_id="p-1", template_id="eval-1", text="Compute 1+1"
    )
    request_args = {
        "job_id": spec.job_id,
        "state": state,
        "prompts": (prompt,),
        "sampling": sampling,
        "seed": 7,
        "samples_per_prompt": 8,
    }
    completion = c.GeneratedCompletion(
        prompt_id="p-1",
        template_id="eval-1",
        sample_idx=0,
        text="Answer: 2",
        token_ids=(2,),
        sampled_logprobs=(-0.1,),
        truncated=False,
    )
    batch_args = {
        "job_id": spec.job_id,
        "state": state,
        "completions": (completion,),
        "metadata": {"seed": 7},
    }
    capability = c.CapabilitySpec(
        dtypes=("float32",),
        quantizations=(),
        max_context_tokens=2048,
        lora=True,
        training_backends=("fixture",),
        generation_backends=("fixture",),
        max_tensor_parallel_size=1,
        importance_sampling_modes=("token_truncate",),
        colocated_generation=True,
    )
    record = c.MetadataRecord(
        namespace="events", key="event-1", value={"job": spec.job_id}
    )
    pointer = c.PointerUpdate(name="current", expected=parent, replacement=state)
    charge = c.BudgetCharge(budget="gate_b", item_id="read-1")
    return [
        state,
        artifact,
        dataset,
        sampling,
        model,
        execution,
        spec,
        evaluation,
        component,
        manifest,
        full_state,
        evidence,
        train_result,
        eval_result,
        error,
        failure,
        c.JobHandle(job_id=spec.job_id, attempt=1),
        prompt,
        c.RolloutRequest(**request_args),
        c.EvalGenerationRequest(
            **request_args, eval_read_id="read-1", fresh_reference=True
        ),
        completion,
        c.RolloutBatch(**batch_args),
        c.EvalGenerationBatch(**batch_args, eval_read_id="read-1"),
        capability,
        c.CapabilityReport(
            backend_id="fixture",
            stack_id=DIGEST,
            capabilities=capability,
            compatible=True,
        ),
        record,
        pointer,
        charge,
        c.AcceptanceEffects(
            run_id="run-1",
            expected_parent=parent,
            records=(record,),
            pointers=(pointer,),
            charges=(charge,),
        ),
        c.AcceptanceReceipt(job_id=spec.job_id, disposition="accepted"),
        c.StagedArtifact(staging_id="stage-1", expected=artifact.artifact_id),
        c.AttemptRecord(
            handle=c.JobHandle(job_id=spec.job_id, attempt=1),
            status="succeeded",
            result=train_result,
        ),
        c.AuditEvent(
            event_id=1,
            run_id="run-1",
            job_id=spec.job_id,
            kind="result_accepted",
            payload={"result_digest": train_result.result_digest},
            created_at=WHEN,
        ),
    ]


@pytest.mark.parametrize("value", examples(), ids=lambda value: type(value).__name__)
def test_every_durable_value_roundtrips(value):
    blob = c.canonical_bytes(value)
    assert type(value).from_bytes(blob) == value
    assert c.canonical_bytes(type(value).from_bytes(blob)) == blob
    assert json.loads(blob)["schema_version"] == 1


def test_roundtrip_examples_cover_every_public_durable_model():
    public = {
        value
        for value in vars(c).values()
        if inspect.isclass(value)
        and issubclass(value, c.DurableModel)
        and value is not c.DurableModel
    }
    assert public == {type(value) for value in examples()}


@pytest.mark.parametrize("make", [training_job, evaluation_job])
def test_retries_exclude_attempt_and_timestamp_from_logical_identity(make):
    first = make()
    retry = make(attempt=2, created_at="2026-11-07T00:00:00Z")
    assert first.job_id == retry.job_id
    assert first.to_bytes() != retry.to_bytes()
    assert make(seed=8).job_id != first.job_id
    assert make(stack_id="b" * 64).job_id != first.job_id
    assert c.decode_job_spec(first.to_bytes()) == first
    bad = json.loads(first.to_bytes())
    bad["seed"] = 100
    with pytest.raises(ValueError, match="immutable semantic inputs"):
        c.decode_job_spec(json.dumps(bad).encode())


@pytest.mark.parametrize("version", [0, 2, -1, True, "1", None])
def test_unsupported_versions_fail_deterministically(version):
    value = json.loads(training_job().to_bytes())
    value["schema_version"] = version
    with pytest.raises(ValueError, match="unsupported schema_version"):
        c.decode_job_spec(json.dumps(value).encode())
    assert c.protocol_compatibility(version) == "rejected"


def test_nested_versions_and_unknown_fields_and_duplicate_keys_rejected():
    value = json.loads(training_job().to_bytes())
    value["start"]["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported schema_version"):
        c.decode_job_spec(json.dumps(value).encode())
    with pytest.raises(ValueError):
        c.StateRef(state_id="state", checkpoint_path="relative/checkpoint")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        c.StateRef.from_bytes(b'{"state_id":"one","state_id":"two"}')


@pytest.mark.parametrize(
    "value",
    [
        Path("checkpoint"),
        object(),
        lambda: None,
        float("nan"),
        float("inf"),
        {1: "non-string key"},
    ],
)
def test_canonical_serialization_rejects_runtime_values(value):
    with pytest.raises(ValueError):
        c.canonical_bytes({"value": value})
    with pytest.raises(ValueError):
        c.MetadataRecord(namespace="test", key="one", value={"value": value})


@pytest.mark.parametrize(
    "value", ["C:\\states\\one", "C:/states/one", "/tmp/one", "\\\\server\\one"]
)
def test_absolute_paths_cannot_cross_contract_boundary(value):
    with pytest.raises(ValueError, match="absolute paths"):
        c.MetadataRecord(namespace="test", key="one", value={"location": value})
    with pytest.raises(ValueError):
        c.StateRef(state_id=value)


def test_canonical_hash_has_full_length_and_unicode_and_stable_key_order():
    assert c.content_hash({"z": 1, "a": "α"}) == c.content_hash({"a": "α", "z": 1})
    assert len(c.content_hash({"x": 1})) == 64
    assert c.canonical_bytes({"z": 1, "a": "α"}) == '{"a":"α","z":1}'.encode()


def test_gate_b_and_result_failure_invariants():
    with pytest.raises(ValueError, match="fresh reference"):
        evaluation_job(fresh_reference=False)
    for value in examples():
        if isinstance(value, (c.TrainingJobResult, c.EvaluationJobResult)):
            assert c.decode_job_result(value.to_bytes()) == value
    result = next(
        value for value in examples() if isinstance(value, c.TrainingJobResult)
    )
    with pytest.raises(ValueError, match="requires an error"):
        c.TrainingJobResult.from_bytes(
            result.to_bytes().replace(b'"succeeded"', b'"failed"')
        )


def test_serialization_revalidates_mutable_maps_and_model_copy():
    record = c.MetadataRecord(namespace="test", key="one", value={})
    record.value["bad"] = "/physical/path"
    with pytest.raises(ValueError, match="absolute paths"):
        record.to_bytes()
    invalid = training_job().model_copy(update={"seed": 999})
    with pytest.raises(ValueError, match="immutable semantic inputs"):
        invalid.to_bytes()


def test_control_plane_state_no_longer_has_path_or_model_profile():
    from autopost import interfaces

    assert "path" not in interfaces.TrainState.model_fields
    assert not hasattr(interfaces, "ModelProfile")
    assert interfaces.TrainState is c.StateManifest


def test_open_connections_and_locks_are_not_durable_values():
    import sqlite3
    import threading

    connection = sqlite3.connect(":memory:")
    try:
        for runtime in (connection, threading.Lock()):
            with pytest.raises(ValueError):
                c.MetadataRecord(
                    namespace="test", key="runtime", value={"runtime": runtime}
                )
    finally:
        connection.close()


def test_measurement_splits_including_hidden_dev_cannot_reach_trainers():
    from autopost.interfaces import MEASUREMENT_SPLITS, Task, assert_not_measurement

    assert "hidden_dev" in MEASUREMENT_SPLITS
    for split in MEASUREMENT_SPLITS:
        task = Task(
            task_id="p",
            cluster_id="c",
            source_id="fixture",
            skill="add",
            problem="1+1",
            answer="2",
            answer_type="int",
            low_entropy=True,
            split=split,
        )
        with pytest.raises(RuntimeError, match="measurement items"):
            assert_not_measurement([task])


def test_result_semantic_digest_excludes_retry_delivery_metadata():
    result = next(
        value for value in examples() if isinstance(value, c.TrainingJobResult)
    )
    retry = result.model_copy(
        update={"attempt": 2, "finished_at": "2026-11-07T00:00:00Z"}
    )
    assert result.to_bytes() != retry.to_bytes()
    assert result.result_digest == retry.result_digest
    assert (
        result.model_copy(update={"metrics": {"kl": 0.1}}).result_digest
        != result.result_digest
    )
