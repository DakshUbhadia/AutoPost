"""Versioned jobs: logical identity excludes delivery/attempt metadata."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

from .base import DurableModel
from .evidence import EvalEvidence
from .models import PositiveInt
from .refs import (
    ArtifactId,
    ArtifactRef,
    ConfigId,
    DatasetRef,
    EvalReadId,
    ExecutionProfileId,
    JobId,
    MeasurementInstrumentId,
    ModelSpecId,
    RunId,
    StackId,
    StateRef,
)
from .serialization import content_hash

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


def _utc_timestamp(value: str) -> str:
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be ISO 8601 UTC values")
    return value


Timestamp = Annotated[str, AfterValidator(_utc_timestamp)]


class JobHandle(DurableModel):
    job_id: JobId
    attempt: PositiveInt


class JobError(DurableModel):
    code: Literal[
        "invalid_spec",
        "unsupported_version",
        "incompatible",
        "executor_failure",
        "cancelled",
        "timeout",
        "missing_artifact",
        "corrupt_artifact",
        "conflict",
        "stale_parent",
    ]
    message: str
    retryable: bool = False


class _JobSpec(DurableModel):
    job_id: JobId
    run_id: RunId
    config_id: ConfigId
    stack_id: StackId
    model_spec_id: ModelSpecId
    execution_profile_id: ExecutionProfileId
    code_revision: str
    attempt: PositiveInt = 1
    created_at: Timestamp
    expected_outputs: tuple[ArtifactId, ...]
    seed: int

    def semantic_inputs(self) -> dict:
        return self.model_dump(
            mode="python", exclude={"job_id", "attempt", "created_at"}
        )

    @model_validator(mode="after")
    def check_job_id(self, info: ValidationInfo):
        if not self.expected_outputs or len(set(self.expected_outputs)) != len(
            self.expected_outputs
        ):
            raise ValueError("expected output identities must be nonempty and unique")
        expected = "job-" + content_hash(self.semantic_inputs())
        if info.context and info.context.get("derive_job_id"):
            object.__setattr__(self, "job_id", expected)
        if self.job_id != expected:
            raise ValueError("job_id does not match immutable semantic inputs")
        return self

    @classmethod
    def create(cls, **inputs):
        return cls.model_validate(
            {**inputs, "job_id": "pending"}, context={"derive_job_id": True}
        )


class TrainingJobSpec(_JobSpec):
    kind: Literal["training"] = "training"
    start: StateRef
    training_slice: DatasetRef
    expected_state: StateRef
    steps: PositiveInt
    verifier_version: str
    trainer: Literal["grpo", "rft"] = "grpo"


class EvaluationJobSpec(_JobSpec):
    kind: Literal["evaluation"] = "evaluation"
    states: tuple[StateRef, ...]
    dataset: DatasetRef
    eval_read_id: EvalReadId
    instrument_id: MeasurementInstrumentId
    template_ids: tuple[str, ...]
    k: PositiveInt
    k_ref: PositiveInt
    fresh_reference: bool
    split: Literal[
        "dev",
        "pilot",
        "screen",
        "gate_a",
        "gate_b",
        "transfer_guard",
        "transfer_sealed",
        "hidden_dev",
        "hidden_audit",
        "hidden_sealed",
    ]

    @model_validator(mode="after")
    def reference_rule(self):
        if self.split == "gate_b" and not self.fresh_reference:
            raise ValueError("Gate-B requires fresh reference generation")
        if not self.states or not self.template_ids:
            raise ValueError("evaluation requires states and templates")
        return self


JobSpec = Annotated[TrainingJobSpec | EvaluationJobSpec, Field(discriminator="kind")]


class _JobResult(DurableModel):
    job_id: JobId
    run_id: RunId
    config_id: ConfigId
    stack_id: StackId
    model_spec_id: ModelSpecId
    execution_profile_id: ExecutionProfileId
    code_revision: str
    attempt: PositiveInt
    started_at: Timestamp
    finished_at: Timestamp
    status: Literal["succeeded", "failed", "cancelled"]
    artifacts: tuple[ArtifactRef, ...] = ()
    error: JobError | None = None

    def semantic_outputs(self) -> dict:
        """Attempt delivery metadata cannot turn identical outputs into conflict."""
        return self.model_dump(
            mode="python", exclude={"attempt", "started_at", "finished_at"}
        )

    @property
    def result_digest(self) -> str:
        return content_hash(self.semantic_outputs())

    @model_validator(mode="after")
    def failure_rule(self):
        if datetime.fromisoformat(self.finished_at) < datetime.fromisoformat(
            self.started_at
        ):
            raise ValueError("finished_at precedes started_at")
        if (self.status == "succeeded") != (self.error is None):
            raise ValueError(
                "success has no error; failure/cancellation requires an error"
            )
        if self.status != "succeeded" and self.artifacts:
            raise ValueError("failed attempts cannot publish finalized outputs")
        return self


class TrainingJobResult(_JobResult):
    kind: Literal["training"] = "training"
    start: StateRef
    training_slice: DatasetRef
    state: StateRef | None = None
    global_step: Annotated[int, Field(ge=0)] | None = None
    rollouts: ArtifactRef | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def outputs_rule(self):
        if self.status == "succeeded" and (
            self.state is None or self.rollouts is None or self.global_step is None
        ):
            raise ValueError(
                "training success requires state, global_step and raw rollouts"
            )
        if self.status != "succeeded" and (
            self.state is not None or self.rollouts is not None
        ):
            raise ValueError("failed attempts cannot publish a state or rollouts")
        return self


class EvaluationJobResult(_JobResult):
    kind: Literal["evaluation"] = "evaluation"
    states: tuple[StateRef, ...]
    dataset: DatasetRef
    eval_read_id: EvalReadId
    evidence: EvalEvidence | None = None

    @model_validator(mode="after")
    def outputs_rule(self):
        if (self.status == "succeeded") != (self.evidence is not None):
            raise ValueError(
                "evaluation success requires raw evidence; failure cannot publish it"
            )
        if self.evidence is not None:
            for name in (
                "job_id",
                "eval_read_id",
                "config_id",
                "stack_id",
                "dataset",
                "states",
            ):
                if getattr(self, name) != getattr(self.evidence, name):
                    raise ValueError(f"evidence {name} differs from result")
        return self


JobResult = Annotated[
    TrainingJobResult | EvaluationJobResult, Field(discriminator="kind")
]


def decode_job_spec(blob: bytes) -> TrainingJobSpec | EvaluationJobSpec:
    # from_bytes also rejects duplicate keys, including nested ones.
    value = TypeAdapter(JobSpec).validate_json(blob)
    return type(value).from_bytes(blob)


def decode_job_result(blob: bytes) -> TrainingJobResult | EvaluationJobResult:
    value = TypeAdapter(JobResult).validate_json(blob)
    return type(value).from_bytes(blob)
