"""Implementation boundaries. These protocols perform no I/O on import."""

from typing import Protocol

from .generation import (
    EvalGenerationBatch,
    EvalGenerationRequest,
    RolloutBatch,
    RolloutRequest,
)
from .jobs import JobHandle, JobResult, JobSpec, JobStatus
from .manifests import ArtifactManifest, StateManifest
from .persistence import (
    AcceptanceEffects,
    AcceptanceReceipt,
    MetadataRecord,
    StagedArtifact,
)
from .refs import ArtifactId, ArtifactRef, JobId, RunId, StateRef


class JobExecutor(Protocol):
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def result(self, handle: JobHandle) -> JobResult: ...
    def cancel(self, handle: JobHandle) -> None: ...


class RolloutProvider(Protocol):
    def generate(self, request: RolloutRequest) -> RolloutBatch: ...


class EvalGenerationProvider(Protocol):
    def generate(self, request: EvalGenerationRequest) -> EvalGenerationBatch: ...


class MetadataStore(Protocol):
    def reserve_job(self, spec: JobSpec) -> JobHandle: ...
    def record_attempt(self, handle: JobHandle, status: JobStatus) -> None: ...
    def get_result(self, job_id: JobId) -> JobResult | None: ...
    def get_state(self, ref: StateRef) -> StateManifest: ...
    def get_record(
        self, run_id: RunId, namespace: str, key: str
    ) -> MetadataRecord | None: ...
    def accept_result(
        self, result: JobResult, effects: AcceptanceEffects
    ) -> AcceptanceReceipt:
        """Atomically accept finalized refs and apply effects once per job.

        Verify identity and inputs against the reserved spec. Compare semantic
        result digests, excluding attempt/timestamps, for duplicate delivery.
        Conflicting/stale/missing artifacts are audited with no effects.
        Stores implement no gate policy.
        """
        ...


class ArtifactStore(Protocol):
    def stage(self, expected: ArtifactId) -> StagedArtifact: ...
    def write(self, staged: StagedArtifact, component: str, data: bytes) -> None: ...
    def validate(self, staged: StagedArtifact, manifest: ArtifactManifest) -> None: ...
    def finalize(
        self, staged: StagedArtifact, manifest: ArtifactManifest
    ) -> ArtifactRef:
        """Validate and atomically publish immutable content or fail without effects."""
        ...

    def resolve(self, ref: ArtifactRef) -> ArtifactManifest: ...
    def read(self, ref: ArtifactRef, component: str) -> bytes: ...
