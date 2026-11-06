"""Generic atomic effects. Champion/promotion policy belongs to the controller."""

from typing import Literal

from pydantic import JsonValue, model_validator

from .base import DurableModel
from .jobs import JobHandle, JobResult, JobStatus
from .manifests import StateManifest
from .models import PositiveInt
from .refs import ArtifactId, JobId, RunId, StateRef

METADATA_SCHEMA_VERSION = 2


class MetadataRecord(DurableModel):
    namespace: str
    key: str
    value: dict[str, JsonValue]


class PointerUpdate(DurableModel):
    name: str
    expected: StateRef | None
    replacement: StateRef


class BudgetCharge(DurableModel):
    budget: str
    item_id: str
    units: PositiveInt = 1


class AcceptanceEffects(DurableModel):
    run_id: RunId
    expected_parent: StateRef | None
    states: tuple[StateManifest, ...] = ()
    records: tuple[MetadataRecord, ...] = ()
    pointers: tuple[PointerUpdate, ...] = ()
    charges: tuple[BudgetCharge, ...] = ()


class AcceptanceReceipt(DurableModel):
    job_id: JobId
    disposition: Literal[
        "accepted",
        "duplicate",
        "conflict",
        "stale_parent",
        "missing_artifact",
        "corrupt_artifact",
        "budget_exhausted",
    ]


class StagedArtifact(DurableModel):
    staging_id: str
    expected: ArtifactId


class AttemptRecord(DurableModel):
    handle: JobHandle
    status: JobStatus
    result: JobResult | None = None

    @model_validator(mode="after")
    def matching_result(self):
        if self.result is not None and (
            self.result.job_id != self.handle.job_id
            or self.result.attempt != self.handle.attempt
            or self.result.status != self.status
        ):
            raise ValueError("attempt handle, status and result must agree")
        return self


class AuditEvent(DurableModel):
    event_id: int
    run_id: RunId
    job_id: JobId | None
    kind: str
    payload: dict[str, JsonValue]
    created_at: str
