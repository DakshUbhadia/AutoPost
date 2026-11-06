"""Generic atomic effects. Champion/promotion policy belongs to the controller."""

from typing import Literal

from pydantic import JsonValue

from .base import DurableModel
from .manifests import StateManifest
from .models import PositiveInt
from .refs import ArtifactId, JobId, RunId, StateRef


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
        "accepted", "duplicate", "conflict", "stale_parent", "missing_artifact"
    ]


class StagedArtifact(DurableModel):
    staging_id: str
    expected: ArtifactId
