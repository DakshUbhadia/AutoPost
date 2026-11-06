"""Opaque, location-independent identities (never storage locators)."""

from typing import Annotated

from pydantic import StringConstraints

from .base import DurableModel

OpaqueId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")
]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RunId = OpaqueId
JobId = OpaqueId
StateId = OpaqueId
ArtifactId = OpaqueId
DatasetId = OpaqueId
PromptId = OpaqueId
ClusterId = OpaqueId
EvalReadId = OpaqueId
MeasurementInstrumentId = OpaqueId
ConfigId = Digest
StackId = Digest
ModelSpecId = Digest
ExecutionProfileId = Digest
VerifierVersion = OpaqueId
ScorerVersion = OpaqueId
PreregistrationId = OpaqueId


class StateRef(DurableModel):
    state_id: StateId


class ArtifactRef(DurableModel):
    artifact_id: ArtifactId
    sha256: Digest


class DatasetRef(DurableModel):
    dataset_id: DatasetId
    sha256: Digest
