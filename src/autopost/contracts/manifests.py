"""Immutable content manifests; physical layout belongs to ArtifactStore."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .base import DurableModel
from .refs import (
    ArtifactRef,
    ConfigId,
    DatasetRef,
    Digest,
    JobId,
    StackId,
    StateRef,
    VerifierVersion,
)
from .serialization import content_hash

# Logical component names, deliberately not relative filesystem paths either.
ComponentName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
]


class ArtifactComponent(DurableModel):
    name: ComponentName
    sha256: Digest
    size_bytes: Annotated[int, Field(ge=0)]
    media_type: str


class ArtifactManifest(DurableModel):
    artifact: ArtifactRef
    kind: Literal["state", "dataset", "rollouts", "completions", "report", "evidence"]
    components: tuple[ArtifactComponent, ...]
    producer_job_id: JobId

    @property
    def content_digest(self) -> str:
        """Hash the complete descriptor, excluding only its self-reference.

        Component hashes cover bytes; this hash also binds names, sizes, media
        types, producer and logical artifact identity to those bytes.
        """
        return content_hash(
            self.model_dump(mode="python", exclude={"artifact": {"sha256"}})
        )

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        kind: str,
        components: tuple[ArtifactComponent, ...],
        producer_job_id: JobId,
    ):
        manifest = cls(
            artifact=ArtifactRef(artifact_id=artifact_id, sha256="0" * 64),
            kind=kind,
            components=tuple(sorted(components, key=lambda item: item.name)),
            producer_job_id=producer_job_id,
        )
        return cls(
            **{
                **manifest.model_dump(mode="python"),
                "artifact": ArtifactRef(
                    artifact_id=artifact_id, sha256=manifest.content_digest
                ),
            }
        )

    @model_validator(mode="after")
    def unique_components(self):
        names = [item.name for item in self.components]
        if len(names) != len(set(names)) or not names:
            raise ValueError("artifact components must be nonempty and uniquely named")
        return self


class StateManifest(DurableModel):
    state: StateRef
    parent: StateRef | None
    job_id: JobId
    global_step: Annotated[int, Field(ge=0)]
    config_id: ConfigId
    stack_id: StackId
    training_slice: DatasetRef
    verifier_version: VerifierVersion
    adapter: ArtifactRef
    optimizer: ArtifactRef
    scheduler: ArtifactRef
    rng: ArtifactRef
    trainer: ArtifactRef
    sampler_cursor: ArtifactRef
