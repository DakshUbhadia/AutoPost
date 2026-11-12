"""Immutable development dataset identities and pinned build settings."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import DurableModel
from .refs import DatasetRef, Digest
from .serialization import content_hash
from .tasks import DEVELOPMENT_SPLITS


class SourceSpec(DurableModel):
    source_id: str
    revision: str
    role: Literal["train", "transfer", "benchmark"]
    family_id: str
    license_id: str
    fixture_only: bool = False
    adapter: Literal["canonical", "gsm8k", "math"] = "canonical"

    @model_validator(mode="after")
    def pinned(self):
        import re

        if any(
            not getattr(self, name).strip()
            for name in ("source_id", "revision", "family_id", "license_id")
        ):
            raise ValueError("source provenance fields must be nonempty")

        if not self.fixture_only and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("curated sources require an exact 40-hex revision")
        return self


class BuildConfig(DurableModel):
    mode: Literal["development"] = "development"
    seed: int = 0
    minhash_ngram: Annotated[int, Field(ge=1, le=10)] = 5
    minhash_permutations: Annotated[int, Field(ge=16, le=256)] = 64
    minhash_threshold: Annotated[float, Field(gt=0, le=1)] = 0.8
    embedding_cos: Annotated[float, Field(gt=0, le=1)] = 0.92
    split_clusters: dict[str, int]
    hidden_instances_per_program: Annotated[int, Field(ge=1, le=10000)] = 8
    hidden_property_trials: Annotated[int, Field(ge=100, le=10000)] = 256

    @model_validator(mode="after")
    def development_only(self):
        if any(
            name not in DEVELOPMENT_SPLITS or name == "hidden_dev"
            for name in self.split_clusters
        ):
            raise ValueError(
                "only disposable development splits may be requested; hidden_dev is program-derived"
            )
        if any(
            type(size) is not int or size < 0 for size in self.split_clusters.values()
        ):
            raise ValueError("split sizes must be nonnegative cluster counts")
        return self


class DatasetManifest(DurableModel):
    dataset: DatasetRef
    mode: Literal["development"] = "development"
    fixture_only: bool
    split: str
    build_id: Digest
    config_id: Digest
    input_digest: Digest
    tasks_sha256: Digest
    report_sha256: Digest
    task_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    sources: tuple[SourceSpec, ...]
    program_digests: tuple[str, ...] = ()

    @property
    def content_digest(self) -> str:
        return content_hash(self.model_dump(mode="python", exclude={"dataset"}))

    @model_validator(mode="after")
    def verify(self):
        if self.split not in DEVELOPMENT_SPLITS:
            raise ValueError("research partitions cannot be built in Sprint 3")
        if (
            len(set(self.task_ids)) != len(self.task_ids)
            or tuple(sorted(set(self.cluster_ids))) != self.cluster_ids
        ):
            raise ValueError("dataset identities must be unique and canonical")
        if (
            self.dataset.sha256 != self.content_digest
            or self.dataset.dataset_id != "dataset-" + self.content_digest
        ):
            raise ValueError("dataset reference does not match manifest content")
        return self

    @classmethod
    def create(cls, **fields):
        payload = {
            "schema_version": 1,
            "mode": "development",
            "program_digests": (),
            **fields,
        }

        def plain(value):
            if isinstance(value, DurableModel):
                return value.model_dump(mode="python")
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value

        digest = content_hash({key: plain(value) for key, value in payload.items()})
        return cls(
            dataset=DatasetRef(dataset_id="dataset-" + digest, sha256=digest), **payload
        )
