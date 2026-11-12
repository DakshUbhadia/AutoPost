"""Versioned training-signal metadata. Raw text lives in evidence artifacts."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import DurableModel
from .refs import ArtifactRef, DatasetRef, Digest, OpaqueId

Count = Annotated[int, Field(ge=0)]


class PassRateObservation(DurableModel):
    group_id: OpaqueId
    task_id: OpaqueId
    task_digest: Digest
    state_id: OpaqueId
    checker_id: OpaqueId
    n: Annotated[int, Field(gt=0)]
    n_correct: Count
    wrong_answers: dict[str, Count]
    evidence: ArtifactRef

    @model_validator(mode="after")
    def counts(self):
        if self.n_correct + sum(self.wrong_answers.values()) > self.n:
            raise ValueError("observation counts exceed samples")
        return self


class LabelReview(DurableModel):
    review_id: OpaqueId
    task_id: OpaqueId
    task_digest: Digest
    method: Literal["human", "deterministic", "second_verifier"]
    checker_id: OpaqueId
    reviewer: OpaqueId
    decision: Literal["confirmed_bad", "confirmed_good", "corrected", "irreparable"]
    reason: Annotated[str, Field(min_length=1)]
    independence_basis: Annotated[str, Field(min_length=1)]
    evidence: ArtifactRef

    @model_validator(mode="after")
    def explanation(self):
        if not self.reason.strip() or not self.independence_basis.strip():
            raise ValueError("review requires a reason and independent evidence basis")
        if self.checker_id.startswith("exact_numeric"):
            raise ValueError("evaluation scorer cannot review training labels")
        return self


class CurriculumConfig(DurableModel):
    group_size: Annotated[int, Field(ge=2, le=256)] = 8
    min_signal_prob: Annotated[float, Field(gt=0, lt=1)] = 0.5
    estimate_k: Annotated[int, Field(ge=1)] = 8
    edge_margin: Annotated[float, Field(ge=0, le=0.5)] = 0.05
    max_share: Annotated[float, Field(gt=0, le=1)] = 0.5
    low_entropy_max_share: Annotated[float, Field(ge=0, le=1)] = 0.1
    cluster_exposure_cap: Annotated[int, Field(ge=1)] = 4
    consensus_share: Annotated[float, Field(gt=0.5, le=1)] = 0.6
    min_estimates: Annotated[int, Field(ge=2)] = 2
    persistent_promotions: Annotated[int, Field(ge=1)] = 3

    @classmethod
    def from_experiment(cls, experiment):
        config = experiment["curriculum"]
        if (
            config["quarantine"]["action"] != "flag"
            or config["quarantine"]["exclusion_requires"] != "independent_verification"
        ):
            raise ValueError("consensus must flag and require independent verification")
        if set(config["quotas"]["by"]) != {"source_id", "skill"}:
            raise ValueError("source and skill quotas are required")
        return cls(
            **{
                key: config[key]
                for key in (
                    "group_size",
                    "min_signal_prob",
                    "estimate_k",
                    "edge_margin",
                    "cluster_exposure_cap",
                )
            },
            max_share=config["quotas"]["max_share"],
            low_entropy_max_share=config["quotas"]["low_entropy_max_share"],
            consensus_share=config["quarantine"]["consensus_share"],
            min_estimates=config["quarantine"]["min_estimates"],
            persistent_promotions=config["audit_flag_persistent_inband_promotions"],
        )


class SliceReservation(DurableModel):
    slice_id: OpaqueId
    dataset: DatasetRef
    config_digest: Digest
    seed: int
    n_prompts: Annotated[int, Field(gt=0)]
    excluded_checkers: tuple[OpaqueId, ...] = ()
    task_ids: tuple[OpaqueId, ...]
