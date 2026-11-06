"""Raw measurement evidence is independent of derived aggregates."""

from pydantic import JsonValue

from .base import DurableModel
from .refs import (
    ArtifactRef,
    ConfigId,
    DatasetRef,
    EvalReadId,
    JobId,
    MeasurementInstrumentId,
    ScorerVersion,
    StackId,
    StateRef,
)


class EvalEvidence(DurableModel):
    eval_read_id: EvalReadId
    job_id: JobId
    instrument_id: MeasurementInstrumentId
    config_id: ConfigId
    stack_id: StackId
    dataset: DatasetRef
    states: tuple[StateRef, ...]
    completions: ArtifactRef
    scorer_version: ScorerVersion
    template_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    fresh_reference: bool
    aggregates: dict[str, JsonValue]
