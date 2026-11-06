"""Deterministic, CPU-only execution of synthetic contract fixtures.

These bytes are engineering fixtures, never model or scientific evidence.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from autopost.contracts import (
    AcceptanceEffects,
    ArtifactComponent,
    ArtifactManifest,
    ArtifactStore,
    BudgetCharge,
    DatasetRef,
    EvalEvidence,
    EvaluationJobResult,
    JobError,
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
    MetadataRecord,
    MetadataStore,
    PointerUpdate,
    StateManifest,
    StateRef,
    TrainingJobResult,
    TrainingJobSpec,
    canonical_bytes,
    content_hash,
    decode_job_spec,
)
from autopost.store.errors import StoreError

FixtureScenario = Literal[
    "success",
    "duplicate_success",
    "conflict",
    "stale_parent",
    "incomplete_artifact",
    "executor_failure",
    "crash_before_finalize",
    "crash_after_finalize",
]
COMPONENTS = ("adapter", "optimizer", "scheduler", "rng", "trainer", "sampler_cursor")


class SimulatedCrash(RuntimeError):
    """An intentional fixture interruption, not a serializable job error."""


def _manifest(
    identity: str, kind: str, producer: str, payload: dict[str, bytes]
) -> ArtifactManifest:
    return ArtifactManifest.create(
        artifact_id=identity,
        kind=kind,
        producer_job_id=producer,
        components=tuple(
            ArtifactComponent(
                name=name,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                media_type="application/json",
            )
            for name, data in payload.items()
        ),
    )


def _publish(
    artifacts: ArtifactStore, manifest: ArtifactManifest, payload: dict[str, bytes]
) -> None:
    staged = artifacts.stage(manifest.artifact.artifact_id)
    for name, data in payload.items():
        artifacts.write(staged, name, data)
    artifacts.validate(staged, manifest)
    artifacts.finalize(staged, manifest)


def initialize_fixture_run(
    metadata: MetadataStore,
    artifacts: ArtifactStore,
    *,
    run_id: str,
    config_id: str,
    stack_id: str,
    dataset: DatasetRef,
    state_id: str = "fixture-base",
    budget: int = 10,
) -> StateManifest:
    """Import a synthetic root and explicitly initialize generic pointers/budget."""
    suffix = content_hash({"run_id": run_id, "state_id": state_id})[:24]
    producer = "fixture-import-" + suffix
    payload = {
        name: canonical_bytes(
            {"fixture_only": True, "role": name, "state_id": state_id}
        )
        for name in COMPONENTS
    }
    manifest = _manifest("fixture-root-" + suffix, "state", producer, payload)
    _publish(artifacts, manifest, payload)
    root = StateManifest(
        state=StateRef(state_id=state_id),
        parent=None,
        job_id=producer,
        global_step=0,
        config_id=config_id,
        stack_id=stack_id,
        training_slice=dataset,
        verifier_version="fixture-v1",
        **{name: manifest.artifact for name in COMPONENTS},
    )
    metadata.register_run(run_id)
    metadata.register_initial_state(run_id, root)
    if metadata.get_pointer(run_id, "cursor") is None:
        metadata.initialize_pointer(run_id, "cursor", root.state)
    metadata.define_budget(run_id, "operations", budget)
    return root


class FixtureExecutor:
    def __init__(
        self,
        metadata: MetadataStore,
        artifacts: ArtifactStore,
        *,
        scenario: FixtureScenario = "success",
    ):
        if scenario not in {
            "success",
            "duplicate_success",
            "conflict",
            "stale_parent",
            "incomplete_artifact",
            "executor_failure",
            "crash_before_finalize",
            "crash_after_finalize",
        }:
            raise ValueError("unknown fixture scenario")
        self.metadata = metadata
        self.artifacts = artifacts
        self.scenario = scenario

    @staticmethod
    def _common(spec):
        return {
            **{
                name: getattr(spec, name)
                for name in (
                    "job_id",
                    "run_id",
                    "config_id",
                    "stack_id",
                    "model_spec_id",
                    "execution_profile_id",
                    "code_revision",
                    "attempt",
                )
            },
            "started_at": spec.created_at,
            "finished_at": spec.created_at,
        }

    def _failure(self, spec, code, status="failed"):
        common = {
            **self._common(spec),
            "status": status,
            "error": JobError(
                code=code,
                message="synthetic fixture " + code,
                retryable=code == "executor_failure",
            ),
        }
        if isinstance(spec, TrainingJobSpec):
            return TrainingJobResult(
                **common, start=spec.start, training_slice=spec.training_slice
            )
        return EvaluationJobResult(
            **common,
            states=spec.states,
            dataset=spec.dataset,
            eval_read_id=spec.eval_read_id,
        )

    def submit(self, spec: JobSpec) -> JobHandle:
        spec = decode_job_spec(spec.to_bytes())
        if isinstance(spec, TrainingJobSpec) and len(spec.expected_outputs) < 2:
            raise StoreError(
                "training fixtures require state and rollout output identities"
            )
        handle = self.metadata.reserve_job(spec)
        attempt = self.metadata.get_attempt(handle)
        if attempt.result is not None:
            return handle
        if attempt.status in {"failed", "cancelled", "succeeded"}:
            raise StoreError(
                "a terminal attempt cannot be executed again; reserve a new attempt"
            )
        self.metadata.record_attempt(handle, "running")
        if self.scenario == "executor_failure":
            self.metadata.record_result(self._failure(spec, "executor_failure"))
            return handle
        outputs = []
        for index, identity in enumerate(spec.expected_outputs):
            if isinstance(spec, TrainingJobSpec) and index == 0:
                kind = "state"
                payload = {
                    name: canonical_bytes(
                        {
                            "fixture_only": True,
                            "job_id": spec.job_id,
                            "role": name,
                            "seed": spec.seed,
                            "steps": spec.steps,
                        }
                    )
                    for name in COMPONENTS
                }
            elif (isinstance(spec, TrainingJobSpec) and index == 1) or (
                not isinstance(spec, TrainingJobSpec) and index == 0
            ):
                kind = (
                    "rollouts" if isinstance(spec, TrainingJobSpec) else "completions"
                )
                payload = {
                    "raw": canonical_bytes(
                        {
                            "fixture_only": True,
                            "job_id": spec.job_id,
                            "seed": spec.seed,
                            "completions": [
                                {
                                    "prompt_id": "fixture-prompt",
                                    "text": "Answer: 2",
                                    "truncated": False,
                                }
                            ],
                        }
                    )
                }
            else:
                kind = "report"
                payload = {
                    "report": canonical_bytes(
                        {"fixture_only": True, "job_id": spec.job_id, "slot": index}
                    )
                }
            manifest = _manifest(identity, kind, spec.job_id, payload)
            outputs.append(manifest.artifact)
            staged = self.artifacts.stage(identity)
            for name, data in payload.items():
                self.artifacts.write(staged, name, data)
            if self.scenario == "crash_before_finalize":
                raise SimulatedCrash("fixture crash before artifact finalization")
            if (
                self.scenario != "incomplete_artifact"
                or index != len(spec.expected_outputs) - 1
            ):
                self.artifacts.validate(staged, manifest)
                self.artifacts.finalize(staged, manifest)
        if self.scenario == "crash_after_finalize":
            raise SimulatedCrash(
                "fixture crash after finalization before result journal/acceptance"
            )
        common = {
            **self._common(spec),
            "status": "succeeded",
            "artifacts": tuple(outputs),
        }
        if isinstance(spec, TrainingJobSpec):
            parent = self.metadata.get_state(spec.start)
            result = TrainingJobResult(
                **common,
                start=spec.start,
                training_slice=spec.training_slice,
                state=spec.expected_state,
                global_step=parent.global_step + spec.steps,
                rollouts=outputs[1],
                metrics={
                    "fixture_only": True,
                    "synthetic_metric": 1.0 if self.scenario == "conflict" else 0.0,
                },
            )
        else:
            evidence = EvalEvidence(
                job_id=spec.job_id,
                eval_read_id=spec.eval_read_id,
                instrument_id=spec.instrument_id,
                config_id=spec.config_id,
                stack_id=spec.stack_id,
                dataset=spec.dataset,
                states=spec.states,
                completions=outputs[0],
                scorer_version="fixture-exact-v1",
                template_ids=spec.template_ids,
                seeds=(spec.seed,),
                fresh_reference=spec.fresh_reference,
                aggregates={
                    "fixture_only": True,
                    "accuracy": 0.5 if self.scenario == "conflict" else 1.0,
                },
            )
            result = EvaluationJobResult(
                **common,
                states=spec.states,
                dataset=spec.dataset,
                eval_read_id=spec.eval_read_id,
                evidence=evidence,
            )
        self.metadata.record_result(result)
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        return self.metadata.get_attempt(handle).status

    def result(self, handle: JobHandle) -> JobResult:
        result = self.metadata.get_attempt(handle).result
        if result is None:
            raise StoreError("fixture attempt has no durable result yet")
        return result

    def cancel(self, handle: JobHandle) -> None:
        attempt = self.metadata.get_attempt(handle)
        if attempt.status in {"succeeded", "failed", "cancelled"}:
            return
        spec = self.metadata.get_job(handle.job_id).model_copy(
            update={"attempt": handle.attempt}
        )
        self.metadata.record_result(self._failure(spec, "cancelled", "cancelled"))

    def effects(
        self,
        handle: JobHandle,
        *,
        pointer: str | None = "cursor",
        budget: str | None = "operations",
    ) -> AcceptanceEffects:
        """Build fixture effects; a real controller will supply its own policy."""
        spec = self.metadata.get_job(handle.job_id)
        result = self.result(handle)
        if result.status != "succeeded":
            raise StoreError("failed fixtures have no semantic effects")
        states, pointers = (), ()
        if isinstance(spec, TrainingJobSpec):
            state = StateManifest(
                state=result.state,
                parent=spec.start,
                job_id=spec.job_id,
                global_step=result.global_step,
                config_id=spec.config_id,
                stack_id=spec.stack_id,
                training_slice=spec.training_slice,
                verifier_version=spec.verifier_version,
                **{name: result.artifacts[0] for name in COMPONENTS},
            )
            states = (state,)
            if pointer is not None:
                expected = (
                    StateRef(state_id="fixture-stale-parent")
                    if self.scenario == "stale_parent"
                    else spec.start
                )
                pointers = (
                    PointerUpdate(
                        name=pointer, expected=expected, replacement=result.state
                    ),
                )
        return AcceptanceEffects(
            run_id=spec.run_id,
            expected_parent=spec.start if isinstance(spec, TrainingJobSpec) else None,
            states=states,
            pointers=pointers,
            records=(
                MetadataRecord(
                    namespace="fixture-results",
                    key=spec.job_id,
                    value={"fixture_only": True, "result_digest": result.result_digest},
                ),
            ),
            charges=(BudgetCharge(budget=budget, item_id=spec.job_id),)
            if budget is not None
            else (),
        )
