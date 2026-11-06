"""Transactional job journal and generic metadata effects over SQLite.

No promotion, restore, pruning or controller policy belongs in this backend.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from pydantic import TypeAdapter

from autopost.contracts import (
    AcceptanceEffects,
    AcceptanceReceipt,
    ArtifactStore,
    AttemptRecord,
    AuditEvent,
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
    MetadataRecord,
    RunId,
    StateManifest,
    StateRef,
    TrainingJobSpec,
    canonical_bytes,
    content_hash,
    decode_job_result,
    decode_job_spec,
)

from .db import connect, migrate_db
from .errors import (
    ConflictError,
    CorruptArtifactError,
    MissingArtifactError,
    StoreError,
)

STATE_COMPONENTS = (
    "adapter",
    "optimizer",
    "scheduler",
    "rng",
    "trainer",
    "sampler_cursor",
)
TERMINAL = {"succeeded", "failed", "cancelled"}


class _Rejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class SQLiteMetadataStore:
    def __init__(self, path: str | Path, artifacts: ArtifactStore):
        self.path = Path(path)
        self.artifacts = artifacts
        migrate_db(self.path)

    @contextmanager
    def _transaction(self):
        connection = connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _reader(self):
        connection = connect(self.path)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _run_id(run_id: RunId) -> str:
        return TypeAdapter(RunId).validate_python(run_id)

    @staticmethod
    def _name(name: str) -> str:
        if type(name) is not str or not name:
            raise StoreError("metadata names must be nonempty strings")
        MetadataRecord(namespace="validation", key=name, value={}).to_bytes()
        return name

    @staticmethod
    def _require_run(db, run_id):
        if not db.execute("SELECT 1 FROM ap_runs WHERE run_id=?", (run_id,)).fetchone():
            raise StoreError("run must be registered before writing metadata")

    @staticmethod
    def _event(db, run_id, job_id, kind, payload):
        db.execute(
            "INSERT INTO ap_events(run_id,job_id,kind,payload) VALUES (?,?,?,?)",
            (run_id, job_id, kind, canonical_bytes(payload)),
        )

    def register_run(self, run_id: RunId) -> None:
        run_id = self._run_id(run_id)
        with self._transaction() as db:
            if db.execute(
                "INSERT OR IGNORE INTO ap_runs(run_id) VALUES (?)", (run_id,)
            ).rowcount:
                self._event(db, run_id, None, "run_registered", {})

    @staticmethod
    def _state_row(db, run_id, ref):
        row = db.execute(
            "SELECT * FROM ap_states WHERE state_id=?", (ref.state_id,)
        ).fetchone()
        if row is None or row["run_id"] != run_id:
            raise _Rejected("stale_parent")
        return row

    def register_initial_state(self, run_id: RunId, manifest: StateManifest) -> None:
        run_id = self._run_id(run_id)
        manifest = StateManifest.from_bytes(manifest.to_bytes())
        if manifest.parent is not None or manifest.global_step != 0:
            raise StoreError("initial imports require a root state at step zero")
        with self._transaction() as db:
            self._require_run(db, run_id)
            old = db.execute(
                "SELECT run_id,manifest FROM ap_states WHERE state_id=?",
                (manifest.state.state_id,),
            ).fetchone()
            if old:
                if old["run_id"] != run_id or old["manifest"] != manifest.to_bytes():
                    raise ConflictError(
                        "initial state identity already has different content"
                    )
                return
            if db.execute("SELECT 1 FROM ap_jobs WHERE run_id=?", (run_id,)).fetchone():
                raise StoreError(
                    "initial states must be imported before reserving jobs"
                )
            manifests = {}
            for name in STATE_COMPONENTS:
                ref = getattr(manifest, name)
                artifact = self.artifacts.resolve(ref)
                if (
                    artifact.kind != "state"
                    or artifact.producer_job_id != manifest.job_id
                ):
                    raise StoreError(
                        "initial state artifact has incompatible provenance"
                    )
                manifests[ref.artifact_id] = artifact
            for artifact in manifests.values():
                self._insert_artifact(db, artifact)
            self._insert_state(db, run_id, manifest)
            self._event(
                db,
                run_id,
                None,
                "initial_state_registered",
                {"state_id": manifest.state.state_id},
            )

    def initialize_pointer(self, run_id: RunId, name: str, state: StateRef) -> None:
        run_id, name = self._run_id(run_id), self._name(name)
        state.to_bytes()
        with self._transaction() as db:
            self._require_run(db, run_id)
            try:
                self._state_row(db, run_id, state)
            except _Rejected as exc:
                raise StoreError(
                    "pointer target is not an accepted state in this run"
                ) from exc
            row = db.execute(
                "SELECT state_id FROM ap_pointers WHERE run_id=? AND name=?",
                (run_id, name),
            ).fetchone()
            if row:
                if row[0] != state.state_id:
                    raise ConflictError(
                        "pointer already initialized; use atomic compare-and-set effects"
                    )
                return
            db.execute(
                "INSERT INTO ap_pointers VALUES (?,?,?)", (run_id, name, state.state_id)
            )
            self._event(
                db,
                run_id,
                None,
                "pointer_initialized",
                {"name": name, "state_id": state.state_id},
            )

    def get_pointer(self, run_id: RunId, name: str) -> StateRef | None:
        with self._reader() as db:
            row = db.execute(
                "SELECT state_id FROM ap_pointers WHERE run_id=? AND name=?",
                (self._run_id(run_id), self._name(name)),
            ).fetchone()
            return StateRef(state_id=row[0]) if row else None

    def define_budget(self, run_id: RunId, name: str, limit: int) -> None:
        run_id, name = self._run_id(run_id), self._name(name)
        if type(limit) is not int or not 0 <= limit <= 2**63 - 1:
            raise StoreError("budget limit must be a nonnegative SQLite integer")
        with self._transaction() as db:
            self._require_run(db, run_id)
            old = db.execute(
                "SELECT limit_value FROM ap_budget_limits WHERE run_id=? AND name=?",
                (run_id, name),
            ).fetchone()
            if old:
                if old[0] != limit:
                    raise ConflictError(
                        "budget definition is immutable; use a new versioned name"
                    )
                return
            db.execute(
                "INSERT INTO ap_budget_limits(run_id,name,limit_value) VALUES (?,?,?)",
                (run_id, name, limit),
            )
            self._event(
                db, run_id, None, "budget_defined", {"name": name, "limit": limit}
            )

    def budget_remaining(self, run_id: RunId, name: str) -> int:
        with self._reader() as db:
            row = db.execute(
                "SELECT limit_value-used FROM ap_budget_limits WHERE run_id=? AND name=?",
                (self._run_id(run_id), self._name(name)),
            ).fetchone()
            if row is None:
                raise StoreError("budget is not defined")
            return row[0]

    @staticmethod
    def _job(db, job_id):
        row = db.execute("SELECT * FROM ap_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise StoreError("logical job is not reserved")
        return row, decode_job_spec(row["spec"])

    def get_job(self, job_id) -> JobSpec:
        with self._reader() as db:
            return self._job(db, job_id)[1]

    def reserve_job(self, spec: JobSpec) -> JobHandle:
        spec = decode_job_spec(spec.to_bytes())
        digest = content_hash(spec.semantic_inputs())
        handle = JobHandle(job_id=spec.job_id, attempt=spec.attempt)
        with self._transaction() as db:
            self._require_run(db, spec.run_id)
            row = db.execute(
                "SELECT semantic_digest FROM ap_jobs WHERE job_id=?", (spec.job_id,)
            ).fetchone()
            if row and row[0] != digest:
                raise ConflictError(
                    "logical job identity has conflicting semantic inputs"
                )
            if row is None:
                db.execute(
                    "INSERT INTO ap_jobs(job_id,run_id,spec,semantic_digest,status) VALUES (?,?,?,?,?)",
                    (spec.job_id, spec.run_id, spec.to_bytes(), digest, "pending"),
                )
            if (
                db.execute(
                    "SELECT 1 FROM ap_attempts WHERE job_id=? AND attempt=?",
                    (spec.job_id, spec.attempt),
                ).fetchone()
                is None
            ):
                db.execute(
                    "INSERT INTO ap_attempts(job_id,attempt,spec,status) VALUES (?,?,?,?)",
                    (spec.job_id, spec.attempt, spec.to_bytes(), "pending"),
                )
                self._refresh_job_status(db, spec.job_id)
                self._event(
                    db,
                    spec.run_id,
                    spec.job_id,
                    "attempt_reserved",
                    {"attempt": spec.attempt},
                )
        return handle

    @staticmethod
    def _refresh_job_status(db, job_id):
        accepted = db.execute(
            "SELECT 1 FROM ap_acceptances WHERE job_id=?", (job_id,)
        ).fetchone()
        status = (
            "succeeded"
            if accepted
            else db.execute(
                "SELECT status FROM ap_attempts WHERE job_id=? ORDER BY attempt DESC LIMIT 1",
                (job_id,),
            ).fetchone()[0]
        )
        db.execute("UPDATE ap_jobs SET status=? WHERE job_id=?", (status, job_id))

    @staticmethod
    def _attempt(db, handle):
        row = db.execute(
            "SELECT * FROM ap_attempts WHERE job_id=? AND attempt=?",
            (handle.job_id, handle.attempt),
        ).fetchone()
        if row is None:
            raise StoreError("job attempt is not reserved")
        return row

    def get_attempt(self, handle: JobHandle) -> AttemptRecord:
        handle.to_bytes()
        with self._reader() as db:
            row = self._attempt(db, handle)
            return AttemptRecord(
                handle=handle,
                status=row["status"],
                result=decode_job_result(row["result"]) if row["result"] else None,
            )

    def record_attempt(self, handle: JobHandle, status: JobStatus) -> None:
        handle.to_bytes()
        TypeAdapter(JobStatus).validate_python(status)
        with self._transaction() as db:
            row = self._attempt(db, handle)
            if row["status"] == status:
                return
            if row["status"] in TERMINAL or status == "pending":
                raise ConflictError(
                    "attempt status cannot move backwards or change terminal status"
                )
            db.execute(
                "UPDATE ap_attempts SET status=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND attempt=?",
                (status, handle.job_id, handle.attempt),
            )
            self._refresh_job_status(db, handle.job_id)
            _, spec = self._job(db, handle.job_id)
            self._event(
                db,
                spec.run_id,
                handle.job_id,
                "attempt_status",
                {"attempt": handle.attempt, "status": status},
            )

    @staticmethod
    def _bind(spec, result):
        for name in (
            "job_id",
            "run_id",
            "config_id",
            "stack_id",
            "model_spec_id",
            "execution_profile_id",
            "code_revision",
            "kind",
        ):
            if getattr(spec, name) != getattr(result, name):
                raise _Rejected("conflict")
        if isinstance(spec, TrainingJobSpec):
            if (
                spec.start != result.start
                or spec.training_slice != result.training_slice
            ):
                raise _Rejected("conflict")
            if result.status == "succeeded" and result.state != spec.expected_state:
                raise _Rejected("conflict")
        else:
            if (
                spec.states != result.states
                or spec.dataset != result.dataset
                or spec.eval_read_id != result.eval_read_id
            ):
                raise _Rejected("conflict")
            if result.evidence is not None:
                for name in ("instrument_id", "template_ids", "fresh_reference"):
                    if getattr(spec, name) != getattr(result.evidence, name):
                        raise _Rejected("conflict")
                if result.evidence.seeds != (spec.seed,):
                    raise _Rejected("conflict")
        if result.status == "succeeded":
            identities = [ref.artifact_id for ref in result.artifacts]
            if len(set(identities)) != len(identities) or set(identities) != set(
                spec.expected_outputs
            ):
                raise _Rejected("conflict")

    def _quarantine(self, db, spec, result, effects_blob, reason):
        db.execute(
            "INSERT INTO ap_quarantine(job_id,attempt,reason,result,effects) VALUES (?,?,?,?,?)",
            (result.job_id, result.attempt, reason, result.to_bytes(), effects_blob),
        )
        self._event(
            db,
            spec.run_id,
            result.job_id,
            "result_quarantined",
            {
                "attempt": result.attempt,
                "reason": reason,
                "result_digest": result.result_digest,
            },
        )

    def record_result(self, result: JobResult) -> None:
        result = decode_job_result(result.to_bytes())
        problem = False
        with self._transaction() as db:
            _, spec = self._job(db, result.job_id)
            row = self._attempt(
                db, JobHandle(job_id=result.job_id, attempt=result.attempt)
            )
            try:
                self._bind(spec, result)
                if (
                    row["result"]
                    and decode_job_result(row["result"]).result_digest
                    != result.result_digest
                ):
                    raise _Rejected("conflict")
                if row["status"] in TERMINAL and row["status"] != result.status:
                    raise _Rejected("conflict")
            except _Rejected:
                self._quarantine(db, spec, result, b"{}", "conflict")
                problem = True
            else:
                if row["result"] is None:
                    db.execute(
                        "UPDATE ap_attempts SET status=?,result=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND attempt=?",
                        (
                            result.status,
                            result.to_bytes(),
                            result.job_id,
                            result.attempt,
                        ),
                    )
                    self._refresh_job_status(db, result.job_id)
                    self._event(
                        db,
                        spec.run_id,
                        result.job_id,
                        "attempt_result",
                        {"attempt": result.attempt, "status": result.status},
                    )
        if problem:
            raise ConflictError(
                "attempt result conflicts with reserved inputs or previous delivery"
            )

    def get_result(self, job_id) -> JobResult | None:
        with self._reader() as db:
            row = db.execute(
                "SELECT result FROM ap_acceptances WHERE job_id=?", (job_id,)
            ).fetchone()
            return decode_job_result(row[0]) if row else None

    def get_state(self, ref: StateRef) -> StateManifest:
        ref.to_bytes()
        with self._reader() as db:
            row = db.execute(
                "SELECT manifest FROM ap_states WHERE state_id=?", (ref.state_id,)
            ).fetchone()
            if row is None:
                raise KeyError(ref.state_id)
            return StateManifest.from_bytes(row[0])

    def get_record(
        self, run_id: RunId, namespace: str, key: str
    ) -> MetadataRecord | None:
        with self._reader() as db:
            row = db.execute(
                "SELECT record FROM ap_records WHERE run_id=? AND namespace=? AND key=?",
                (self._run_id(run_id), namespace, key),
            ).fetchone()
            return MetadataRecord.from_bytes(row[0]) if row else None

    def events(self, run_id: RunId) -> tuple[AuditEvent, ...]:
        with self._reader() as db:
            return tuple(
                AuditEvent(
                    event_id=row["event_id"],
                    run_id=row["run_id"],
                    job_id=row["job_id"],
                    kind=row["kind"],
                    payload=json.loads(row["payload"]),
                    created_at=row["created_at"],
                )
                for row in db.execute(
                    "SELECT * FROM ap_events WHERE run_id=? ORDER BY event_id",
                    (self._run_id(run_id),),
                )
            )

    def quarantined_results(self, job_id) -> tuple[JobResult, ...]:
        with self._reader() as db:
            return tuple(
                decode_job_result(row[0])
                for row in db.execute(
                    "SELECT result FROM ap_quarantine WHERE job_id=? ORDER BY quarantine_id",
                    (job_id,),
                )
            )

    @staticmethod
    def _insert_artifact(db, manifest):
        old = db.execute(
            "SELECT manifest FROM ap_artifacts WHERE artifact_id=?",
            (manifest.artifact.artifact_id,),
        ).fetchone()
        if old and old[0] != manifest.to_bytes():
            raise ConflictError("accepted artifact is immutable")
        if old is None:
            db.execute(
                "INSERT INTO ap_artifacts VALUES (?,?,?)",
                (
                    manifest.artifact.artifact_id,
                    manifest.artifact.sha256,
                    manifest.to_bytes(),
                ),
            )

    @staticmethod
    def _insert_state(db, run_id, manifest):
        db.execute(
            "INSERT INTO ap_states VALUES (?,?,?,?,?,?)",
            (
                manifest.state.state_id,
                run_id,
                manifest.parent.state_id if manifest.parent else None,
                manifest.job_id,
                manifest.global_step,
                manifest.to_bytes(),
            ),
        )

    def _resolved_outputs(self, db, spec, result, effects):
        refs = {ref.artifact_id: ref for ref in result.artifacts}
        owned = set(refs)
        dependencies = [
            getattr(state, name)
            for state in effects.states
            for name in STATE_COMPONENTS
        ]
        if isinstance(spec, TrainingJobSpec):
            if result.rollouts.artifact_id not in owned:
                raise _Rejected("conflict")
            dependencies.append(result.rollouts)
        else:
            dependencies.append(result.evidence.completions)
        for ref in dependencies:
            old_ref = refs.get(ref.artifact_id)
            if old_ref and old_ref != ref:
                raise _Rejected("conflict")
            refs[ref.artifact_id] = ref
        manifests = {}
        for identity, ref in refs.items():
            try:
                manifest = self.artifacts.resolve(ref)
            except MissingArtifactError as exc:
                raise _Rejected("missing_artifact") from exc
            except CorruptArtifactError as exc:
                raise _Rejected("corrupt_artifact") from exc
            except ConflictError as exc:
                raise _Rejected("conflict") from exc
            old = db.execute(
                "SELECT manifest FROM ap_artifacts WHERE artifact_id=?", (identity,)
            ).fetchone()
            if old and old[0] != manifest.to_bytes():
                raise _Rejected("conflict")
            if identity in owned:
                if manifest.producer_job_id != result.job_id:
                    raise _Rejected("conflict")
            elif old is None:
                raise _Rejected("missing_artifact")
            manifests[identity] = manifest
        if isinstance(spec, TrainingJobSpec):
            if manifests[result.rollouts.artifact_id].kind != "rollouts":
                raise _Rejected("conflict")
            for state in effects.states:
                if any(
                    manifests[getattr(state, name).artifact_id].kind != "state"
                    for name in STATE_COMPONENTS
                ):
                    raise _Rejected("conflict")
        else:
            raw = manifests[result.evidence.completions.artifact_id]
            if raw.kind != "completions" or (
                spec.fresh_reference and raw.producer_job_id != spec.job_id
            ):
                raise _Rejected("conflict")
        return manifests

    def _validate_effects(self, db, spec, result, effects):
        if effects.run_id != spec.run_id:
            raise _Rejected("conflict")
        if isinstance(spec, TrainingJobSpec):
            parent = self._state_row(db, spec.run_id, spec.start)
            if effects.expected_parent != spec.start or len(effects.states) != 1:
                raise _Rejected("stale_parent")
            state = effects.states[0]
            if state.parent != spec.start:
                raise _Rejected("stale_parent")
            if (
                state.state != result.state
                or state.job_id != spec.job_id
                or state.config_id != spec.config_id
                or state.stack_id != spec.stack_id
                or state.training_slice != spec.training_slice
                or state.verifier_version != spec.verifier_version
                or state.global_step != result.global_step
                or state.global_step != parent["global_step"] + spec.steps
            ):
                raise _Rejected("conflict")
        else:
            if effects.states or effects.expected_parent is not None:
                raise _Rejected("conflict")
            for ref in spec.states:
                self._state_row(db, spec.run_id, ref)
        for state in effects.states:
            if db.execute(
                "SELECT 1 FROM ap_states WHERE state_id=?", (state.state.state_id,)
            ).fetchone():
                raise _Rejected("conflict")
        keys = [state.state.state_id for state in effects.states]
        if len(keys) != len(set(keys)):
            raise _Rejected("conflict")
        pointer_names = [pointer.name for pointer in effects.pointers]
        record_keys = [(record.namespace, record.key) for record in effects.records]
        charge_keys = [(charge.budget, charge.item_id) for charge in effects.charges]
        if any(
            len(items) != len(set(items))
            for items in (pointer_names, record_keys, charge_keys)
        ):
            raise _Rejected("conflict")
        for pointer in effects.pointers:
            row = db.execute(
                "SELECT state_id FROM ap_pointers WHERE run_id=? AND name=?",
                (spec.run_id, pointer.name),
            ).fetchone()
            current = StateRef(state_id=row[0]) if row else None
            if current != pointer.expected:
                raise _Rejected("stale_parent")
            if pointer.replacement.state_id not in keys:
                self._state_row(db, spec.run_id, pointer.replacement)
        for record in effects.records:
            row = db.execute(
                "SELECT record FROM ap_records WHERE run_id=? AND namespace=? AND key=?",
                (spec.run_id, record.namespace, record.key),
            ).fetchone()
            if row and row[0] != record.to_bytes():
                raise _Rejected("conflict")
        totals = {}
        for charge in effects.charges:
            if db.execute(
                "SELECT 1 FROM ap_budget_charges WHERE run_id=? AND budget=? AND item_id=?",
                (spec.run_id, charge.budget, charge.item_id),
            ).fetchone():
                raise _Rejected("conflict")
            totals[charge.budget] = totals.get(charge.budget, 0) + charge.units
        for budget, units in totals.items():
            row = db.execute(
                "SELECT limit_value-used FROM ap_budget_limits WHERE run_id=? AND name=?",
                (spec.run_id, budget),
            ).fetchone()
            if row is None or row[0] < units:
                raise _Rejected("budget_exhausted")

    def _before_accept_commit(self) -> None:
        """Failure-injection seam, after all writes but before transaction commit."""

    def accept_result(
        self, result: JobResult, effects: AcceptanceEffects
    ) -> AcceptanceReceipt:
        result = decode_job_result(result.to_bytes())
        effects = AcceptanceEffects.from_bytes(effects.to_bytes())
        if result.status != "succeeded":
            raise StoreError(
                "only successful results can apply effects; journal failures with record_result"
            )
        disposition = "accepted"
        with self._transaction() as db:
            _, spec = self._job(db, result.job_id)
            attempt = self._attempt(
                db, JobHandle(job_id=result.job_id, attempt=result.attempt)
            )
            accepted = db.execute(
                "SELECT * FROM ap_acceptances WHERE job_id=?", (result.job_id,)
            ).fetchone()
            try:
                self._bind(spec, result)
                if attempt["status"] in TERMINAL and attempt["status"] != result.status:
                    raise _Rejected("conflict")
                if (
                    attempt["result"]
                    and decode_job_result(attempt["result"]).result_digest
                    != result.result_digest
                ):
                    raise _Rejected("conflict")
                if accepted:
                    if accepted["result_digest"] != result.result_digest or accepted[
                        "effects_digest"
                    ] != content_hash(effects):
                        raise _Rejected("conflict")
                    self._resolved_outputs(db, spec, result, effects)
                    disposition = "duplicate"
                else:
                    self._validate_effects(db, spec, result, effects)
                    manifests = self._resolved_outputs(db, spec, result, effects)
            except _Rejected as exc:
                disposition = exc.reason
                self._quarantine(db, spec, result, effects.to_bytes(), disposition)
            else:
                if disposition == "accepted":
                    for manifest in manifests.values():
                        self._insert_artifact(db, manifest)
                    for state in effects.states:
                        self._insert_state(db, spec.run_id, state)
                    for record in effects.records:
                        db.execute(
                            "INSERT OR IGNORE INTO ap_records VALUES (?,?,?,?)",
                            (
                                spec.run_id,
                                record.namespace,
                                record.key,
                                record.to_bytes(),
                            ),
                        )
                    for pointer in effects.pointers:
                        db.execute(
                            "INSERT INTO ap_pointers VALUES (?,?,?) ON CONFLICT(run_id,name) DO UPDATE SET state_id=excluded.state_id",
                            (spec.run_id, pointer.name, pointer.replacement.state_id),
                        )
                    for charge in effects.charges:
                        db.execute(
                            "INSERT INTO ap_budget_charges VALUES (?,?,?,?,?)",
                            (
                                spec.run_id,
                                charge.budget,
                                charge.item_id,
                                result.job_id,
                                charge.units,
                            ),
                        )
                        db.execute(
                            "UPDATE ap_budget_limits SET used=used+? WHERE run_id=? AND name=?",
                            (charge.units, spec.run_id, charge.budget),
                        )
                    db.execute(
                        "INSERT INTO ap_acceptances(job_id,result_digest,effects_digest,result,effects) VALUES (?,?,?,?,?)",
                        (
                            result.job_id,
                            result.result_digest,
                            content_hash(effects),
                            result.to_bytes(),
                            effects.to_bytes(),
                        ),
                    )
                if attempt["result"] is None:
                    db.execute(
                        "UPDATE ap_attempts SET status='succeeded',result=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND attempt=?",
                        (result.to_bytes(), result.job_id, result.attempt),
                    )
                self._refresh_job_status(db, result.job_id)
                self._event(
                    db,
                    spec.run_id,
                    result.job_id,
                    "result_" + disposition,
                    {"attempt": result.attempt, "result_digest": result.result_digest},
                )
                self._before_accept_commit()
        return AcceptanceReceipt(job_id=result.job_id, disposition=disposition)
