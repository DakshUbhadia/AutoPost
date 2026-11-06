import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from autopost.contracts import BudgetCharge, MetadataRecord, PointerUpdate, StateRef
from autopost.execution.fixture import (
    FixtureExecutor,
    SimulatedCrash,
    initialize_fixture_run,
)
from autopost.store.errors import ConflictError, StoreError
from autopost.store.metadata import SQLiteMetadataStore


def counts(storage):
    with closing(sqlite3.connect(storage.database)) as db:
        return {
            name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in (
                "ap_states",
                "ap_acceptances",
                "ap_budget_charges",
                "ap_records",
                "ap_artifacts",
            )
        }


def execute(storage, scenario="success", **inputs):
    executor = FixtureExecutor(storage.metadata, storage.artifacts, scenario=scenario)
    spec = storage.training(**inputs)
    handle = executor.submit(spec)
    return spec, executor, handle, executor.result(handle), executor.effects(handle)


def test_acceptance_is_durable_and_applies_state_pointer_budget_and_records_together(
    storage,
):
    spec, executor, handle, result, effects = execute(storage)
    assert executor.status(handle) == "succeeded"
    assert storage.metadata.get_result(spec.job_id) is None
    assert counts(storage)["ap_states"] == 1
    assert storage.metadata.accept_result(result, effects).disposition == "accepted"
    reopened = SQLiteMetadataStore(storage.database, storage.artifacts)
    assert reopened.get_result(spec.job_id) == result
    assert reopened.get_state(result.state) == effects.states[0]
    assert reopened.get_state(result.state).parent == storage.base.state
    assert reopened.get_pointer(storage.run_id, "cursor") == result.state
    assert reopened.budget_remaining(storage.run_id, "operations") == 9
    assert (
        reopened.get_record(storage.run_id, "fixture-results", spec.job_id)
        == effects.records[0]
    )
    assert counts(storage) == {
        "ap_states": 2,
        "ap_acceptances": 1,
        "ap_budget_charges": 1,
        "ap_records": 1,
        "ap_artifacts": 3,
    }


def test_retries_and_duplicate_deliveries_apply_exactly_once(storage):
    spec, _, _, first, effects = execute(storage)
    assert storage.metadata.accept_result(first, effects).disposition == "accepted"
    before = counts(storage)
    retry_executor = FixtureExecutor(
        storage.metadata, storage.artifacts, scenario="duplicate_success"
    )
    retry = retry_executor.submit(
        spec.model_copy(update={"attempt": 2, "created_at": "2026-11-07T00:00:00Z"})
    )
    second = retry_executor.result(retry)
    assert first.to_bytes() != second.to_bytes()
    assert first.result_digest == second.result_digest
    assert (
        storage.metadata.accept_result(
            second, retry_executor.effects(retry)
        ).disposition
        == "duplicate"
    )
    assert storage.metadata.accept_result(first, effects).disposition == "duplicate"
    assert counts(storage) == before
    assert storage.metadata.get_result(spec.job_id) == first
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9
    assert storage.metadata.get_attempt(retry).result == second


def test_duplicate_is_recognized_after_pointer_advances_and_budget_is_spent(storage):
    _, _, _, first, first_effects = execute(storage)
    storage.metadata.accept_result(first, first_effects)
    _, _, _, second, second_effects = execute(
        storage, start=first.state, expected_state=StateRef(state_id="third")
    )
    storage.metadata.accept_result(second, second_effects)
    assert (
        storage.metadata.accept_result(first, first_effects).disposition == "duplicate"
    )
    assert storage.metadata.get_pointer(storage.run_id, "cursor") == second.state
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 8


def test_conflicting_result_is_quarantined_without_overwriting_accepted_history(
    storage,
):
    spec, _, _, accepted, effects = execute(storage)
    storage.metadata.accept_result(accepted, effects)
    before = counts(storage)
    _, _, _, conflict, conflict_effects = execute(storage, "conflict", attempt=2)
    assert (
        storage.metadata.accept_result(conflict, conflict_effects).disposition
        == "conflict"
    )
    assert counts(storage) == before
    assert storage.metadata.get_result(spec.job_id) == accepted
    assert storage.metadata.quarantined_results(spec.job_id) == (conflict,)
    event = storage.metadata.events(storage.run_id)[-1]
    assert event.kind == "result_quarantined" and event.payload["reason"] == "conflict"


def test_same_result_cannot_replay_different_budget_or_pointer_effects(storage):
    spec, _, _, result, effects = execute(storage)
    storage.metadata.accept_result(result, effects)
    changed = effects.model_copy(
        update={
            "charges": (BudgetCharge(budget="operations", item_id="other", units=2),)
        }
    )
    assert storage.metadata.accept_result(result, changed).disposition == "conflict"
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9
    assert len(storage.metadata.quarantined_results(spec.job_id)) == 1


def test_late_result_fails_compare_and_set_with_no_partial_effects(storage):
    spec, _, _, late, effects = execute(storage)
    _, _, _, earlier, earlier_effects = execute(
        storage, expected_state=StateRef(state_id="earlier")
    )
    storage.metadata.accept_result(earlier, earlier_effects)
    before = counts(storage)
    assert storage.metadata.accept_result(late, effects).disposition == "stale_parent"
    assert counts(storage) == before
    assert storage.metadata.get_result(spec.job_id) is None
    assert storage.metadata.get_pointer(storage.run_id, "cursor") == earlier.state
    assert storage.metadata.quarantined_results(spec.job_id) == (late,)


def test_fixture_stale_parent_mode_is_audited(storage):
    _, _, _, result, effects = execute(storage, "stale_parent")
    assert storage.metadata.accept_result(result, effects).disposition == "stale_parent"
    assert counts(storage)["ap_states"] == 1
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10


def test_incomplete_outputs_are_nonsemantic_and_can_be_recovered_on_retry(storage):
    spec, _, _, result, effects = execute(storage, "incomplete_artifact")
    assert (
        storage.metadata.accept_result(result, effects).disposition
        == "missing_artifact"
    )
    assert counts(storage) == {
        "ap_states": 1,
        "ap_acceptances": 0,
        "ap_budget_charges": 0,
        "ap_records": 0,
        "ap_artifacts": 1,
    }
    assert storage.metadata.get_result(spec.job_id) is None
    _, executor, handle, recovered, recovered_effects = execute(storage, attempt=2)
    assert recovered.result_digest == result.result_digest
    assert (
        storage.metadata.accept_result(recovered, recovered_effects).disposition
        == "accepted"
    )
    assert executor.result(handle) == recovered
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


@pytest.mark.parametrize("corruption", ["manifest", "payload"])
def test_corrupt_finalized_outputs_are_quarantined_before_any_effect(
    storage, corruption
):
    spec, _, _, result, effects = execute(storage)
    directory = storage.artifacts._object_directory(result.rollouts.artifact_id)
    path = (
        directory / "manifest.json"
        if corruption == "manifest"
        else next((directory / "payload").iterdir())
    )
    path.write_bytes(b"corrupt")
    assert (
        storage.metadata.accept_result(result, effects).disposition
        == "corrupt_artifact"
    )
    assert storage.metadata.get_result(spec.job_id) is None
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10
    assert counts(storage)["ap_states"] == 1


def test_failed_executor_result_is_journaled_and_cannot_apply_effects(storage):
    spec = storage.training()
    executor = FixtureExecutor(
        storage.metadata, storage.artifacts, scenario="executor_failure"
    )
    handle = executor.submit(spec)
    result = executor.result(handle)
    assert (
        executor.status(handle) == "failed" and result.error.code == "executor_failure"
    )
    assert storage.metadata.get_result(spec.job_id) is None
    from autopost.contracts import AcceptanceEffects

    with pytest.raises(StoreError, match="only successful"):
        storage.metadata.accept_result(
            result,
            AcceptanceEffects(
                run_id=storage.run_id, expected_parent=storage.base.state
            ),
        )
    reopened = SQLiteMetadataStore(storage.database, storage.artifacts)
    assert reopened.get_attempt(handle).result == result
    assert counts(storage)["ap_states"] == 1


def test_cancel_and_retry_interrupted_attempt(storage):
    spec = storage.training()
    executor = FixtureExecutor(
        storage.metadata, storage.artifacts, scenario="crash_before_finalize"
    )
    with pytest.raises(SimulatedCrash):
        executor.submit(spec)
    handle = storage.metadata.reserve_job(spec)
    executor.cancel(handle)
    assert executor.status(handle) == "cancelled"
    assert executor.result(handle).error.code == "cancelled"
    executor.cancel(handle)
    _, _, _, result, effects = execute(storage, attempt=2)
    assert storage.metadata.accept_result(result, effects).disposition == "accepted"
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


@pytest.mark.parametrize(
    "field,value",
    [
        ("stack_id", "b" * 64),
        ("code_revision", "other"),
        ("state", StateRef(state_id="unexpected")),
        ("global_step", 1000),
    ],
)
def test_result_input_and_state_binding_mismatch_is_quarantined(storage, field, value):
    _, _, _, original, effects = execute(storage)
    bad = original.model_copy(update={field: value})
    assert storage.metadata.accept_result(bad, effects).disposition == "conflict"
    assert counts(storage)["ap_states"] == 1
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10


def test_state_manifest_must_match_segment_provenance(storage):
    _, _, _, result, effects = execute(storage)
    bad_state = effects.states[0].model_copy(update={"verifier_version": "unexpected"})
    bad_effects = effects.model_copy(update={"states": (bad_state,)})
    assert storage.metadata.accept_result(result, bad_effects).disposition == "conflict"
    assert counts(storage)["ap_states"] == 1


def test_budget_totals_are_checked_atomically_and_charge_ids_are_unique(storage):
    _, _, _, result, effects = execute(storage)
    bad = effects.model_copy(
        update={
            "charges": (
                BudgetCharge(budget="operations", item_id="one", units=6),
                BudgetCharge(budget="operations", item_id="two", units=5),
            )
        }
    )
    assert storage.metadata.accept_result(result, bad).disposition == "budget_exhausted"
    assert counts(storage)["ap_budget_charges"] == 0
    assert storage.metadata.accept_result(result, effects).disposition == "accepted"
    _, _, _, second, second_effects = execute(
        storage, start=result.state, expected_state=StateRef(state_id="second")
    )
    bad = second_effects.model_copy(update={"charges": effects.charges})
    assert storage.metadata.accept_result(second, bad).disposition == "conflict"
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


def test_immutable_records_and_duplicate_effect_keys_cannot_partially_apply(storage):
    _, _, _, first, effects = execute(storage)
    shared = MetadataRecord(namespace="log", key="shared", value={"value": 1})
    effects = effects.model_copy(update={"records": (shared,)})
    storage.metadata.accept_result(first, effects)
    _, _, _, second, effects2 = execute(
        storage, start=first.state, expected_state=StateRef(state_id="second")
    )
    for records in (
        (shared.model_copy(update={"value": {"value": 2}}),),
        (shared, shared),
    ):
        assert (
            storage.metadata.accept_result(
                second, effects2.model_copy(update={"records": records})
            ).disposition
            == "conflict"
        )
    assert storage.metadata.get_record(storage.run_id, "log", "shared") == shared
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


def test_cross_run_pointer_targets_are_rejected(storage):
    other = initialize_fixture_run(
        storage.metadata,
        storage.artifacts,
        run_id="other-run",
        config_id="a" * 64,
        stack_id="a" * 64,
        dataset=storage.dataset,
        state_id="other-base",
    )
    _, _, _, result, effects = execute(storage)
    changed = effects.model_copy(
        update={
            "pointers": (
                PointerUpdate(
                    name="cursor", expected=storage.base.state, replacement=other.state
                ),
            )
        }
    )
    assert storage.metadata.accept_result(result, changed).disposition == "stale_parent"
    assert storage.metadata.get_pointer(storage.run_id, "cursor") == storage.base.state


def test_concurrent_delivery_of_one_job_has_one_semantic_commit(storage):
    _, _, _, result, effects = execute(storage)

    def deliver(_):
        store = SQLiteMetadataStore(storage.database, storage.artifacts)
        return store.accept_result(result, effects).disposition

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(deliver, range(8)))
    assert outcomes.count("accepted") == 1 and outcomes.count("duplicate") == 7
    assert counts(storage)["ap_states"] == 2
    assert counts(storage)["ap_budget_charges"] == 1


def test_concurrent_distinct_jobs_cannot_both_advance_the_same_pointer(storage):
    _, _, _, left, le = execute(storage)
    _, _, _, right, re = execute(storage, expected_state=StateRef(state_id="right"))

    def deliver(pair):
        return (
            SQLiteMetadataStore(storage.database, storage.artifacts)
            .accept_result(*pair)
            .disposition
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(deliver, ((left, le), (right, re))))
    assert sorted(outcomes) == ["accepted", "stale_parent"]
    assert (
        counts(storage)["ap_states"] == 2
        and storage.metadata.budget_remaining(storage.run_id, "operations") == 9
    )


def test_attempt_terminal_status_and_first_result_are_immutable(storage):
    spec, _, handle, result, _ = execute(storage)
    with pytest.raises(ConflictError):
        storage.metadata.record_attempt(handle, "running")
    with pytest.raises(ConflictError):
        storage.metadata.record_result(
            result.model_copy(update={"metrics": {"changed": True}})
        )
    assert storage.metadata.get_attempt(handle).result == result
    assert storage.metadata.get_result(spec.job_id) is None


def test_history_cannot_be_overwritten_even_by_direct_sql(storage):
    _, _, _, result, effects = execute(storage)
    storage.metadata.accept_result(result, effects)
    with closing(sqlite3.connect(storage.database)) as db:
        for table in ("ap_states", "ap_artifacts", "ap_acceptances", "ap_records"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                db.execute(f"DELETE FROM {table}")
            db.rollback()
    assert storage.metadata.get_state(result.state) == effects.states[0]


def test_unsupported_wire_versions_fail_before_journal_side_effects(storage):
    before = counts(storage)
    bad = storage.training().model_copy(update={"schema_version": 99})
    with pytest.raises(ValueError, match="unsupported schema_version"):
        storage.metadata.reserve_job(bad)
    assert counts(storage) == before
    with closing(sqlite3.connect(storage.database)) as db:
        assert db.execute("SELECT count(*) FROM ap_jobs").fetchone()[0] == 0


def test_evaluation_raw_manifest_is_independent_of_aggregate_scores(storage):
    spec = storage.evaluation()
    executor = storage.executor
    first_handle = executor.submit(spec)
    first = executor.result(first_handle)
    raw_manifest = storage.artifacts.resolve(first.evidence.completions)
    raw_bytes = storage.artifacts.read(first.evidence.completions, "raw")
    assert (
        storage.metadata.accept_result(
            first, executor.effects(first_handle)
        ).disposition
        == "accepted"
    )
    changed = FixtureExecutor(storage.metadata, storage.artifacts, scenario="conflict")
    handle = changed.submit(spec.model_copy(update={"attempt": 2}))
    second = changed.result(handle)
    assert first.evidence.aggregates != second.evidence.aggregates
    assert first.evidence.completions == second.evidence.completions
    assert (
        storage.metadata.accept_result(second, changed.effects(handle)).disposition
        == "conflict"
    )
    assert storage.artifacts.resolve(second.evidence.completions) == raw_manifest
    assert storage.artifacts.read(second.evidence.completions, "raw") == raw_bytes
    assert storage.metadata.get_result(spec.job_id) == first
    assert counts(storage)["ap_states"] == 1
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


def test_database_failure_after_writes_rolls_back_every_semantic_effect(
    storage, monkeypatch
):
    spec, _, _, result, effects = execute(storage)
    before = counts(storage)

    def fail():
        raise sqlite3.OperationalError("injected commit failure")

    monkeypatch.setattr(storage.metadata, "_before_accept_commit", fail)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        storage.metadata.accept_result(result, effects)
    assert counts(storage) == before
    assert storage.metadata.get_result(spec.job_id) is None
    assert storage.metadata.get_pointer(storage.run_id, "cursor") == storage.base.state
    assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10
    monkeypatch.setattr(storage.metadata, "_before_accept_commit", lambda: None)
    assert storage.metadata.accept_result(result, effects).disposition == "accepted"
