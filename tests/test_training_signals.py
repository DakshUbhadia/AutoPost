"""CPU integration tests use explicitly synthetic rollout/verifier evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction

import pytest

from autopost.contracts.signals import (
    CurriculumConfig,
    LabelReview,
    PassRateObservation,
)
from autopost.curriculum import CurriculumSampler, PoolExhaustedError, choose
from autopost.data.access import TrainingDataAccess
from autopost.interfaces import (
    Completion,
    OfflineExample,
    Verdict,
    band_for,
    posterior_pass_rate,
)
from autopost.store.errors import ConflictError, StoreError
from autopost.store.signals import TrainingSignalRepository, task_digest

SYNTHETIC_CHECKER = "synthetic-train-verifier@1"


@pytest.fixture
def signals(catalog):
    storage = catalog.storage
    repo = TrainingSignalRepository(storage.database, storage.artifacts)
    tasks = catalog.repo.training_tasks(catalog.refs["train"], fixture_only=True)
    return repo, catalog.refs["train"], tasks


def group(repo, ref, task, identity, *, correct=4, n=8, checker=SYNTHETIC_CHECKER):
    pairs = tuple(
        (
            Completion(
                f"{identity}-sample-{i}",
                task.task_id,
                "fixture-base",
                "T_train",
                0,
                i,
                "Answer: " + (task.answer if i < correct else "999999"),
                12,
                False,
            ),
            Verdict(
                i < correct,
                "ok" if i < correct else "wrong",
                task.answer if i < correct else "999999",
                checker,
            ),
        )
        for i in range(n)
    )
    return repo.record_group(ref, task, identity, pairs, fixture_only=True), pairs


def config(**updates):
    # The development fixture deliberately has only one source/skill. Production
    # defaults are stricter and are tested independently with balanced tasks.
    return CurriculumConfig(max_share=1.0, low_entropy_max_share=1.0, **updates)


def review(repo, task, identity, decision, **updates):
    evidence = repo.retain_evidence(
        {
            "synthetic_test_only": True,
            "task_digest": task_digest(task),
            "derivation": "Synthetic independent test attestation",
        }
    )
    return LabelReview(
        **{
            "review_id": identity,
            "task_id": task.task_id,
            "task_digest": task_digest(task),
            "method": "deterministic",
            "checker_id": "synthetic-independent-derivation@1",
            "reviewer": "test-reviewer",
            "decision": decision,
            "reason": "Synthetic test: independent recomputation confirms this decision",
        "independence_basis": "Synthetic deterministic recomputation; no model consensus used",
            "evidence": evidence,
            **updates,
        }
    )


def test_observations_retain_raw_evidence_roundtrip_retry_and_checker_exclusion(
    signals,
):
    repo, ref, tasks = signals
    observation, pairs = group(repo, ref, tasks[0], "group-1", correct=6)
    assert PassRateObservation.from_bytes(observation.to_bytes()) == observation
    assert (
        repo.record_group(ref, tasks[0], "group-1", pairs, fixture_only=True)
        == observation
    )
    assert repo.estimate(tasks[0]) == 0.7
    assert repo.estimate(tasks[0], excluded_checkers=(SYNTHETIC_CHECKER,)) == 0.5
    assert b'"completion"' in repo.artifacts.read(observation.evidence, "evidence.json")
    with pytest.raises(ConflictError, match="different"):
        group(repo, ref, tasks[0], "group-1", correct=3)
    with pytest.raises(ConflictError, match="already counted"):
        repo.record_group(ref, tasks[0], "new-group", pairs, fixture_only=True)
    assert repo.estimate(tasks[0]) == 0.7  # conflict transaction rolled back


def test_consensus_only_flags_independent_review_excludes_and_good_review_restores(
    signals,
):
    repo, ref, tasks = signals
    task = tasks[0]
    group(repo, ref, task, "consensus-1", correct=2)
    group(repo, ref, task, "consensus-2", correct=2)
    assert "suspected_label_error" in [
        event["kind"] for event in repo.audit(task.task_id)
    ]
    access = TrainingDataAccess(repo.datasets, fixture_only=True)
    assert task in access.on_policy(ref)
    bad = review(repo, task, "bad-label", "confirmed_bad")
    assert LabelReview.from_bytes(bad.to_bytes()) == bad
    repo.review(ref, task, bad, fixture_only=True)
    repo.review(ref, task, bad, fixture_only=True)
    assert task not in access.on_policy(ref)
    with pytest.raises(StoreError):
        access.offline(
            ref,
            (
                OfflineExample(
                    task,
                    "solution",
                    "fixture",
                    Verdict(True, "ok", task.answer, SYNTHETIC_CHECKER),
                ),
            ),
        )
    with pytest.raises(ValueError, match="differ"):
        repo.review(
            ref,
            task,
            review(
                repo,
                task,
                "same-checker",
                "confirmed_bad",
                method="second_verifier",
                checker_id=SYNTHETIC_CHECKER,
            ),
            fixture_only=True,
        )
    repo.review(
        ref, task, review(repo, task, "good-label", "confirmed_good"), fixture_only=True
    )
    assert task in access.on_policy(ref)
    repo.review(
        ref, task, review(repo, task, "fixed-label", "corrected"), fixture_only=True
    )
    assert repo.estimate(task) == 0.5
    repo.review(
        ref, task, review(repo, task, "later-good", "confirmed_good"), fixture_only=True
    )
    assert task not in access.on_policy(
        ref
    )  # old immutable corrected label stays retired


def test_review_requires_exact_task_binding_and_real_finalized_evidence(signals):
    repo, ref, tasks = signals
    bad = review(repo, tasks[0], "bad", "confirmed_bad")
    with pytest.raises(ValueError, match="bind"):
        repo.review(ref, tasks[1], bad, fixture_only=True)
    missing = bad.model_copy(
        update={"evidence": bad.evidence.model_copy(update={"artifact_id": "missing"})}
    )
    with pytest.raises(StoreError):
        repo.review(ref, tasks[0], missing, fixture_only=True)


def test_atomic_slice_restart_retry_caps_and_failure_without_partial_exposure(signals):
    repo, ref, tasks = signals
    for i, task in enumerate(tasks):
        group(repo, ref, task, f"estimate-{i}")
    policy = config(cluster_exposure_cap=1)
    sampler = CurriculumSampler(repo, ref, policy, fixture_only=True)
    selected = sampler.reserve("slice-one", len(tasks), 17)
    assert set(selected) == set(tasks)
    exposures = repo.exposures()
    restarted = TrainingSignalRepository(repo.path, repo.artifacts)
    assert (
        CurriculumSampler(restarted, ref, policy, fixture_only=True).reserve(
            "slice-one", len(tasks), 17
        )
        == selected
    )
    assert restarted.exposures() == exposures
    with pytest.raises(ConflictError):
        sampler.reserve("slice-one", len(tasks), 18)
    with pytest.raises(PoolExhaustedError):
        sampler.reserve("slice-two", len(tasks), 18)
    assert repo.exposures() == exposures
    repo.review(
        ref,
        tasks[0],
        review(repo, tasks[0], "exclude-after-reserve", "confirmed_bad"),
        fixture_only=True,
    )
    with pytest.raises(StoreError, match="excluded"):
        sampler.reserve("slice-one", len(tasks), 17)


def test_two_concurrent_writers_cannot_overspend_caps(signals):
    repo, ref, tasks = signals
    for i, task in enumerate(tasks):
        group(repo, ref, task, f"estimate-{i}")
    policy = config(cluster_exposure_cap=1)

    def reserve(index):
        try:
            return repo.reserve_slice(
                ref,
                f"concurrent-{index}",
                len(tasks),
                0,
                policy,
                excluded_checkers=(),
                fixture_only=True,
            )
        except PoolExhaustedError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert sum(result is not None for result in results) == 1
    assert max(repo.exposures().values()) == 1


def test_unknown_estimates_do_not_train_and_measurement_items_cannot_enter(
    signals, catalog
):
    repo, ref, tasks = signals
    with pytest.raises(PoolExhaustedError):
        repo.reserve_slice(
            ref, "unknown", 1, 0, config(), excluded_checkers=(), fixture_only=True
        )
    assert repo.exposures() == {}
    measurement_ref = catalog.refs["pilot"]
    measured = catalog.repo.inspect(measurement_ref)[0]
    with pytest.raises(StoreError, match="measurement"):
        group(repo, measurement_ref, measured, "measurement-group")
    with pytest.raises(StoreError, match="membership|outside"):
        group(
            repo, ref, replace(tasks[0], answer="123", answer_type="int"), "relabeled"
        )


def test_promotion_persistence_only_flags_and_edges_are_refreshed(signals):
    repo, ref, tasks = signals
    policy = config()
    group(repo, ref, tasks[0], "in-band")
    group(repo, ref, tasks[1], "edge", correct=0)
    for i in range(3):
        edges = repo.promotion_refresh(ref, f"promotion-{i}", policy, fixture_only=True)
        assert tasks[1] in edges
    assert "persistent_in_band" in [
        event["kind"] for event in repo.audit(tasks[0].task_id)
    ]
    assert tasks[0] in repo.datasets.training_tasks(ref, fixture_only=True)
    before = repo.audit(tasks[0].task_id)
    assert (
        repo.promotion_refresh(ref, "promotion-2", policy, fixture_only=True) == edges
    )
    assert repo.audit(tasks[0].task_id) == before


def test_sampler_enforces_source_skill_entropy_quotas_and_determinism(signals):
    _, _, tasks = signals
    balanced = tuple(
        replace(
            tasks[0],
            task_id=f"t{i}",
            cluster_id=f"c{i}",
            source_id=f"s{i % 2}",
            skill=f"k{i // 2 % 2}",
            low_entropy=i == 0,
        )
        for i in range(20)
    )
    policy = CurriculumConfig()
    picked = choose(balanced, 10, 42, policy, {})
    assert picked == choose(tuple(reversed(balanced)), 10, 42, policy, {})
    for field in ("source_id", "skill"):
        assert (
            max(
                sum(getattr(t, field) == getattr(p, field) for t in picked)
                for p in picked
            )
            <= 5
        )
    assert sum(t.low_entropy for t in picked) <= 1
    assert band_for(8, 0.5)[0] == pytest.approx(0.083, abs=0.001)
    assert posterior_pass_rate(4, 8) == Fraction(1, 2)
    with pytest.raises(ValueError):
        posterior_pass_rate(9, 8)
