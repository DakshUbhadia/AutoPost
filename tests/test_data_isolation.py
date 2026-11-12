import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

from autopost.contracts import DatasetRef
from autopost.contracts.tasks import MEASUREMENT_SPLITS, SEALED_SPLITS
from autopost.data.access import TrainingDataAccess
from autopost.data.builder import build
from autopost.interfaces import OfflineExample, Verdict, assert_not_measurement
from autopost.store.errors import ConflictError, StoreError


def test_dataset_registration_is_idempotent_and_keeps_canonical_provenance(catalog):
    refs2 = catalog.repo.publish(catalog.result)
    assert refs2 == catalog.refs
    for split, ref in refs2.items():
        manifest = catalog.repo.manifest(ref)
        assert manifest.split == split and catalog.repo.inspect(ref)
    with closing(sqlite3.connect(catalog.storage.database)) as db:
        count = sum(len(bundle.tasks) for bundle in catalog.result.bundles)
        assert (
            db.execute("SELECT count(*) FROM ap_dataset_tasks").fetchone()[0] == count
        )
        assert db.execute("SELECT count(*) FROM ap_datasets").fetchone()[0] == len(
            refs2
        )
        for source_ver, measurable, answer_type in db.execute(
            "SELECT source_ver,measurable,answer_type FROM ap_dataset_tasks"
        ):
            assert source_ver and bool(measurable) == (
                answer_type in {"int", "rational", "decimal"}
            )


def test_both_trainer_gateways_reject_every_measurement_partition(catalog):
    access = TrainingDataAccess(catalog.repo, fixture_only=True)
    train_ref = catalog.refs["train"]
    prompts = access.on_policy(train_ref)
    assert prompts and all(task.split == "train" for task in prompts)
    good = OfflineExample(
        prompts[0],
        "verified fixture",
        "fixture-self-rft",
        Verdict(True, "ok", "12", "fixture-verifier"),
    )
    assert access.offline(train_ref, (good,)) == (good,)
    for split, ref in catalog.refs.items():
        if split not in MEASUREMENT_SPLITS:
            continue
        with pytest.raises(StoreError):
            access.on_policy(ref)
        task = catalog.repo.inspect(ref)[0]
        example = OfflineExample(
            task,
            "attempted leakage",
            "fixture",
            Verdict(True, "ok", task.answer, "fixture"),
        )
        with pytest.raises((StoreError, RuntimeError)):
            access.offline(train_ref, (example,))
        if task.source_role == "hidden":
            with pytest.raises(ValueError):
                replace(task, split="train", source_role="train")
        else:
            relabeled = replace(task, split="train", source_role="train")
            with pytest.raises(StoreError):
                catalog.repo.assert_training_tasks(
                    train_ref, (relabeled,), fixture_only=True
                )
            with pytest.raises(RuntimeError):
                assert_not_measurement(
                    (relabeled,), measurement_clusters=frozenset({task.cluster_id})
                )


def test_job_reservation_refuses_measurement_slices_before_any_attempt(
    storage, catalog
):
    with closing(sqlite3.connect(storage.database)) as db:
        before = db.execute("SELECT count(*) FROM ap_jobs").fetchone()[0]
    for split, ref in catalog.refs.items():
        if split not in MEASUREMENT_SPLITS:
            continue
        with pytest.raises(StoreError):
            storage.metadata.reserve_job(storage.training(training_slice=ref))
    with closing(sqlite3.connect(storage.database)) as db:
        assert db.execute("SELECT count(*) FROM ap_jobs").fetchone()[0] == before
    handle = storage.metadata.reserve_job(
        storage.training(training_slice=catalog.refs["train"])
    )
    assert handle.job_id


def test_production_run_cannot_bypass_dataset_resolution_or_use_synthetic_inputs(
    storage, catalog
):
    storage.metadata.register_run("production")
    for ref in (
        DatasetRef(dataset_id="missing", sha256="b" * 64),
        catalog.refs["train"],
    ):
        with pytest.raises(StoreError):
            storage.metadata.reserve_job(
                storage.training(run_id="production", training_slice=ref)
            )
    with pytest.raises(ConflictError):
        storage.metadata.register_run("production", fixture_only=True)


def test_future_builds_cannot_launder_prior_measurement_clusters_or_problem_text(
    catalog, development_result
):
    sources, embeddings, config, programs, first = development_result
    first_measurements = {
        task.task_id
        for bundle in first.bundles
        if bundle.manifest.split in MEASUREMENT_SPLITS
        for task in bundle.tasks
    }
    # Search a fixed engineering seed schedule, not a scientific calibration.
    for seed in range(1, 20):
        result = build(
            sources,
            embeddings,
            config.model_copy(update={"seed": seed}),
            programs=programs,
        )
        train = next(
            bundle for bundle in result.bundles if bundle.manifest.split == "train"
        )
        if first_measurements & {task.task_id for task in train.tasks}:
            break
    else:
        raise AssertionError("fixture must exercise cross-build overlap")
    with pytest.raises(StoreError, match="cross-build"):
        catalog.repo.publish(result)


def test_database_constraints_and_immutable_catalog_prevent_relabeling(catalog):
    with closing(sqlite3.connect(catalog.storage.database)) as db:
        row = list(
            db.execute(
                "SELECT * FROM ap_dataset_tasks WHERE split='gate_a' LIMIT 1"
            ).fetchone()
        )
        row[1] = "forged"
        row[8] = 0
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO ap_dataset_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row
            )
        db.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE ap_dataset_tasks SET split='train'")
        db.rollback()
        for table in ("ap_datasets", "ap_measurement_blocks"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"DELETE FROM {table}")
            db.rollback()


def test_hidden_and_sealed_jobs_cannot_bypass_review_and_development_partition_guards(
    storage, catalog
):
    ref = catalog.refs["hidden_dev"]
    with pytest.raises(StoreError, match="quarantined"):
        catalog.repo.evaluation_tasks(ref, fixture_only=True)
    with pytest.raises(StoreError, match="quarantined"):
        storage.metadata.reserve_job(
            storage.evaluation(dataset=ref, split="hidden_dev")
        )
    for split in SEALED_SPLITS:
        with pytest.raises(StoreError, match="unavailable"):
            storage.metadata.reserve_job(storage.evaluation(split=split))
    with closing(sqlite3.connect(storage.database)) as db:
        assert db.execute("SELECT count(*) FROM ap_jobs").fetchone()[0] == 0


def test_offline_examples_need_verifier_evidence_and_real_training_membership(catalog):
    task = catalog.repo.training_tasks(catalog.refs["train"], fixture_only=True)[0]
    access = TrainingDataAccess(catalog.repo, fixture_only=True)
    for verdict, provenance in (
        (Verdict(False, "wrong", None, "fixture"), "rft"),
        (Verdict(True, "ok", task.answer, "fixture"), ""),
    ):
        with pytest.raises(ValueError):
            access.offline(
                catalog.refs["train"],
                (OfflineExample(task, "text", provenance, verdict),),
            )


def test_training_artifact_graph_contains_no_measurement_rows_or_raw_audit_sources(
    catalog,
):
    import json

    from autopost.contracts import ArtifactRef

    ref = catalog.refs["train"]
    with closing(sqlite3.connect(catalog.storage.database)) as db:
        row = db.execute(
            "SELECT artifact_ref,audit_ref FROM ap_datasets WHERE dataset_id=?",
            (ref.dataset_id,),
        ).fetchone()
    artifact, audit = (ArtifactRef.from_bytes(blob) for blob in row)
    assert artifact != audit
    manifest = catalog.storage.artifacts.resolve(artifact)
    assert {component.name for component in manifest.components} == {
        "tasks",
        "dataset-manifest",
    }
    tasks = [
        json.loads(line)
        for line in catalog.storage.artifacts.read(artifact, "tasks").splitlines()
    ]
    assert tasks and all(task["split"] == "train" for task in tasks)
    declared = json.loads(catalog.storage.artifacts.read(artifact, "dataset-manifest"))
    assert set(declared["task_ids"]) == {task["task_id"] for task in tasks}
    assert "hidden_programs" in json.loads(
        catalog.storage.artifacts.read(audit, "build-report")
    )


def test_cross_build_embedding_paraphrase_cannot_evade_protected_measurement_catalog(
    catalog, development_result
):
    from autopost.contracts.datasets import BuildConfig, SourceSpec
    from autopost.data.dedup import EmbeddingTable
    from autopost.data.normalize import problem_key
    from autopost.data.sources import SourceRow, SourceSnapshot

    sources, embeddings, _, _, _ = development_result
    protected = catalog.repo.inspect(catalog.refs["gate_a"])[0]
    text = "A wholly rephrased mathematical exercise whose cached semantics match a protected item."
    spec = SourceSpec(
        source_id="fresh-source",
        revision="fixture-v2",
        role="train",
        family_id="fresh-family",
        license_id="fixture",
        fixture_only=True,
    )
    fresh = SourceSnapshot(
        spec, (SourceRow("fresh", text, protected.answer, "fixture-math"),)
    )
    snapshots = (
        fresh,
        *(
            source
            for source in sources
            if source.spec.role in {"transfer", "benchmark"}
        ),
    )
    cache = EmbeddingTable(
        embeddings.model_id,
        embeddings.revision,
        {
            **embeddings.vectors,
            problem_key(text): embeddings.vectors[problem_key(protected.problem)],
        },
        True,
    )
    result = build(
        snapshots, cache, BuildConfig(split_clusters={"train": 1, "transfer_guard": 1})
    )
    with pytest.raises(StoreError, match="near-duplicate measurement"):
        catalog.repo.publish(result)
    assert (
        catalog.repo.catalog_report(catalog.result.build_id)[
            "forbidden_training_overlap"
        ]
        is False
    )
