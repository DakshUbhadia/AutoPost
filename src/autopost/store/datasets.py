"""Immutable dataset catalog, cross-build isolation and reviewed hidden access."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from autopost.contracts import (
    ArtifactComponent,
    ArtifactManifest,
    ArtifactRef,
    ArtifactStore,
    DatasetRef,
)
from autopost.contracts.datasets import BuildConfig, DatasetManifest, SourceSpec
from autopost.contracts.serialization import canonical_bytes, content_hash
from autopost.contracts.tasks import MEASUREMENT_SPLITS, SEALED_SPLITS, Task
from autopost.data.builder import BuildResult, task_bytes
from autopost.data.dedup import EmbeddingTable, cluster
from autopost.data.hidden import (
    HiddenProgram,
    HiddenTestReport,
    HumanReview,
    generate,
    test_program,
    verify_review,
)
from autopost.data.normalize import normalize, problem_key
from autopost.data.sources import SourceRow, SourceSnapshot

from .db import connect, migrate_db
from .errors import ConflictError, StoreError


def task_keys(task: Task, manifest: DatasetManifest):
    keys = {
        ("prompt", task.task_id),
        ("cluster", task.cluster_id),
        ("problem", problem_key(task.problem)),
        *(("lineage", key) for key in task.lineage_keys),
    }
    spec = next(
        (source for source in manifest.sources if source.source_id == task.source_id),
        None,
    )
    if spec is not None:
        keys.add(("family", spec.family_id))
    return keys


def training_rows(
    db, ref: DatasetRef, *, fixture_only: bool = False
) -> tuple[Task, ...]:
    ref.to_bytes()
    dataset = db.execute(
        "SELECT * FROM ap_datasets WHERE dataset_id=?", (ref.dataset_id,)
    ).fetchone()
    if dataset is None:
        if fixture_only:
            return ()  # Explicit durable fixture-run exception; never for production.
        raise StoreError(
            "training input must be a registered immutable training dataset"
        )
    if dataset["sha256"] != ref.sha256 or dataset["split"] not in {"train", "dev"}:
        raise StoreError("measurement or mismatched dataset cannot reach a trainer")
    if dataset["fixture_only"] and not fixture_only:
        raise StoreError(
            "synthetic development fixtures require an explicitly marked fixture run"
        )
    overlap = db.execute(
        "SELECT 1 FROM ap_task_keys k JOIN ap_measurement_blocks b ON k.kind=b.kind AND k.value=b.value "
        "WHERE k.dataset_id=? LIMIT 1",
        (ref.dataset_id,),
    ).fetchone()
    if overlap:
        raise StoreError(
            "training input overlaps protected measurement identity, cluster, problem or source lineage"
        )
    tasks = tuple(
        Task.from_bytes(row[0])
        for row in db.execute(
            "SELECT record FROM ap_dataset_tasks WHERE dataset_id=? ORDER BY task_id",
            (ref.dataset_id,),
        )
    )
    if not tasks or any(
        task.split not in {"train", "dev"} or task.source_role != "train"
        for task in tasks
    ):
        raise StoreError("invalid training dataset membership")
    return tasks


def evaluation_rows(
    db, ref: DatasetRef, split: str, *, fixture_only=False
) -> tuple[Task, ...]:
    if split in SEALED_SPLITS:
        raise StoreError(
            "research hidden/sealed partitions are unavailable during development"
        )
    row = db.execute(
        "SELECT sha256,split,fixture_only FROM ap_datasets WHERE dataset_id=?",
        (ref.dataset_id,),
    ).fetchone()
    if row is None:
        if fixture_only:
            return ()
        raise StoreError("evaluation input must be a registered immutable dataset")
    if row[0] != ref.sha256 or row[1] != split or split in {"train"}:
        raise StoreError(
            "evaluation identity and split must match the dataset manifest"
        )
    if row[2] and not fixture_only:
        raise StoreError(
            "synthetic evaluation fixtures require explicit fixture provenance"
        )
    tasks = tuple(
        Task.from_bytes(row[0])
        for row in db.execute(
            "SELECT record FROM ap_dataset_tasks WHERE dataset_id=? ORDER BY task_id",
            (ref.dataset_id,),
        )
    )
    for task in tasks:
        if task.source_role == "hidden":
            review = db.execute(
                "SELECT record FROM ap_hidden_reviews WHERE program_digest=? AND report_digest=?",
                (task.program_digest, task.property_report_digest),
            ).fetchone()
            if (
                review is None
                or HumanReview.from_bytes(review[0]).decision != "approved"
            ):
                raise StoreError(
                    "hidden_dev is quarantined until generated-instance human review is approved"
                )
    return tasks


class DatasetRepository:
    def __init__(self, path: str | Path, artifacts: ArtifactStore):
        self.path = Path(path)
        self.artifacts = artifacts
        migrate_db(self.path)

    @contextmanager
    def _connection(self, *, write=False):
        db = connect(self.path)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def publish(self, result: BuildResult) -> dict[str, DatasetRef]:
        if not result.bundles:
            raise StoreError("build contains no nonempty development partitions")
        declared = {
            split
            for split, counts in result.report["splits"].items()
            if counts["items"]
        }
        if declared != {bundle.manifest.split for bundle in result.bundles}:
            raise StoreError("all build partitions must be registered together")
        # Raw sources, hidden previews and audit reports must never be placed
        # inside the artifact delivered as a trainer's input dataset.
        hidden_instances = {}
        for packet in result.report["hidden_programs"]:
            program = HiddenProgram.model_validate_json(json.dumps(packet["program"]))
            campaign = HiddenTestReport.model_validate_json(
                json.dumps(packet["report"])
            )
            if not campaign.passed or content_hash(
                test_program(program, seed=campaign.seed, trials=campaign.trials)
            ) != content_hash(campaign):
                raise StoreError("hidden property campaign is not reproducible")
            instances, _ = generate(
                program,
                seed=campaign.seed,
                count=result.report["config"]["hidden_instances_per_program"],
            )
            for instance in instances:
                hidden_instances[instance.instance_id] = (instance, program, campaign)
        expected_inputs = content_hash(
            {
                "sources": [source.digest for source in result.bundles[0].sources],
                "embeddings": result.embeddings.digest,
                "programs": [
                    HiddenProgram.model_validate_json(
                        json.dumps(packet["program"])
                    ).digest
                    for packet in result.report["hidden_programs"]
                ],
            }
        )
        if expected_inputs != result.report["input_digest"]:
            raise StoreError(
                "source snapshots or embedding cache differ from build inputs"
            )
        audit_content = {
            "build-report": canonical_bytes(result.report),
            "source-snapshots": canonical_bytes(
                [source.to_dict() for source in result.bundles[0].sources]
            ),
            "embedding-cache": canonical_bytes(
                {
                    "model_id": result.embeddings.model_id,
                    "revision": result.embeddings.revision,
                    "fixture_only": result.embeddings.fixture_only,
                    "vectors": result.embeddings.vectors,
                }
            ),
        }
        audit = ArtifactManifest.create(
            artifact_id="dataset-audit-" + result.build_id,
            kind="report",
            producer_job_id="build-" + result.build_id,
            components=tuple(
                ArtifactComponent(
                    name=name,
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                    media_type="application/json",
                )
                for name, data in audit_content.items()
            ),
        )
        staged_audit = self.artifacts.stage(audit.artifact.artifact_id)
        for name, data in audit_content.items():
            self.artifacts.write(staged_audit, name, data)
        self.artifacts.finalize(staged_audit, audit)
        prepared = []
        for bundle in result.bundles:
            manifest = DatasetManifest.from_bytes(bundle.manifest.to_bytes())
            payload = task_bytes(bundle.tasks)
            if (
                manifest.build_id != result.build_id
                or hashlib.sha256(payload).hexdigest() != manifest.tasks_sha256
                or content_hash(bundle.report) != manifest.report_sha256
                or bundle.report != result.report
                or tuple(task.task_id for task in bundle.tasks) != manifest.task_ids
                or tuple(sorted({task.cluster_id for task in bundle.tasks}))
                != manifest.cluster_ids
            ):
                raise StoreError(
                    "dataset payload differs from immutable build manifest"
                )
            if bundle.report["splits"][manifest.split] != {
                "items": len(bundle.tasks),
                "clusters": len(manifest.cluster_ids),
            }:
                raise StoreError("dataset row counts differ from its build report")
            for task in bundle.tasks:
                Task.from_bytes(task.to_bytes())
                if (
                    task.split != manifest.split
                    or task.split in SEALED_SPLITS
                    or task.source_role == "benchmark"
                ):
                    raise StoreError("forbidden dataset membership")
                if task.source_role == "hidden":
                    if (
                        task.program_digest not in manifest.program_digests
                        or task.split != "hidden_dev"
                    ):
                        raise StoreError(
                            "hidden data requires disposable program provenance"
                        )
                    expected = hidden_instances.get(task.task_id)
                    if expected is None:
                        raise StoreError(
                            "hidden instance is outside the validated generation campaign"
                        )
                    instance, program, campaign = expected
                    if (
                        task.problem != instance.problem
                        or task.answer != str(Fraction(instance.answer))
                        or task.program_digest != program.digest
                        or task.property_report_digest != content_hash(campaign)
                        or task.lineage_keys != program.parent_lineage_keys
                        or task.source_ver != program.digest
                    ):
                        raise StoreError(
                            "hidden task differs from execution-derived label or tested provenance"
                        )
                else:
                    source = next(
                        (
                            source
                            for source in manifest.sources
                            if source.source_id == task.source_id
                        ),
                        None,
                    )
                    if (
                        source is None
                        or source.revision != task.source_ver
                        or source.role != task.source_role
                    ):
                        raise StoreError(
                            "task source revision or role does not match dataset provenance"
                        )
            content = {"tasks": payload, "dataset-manifest": manifest.to_bytes()}
            artifact = ArtifactManifest.create(
                artifact_id=manifest.dataset.dataset_id,
                kind="dataset",
                producer_job_id="build-" + result.build_id,
                components=tuple(
                    ArtifactComponent(
                        name=name,
                        sha256=hashlib.sha256(data).hexdigest(),
                        size_bytes=len(data),
                        media_type="application/jsonl"
                        if name == "tasks"
                        else "application/json",
                    )
                    for name, data in content.items()
                ),
            )
            staged = self.artifacts.stage(artifact.artifact.artifact_id)
            for name, data in content.items():
                self.artifacts.write(staged, name, data)
            self.artifacts.finalize(staged, artifact)
            prepared.append((bundle, artifact.artifact))
        with self._connection(write=True) as db:
            # Reconcile cluster aliases across registered builds. Hidden
            # instances inherit protected parent lineage and exact problem
            # keys; their curated parents participate in semantic clustering.
            old_rows = db.execute("SELECT record FROM ap_dataset_tasks").fetchall()
            old_tasks = [Task.from_bytes(row[0]) for row in old_rows]
            new_tasks = [task for bundle, _ in prepared for task in bundle.tasks]
            new_ids = {bundle.manifest.dataset.dataset_id for bundle, _ in prepared}
            existing_ids = {
                row[0] for row in db.execute("SELECT dataset_id FROM ap_datasets")
            }
            if not new_ids <= existing_ids:
                tasks = {task.task_id: task for task in (*old_tasks, *new_tasks)}
                measured_ids = {
                    task.task_id
                    for task in (*old_tasks, *new_tasks)
                    if task.split in MEASUREMENT_SPLITS
                }
                current_training = {
                    task.task_id for task in new_tasks if task.split in {"train", "dev"}
                }
                for bundle, _ in prepared:
                    for task in bundle.tasks:
                        if task.task_id in current_training and any(
                            db.execute(
                                "SELECT 1 FROM ap_measurement_blocks WHERE kind=? AND value=?",
                                key,
                            ).fetchone()
                            for key in task_keys(task, bundle.manifest)
                        ):
                            raise StoreError(
                                "cross-build training input overlaps protected measurement provenance"
                            )
                vectors = dict(result.embeddings.vectors)
                snapshots = [source.to_dict() for source in result.bundles[0].sources]
                for row in db.execute("SELECT DISTINCT audit_ref FROM ap_datasets"):
                    audit_ref = ArtifactRef.from_bytes(row[0])
                    cache = json.loads(
                        self.artifacts.read(audit_ref, "embedding-cache")
                    )
                    snapshots.extend(
                        json.loads(self.artifacts.read(audit_ref, "source-snapshots"))
                    )
                    if (
                        cache["model_id"] == result.embeddings.model_id
                        and cache["revision"] == result.embeddings.revision
                    ):
                        for key, vector in cache["vectors"].items():
                            vectors.setdefault(key, tuple(vector))
                for snapshot in snapshots:
                    if snapshot["source"]["role"] != "benchmark":
                        continue
                    spec = SourceSpec.model_validate_json(
                        json.dumps(snapshot["source"])
                    )
                    for value in snapshot["rows"]:
                        item = SourceRow(
                            **{
                                **value,
                                "lineage_keys": tuple(value.get("lineage_keys", ())),
                            }
                        )
                        normalized = normalize(SourceSnapshot(spec, (item,)), item)
                        tasks[normalized.prompt_id] = SimpleNamespace(
                            task_id=normalized.prompt_id,
                            cluster_id="benchmark-" + normalized.prompt_id,
                            problem=normalized.problem,
                            lineage_keys=normalized.lineage,
                            source_role="benchmark",
                        )
                        measured_ids.add(normalized.prompt_id)
                items = tuple(
                    SimpleNamespace(
                        prompt_id=task.task_id,
                        problem=task.problem,
                        lineage=task.lineage_keys,
                    )
                    for task in tasks.values()
                    if task.source_role != "hidden"
                )
                cache = EmbeddingTable(
                    result.embeddings.model_id,
                    result.embeddings.revision,
                    vectors,
                    result.embeddings.fixture_only,
                )
                settings = BuildConfig.model_validate_json(
                    json.dumps(result.report["config"])
                )
                groups, catalog_edges = cluster(items, cache, settings)
                blocked_clusters = set()
                for group in groups:
                    if not measured_ids.intersection(item.prompt_id for item in group):
                        continue
                    if current_training.intersection(item.prompt_id for item in group):
                        raise StoreError(
                            "cross-build near-duplicate measurement cluster cannot reach training"
                        )
                    for item in group:
                        task = tasks[item.prompt_id]
                        blocked_clusters.add(task.cluster_id)
                        keys = {
                            ("prompt", task.task_id),
                            ("cluster", task.cluster_id),
                            ("problem", problem_key(task.problem)),
                            *(("lineage", key) for key in task.lineage_keys),
                        }
                        for key in keys:
                            db.execute(
                                "INSERT OR IGNORE INTO ap_measurement_blocks VALUES (?,?)",
                                key,
                            )
                db.execute(
                    "INSERT INTO ap_data_catalog_checks VALUES (?,?)",
                    (
                        result.build_id,
                        canonical_bytes(
                            {
                                "schema": "autopost.catalog-check/1",
                                "build_id": result.build_id,
                                "prior_dataset_ids": sorted(existing_ids),
                                "model_id": cache.model_id,
                                "revision": cache.revision,
                                "edges": catalog_edges,
                                "blocked_clusters": sorted(blocked_clusters),
                                "forbidden_training_overlap": False,
                            }
                        ),
                    ),
                )
            for bundle, ref in prepared:
                manifest = bundle.manifest
                old = db.execute(
                    "SELECT manifest,artifact_ref,audit_ref FROM ap_datasets WHERE dataset_id=?",
                    (manifest.dataset.dataset_id,),
                ).fetchone()
                if old:
                    if (
                        old[0] != manifest.to_bytes()
                        or old[1] != ref.to_bytes()
                        or old[2] != audit.artifact.to_bytes()
                    ):
                        raise ConflictError("dataset identity is immutable")
                    continue
                db.execute(
                    "INSERT INTO ap_datasets VALUES (?,?,?,?,?,?,?)",
                    (
                        manifest.dataset.dataset_id,
                        manifest.dataset.sha256,
                        manifest.split,
                        int(manifest.fixture_only),
                        manifest.to_bytes(),
                        ref.to_bytes(),
                        audit.artifact.to_bytes(),
                    ),
                )
                for task in bundle.tasks:
                    db.execute(
                        "INSERT INTO ap_dataset_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            manifest.dataset.dataset_id,
                            task.task_id,
                            task.cluster_id,
                            task.source_id,
                            task.source_ver,
                            task.source_item_id,
                            task.source_role,
                            task.answer_type,
                            int(task.measurable),
                            task.split,
                            task.schema_version,
                            task.to_bytes(),
                        ),
                    )
                    keys = task_keys(task, manifest)
                    for kind, value in keys:
                        db.execute(
                            "INSERT INTO ap_task_keys VALUES (?,?,?,?)",
                            (manifest.dataset.dataset_id, task.task_id, kind, value),
                        )
                        if task.split in MEASUREMENT_SPLITS and (
                            kind != "family" or task.source_role == "transfer"
                        ):
                            db.execute(
                                "INSERT OR IGNORE INTO ap_measurement_blocks VALUES (?,?)",
                                (kind, value),
                            )
        return {
            bundle.manifest.split: bundle.manifest.dataset for bundle, _ in prepared
        }

    def manifest(self, ref: DatasetRef) -> DatasetManifest:
        ref.to_bytes()
        with self._connection() as db:
            row = db.execute(
                "SELECT manifest,artifact_ref FROM ap_datasets WHERE dataset_id=? AND sha256=?",
                (ref.dataset_id, ref.sha256),
            ).fetchone()
        if row is None:
            raise StoreError("unknown or mismatched dataset reference")
        manifest = DatasetManifest.from_bytes(row[0])
        artifact = ArtifactRef.from_bytes(row[1])
        if self.artifacts.read(artifact, "dataset-manifest") != manifest.to_bytes():
            raise StoreError("catalog differs from finalized dataset manifest")
        if (
            hashlib.sha256(self.artifacts.read(artifact, "tasks")).hexdigest()
            != manifest.tasks_sha256
        ):
            raise StoreError("dataset task digest mismatch")
        return manifest

    def inspect(self, ref: DatasetRef) -> tuple[Task, ...]:
        self.manifest(ref)
        with self._connection() as db:
            return tuple(
                Task.from_bytes(row[0])
                for row in db.execute(
                    "SELECT record FROM ap_dataset_tasks WHERE dataset_id=? ORDER BY task_id",
                    (ref.dataset_id,),
                )
            )

    def training_tasks(
        self, ref: DatasetRef, *, fixture_only=False
    ) -> tuple[Task, ...]:
        self.manifest(ref)
        with self._connection() as db:
            return training_rows(db, ref, fixture_only=fixture_only)

    def assert_training_tasks(
        self, ref: DatasetRef, tasks: tuple[Task, ...], *, fixture_only=False
    ) -> None:
        canonical = {
            task.task_id: task.to_bytes()
            for task in self.training_tasks(ref, fixture_only=fixture_only)
        }
        if any(canonical.get(task.task_id) != task.to_bytes() for task in tasks):
            raise StoreError(
                "caller-supplied or relabeled task is outside the canonical training slice"
            )

    def report(self, ref: DatasetRef):
        manifest = self.manifest(ref)
        with self._connection() as db:
            row = db.execute(
                "SELECT audit_ref FROM ap_datasets WHERE dataset_id=?",
                (ref.dataset_id,),
            ).fetchone()
        report = json.loads(
            self.artifacts.read(ArtifactRef.from_bytes(row[0]), "build-report")
        )
        if content_hash(report) != manifest.report_sha256:
            raise StoreError("build report digest mismatch")
        return report

    def catalog_report(self, build_id: str):
        with self._connection() as db:
            row = db.execute(
                "SELECT report FROM ap_data_catalog_checks WHERE build_id=?",
                (build_id,),
            ).fetchone()
        if row is None:
            raise StoreError("catalog check is not registered")
        return json.loads(row[0])

    def record_review(
        self, program: HiddenProgram, report: HiddenTestReport, review: HumanReview
    ):
        program = HiddenProgram.from_bytes(program.to_bytes())
        report = HiddenTestReport.from_bytes(report.to_bytes())
        review = HumanReview.from_bytes(review.to_bytes())
        if content_hash(
            test_program(program, seed=report.seed, trials=report.trials)
        ) != content_hash(report):
            raise StoreError("hidden test report cannot be reconstructed")
        verify_review(program, report, review)
        with self._connection(write=True) as db:
            previous = db.execute(
                "SELECT record FROM ap_hidden_reviews WHERE program_digest=? AND report_digest=?",
                (program.digest, content_hash(report)),
            ).fetchone()
            if previous and previous[0] != review.to_bytes():
                raise ConflictError(
                    "review history is immutable; revise and retest the program"
                )
            db.execute(
                "INSERT OR IGNORE INTO ap_hidden_reviews VALUES (?,?,?)",
                (program.digest, content_hash(report), review.to_bytes()),
            )

    def evaluation_tasks(
        self, ref: DatasetRef, *, fixture_only=False
    ) -> tuple[Task, ...]:
        manifest = self.manifest(ref)
        if manifest.split not in MEASUREMENT_SPLITS or manifest.split in SEALED_SPLITS:
            raise StoreError("only disposable measurement partitions are available")
        with self._connection() as db:
            return evaluation_rows(db, ref, manifest.split, fixture_only=fixture_only)
