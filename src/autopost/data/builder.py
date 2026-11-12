"""Deterministic cluster-level development datasets with complete exclusions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

from autopost.contracts.datasets import BuildConfig, DatasetManifest
from autopost.contracts.serialization import content_hash
from autopost.contracts.tasks import MEASUREMENT_SPLITS, Task

from .dedup import EmbeddingTable, cluster
from .hidden import HiddenProgram, generate, test_program
from .normalize import normalize, problem_key, type_answer
from .sources import SourceSnapshot

BUILDER_VERSION = "data-builder-1"


@dataclass(frozen=True)
class DatasetBundle:
    manifest: DatasetManifest
    tasks: tuple[Task, ...]
    report: dict
    sources: tuple[SourceSnapshot, ...]


@dataclass(frozen=True)
class BuildResult:
    build_id: str
    bundles: tuple[DatasetBundle, ...]
    report: dict
    embeddings: EmbeddingTable


def build(
    sources: tuple[SourceSnapshot, ...],
    embeddings: EmbeddingTable,
    config: BuildConfig,
    *,
    programs: tuple[HiddenProgram, ...] = (),
) -> BuildResult:
    config = BuildConfig.from_bytes(config.to_bytes())
    sources = tuple(
        sorted(sources, key=lambda item: (item.spec.source_id, item.spec.revision))
    )
    if len({source.spec.source_id for source in sources}) != len(sources):
        raise ValueError("each source identity must have exactly one pinned snapshot")
    if not {"train", "transfer", "benchmark"} <= {
        source.spec.role for source in sources
    }:
        raise ValueError(
            "training, transfer and public-benchmark reference snapshots are required"
        )
    fixture_only = any(source.spec.fixture_only for source in sources)
    if embeddings.fixture_only and not all(
        source.spec.fixture_only for source in sources
    ):
        raise ValueError("synthetic embeddings cannot process real source snapshots")
    programs = tuple(sorted(programs, key=lambda item: item.program_id))
    if len({program.program_id for program in programs}) != len(programs):
        raise ValueError("duplicate hidden program identity")
    semantic_programs = [
        content_hash(
            program.model_dump(exclude={"program_id", "origin", "parent_lineage_keys"})
        )
        for program in programs
    ]
    if len(set(semantic_programs)) != len(semantic_programs):
        raise ValueError("duplicate hidden program semantics")
    input_digest = content_hash(
        {
            "sources": [source.digest for source in sources],
            "embeddings": embeddings.digest,
            "programs": [program.digest for program in programs],
        }
    )
    build_id = content_hash(
        {
            "version": BUILDER_VERSION,
            "inputs": input_digest,
            "config": config.model_dump(mode="json"),
        }
    )
    rows, rejections = [], []

    def reject(row, stage, reason):
        rejections.append(
            {
                "key": f"{row.source.spec.source_id}@{row.source.spec.revision}:{row.item.source_item_id}",
                "stage": stage,
                "reason": reason,
            }
        )

    for source in sources:
        seen = set()
        for item in sorted(source.rows, key=lambda value: value.source_item_id):
            if item.source_item_id in seen:
                raise ValueError("duplicate source item identity in pinned snapshot")
            seen.add(item.source_item_id)
            try:
                rows.append(normalize(source, item))
            except ValueError as exc:
                if source.spec.role == "benchmark":
                    raise ValueError(
                        "invalid benchmark reference; refusing incomplete decontamination"
                    ) from exc
                rejections.append(
                    {
                        "key": f"{source.spec.source_id}@{source.spec.revision}:{item.source_item_id}",
                        "stage": "normalize",
                        "reason": str(exc),
                    }
                )
    groups, edges = cluster(tuple(rows), embeddings, config)
    protected_families = {
        source.spec.family_id for source in sources if source.spec.role == "transfer"
    }
    held_out = {key for program in programs for key in program.parent_lineage_keys}
    if not held_out <= {key for row in rows for key in row.lineage}:
        raise ValueError("hidden programs require known held-out source parents")
    candidates, transfer, forced_pilot = [], [], []
    for group in groups:
        identity = "cluster-" + content_hash(sorted(row.prompt_id for row in group))
        roles = {row.source.spec.role for row in group}
        if "benchmark" in roles:
            for row in group:
                reject(
                    row,
                    "decontaminate",
                    "benchmark_reference_only"
                    if row.source.spec.role == "benchmark"
                    else "benchmark_overlap",
                )
            continue
        if len({row.answer for row in group}) > 1:
            for row in group:
                reject(row, "deduplicate", "conflicting_reference_labels")
            continue
        eligible = []
        for row in group:
            if row.source.spec.role == "train" and (
                "transfer" in roles or row.source.spec.family_id in protected_families
            ):
                reject(row, "decontaminate", "transfer_source_lineage")
            else:
                eligible.append(row)
        if not eligible:
            continue
        representative = min(eligible, key=lambda row: row.prompt_id)
        for row in eligible:
            if row is not representative:
                reject(row, "deduplicate", "duplicate_in_cluster")
        representative = replace(
            representative,
            lineage=tuple(sorted({key for row in group for key in row.lineage})),
        )
        entry = (identity, representative)
        if "transfer" in roles:
            if representative.measurable:
                transfer.append(entry)
            else:
                reject(representative, "measurement_filter", "non_numeric_measurement")
        elif any(held_out & set(row.lineage) for row in group):
            if not representative.measurable:
                raise ValueError(
                    "hidden parent must be an exact-numeric held-out source item"
                )
            forced_pilot.append(entry)
        else:
            candidates.append(entry)
    order = lambda entry: content_hash({"seed": config.seed, "cluster": entry[0]})
    candidates.sort(key=order)
    transfer.sort(key=order)
    forced_pilot.sort(key=order)
    if len(forced_pilot) > config.split_clusters.get("pilot", 0):
        raise ValueError("pilot quota cannot hold all hidden-program parents")
    assignments = {"pilot": list(forced_pilot)}
    for split in (
        "pilot",
        "screen",
        "gate_a",
        "gate_a_reservoir",
        "gate_b",
        "gate_b_reservoir",
        "dev",
        "train",
        "transfer_guard",
    ):
        requested = config.split_clusters.get(split, 0) - len(
            assignments.get(split, ())
        )
        pool = transfer if split == "transfer_guard" else candidates
        eligible = [
            entry
            for entry in pool
            if split not in MEASUREMENT_SPLITS or entry[1].measurable
        ]
        if len(eligible) < requested:
            raise ValueError(f"insufficient eligible clusters for {split}")
        selected = eligible[:requested]
        assignments.setdefault(split, []).extend(selected)
        for entry in selected:
            pool.remove(entry)
    for _, row in (*candidates, *transfer):
        reject(row, "split", "unallocated_cluster")
    tasks_by_split = {}
    for split, entries in assignments.items():
        tasks_by_split[split] = [
            Task(
                task_id=row.prompt_id,
                cluster_id=identity,
                source_id=row.source.spec.source_id,
                source_ver=row.source.spec.revision,
                source_item_id=row.item.source_item_id,
                source_role=row.source.spec.role,
                lineage_keys=row.lineage,
                skill=row.item.skill,
                problem=row.problem,
                answer=row.answer,
                answer_type=row.answer_type,
                low_entropy=row.low_entropy,
                split=split,
            )
            for identity, row in entries
        ]
    hidden_reports, hidden_tasks = [], []
    pilot_lineage = {key for _, row in assignments["pilot"] for key in row.lineage}
    for program in programs:
        if not set(program.parent_lineage_keys) <= pilot_lineage:
            raise ValueError("hidden parent was excluded or cannot be held out")
        report = test_program(
            program, seed=config.seed, trials=config.hidden_property_trials
        )
        if not report.passed:
            raise ValueError("hidden program failed invariant/property campaign")
        instances, rejected = generate(
            program, seed=config.seed, count=config.hidden_instances_per_program
        )
        if len(instances) != config.hidden_instances_per_program:
            raise ValueError("hidden program could not generate its requested pool")
        report_digest = content_hash(report)
        hidden_reports.append(
            {
                "program": program.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "review_status": "pending_human_review",
                "generation_rejections": rejected,
            }
        )
        for instance in instances:
            answer, kind = type_answer(instance.answer)
            hidden_tasks.append(
                Task(
                    task_id=instance.instance_id,
                    cluster_id=program.program_id,
                    source_id="hidden-dev",
                    source_ver=program.digest,
                    source_item_id=instance.instance_id,
                    source_role="hidden",
                    lineage_keys=program.parent_lineage_keys,
                    skill=program.skill,
                    problem=instance.problem,
                    answer=answer,
                    answer_type=kind,
                    low_entropy=Fraction(answer).denominator == 1
                    and abs(Fraction(answer)) <= 10,
                    split="hidden_dev",
                    program_id=program.program_id,
                    program_digest=program.digest,
                    property_report_digest=report_digest,
                )
            )
    if hidden_tasks:
        tasks_by_split["hidden_dev"] = hidden_tasks
    measured = [
        task
        for split, tasks in tasks_by_split.items()
        if split in MEASUREMENT_SPLITS
        for task in tasks
    ]
    protected_problems = {problem_key(task.problem) for task in measured}
    protected_lineage = {key for task in measured for key in task.lineage_keys}
    for split in ("train", "dev"):
        if any(
            problem_key(task.problem) in protected_problems
            or protected_lineage.intersection(task.lineage_keys)
            for task in tasks_by_split.get(split, ())
        ):
            raise ValueError(
                "final hidden/measurement decontamination failed; no training artifact can be published"
            )
    by_cluster = {}
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            previous = by_cluster.setdefault(task.cluster_id, split)
            if previous != split:
                raise ValueError("forbidden cluster overlap")
    report = {
        "schema": "autopost.data-build/1",
        "builder_version": BUILDER_VERSION,
        "build_id": build_id,
        "config": config.model_dump(mode="json"),
        "input_digest": input_digest,
        "fixture_only": fixture_only,
        "embedding": {
            "model_id": embeddings.model_id,
            "revision": embeddings.revision,
            "digest": embeddings.digest,
            "fixture_only": embeddings.fixture_only,
        },
        "dedup_edges": sorted(edges, key=content_hash),
        "rejections": sorted(rejections, key=content_hash),
        "contamination": {
            "forbidden_cluster_overlap": [],
            "transfer_families": sorted(protected_families),
        },
        "hidden_programs": hidden_reports,
        "splits": {
            split: {
                "items": len(tasks),
                "clusters": len({task.cluster_id for task in tasks}),
            }
            for split, tasks in sorted(tasks_by_split.items())
        },
    }
    bundles = []
    specs = tuple(source.spec for source in sources)
    for split, tasks in sorted(tasks_by_split.items()):
        if not tasks:
            continue
        tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
        payload = task_bytes(tasks)
        manifest = DatasetManifest.create(
            fixture_only=fixture_only,
            split=split,
            build_id=build_id,
            config_id=content_hash(config),
            input_digest=input_digest,
            tasks_sha256=hashlib.sha256(payload).hexdigest(),
            report_sha256=content_hash(report),
            task_ids=tuple(task.task_id for task in tasks),
            cluster_ids=tuple(sorted({task.cluster_id for task in tasks})),
            sources=specs,
            program_digests=tuple(program.digest for program in programs)
            if split == "hidden_dev"
            else (),
        )
        bundles.append(DatasetBundle(manifest, tasks, report, sources))
    return BuildResult(build_id, tuple(bundles), report, embeddings)


def task_bytes(tasks: tuple[Task, ...]) -> bytes:
    return b"".join(
        task.to_bytes() + b"\n" for task in sorted(tasks, key=lambda task: task.task_id)
    )
