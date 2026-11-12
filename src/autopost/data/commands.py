"""Filesystem/CLI adapters for reproducible development builds and reviews."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from autopost.contracts.datasets import BuildConfig
from autopost.contracts.serialization import canonical_bytes, content_hash
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.datasets import DatasetRepository
from autopost.store.errors import ConflictError

from .builder import build
from .dedup import EmbeddingTable
from .fixtures import development_inputs
from .hidden import HiddenProgram, HiddenTestReport, HumanReview
from .sources import read_jsonl


def _write_once(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = canonical_bytes(value) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(blob)
    except FileExistsError:
        if path.read_bytes() != blob:
            raise ConflictError(
                "immutable build report already has different bytes"
            ) from None


def build_command(
    root: Path,
    *,
    seed=0,
    source_paths=(),
    embedding_path=None,
    config_path=None,
    programs_path=None,
):
    if not source_paths:
        sources, embeddings, config, programs = development_inputs(seed)
    else:
        if embedding_path is None or config_path is None:
            raise ValueError(
                "local sources require explicit embedding and build-config files"
            )
        sources = tuple(read_jsonl(path) for path in source_paths)
        value = json.loads(Path(embedding_path).read_text(encoding="utf-8"))
        embeddings = EmbeddingTable(
            **{
                **value,
                "vectors": {
                    key: tuple(vector) for key, vector in value["vectors"].items()
                },
            }
        )
        config = BuildConfig.model_validate_json(
            json.dumps(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
        )
        programs = ()
    if config_path:
        config = BuildConfig.model_validate_json(
            json.dumps(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
        )
    if programs_path:
        programs = tuple(
            HiddenProgram.model_validate_json(json.dumps(value))
            for value in json.loads(Path(programs_path).read_text(encoding="utf-8"))
        )
    if embedding_path and not source_paths:
        raise ValueError("custom embedding files require explicit source snapshots")
    result = build(sources, embeddings, config, programs=programs)
    repo = DatasetRepository(
        root / "datasets.sqlite", LocalArtifactStore(root / "artifacts")
    )
    refs = repo.publish(result)
    index = {
        "schema": "autopost.development-data/1",
        "build_id": result.build_id,
        "fixture_only": result.report["fixture_only"],
        "splits": {name: ref.model_dump(mode="json") for name, ref in refs.items()},
        "counts": result.report["splits"],
        "hidden_review_status": "pending_human_review"
        if programs
        else "not_applicable",
    }
    directory = root / result.build_id
    _write_once(directory / "index.json", index)
    _write_once(
        directory / "contamination-report.json",
        {
            key: result.report[key]
            for key in ("contamination", "dedup_edges", "rejections")
        },
    )
    _write_once(
        directory / "hidden-review-packets.json", result.report["hidden_programs"]
    )
    _write_once(directory / "build-report.json", result.report)
    return index


def review_command(
    root: Path,
    packet_path: Path,
    program_id: str,
    reviewer: str,
    decision: str,
    notes: str,
):
    packets = json.loads(packet_path.read_text(encoding="utf-8"))
    packet = next(
        (packet for packet in packets if packet["program"]["program_id"] == program_id),
        None,
    )
    if packet is None:
        raise ValueError("program not found in generated-instance review packet")
    program = HiddenProgram.model_validate_json(json.dumps(packet["program"]))
    report = HiddenTestReport.model_validate_json(json.dumps(packet["report"]))
    review = HumanReview(
        program_digest=program.digest,
        report_digest=content_hash(report),
        reviewer=reviewer,
        decision=decision,
        notes=notes,
        sampled_instance_ids=tuple(item.instance_id for item in report.sample),
    )
    repo = DatasetRepository(
        root / "datasets.sqlite", LocalArtifactStore(root / "artifacts")
    )
    repo.record_review(program, report, review)
    return review.model_dump(mode="json")
