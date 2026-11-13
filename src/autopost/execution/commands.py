"""Filesystem/CLI adapters for local-executor and development GPU checks."""

import json
import time
from pathlib import Path

from autopost.config import load_config, stack_hash
from autopost.contracts import (
    DatasetRef,
    ExecutionProfile,
    ModelSpec,
    StateRef,
    TrainingJobSpec,
    content_hash,
)
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.metadata import SQLiteMetadataStore

from .capabilities import algorithm_controls, discover, runtime_snapshot
from .fixture import initialize_fixture_run
from .spike import run_spike
from .subprocess import LocalSubprocessExecutor


def execute_command(args):
    if args.action == "spike":
        report = run_spike(
            args.root,
            base_yaml=args.base_yaml,
            execution=args.execution,
            run_gpu=args.run_gpu,
        )
        return report, 0 if report["status"] == "PASS" else 2
    if args.action == "capabilities":
        cfg = load_config(args.base_yaml, execution=args.execution)
        snapshot = runtime_snapshot()
        report = discover(
            ModelSpec.model_validate_json(json.dumps(cfg["model"])),
            ExecutionProfile.model_validate_json(json.dumps(cfg["execution"])),
            stack_hash(cfg, lock_sha256=content_hash(snapshot["versions"])),
            snapshot=snapshot,
        )
        profile = ExecutionProfile.model_validate_json(json.dumps(cfg["execution"]))
        algorithm_controls(cfg["experiment"], profile)
        return {
            "report": report.model_dump(mode="json"),
            "runtime": snapshot,
        }, 0 if report.compatible else 2
    root = Path(args.root)
    artifacts = LocalArtifactStore(root / "artifacts")
    metadata = SQLiteMetadataStore(root / "metadata.sqlite", artifacts)
    digest = content_hash({"fixture_only": True, "subprocess": True})
    dataset = DatasetRef(dataset_id="fixture-training", sha256=digest)
    base = initialize_fixture_run(
        metadata,
        artifacts,
        run_id="subprocess-fixture",
        config_id=digest,
        stack_id=digest,
        dataset=dataset,
    )
    spec = TrainingJobSpec.create(
        run_id="subprocess-fixture",
        config_id=digest,
        stack_id=digest,
        model_spec_id=digest,
        execution_profile_id=digest,
        code_revision="subprocess-fixture@1",
        created_at="2026-11-12T00:00:00Z",
        seed=17,
        start=base.state,
        training_slice=dataset,
        expected_state=StateRef(state_id="subprocess-next"),
        steps=1,
        verifier_version="fixture-v1",
        expected_outputs=("subprocess-state", "subprocess-rollouts"),
    )
    with LocalSubprocessExecutor(
        metadata,
        artifacts,
        root / "jobs",
        handlers={"training": "autopost.execution.fixture_worker:execute"},
    ) as executor:
        handle = executor.submit(spec)
        while executor.status(handle) in {"pending", "running"}:
            time.sleep(0.02)
        result = executor.result(handle)
        receipt = (
            metadata.accept_result(result, executor.effects(handle))
            if result.status == "succeeded"
            else None
        )
        return {
            "fixture_only": True,
            "handle": handle.model_dump(mode="json"),
            "status": result.status,
            "acceptance": receipt.disposition if receipt else None,
            "logs": executor.logs(handle),
            "remaining_budget": metadata.budget_remaining(spec.run_id, "operations"),
        }, 0 if result.status == "succeeded" else 2
