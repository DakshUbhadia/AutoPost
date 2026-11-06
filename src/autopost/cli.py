"""Command line interface for repository setup and preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from autopost.config import (
    config_hash,
    load_config,
    merged_config_hash,
    stack_hash,
    unresolved_fields,
)
from autopost.prereg import freeze, make_draft, validate, verify, warnings
from autopost.store.db import init_db, migrate_db, schema_version


def _run_fixture(root: Path, scenario: str) -> dict:
    from autopost.contracts import DatasetRef, StateRef, TrainingJobSpec, content_hash
    from autopost.execution.fixture import FixtureExecutor, initialize_fixture_run
    from autopost.store.artifacts import LocalArtifactStore
    from autopost.store.metadata import SQLiteMetadataStore

    directory = root / scenario
    artifacts = LocalArtifactStore(directory / "artifacts")
    metadata = SQLiteMetadataStore(directory / "metadata.sqlite", artifacts)
    identity = content_hash({"fixture_only": True})
    dataset = DatasetRef(dataset_id="fixture-train", sha256=identity)
    base = initialize_fixture_run(
        metadata,
        artifacts,
        run_id="fixture-run",
        config_id=identity,
        stack_id=identity,
        dataset=dataset,
    )
    spec = TrainingJobSpec.create(
        run_id="fixture-run",
        config_id=identity,
        stack_id=identity,
        model_spec_id=identity,
        execution_profile_id=identity,
        code_revision="fixture-v1",
        created_at="2026-11-06T00:00:00Z",
        seed=0,
        start=base.state,
        training_slice=dataset,
        expected_state=StateRef(state_id="fixture-next"),
        expected_outputs=("fixture-state", "fixture-rollouts"),
        steps=5,
        verifier_version="fixture-v1",
    )
    dispositions = []
    if scenario in {"duplicate_success", "conflict"}:
        first = FixtureExecutor(metadata, artifacts)
        handle = first.submit(spec)
        dispositions.append(
            metadata.accept_result(
                first.result(handle), first.effects(handle)
            ).disposition
        )
        spec = spec.model_copy(update={"attempt": 2})
    executor = FixtureExecutor(metadata, artifacts, scenario=scenario)
    handle = executor.submit(spec)
    result = executor.result(handle)
    if result.status == "succeeded":
        dispositions.append(
            metadata.accept_result(result, executor.effects(handle)).disposition
        )
    report = {
        "schema": "autopost.fixture-report/1",
        "fixture_only": True,
        "scenario": scenario,
        "job_id": spec.job_id,
        "attempt_status": executor.status(handle),
        "dispositions": dispositions,
        "remaining_budget": metadata.budget_remaining("fixture-run", "operations"),
        "accepted_result": metadata.get_result(spec.job_id) is not None,
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autopost")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config").add_subparsers(dest="action", required=True)
    config_commands = [
        config.add_parser(action) for action in ("placeholders", "show", "hashes")
    ]
    db = commands.add_parser("db").add_subparsers(dest="action", required=True)
    init = db.add_parser("init")
    init.add_argument("--path", default="runs/autopost.sqlite")
    for action in ("migrate", "status"):
        db.add_parser(action).add_argument("--path", default="runs/autopost.sqlite")
    fixture = commands.add_parser("fixture").add_subparsers(
        dest="action", required=True
    )
    fixture_run = fixture.add_parser("run")
    fixture_run.add_argument("--root", default="runs/sprint2-fixture")
    fixture_run.add_argument(
        "--scenario",
        choices=(
            "success",
            "duplicate_success",
            "conflict",
            "stale_parent",
            "incomplete_artifact",
            "executor_failure",
        ),
        default="success",
    )
    prereg = commands.add_parser("prereg").add_subparsers(dest="action", required=True)
    draft = prereg.add_parser("init")
    draft.add_argument("--version", default="stageB-p1-v1")
    draft.add_argument("--phase", default="phase1")
    draft.add_argument("--base-yaml", default="configs/base.yaml")
    draft.add_argument("--lock", default="requirements.lock")
    valid = prereg.add_parser("validate")
    valid.add_argument("path")
    valid.add_argument("--base-yaml", default="configs/base.yaml")
    fr = prereg.add_parser("freeze")
    fr.add_argument("path")
    fr.add_argument("--db", default="runs/autopost.sqlite")
    fr.add_argument("--base-yaml", default="configs/base.yaml")
    fr.add_argument("--output-dir", default="configs/prereg")
    valid.add_argument("--lock", default="requirements.lock")
    fr.add_argument("--lock", default="requirements.lock")
    for command in config_commands:
        command.add_argument("--base-yaml", default="configs/base.yaml")
        command.add_argument("--lock", default="requirements.lock")
    for command in [*config_commands, draft, valid, fr]:
        for name in ("model", "execution", "data"):
            command.add_argument(
                f"--{name}", help="profile path relative to --base-yaml (or absolute)"
            )
    ver = prereg.add_parser("verify")
    ver.add_argument("--db", default="runs/autopost.sqlite")
    ver.add_argument("--prereg-dir", default="configs/prereg")
    args = parser.parse_args(argv)
    profiles = {
        name: getattr(args, name, None) for name in ("model", "execution", "data")
    }
    try:
        if args.command == "config":
            cfg = load_config(args.base_yaml, **profiles)
            if args.action == "show":
                print(json.dumps(cfg, indent=2, ensure_ascii=False))
            elif args.action == "hashes":
                lock_digest = hashlib.sha256(Path(args.lock).read_bytes()).hexdigest()
                print(
                    json.dumps(
                        {
                            "config_id": config_hash(cfg),
                            "merged_config_sha256": merged_config_hash(cfg),
                            "stack_id": stack_hash(cfg, lock_sha256=lock_digest),
                        },
                        indent=2,
                    )
                )
            else:
                for path, kind in unresolved_fields(cfg):
                    print(f"{path}\t{kind}")
        elif args.command == "db":
            if args.action == "init":
                init_db(args.path)
            elif args.action == "migrate":
                print(f"metadata schema {migrate_db(args.path)}")
            else:
                print(f"metadata schema {schema_version(args.path)}")
        elif args.command == "fixture":
            print(
                json.dumps(
                    _run_fixture(Path(args.root), args.scenario),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "prereg" and args.action == "init":
            if args.phase != "phase1":
                parser.error("Phase II pre-registration is built in a later step")
            path = Path("configs/prereg") / f"{args.version}.draft.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                __import__(
                    "autopost.prereg", fromlist=["canonical_bytes"]
                ).canonical_bytes(
                    make_draft(
                        args.version, Path(args.base_yaml), Path(args.lock), **profiles
                    )
                )
            )
            print(path)
        elif args.command == "prereg" and args.action == "validate":
            doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
            problems = validate(
                doc, Path(args.base_yaml), lock=Path(args.lock), **profiles
            )
            for warning in warnings(doc):
                print(f"WARNING: {warning}")
            for problem in problems:
                print(f"ERROR: {problem}")
            if problems:
                return 1
            print("valid")
        elif args.command == "prereg" and args.action == "freeze":
            doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
            output = Path(args.output_dir) / f"{doc['meta']['version']}.json"
            print(
                f"sha256 {freeze(doc, Path(args.base_yaml), output, Path(args.db), lock=Path(args.lock), **profiles)}"
            )
            print("Create the prereg-stageB-p1-v1 git tag after review.")
        elif args.command == "prereg" and args.action == "verify":
            results = verify(Path(args.db), Path(args.prereg_dir))
            for version, ok in results:
                print(f"{'OK' if ok else 'MISMATCH'} {version}")
            if not results:
                print("no pre-registrations")
            return 0 if all(ok for _, ok in results) else 1
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
