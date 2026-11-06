"""Shared CPU-only harness for store/executor contract and failure tests."""

from types import SimpleNamespace

import pytest

from autopost.contracts import DatasetRef, EvaluationJobSpec, StateRef, TrainingJobSpec
from autopost.execution.fixture import FixtureExecutor, initialize_fixture_run
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.metadata import SQLiteMetadataStore

DIGEST = "a" * 64


@pytest.fixture
def storage(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    database = tmp_path / "metadata.sqlite"
    metadata = SQLiteMetadataStore(database, artifacts)
    dataset = DatasetRef(dataset_id="fixture-train", sha256=DIGEST)
    root = initialize_fixture_run(
        metadata,
        artifacts,
        run_id="fixture-run",
        config_id=DIGEST,
        stack_id=DIGEST,
        dataset=dataset,
    )
    common = {
        "run_id": "fixture-run",
        "config_id": DIGEST,
        "stack_id": DIGEST,
        "model_spec_id": DIGEST,
        "execution_profile_id": DIGEST,
        "code_revision": "fixture-rev",
        "created_at": "2026-11-06T00:00:00Z",
        "seed": 17,
    }

    def training(**overrides):
        state = overrides.pop("expected_state", StateRef(state_id="fixture-next"))
        inputs = {
            **common,
            "start": root.state,
            "training_slice": dataset,
            "expected_state": state,
            "steps": 5,
            "verifier_version": "fixture-v1",
            "expected_outputs": (
                state.state_id + "-bundle",
                state.state_id + "-rollouts",
            ),
            **overrides,
        }
        return TrainingJobSpec.create(**inputs)

    def evaluation(**overrides):
        return EvaluationJobSpec.create(
            **{
                **common,
                "states": (root.state,),
                "dataset": dataset,
                "eval_read_id": "fixture-read",
                "instrument_id": "fixture-instrument",
                "template_ids": ("eval-only",),
                "k": 2,
                "k_ref": 4,
                "fresh_reference": True,
                "split": "gate_b",
                "expected_outputs": ("fixture-eval-raw",),
                **overrides,
            }
        )

    return SimpleNamespace(
        root=tmp_path,
        database=database,
        artifacts=artifacts,
        metadata=metadata,
        base=root,
        dataset=dataset,
        training=training,
        evaluation=evaluation,
        executor=FixtureExecutor(metadata, artifacts),
        run_id="fixture-run",
    )
