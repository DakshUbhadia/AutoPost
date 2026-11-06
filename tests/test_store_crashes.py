"""Real process exits verify filesystem publication and SQLite WAL recovery."""

import subprocess
import sys

import pytest

from autopost.execution.fixture import FixtureExecutor
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.metadata import SQLiteMetadataStore

CHILD = r"""
import os
import sys
from pathlib import Path
from autopost.contracts import decode_job_spec
from autopost.execution.fixture import FixtureExecutor, SimulatedCrash
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.metadata import SQLiteMetadataStore

class CrashStore(SQLiteMetadataStore):
    def _before_accept_commit(self):
        os._exit(23)

root, phase = Path(sys.argv[1]), sys.argv[2]
artifacts = LocalArtifactStore(root / "artifacts")
store_type = CrashStore if phase == "during_metadata_commit" else SQLiteMetadataStore
metadata = store_type(root / "metadata.sqlite", artifacts)
scenario = phase if phase != "during_metadata_commit" else "success"
executor = FixtureExecutor(metadata, artifacts, scenario=scenario)
spec = decode_job_spec((root / "job.json").read_bytes())
try:
    handle = executor.submit(spec)
    metadata.accept_result(executor.result(handle), executor.effects(handle))
except SimulatedCrash:
    os._exit(23)
raise RuntimeError("crash injection did not fire")
"""


@pytest.mark.parametrize(
    "phase", ["crash_before_finalize", "crash_after_finalize", "during_metadata_commit"]
)
def test_abrupt_process_crash_has_no_effect_and_retry_recovers_once(storage, phase):
    spec = storage.training()
    (storage.root / "job.json").write_bytes(spec.to_bytes())
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(storage.root), phase],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 23, child.stdout + child.stderr
    artifacts = LocalArtifactStore(storage.root / "artifacts")
    metadata = SQLiteMetadataStore(storage.database, artifacts)
    assert metadata.get_result(spec.job_id) is None
    assert metadata.get_pointer(storage.run_id, "cursor") == storage.base.state
    assert metadata.budget_remaining(storage.run_id, "operations") == 10
    with pytest.raises(KeyError):
        metadata.get_state(spec.expected_state)
    executor = FixtureExecutor(metadata, artifacts)
    handle = executor.submit(spec)
    result = executor.result(handle)
    effects = executor.effects(handle)
    assert metadata.accept_result(result, effects).disposition == "accepted"
    assert metadata.accept_result(result, effects).disposition == "duplicate"
    assert metadata.get_state(spec.expected_state) == effects.states[0]
    assert metadata.budget_remaining(storage.run_id, "operations") == 9
