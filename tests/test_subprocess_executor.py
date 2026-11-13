"""Real process lifecycle tests; all payloads are explicitly synthetic."""

import time

import pytest

from autopost.execution.subprocess import LocalSubprocessExecutor
from autopost.store.errors import StoreError

HANDLERS = {
    kind: "autopost.execution.fixture_worker:execute"
    for kind in ("training", "evaluation")
}


def executor(storage, **options):
    return LocalSubprocessExecutor(
        storage.metadata,
        storage.artifacts,
        storage.root / "jobs",
        handlers=HANDLERS,
        **options,
    )


def wait(executor, handle):
    deadline = time.monotonic() + 30
    while executor.status(handle) not in {"succeeded", "failed", "cancelled"}:
        if time.monotonic() > deadline:
            pytest.fail("subprocess did not finish")
        time.sleep(0.03)
    return executor.result(handle)


@pytest.mark.parametrize("kind", ["training", "evaluation"])
def test_success_duplicate_and_restart_collect_with_explicit_acceptance(storage, kind):
    with executor(storage) as worker:
        spec = getattr(storage, kind)()
        handle = worker.submit(spec)
        assert worker.submit(spec) == handle
        result = wait(worker, handle)
        assert result.status == "succeeded"
        assert storage.metadata.get_result(spec.job_id) is None
        receipt = storage.metadata.accept_result(result, worker.effects(handle))
        assert receipt.disposition == "accepted"
        assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9
        assert "synthetic fixture worker started" in worker.logs(handle)["stdout"]
    with executor(storage) as restarted:
        assert restarted.submit(spec) == handle
        assert restarted.result(handle) == result
        assert (
            storage.metadata.accept_result(
                result, restarted.effects(handle)
            ).disposition
            == "duplicate"
        )
        assert storage.metadata.budget_remaining(storage.run_id, "operations") == 9


@pytest.mark.parametrize(
    "options,timeout,code",
    [
        ({"raise_error": True}, 30, "executor_failure"),
        ({"delay_seconds": 5}, 0.1, "timeout"),
    ],
)
def test_failure_timeout_and_new_attempt(storage, options, timeout, code):
    with executor(storage, options=options, timeout_seconds=timeout) as worker:
        spec = storage.training()
        handle = worker.submit(spec)
        result = wait(worker, handle)
        assert result.error.code == code
        assert result.artifacts == ()
        assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10
        if options.get("raise_error"):
            assert "synthetic worker failure" in worker.logs(handle)["stderr"]
    with executor(storage) as healthy:
        retry = spec.model_copy(update={"attempt": 2})
        assert wait(healthy, healthy.submit(retry)).status == "succeeded"


def test_cancel_and_ownership_across_instances(storage):
    with executor(storage, options={"delay_seconds": 10}) as owner:
        spec = storage.training()
        handle = owner.submit(spec)
        with (
            LocalSubprocessExecutor(
                storage.metadata,
                storage.artifacts,
                storage.root / "other",
                handlers=HANDLERS,
            ) as other,
            pytest.raises(StoreError, match="claimed"),
        ):
            other.submit(spec)
        owner.cancel(handle)
        owner.cancel(handle)
        assert wait(owner, handle).status == "cancelled"
        assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10


def test_spawn_error_and_missing_handler(storage):
    with executor(
        storage, python_executable=str(storage.root / "missing-python")
    ) as worker:
        assert (
            wait(worker, worker.submit(storage.training())).error.code
            == "executor_failure"
        )
    with (
        LocalSubprocessExecutor(
            storage.metadata, storage.artifacts, storage.root / "none", handlers={}
        ) as worker,
        pytest.raises(ValueError, match="no installed handler"),
    ):
        worker.submit(storage.evaluation())


def test_late_response_cannot_win_against_a_delayed_watchdog(storage, monkeypatch):
    original = LocalSubprocessExecutor._watch

    def delayed(self, *args):
        time.sleep(1.0)
        return original(self, *args)

    monkeypatch.setattr(LocalSubprocessExecutor, "_watch", delayed)
    with executor(storage, timeout_seconds=0.05) as worker:
        result = wait(worker, worker.submit(storage.training()))
        assert result.error.code == "timeout"
        assert storage.metadata.budget_remaining(storage.run_id, "operations") == 10
