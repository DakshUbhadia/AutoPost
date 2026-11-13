"""Local process placement. Only this backend handles process/file locators."""

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from autopost.contracts import (
    AcceptanceEffects,
    EvaluationJobResult,
    JobError,
    TrainingJobResult,
    TrainingJobSpec,
    decode_job_result,
    decode_job_spec,
)
from autopost.store.errors import StoreError


def failure(spec, code, started_at, *, message=None):
    common = {
        name: getattr(spec, name)
        for name in (
            "job_id",
            "run_id",
            "config_id",
            "stack_id",
            "model_spec_id",
            "execution_profile_id",
            "code_revision",
            "attempt",
        )
    }
    common.update(
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        status="cancelled" if code == "cancelled" else "failed",
        error=JobError(
            code=code,
            message=message or code,
            retryable=code in {"executor_failure", "timeout"},
        ),
    )
    if isinstance(spec, TrainingJobSpec):
        return TrainingJobResult(
            **common, start=spec.start, training_slice=spec.training_slice
        )
    return EvaluationJobResult(
        **common,
        states=spec.states,
        dataset=spec.dataset,
        eval_read_id=spec.eval_read_id,
    )


class LocalSubprocessExecutor:
    """Trusted installed handlers run behind the versioned JobSpec boundary.

    Result journaling never accepts state/budget effects. The controller must
    explicitly call accept_result. Active processes belong to this instance;
    completed results can be collected after restart from atomic response files.
    """

    def __init__(
        self,
        metadata,
        artifacts,
        root,
        *,
        handlers,
        options=None,
        timeout_seconds=600.0,
        python_executable=None,
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.metadata, self.artifacts = metadata, artifacts
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.handlers, self.options = dict(handlers), dict(options or {})
        self.timeout_seconds = timeout_seconds
        self.python = str(python_executable or sys.executable)
        self._active = {}
        self._threads = []
        self._lock = threading.RLock()

    def _directory(self, handle):
        handle.to_bytes()
        return self.root / handle.job_id / str(handle.attempt)

    def submit(self, spec):
        spec = decode_job_spec(spec.to_bytes())
        if spec.kind not in self.handlers:
            raise ValueError(
                f"no installed handler for {spec.kind}; training backends arrive in Sprint 6"
            )
        with self._lock:
            handle = self.metadata.reserve_job(spec)
            if (
                self.metadata.get_attempt(handle).result is not None
                or handle in self._active
            ):
                return handle
            directory = self._directory(handle)
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                self._collect(handle)
                if self.metadata.get_attempt(handle).result is None:
                    raise StoreError(
                        "attempt already claimed; collect it or use a new attempt after recovery"
                    ) from None
                return handle
            started = datetime.now(UTC).isoformat()
            request = {
                "spec": spec.model_dump(mode="json"),
                "handler": self.handlers[spec.kind],
                "options": self.options,
                "metadata": str(self.metadata.path.resolve()),
                "artifacts": str(self.artifacts.root),
                "started_at": started,
                "deadline_monotonic": time.monotonic() + self.timeout_seconds,
            }
            (directory / "request.json").write_text(
                json.dumps(request), encoding="utf-8"
            )
            if not self.metadata.start_attempt(handle):
                raise StoreError(
                    "attempt already claimed in metadata; do not execute it twice"
                )
            stdout = (directory / "stdout.log").open("wb")
            stderr = (directory / "stderr.log").open("wb")
            try:
                process = subprocess.Popen(
                    [
                        self.python,
                        "-I",
                        "-m",
                        "autopost.execution.worker",
                        str(directory),
                    ],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError:
                stdout.close()
                stderr.close()
                self.metadata.record_result(
                    failure(
                        spec,
                        "executor_failure",
                        started,
                        message="worker could not start; see backend configuration",
                    )
                )
                return handle
            self._active[handle] = (process, spec, started)
            thread = threading.Thread(
                target=self._watch, args=(handle, stdout, stderr), daemon=True
            )
            self._threads.append(thread)
            thread.start()
            return handle

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
            if process.poll() is None:
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()

    def _watch(self, handle, stdout, stderr):
        process, spec, started = self._active[handle]
        timed_out = False
        try:
            try:
                request = json.loads(
                    (self._directory(handle) / "request.json").read_bytes()
                )
                process.wait(
                    timeout=max(0.0, request["deadline_monotonic"] - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
            with self._lock:
                if self.metadata.get_attempt(handle).result is None:
                    if timed_out:
                        self.metadata.record_result(failure(spec, "timeout", started))
                    else:
                        self._collect(handle)
                        if self.metadata.get_attempt(handle).result is None:
                            self.metadata.record_result(
                                failure(
                                    spec,
                                    "executor_failure",
                                    started,
                                    message="worker exited without a valid result; inspect captured logs",
                                )
                            )
        finally:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout.close()
            stderr.close()
            with self._lock:
                self._active.pop(handle, None)

    def _collect(self, handle):
        if self.metadata.get_attempt(handle).result is not None:
            return
        directory = self._directory(handle)
        response = directory / "response.json"
        if not response.is_file():
            return
        try:
            value = json.loads(response.read_bytes())
            request = json.loads((directory / "request.json").read_bytes())
            if value["finished_monotonic"] > request["deadline_monotonic"]:
                active = self._active.get(handle)
                if active is not None:
                    self._terminate(active[0])
                spec = self.metadata.get_job(handle.job_id).model_copy(
                    update={"attempt": handle.attempt}
                )
                self.metadata.record_result(
                    failure(spec, "timeout", request["started_at"])
                )
                return
            result = decode_job_result(json.dumps(value["result"]).encode())
            if result.job_id != handle.job_id or result.attempt != handle.attempt:
                raise ValueError("worker result belongs to another attempt")
            if value["effects"] is not None:
                effects = AcceptanceEffects.from_bytes(
                    json.dumps(value["effects"]).encode()
                )
                if effects.run_id != result.run_id:
                    raise ValueError("worker effects belong to another run")
            self.metadata.record_result(result)
        except (ValueError, KeyError, StoreError) as exc:
            spec = self.metadata.get_job(handle.job_id).model_copy(
                update={"attempt": handle.attempt}
            )
            started = json.loads((directory / "request.json").read_bytes())[
                "started_at"
            ]
            self.metadata.record_result(
                failure(
                    spec,
                    "executor_failure",
                    started,
                    message="invalid worker response: " + type(exc).__name__,
                )
            )

    def status(self, handle):
        with self._lock:
            self._collect(handle)
            return self.metadata.get_attempt(handle).status

    def result(self, handle):
        self.status(handle)
        result = self.metadata.get_attempt(handle).result
        if result is None:
            raise StoreError("job has no terminal result yet")
        return result

    def effects(self, handle):
        if self.result(handle).status != "succeeded":
            raise StoreError("failed attempts have no acceptance effects")
        value = json.loads((self._directory(handle) / "response.json").read_bytes())
        if value["effects"] is None:
            return AcceptanceEffects(
                run_id=self.result(handle).run_id, expected_parent=None
            )
        return AcceptanceEffects.from_bytes(json.dumps(value["effects"]).encode())

    def cancel(self, handle):
        with self._lock:
            if self.status(handle) in {"succeeded", "failed", "cancelled"}:
                return
            active = self._active.get(handle)
            if active is None:
                raise StoreError("this executor does not own the active process")
            process, spec, started = active
            self._terminate(process)
            self.metadata.record_result(failure(spec, "cancelled", started))

    def logs(self, handle):
        self.metadata.get_attempt(handle)
        directory = self._directory(handle)
        return {
            name: (directory / (name + ".log")).read_text(
                encoding="utf-8", errors="replace"
            )
            for name in ("stdout", "stderr")
        }

    def close(self):
        for handle in tuple(self._active):
            self.cancel(handle)
        for thread in self._threads:
            thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
