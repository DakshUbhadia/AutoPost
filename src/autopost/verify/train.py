"""Bounded process pool for training only; no measurement imports."""

import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore

from autopost.interfaces import Completion, Task, TrainVerifier, Verdict

PINS = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "sympy": "1.14.0",
    "antlr4-python3-runtime": "4.13.2",
    "mpmath": "1.3.0",
}
CHECKER_ID = "math-verify-0.9.0:latex-1.11.0:sympy-1.14.0:antlr-4.13.2:mpmath-1.3.0:policy-1:deadline-10"


class MathVerifyPool(TrainVerifier):
    checker_id = CHECKER_ID

    def __init__(self, *, workers=2, timeout_seconds=10.0):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workers must be in [1,32]")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for package, pin in PINS.items():
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise RuntimeError(
                    "install requirements/train-verify.lock before using the training verifier"
                ) from exc
            if installed != pin:
                raise RuntimeError(
                    f"unsupported {package} version {installed}; required {pin}"
                )
        self.workers, self.timeout_seconds = workers, timeout_seconds
        self.checker_id = (
            CHECKER_ID.rsplit(":deadline-", 1)[0] + f":deadline-{timeout_seconds:g}"
        )
        self._slots = BoundedSemaphore(workers)

    def check(self, task: Task, completion: Completion) -> Verdict:
        if completion.task_id != task.task_id:
            raise ValueError("completion belongs to a different task")
        if completion.truncated:
            return Verdict(False, "truncated", None, self.checker_id)
        if not completion.text.strip():
            return Verdict(False, "no_answer", None, self.checker_id)
        if len(completion.text) > 200_000:
            return Verdict(False, "parse_error", None, self.checker_id)
        # UTF-8 size is also bounded; never pass arbitrary runtime objects.
        payload = json.dumps(
            {"text": completion.text, "reference": task.answer}, ensure_ascii=True
        ).encode("ascii")
        if len(payload) > 235_000:
            return Verdict(False, "parse_error", None, self.checker_id)
        worker = Path(__file__).with_name("train_worker.py").resolve()
        with self._slots, TemporaryDirectory(prefix="autopost-verifier-") as directory:
            # -I ignores PYTHONPATH/user site; no inherited credentials or user
            # Python startup hooks. Text is JSON stdin, never shell code.
            environment = {
                key: value
                for key, value in os.environ.items()
                if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}
            }
            try:
                result = subprocess.run(
                    [sys.executable, "-I", str(worker)],
                    input=payload,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=directory,
                    env=environment,
                    timeout=self.timeout_seconds,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return Verdict(False, "timeout", None, self.checker_id)
            except OSError as exc:
                raise RuntimeError("training verifier worker could not start") from exc
        try:
            value = json.loads(result.stdout)
            if (
                result.returncode != 0
                or type(value["correct"]) is not bool
                or value["status"]
                not in {"ok", "wrong", "no_answer", "multiple_answers", "parse_error"}
                or value["correct"] != (value["status"] == "ok")
                or (
                    value["extracted"] is not None
                    and not isinstance(value["extracted"], str)
                )
            ):
                raise ValueError("invalid verifier result")
            return Verdict(**value, checker_id=self.checker_id)
        except (KeyError, TypeError, ValueError):
            return Verdict(False, "parse_error", None, self.checker_id)

    def check_many(self, pairs):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return tuple(pool.map(lambda pair: self.check(*pair), pairs))
