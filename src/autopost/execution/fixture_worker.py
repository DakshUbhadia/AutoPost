"""Explicitly selected synthetic handler for subprocess conformance tests."""

import time

from autopost.contracts import AttemptRecord

from .fixture import FixtureExecutor


class _BufferedJournal:
    """Let fixtures return results to the dispatcher without early publication."""

    def __init__(self, metadata):
        self.metadata = metadata
        self.result = None

    def __getattr__(self, name):
        return getattr(self.metadata, name)

    def record_result(self, result):
        self.result = result

    def get_attempt(self, handle):
        if self.result is not None:
            return AttemptRecord(
                handle=handle, status=self.result.status, result=self.result
            )
        return self.metadata.get_attempt(handle)


def execute(spec, metadata, artifacts, options):
    delay = options.get("delay_seconds", 0)
    print("synthetic fixture worker started", flush=True)
    time.sleep(delay)
    if options.get("raise_error"):
        raise RuntimeError("synthetic worker failure")
    executor = FixtureExecutor(
        _BufferedJournal(metadata),
        artifacts,
        scenario=options.get("scenario", "success"),
    )
    handle = executor.submit(spec)
    result = executor.result(handle)
    effects = executor.effects(handle) if result.status == "succeeded" else None
    return result, effects
