"""Trusted local backend dispatcher. This module is a subprocess entrypoint."""

import importlib
import json
import os
import sys
import time
from pathlib import Path

from autopost.contracts import decode_job_spec
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.metadata import SQLiteMetadataStore

from .subprocess import failure


def main():
    directory = Path(sys.argv[1])
    request = json.loads((directory / "request.json").read_bytes())
    spec = decode_job_spec(json.dumps(request["spec"]).encode())
    artifacts = LocalArtifactStore(request["artifacts"])
    metadata = SQLiteMetadataStore(request["metadata"], artifacts)
    try:
        module, name = request["handler"].split(":")
        handler = getattr(importlib.import_module(module), name)
        result, effects = handler(spec, metadata, artifacts, request["options"])
    except Exception:  # noqa: BLE001 -- backend exceptions must become stable results
        import traceback

        traceback.print_exc()
        result = failure(
            spec,
            "executor_failure",
            request["started_at"],
            message="backend handler failed; inspect stderr log",
        )
        effects = None
    value = {
        "finished_monotonic": time.monotonic(),
        "result": result.model_dump(mode="json"),
        "effects": effects.model_dump(mode="json") if effects is not None else None,
    }
    with (directory / "response.tmp").open("x", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(directory / "response.tmp", directory / "response.json")


if __name__ == "__main__":
    main()
