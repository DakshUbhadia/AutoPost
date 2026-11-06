"""Canonical JSON: UTF-8, sorted string keys, compact separators, no NaN.

Never stringify unknown objects: that hides process-local identity in hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


def validate_json(value: object, *, paths: bool = False) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str:
        if paths and (
            value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", value)
        ):
            raise ValueError(
                "physical absolute paths are forbidden in durable contracts"
            )
        return
    if type(value) in (list, tuple):
        for child in value:
            validate_json(child, paths=paths)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, child in value.items():
            validate_json(key, paths=paths)
            validate_json(child, paths=paths)
        return
    raise ValueError(f"not a finite JSON value: {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    # Import lazily so the serializer itself has no framework dependency.
    from .base import DurableModel

    if isinstance(value, DurableModel):
        model = value
        value = model.model_dump(mode="python")
        validate_json(value, paths=True)
        # Frozen top-level models can still contain mutable JSON maps. Recheck
        # invariants here so mutation/model_copy cannot bypass the boundary.
        type(model).model_validate_json(json.dumps(value, allow_nan=False))
    validate_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
