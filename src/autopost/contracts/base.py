"""Strict, versioned, project-owned durable values."""

from __future__ import annotations

import json
from typing import ClassVar, Literal, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from .errors import UnsupportedVersionError
from .serialization import canonical_bytes, validate_json


def protocol_compatibility(version: object) -> Literal["supported", "rejected"]:
    # No previous durable job/ref/manifest protocol has shipped. In particular,
    # Revision 2 Python records are not a version-0 wire protocol to migrate.
    return "supported" if type(version) is int and version == 1 else "rejected"


def _boundary_value(value: object) -> object:
    if isinstance(value, DurableModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return {key: _boundary_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_boundary_value(child) for child in value]
    return value


class DurableModel(BaseModel):
    opaque_text_fields: ClassVar[frozenset[str]] = frozenset()
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )
    schema_version: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def check_boundary(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, dict):
            version = value.get("schema_version", 1)
            if protocol_compatibility(version) != "supported":
                raise UnsupportedVersionError(
                    f"{cls.__name__}: unsupported schema_version {version!r}"
                )
        payload = _boundary_value(value)
        validate_json(payload)
        if isinstance(payload, dict):

            def nested_model(annotation):
                return (
                    isinstance(annotation, type)
                    and issubclass(annotation, DurableModel)
                    or any(nested_model(arg) for arg in get_args(annotation))
                )

            for key, child in payload.items():
                field = cls.model_fields.get(key)
                # Nested models validate their own fields. Opaque generated
                # text may contain slash-prefixed math without being a locator.
                if key not in cls.opaque_text_fields and not (
                    field and nested_model(field.annotation)
                ):
                    validate_json(child, paths=True)
        if info.mode == "json" and isinstance(value, dict):
            # A before-model validator materializes JSON arrays as lists.
            # Normalize only tuple fields, leaving strict scalar checks intact.
            value = dict(value)
            for name, field in cls.model_fields.items():
                if get_origin(field.annotation) is tuple and isinstance(
                    value.get(name), list
                ):
                    value[name] = tuple(value[name])
        return value

    def to_bytes(self) -> bytes:
        return canonical_bytes(self)

    @classmethod
    def from_bytes(cls, blob: bytes) -> Self:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(blob, object_pairs_hook=pairs)
        # Run version checks before any backend could receive the value.
        return cls.model_validate_json(json.dumps(value, allow_nan=False))
