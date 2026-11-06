"""Stable wire error codes; backend exception types never cross boundaries."""

from enum import Enum


class ErrorCode(str, Enum):
    INVALID_SPEC = "invalid_spec"
    UNSUPPORTED_VERSION = "unsupported_version"
    INCOMPATIBLE = "incompatible"
    EXECUTOR_FAILURE = "executor_failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    MISSING_ARTIFACT = "missing_artifact"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    CONFLICT = "conflict"
    STALE_PARENT = "stale_parent"


class CompatibilityError(ValueError):
    code = ErrorCode.INCOMPATIBLE


class UnsupportedVersionError(ValueError):
    code = ErrorCode.UNSUPPORTED_VERSION
