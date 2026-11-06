"""Backend exceptions carry stable project error codes, never library types."""

from autopost.contracts.errors import ErrorCode


class StoreError(RuntimeError):
    code = ErrorCode.INVALID_SPEC


class SchemaVersionError(StoreError):
    code = ErrorCode.UNSUPPORTED_VERSION


class MissingArtifactError(StoreError):
    code = ErrorCode.MISSING_ARTIFACT


class CorruptArtifactError(StoreError):
    code = ErrorCode.CORRUPT_ARTIFACT


class ConflictError(StoreError):
    code = ErrorCode.CONFLICT


class StaleParentError(StoreError):
    code = ErrorCode.STALE_PARENT
