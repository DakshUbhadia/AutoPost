"""Revision 3 composition and scientific/stack identities.

Includes resolve in this I/O adapter; source paths never enter merged values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from autopost.contracts import ExecutionProfile, ModelSpec, content_hash


def _mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")  # noqa: TRY004
    return value


def load_config(
    path: str | Path = "configs/base.yaml",
    *,
    model: str | Path | None = None,
    execution: str | Path | None = None,
    data: str | Path | None = None,
) -> dict:
    path = Path(path)
    experiment = _mapping(path)
    if "profiles" not in experiment:
        # Read-only transition for historical preregistration v1 inputs.
        if "schema_version" in experiment:
            raise ValueError("versioned configuration requires profile selections")
        return experiment
    profiles = experiment.pop("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "model",
        "execution",
        "data",
    }:
        raise ValueError("profiles must select exactly model, execution and data")
    if (
        type(experiment.get("schema_version")) is not int
        or experiment["schema_version"] != 1
    ):
        raise ValueError("unsupported configuration schema_version")
    values = {}
    for name, override in (("model", model), ("execution", execution), ("data", data)):
        source = Path(override if override is not None else profiles[name])
        values[name] = _mapping(
            source if source.is_absolute() else path.parent / source
        )
    values["model"] = ModelSpec.model_validate_json(
        json.dumps(values["model"])
    ).model_dump(mode="json")
    values["execution"] = ExecutionProfile.model_validate_json(
        json.dumps(values["execution"])
    ).model_dump(mode="json")
    if (
        type(values["data"].get("schema_version")) is not int
        or values["data"]["schema_version"] != 1
    ):
        raise ValueError("unsupported data configuration schema_version")
    return {"schema_version": 1, "experiment": experiment, **values}


def merged_config_hash(cfg: dict) -> str:
    """Audit identity of all composed values, independent of YAML formatting."""
    return content_hash(cfg)


def config_hash(cfg: dict) -> str:
    """Scientific semantics. Execution changes do not change this identity."""
    if "experiment" not in cfg:
        return content_hash(cfg)
    return content_hash(
        {key: cfg[key] for key in ("schema_version", "experiment", "model", "data")}
    )


def stack_hash(cfg: dict, *, lock_sha256: str) -> str:
    """All execution settings, dependency lock content, and wire versions.

    Parallelism, offload, checkpointing and sampling can affect numerics, so
    topology is conservatively included. Physical deployment locations are not.
    """
    return content_hash(
        {
            "config_id": config_hash(cfg),
            "execution": cfg.get("execution", {}),
            "lock_sha256": lock_sha256,
            "contracts": {
                name: 1
                for name in (
                    "JobSpec",
                    "JobResult",
                    "StateManifest",
                    "ArtifactManifest",
                    "EvalEvidence",
                    "refs",
                )
            },
        }
    )


def unresolved_fields(cfg: dict) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if value is None:
            found.append((path, "null"))
        elif isinstance(value, str) and "<pin" in value:
            found.append((path, "pin"))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(cfg, "")
    return found
