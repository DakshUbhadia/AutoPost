"""Configuration loading and placeholder inspection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autopost.interfaces import stable_hash


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return value


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg)


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
