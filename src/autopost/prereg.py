"""Phase I pre-registration creation, validation, and freezing."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from autopost.config import (
    config_hash,
    load_config,
    merged_config_hash,
    stack_hash,
    unresolved_fields,
)
from autopost.contracts.serialization import canonical_bytes as canonical_json
from autopost.store.db import connect


class Prereg(BaseModel):
    schema_: Literal["autopost.prereg/1", "autopost.prereg/2"] = Field(
        default="autopost.prereg/1", alias="schema"
    )
    meta: dict[str, Any]
    stack: dict[str, Any]
    judgements: dict[str, Any]
    statistics: dict[str, Any]
    phase1_plan: dict[str, Any]
    go_no_go: dict[str, Any]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_draft(version: str, base_yaml: Path, lock: Path, **profiles) -> dict:
    merged = load_config(base_yaml, **profiles)
    cfg = merged.get("experiment", merged)
    profiles = cfg.get("model_profiles", {})
    model_revisions = {
        name: {
            key: profile.get(key)
            for key in ("hf_repo", "revision", "tokenizer_revision")
        }
        for name, profile in profiles.items()
    }
    margins = {item["name"]: None for item in cfg.get("gate", {}).get("guardrails", [])}
    draft = {
        "schema": "autopost.prereg/1",
        "meta": {
            "version": version,
            "phase": "phase1",
            "stage": "B",
            "created_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
        },
        "stack": {
            "base_yaml_sha256": _sha(base_yaml),
            "lock_sha256": _sha(lock),
            "models": model_revisions,
            "embedding_model": cfg.get("data", {}).get("dedup", {}).get("embed_model"),
            "embedding_revision": cfg.get("data", {})
            .get("dedup", {})
            .get("embed_model_revision"),
        },
        "judgements": {
            "floor": None,
            "guardrail_margins": margins,
            "phase1_compute_budget": {
                "hardware": None,
                "gpu_hours_total": None,
                "calendar_limit_days": None,
            },
            "e1a_learning_rates": [2.0e-5, None],
        },
        "statistics": {
            "alpha_a": 0.05,
            "alpha_b": 0.05,
            "guardrail_alpha": 0.05,
            "n_boot": 4000,
            "bootstrap_unit": "cluster",
            "k_ref_multiplier": 2,
            "per_set_power_target": 0.90,
            "guardrail_power_target": 0.97,
            "target_joint_power": 0.80,
            "nominal_false_promotion": 0.0025,
        },
        "phase1_plan": {
            "e1a": {"rates": [2.0e-5, None], "steps_per_rate": 200, "split": "dev"},
            "e1": {"steps": 1000, "checkpoint_every_steps": 50, "split": "pilot"},
            "e2_compute_matched_to": "e1",
            "e9_compute_matched_to": "e1",
        },
        "go_no_go": {
            "signal": {
                "comparison": "GRPO minus random-reward on pilot",
                "interval": "cluster-bootstrap excludes 0",
                "minimum_gain": "floor",
                "within_step_budget": True,
            },
            "power": {
                "promote_at_floor_at_least": 0.80,
                "promote_under_no_effect_at_most": 0.0025,
            },
            "controller": {
                "expected_false_halts_at_most": 1.0,
                "halt_improving_run": False,
            },
            "budget": {"main_run_plus_replication_and_evaluation_within_budget": True},
        },
    }
    if "experiment" in merged:
        draft["schema"] = "autopost.prereg/2"
        mode = merged["execution"]["validation_mode"]
        draft["meta"].update(stage=mode, mode=mode)
        model = merged["model"]
        draft["stack"] = {
            "config_id": config_hash(merged),
            "merged_config_sha256": merged_config_hash(merged),
            "stack_id": stack_hash(merged, lock_sha256=_sha(lock)),
            "lock_sha256": _sha(lock),
            "config": merged,
            "models": {
                "selected": {
                    "hf_repo": model["model_id"],
                    "revision": model["revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                }
            },
            "embedding_model": merged["data"].get("dedup", {}).get("embed_model"),
            "embedding_revision": merged["data"]
            .get("dedup", {})
            .get("embed_model_revision"),
        }
    return draft


def canonical_bytes(prereg: dict) -> bytes:
    # Preserve the historical prereg v1 trailing newline, using the shared JSON rules.
    return canonical_json(prereg) + b"\n"


def validate(
    prereg: dict, base_yaml: Path, *, lock: Path = Path("requirements.lock"), **profiles
) -> list[str]:
    problems: list[str] = []
    try:
        Prereg(**prereg)
    except ValidationError as exc:
        problems.append(f"invalid prereg structure: {exc}")
    schema = prereg.get("schema")
    if schema not in {"autopost.prereg/1", "autopost.prereg/2"}:
        problems.append("unsupported prereg schema")
    j = prereg.get("judgements", {})
    stats = prereg.get("statistics", {})
    for key in (
        "floor",
        "guardrail_margins",
        "phase1_compute_budget",
        "e1a_learning_rates",
    ):
        if key not in j or _has_null(j[key]):
            problems.append(f"judgements.{key} contains a required null or is missing")
    floor = j.get("floor")
    if not _fraction(floor):
        problems.append("judgements.floor must be in (0, 1)")
    for name, margin in j.get("guardrail_margins", {}).items():
        if not _fraction(margin):
            problems.append(f"guardrail margin {name} must be in (0, 1)")
    for name in ("alpha_a", "alpha_b", "guardrail_alpha"):
        if not _fraction(stats.get(name)):
            problems.append(f"statistics.{name} must be in (0, 1)")
    try:
        merged = load_config(base_yaml, **profiles)
        cfg = merged.get("experiment", merged)
        if any(kind == "pin" for _, kind in unresolved_fields(merged)):
            problems.append("configuration still contains <pin placeholders")
        if "experiment" in merged:
            if schema != "autopost.prereg/2":
                problems.append("split configuration requires autopost.prereg/2")
            stack = prereg.get("stack", {})
            _same(stack.get("config_id"), config_hash(merged), "config_id", problems)
            _same(
                stack.get("merged_config_sha256"),
                merged_config_hash(merged),
                "merged_config_sha256",
                problems,
            )
            _same(stack.get("config"), merged, "config snapshot", problems)
            lock_digest = stack.get("lock_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", str(lock_digest)):
                problems.append("stack.lock_sha256 must be a SHA-256 digest")
            _same(lock_digest, _sha(lock), "lock_sha256", problems)
            _same(
                stack.get("stack_id"),
                stack_hash(merged, lock_sha256=lock_digest),
                "stack_id",
                problems,
            )
            model = merged["model"]
            _same(
                stack.get("models"),
                {
                    "selected": {
                        "hf_repo": model["model_id"],
                        "revision": model["revision"],
                        "tokenizer_revision": model["tokenizer_revision"],
                    }
                },
                "stack.models",
                problems,
            )
            _same(
                stack.get("embedding_model"),
                merged["data"].get("dedup", {}).get("embed_model"),
                "stack.embedding_model",
                problems,
            )
            _same(
                stack.get("embedding_revision"),
                merged["data"].get("dedup", {}).get("embed_model_revision"),
                "stack.embedding_revision",
                problems,
            )
            mode = merged["execution"]["validation_mode"]
            _same(prereg.get("meta", {}).get("mode"), mode, "meta.mode", problems)
            if mode != "research":
                problems.append(
                    "scientific Phase I freeze requires a research execution profile"
                )
            for key in ("revision", "tokenizer_revision"):
                if not re.fullmatch(r"[0-9a-fA-F]{40}", merged["model"][key]):
                    problems.append(f"model.{key} must be a 40-hex revision")
            if merged["model"]["max_context_tokens"] is None:
                problems.append("model.max_context_tokens requires spike validation")
            for name in ("rollout_sampling", "eval_sampling"):
                if merged["execution"][name]["max_new_tokens"] is None:
                    problems.append(
                        f"execution.{name}.max_new_tokens must be calibrated"
                    )
        elif schema == "autopost.prereg/2":
            problems.append("autopost.prereg/2 requires split configuration")
        for name, profile in cfg.get("model_profiles", {}).items():
            for key in ("revision", "tokenizer_revision"):
                if not re.fullmatch(r"[0-9a-fA-F]{40}", str(profile.get(key, ""))):
                    problems.append(
                        f"model_profiles.{name}.{key} must be a 40-hex revision"
                    )
        for name, profile in prereg.get("stack", {}).get("models", {}).items():
            for key in ("revision", "tokenizer_revision"):
                if not re.fullmatch(r"[0-9a-fA-F]{40}", str(profile.get(key, ""))):
                    problems.append(
                        f"stack.models.{name}.{key} must be a 40-hex revision"
                    )
        embedding_revision = prereg.get("stack", {}).get("embedding_revision")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(embedding_revision or "")):
            problems.append("stack.embedding_revision must be a 40-hex revision")
        gate = cfg.get("gate", {})
        _same(gate.get("floor"), floor, "gate.floor", problems)
        yaml_margins = {
            g.get("name"): g.get("margin") for g in gate.get("guardrails", [])
        }
        for name, margin in j.get("guardrail_margins", {}).items():
            _same(
                yaml_margins.get(name),
                margin,
                f"gate.guardrails.{name}.margin",
                problems,
            )
        _same(
            gate.get("primary", {}).get("alpha_a"),
            stats.get("alpha_a"),
            "alpha_a",
            problems,
        )
        _same(
            gate.get("primary", {}).get("alpha_b"),
            stats.get("alpha_b"),
            "alpha_b",
            problems,
        )
        if any(
            g.get("alpha") != stats.get("guardrail_alpha")
            for g in gate.get("guardrails", [])
        ):
            problems.append("base.yaml guardrail alpha values disagree with prereg")
    except (
        OSError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        yaml.YAMLError,
    ) as exc:
        problems.append(f"could not compare base.yaml values: {exc}")
    return problems


def warnings(prereg: dict) -> list[str]:
    judgements = prereg.get("judgements", {})
    notes = []
    if isinstance(judgements.get("floor"), (int, float)) and judgements["floor"] < 0.01:
        notes.append("floor < 0.01; evaluation cost scales roughly with 1/floor²")
    for name, margin in judgements.get("guardrail_margins", {}).items():
        if isinstance(margin, (int, float)) and margin < 0.005:
            notes.append(
                f"{name} margin < 0.005; evaluation cost scales roughly with 1/floor²"
            )
    return notes


def freeze(
    prereg: dict,
    base_yaml: Path,
    destination: Path,
    db_path: Path,
    *,
    lock: Path = Path("requirements.lock"),
    **profiles,
) -> str:
    problems = validate(prereg, base_yaml, lock=lock, **profiles)
    if problems:
        raise ValueError("; ".join(problems))
    version = prereg["meta"]["version"]
    blob = canonical_bytes(prereg)
    digest = hashlib.sha256(blob).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM prereg WHERE version=?", (version,)
        ).fetchone():
            raise ValueError(f"pre-registration version already exists: {version}")
        destination.write_bytes(blob)
        destination.chmod(0o444)
        connection.execute(
            "INSERT INTO prereg(version, sha256) VALUES (?, ?)", (version, digest)
        )
        connection.execute(
            "INSERT INTO events(kind, payload) VALUES (?, ?)",
            (
                "prereg_frozen",
                json.dumps(
                    {"version": version, "sha256": digest, "path": str(destination)},
                    sort_keys=True,
                ),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return digest


def verify(db_path: Path, prereg_dir: Path) -> list[tuple[str, bool]]:
    connection = connect(db_path)
    try:
        records = connection.execute(
            "SELECT version, sha256 FROM prereg ORDER BY version"
        ).fetchall()
    finally:
        connection.close()
    results = []
    for row in records:
        path = prereg_dir / f"{row['version']}.json"
        ok = (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        )
        results.append((row["version"], ok))
    return results


def _has_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_has_null(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_null(item) for item in value)
    return False


def _fraction(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 < value < 1
    )


def _same(left: Any, right: Any, label: str, problems: list[str]) -> None:
    if left != right:
        problems.append(f"base.yaml {label} disagrees with prereg")
