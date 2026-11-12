"""Versioned task values shared by ingestion, storage and trainer boundaries."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Literal

from .serialization import canonical_bytes

AnswerType = Literal["int", "rational", "decimal", "expr", "interval", "tuple"]
MEASURABLE_TYPES = frozenset({"int", "rational", "decimal"})
Split = Literal[
    "train",
    "dev",
    "pilot",
    "screen",
    "gate_a",
    "gate_a_reservoir",
    "gate_b",
    "gate_b_reservoir",
    "transfer_guard",
    "transfer_sealed",
    "hidden_dev",
    "hidden_audit",
    "hidden_sealed",
]
MEASUREMENT_SPLITS = frozenset(
    {
        "pilot",
        "screen",
        "gate_a",
        "gate_a_reservoir",
        "gate_b",
        "gate_b_reservoir",
        "transfer_guard",
        "transfer_sealed",
        "hidden_dev",
        "hidden_audit",
        "hidden_sealed",
    }
)
DEVELOPMENT_SPLITS = frozenset(
    {
        "train",
        "dev",
        "pilot",
        "screen",
        "gate_a",
        "gate_a_reservoir",
        "gate_b",
        "gate_b_reservoir",
        "transfer_guard",
        "hidden_dev",
    }
)
SEALED_SPLITS = frozenset({"hidden_audit", "hidden_sealed", "transfer_sealed"})


@dataclass(frozen=True)
class Task:
    task_id: str
    cluster_id: str
    source_id: str
    skill: str
    problem: str
    answer: str
    answer_type: AnswerType
    low_entropy: bool
    split: Split
    source_ver: str = "legacy-unversioned"
    source_item_id: str = "legacy"
    lineage_keys: tuple[str, ...] = ()
    source_role: Literal["train", "transfer", "benchmark", "hidden"] = "train"
    program_id: str | None = None
    program_digest: str | None = None
    property_report_digest: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported Task schema_version")
        for name in (
            "task_id",
            "cluster_id",
            "source_id",
            "source_ver",
            "source_item_id",
            "skill",
            "problem",
            "answer",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"Task.{name} must be a nonempty string")
        if self.answer_type not in {
            "int",
            "rational",
            "decimal",
            "expr",
            "interval",
            "tuple",
        }:
            raise ValueError("invalid answer type")
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}", getattr(self, name))
            for name in ("task_id", "cluster_id")
        ):
            raise ValueError("task and cluster identities must be opaque values")
        if self.measurable:
            if len(self.answer) > 200:
                raise ValueError(
                    "numeric reference exceeds the exact scorer's supported size"
                )
            try:
                numeric = Fraction(self.answer)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(
                    "numeric answer type requires an exact numeric label"
                ) from exc
            if self.answer_type == "int" and numeric.denominator != 1:
                raise ValueError("integer answer type requires an integer")
            denominator = numeric.denominator
            for factor in (2, 5):
                while denominator % factor == 0:
                    denominator //= factor
            if self.answer_type == "decimal" and denominator != 1:
                raise ValueError("decimal answer type requires a terminating decimal")
        if self.split not in DEVELOPMENT_SPLITS | SEALED_SPLITS:
            raise ValueError("invalid task split")
        if (
            type(self.low_entropy) is not bool
            or type(self.lineage_keys) is not tuple
            or any(type(key) is not str for key in self.lineage_keys)
        ):
            raise ValueError("invalid task flags or lineage")
        if self.source_role not in {"train", "transfer", "benchmark", "hidden"}:
            raise ValueError("invalid source role")
        if self.split in MEASUREMENT_SPLITS and not self.measurable:
            raise ValueError("measurement tasks require exact-numeric answer types")
        if self.source_role == "hidden" and (
            not self.program_id
            or self.cluster_id != self.program_id
            or not self.program_digest
            or not self.property_report_digest
        ):
            raise ValueError(
                "hidden instances require program identity and validation provenance"
            )
        if self.source_role != "hidden" and any(
            value is not None
            for value in (
                self.program_id,
                self.program_digest,
                self.property_report_digest,
            )
        ):
            raise ValueError(
                "hidden program provenance cannot masquerade as curated training data"
            )

    @property
    def measurable(self) -> bool:
        return self.answer_type in MEASURABLE_TYPES

    def to_dict(self) -> dict:
        return {**asdict(self), "measurable": self.measurable}

    def to_bytes(self) -> bytes:
        # Problem/answer text is opaque evidence, not a filesystem locator.
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> Task:
        value = dict(value)
        measurable = value.pop("measurable", None)
        if isinstance(value.get("lineage_keys"), list):
            value["lineage_keys"] = tuple(value["lineage_keys"])
        task = cls(**value)
        if measurable is not None and (
            type(measurable) is not bool or measurable != task.measurable
        ):
            raise ValueError("stored measurable flag disagrees with answer type")
        return task

    @classmethod
    def from_bytes(cls, blob: bytes) -> Task:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate task JSON key")
                result[key] = value
            return result

        return cls.from_dict(json.loads(blob, object_pairs_hook=pairs))
