"""Pinned local JSONL snapshots; no implicit network access or label generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autopost.contracts.datasets import SourceSpec
from autopost.contracts.serialization import canonical_bytes, content_hash


@dataclass(frozen=True)
class SourceRow:
    source_item_id: str
    problem: str
    answer: str
    skill: str
    lineage_keys: tuple[str, ...] = ()
    multiple_choice: bool = False
    label_origin: str = "curated_reference"
    source_payload: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(
            type(getattr(self, name)) is not str
            for name in ("source_item_id", "problem", "answer", "skill")
        ):
            raise ValueError("source text fields must be strings")
        if (
            not self.source_item_id
            or not self.skill
            or type(self.multiple_choice) is not bool
        ):
            raise ValueError("invalid source identity, skill or flags")
        if self.label_origin != "curated_reference":
            raise ValueError("unverified/model-generated source labels are forbidden")
        if type(self.lineage_keys) is not tuple or any(
            type(key) is not str or not key for key in self.lineage_keys
        ):
            raise ValueError("invalid source lineage")
        canonical_bytes(self.source_payload)


def _boxed_reference(solution: str) -> str:
    start = solution.rfind("\\boxed{")
    if start < 0:
        return ""
    start += len("\\boxed{")
    depth = 1
    for index in range(start, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
        if depth == 0:
            return solution[start:index]
    return ""


def _adapt(spec: SourceSpec, value: dict) -> SourceRow:
    if spec.adapter == "canonical":
        if isinstance(value.get("lineage_keys"), list):
            value["lineage_keys"] = tuple(value["lineage_keys"])
        return SourceRow(**value)
    item_id = str(value.get("id", content_hash(value)))
    if spec.adapter == "gsm8k":
        question, solution = value["question"], value["answer"]
        answer = solution.rsplit("####", 1)[-1].strip() if "####" in solution else ""
        return SourceRow(item_id, question, answer, "arithmetic", source_payload=value)
    solution = value.get("solution", "")
    answer = value.get("answer", _boxed_reference(solution))
    return SourceRow(
        item_id,
        value["problem"],
        answer,
        value.get("type", "mathematics"),
        source_payload=value,
    )


@dataclass(frozen=True)
class SourceSnapshot:
    spec: SourceSpec
    rows: tuple[SourceRow, ...]

    def to_dict(self):
        return {
            "source": self.spec.model_dump(mode="json"),
            "rows": sorted((asdict(row) for row in self.rows), key=content_hash),
        }

    @property
    def digest(self):
        return content_hash(self.to_dict())


def read_jsonl(path: str | Path) -> SourceSnapshot:
    """First nonempty line: {source: SourceSpec}; remaining lines: SourceRow."""
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("empty source snapshot")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or set(header) != {"source"}:
        raise ValueError("source JSONL must begin with a pinned source header")
    spec = SourceSpec.model_validate_json(json.dumps(header["source"]))
    rows = []
    for index, line in enumerate(lines[1:], start=2):
        try:
            value = json.loads(line)
            rows.append(_adapt(spec, value))
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            # Invalid envelopes cannot be silently omitted from protected data.
            raise ValueError(f"invalid source row {index}") from exc
    return SourceSnapshot(spec, tuple(rows))
