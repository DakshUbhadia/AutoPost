"""Conservative answer typing. This module never executes answer expressions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

from autopost.contracts.serialization import content_hash
from autopost.contracts.tasks import MEASURABLE_TYPES

from .sources import SourceRow, SourceSnapshot

NORMALIZER_VERSION = "normalize-1"


def normalize_text(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text)
        .replace("\u200b", "")
        .replace("−", "-")
        .split()
    )


def problem_key(text: str) -> str:
    return content_hash(normalize_text(text).casefold())


def type_answer(raw: str) -> tuple[str, str]:
    answer = normalize_text(raw).replace("⁄", "/").strip("$").strip()
    if not answer or len(answer) > 2000:
        raise ValueError("empty_or_oversized_answer")
    match = re.fullmatch(r"\\(?:d|t)?frac\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}", answer)
    numeric = f"{match[1]}/{match[2]}" if match else answer
    kind = (
        "rational"
        if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", numeric)
        else (
            "int"
            if re.fullmatch(r"[+-]?\d+", numeric)
            else "decimal"
            if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", numeric)
            else None
        )
    )
    if kind:
        if len(numeric) > 200:
            raise ValueError("oversized_numeric_answer")
        try:
            value = Fraction(numeric.replace(" ", ""))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid_numeric_answer") from exc
        canonical = (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        )
        if len(canonical) > 200:
            raise ValueError("oversized_numeric_answer")
        return canonical, kind
    if answer.casefold() in {"yes", "no", "true", "false"}:
        raise ValueError("yes_no_answer")
    if re.fullmatch(r"[+-]?(?:nan|inf|infinity)", answer, re.IGNORECASE):
        raise ValueError("nonfinite_answer")
    if answer.startswith(("[", "(")) and "," in answer:
        return answer, "interval" if answer.startswith("[") else "tuple"
    return answer, "expr"


@dataclass(frozen=True)
class NormalizedRow:
    prompt_id: str
    source: SourceSnapshot
    item: SourceRow
    problem: str
    answer: str
    answer_type: str
    low_entropy: bool
    lineage: tuple[str, ...]

    @property
    def measurable(self):
        return self.answer_type in MEASURABLE_TYPES


def normalize(snapshot: SourceSnapshot, row: SourceRow) -> NormalizedRow:
    problem = normalize_text(row.problem)
    if not problem or len(problem) > 50000:
        raise ValueError("empty_or_oversized_problem")
    if row.multiple_choice or len(re.findall(r"(?:^|\s)[A-D][).]\s", problem)) >= 2:
        raise ValueError("multiple_choice")
    answer, kind = type_answer(row.answer)
    origin = f"{snapshot.spec.source_id}@{snapshot.spec.revision}:{row.source_item_id}"
    lineage = tuple(sorted({origin, *row.lineage_keys}))
    prompt_id = "prompt-" + content_hash(
        {
            "normalizer": NORMALIZER_VERSION,
            "source": origin,
            "problem": problem,
            "answer": answer,
            "skill": row.skill,
        }
    )
    low = (
        kind in MEASURABLE_TYPES
        and Fraction(answer).denominator == 1
        and abs(Fraction(answer)) <= 10
    )
    return NormalizedRow(
        prompt_id, snapshot, row, problem, answer, kind, bool(low), lineage
    )
