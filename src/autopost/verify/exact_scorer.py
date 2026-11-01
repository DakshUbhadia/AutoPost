"""autopost/verify/exact_scorer.py

The measurement-channel scorer (EvalScorer), Revision 2.

Scope: measurement sets contain only problems whose reference answer is an
integer or a rational number. That keeps this scorer small enough to test
exhaustively, keeps it independent of the training verifier (it must not
import math_verify, sympy, or any train-verifier module; enforced by
tests/test_independence.py), and avoids running any parser that can execute
code. Training data may still contain symbolic answers; they are checked by
the train verifier only.

Two operations:
  score_content(completion)  format-agnostic correctness (the primary metric)
  format_compliant(completion, marker)   did the output use the instructed
                                         answer marker exactly once
                                         (a guardrail, not a capability metric)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

CHECKER_ID = "exact_numeric@2"
MAX_CHARS = 200_000                 # refuse pathological inputs
Marker = Literal["boxed", "answer_line", "final_answer_phrase"]


@dataclass(frozen=True)
class NumericVerdict:
    correct: bool
    status: Literal["ok", "wrong", "no_answer", "multiple_answers",
                    "parse_error", "truncated"]
    extracted: str | None
    checker_id: str = CHECKER_ID


# ---------------------------------------------------------------- parsing
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_PLAIN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
_SLASH = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_FRAC = re.compile(r"^([+-]?)\\[dt]?frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}$")
_FRAC_SHORT = re.compile(r"^([+-]?)\\[dt]?frac\s*(\d)\s*(\d)$")
_LEADING_VAR = re.compile(r"^[A-Za-z]\w*\s*=\s*")
_WRAPPERS = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", "$"]
_TEXT_CMD = re.compile(r"\\(text|mathrm|textbf|mbox)\s*\{[^{}]*\}")


def parse_number(raw: str) -> Fraction | None:
    """Parse one answer string to an exact Fraction, or None."""
    s = raw.strip()
    if not s or len(s) > 200:
        return None
    for w in _WRAPPERS:
        s = s.replace(w, "")
    s = _TEXT_CMD.sub("", s)                 # units such as \text{ apples}
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.strip().rstrip(".").strip()
    s = _LEADING_VAR.sub("", s)               # "x = 5" -> "5"
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1].strip()
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")" and "," not in s:
        s = s[1:-1].strip()
    try:
        if _THOUSANDS.match(s):
            return Fraction(s.replace(",", ""))
        if _PLAIN.match(s):
            return Fraction(s)
        m = _SLASH.match(s)
        if m:
            den = int(m.group(2))
            return None if den == 0 else Fraction(int(m.group(1)), den)
        m = _FRAC.match(s) or _FRAC_SHORT.match(s)
        if m:
            sign, num, den = m.group(1), int(m.group(2)), int(m.group(3))
            if den == 0:
                return None
            v = Fraction(num, den)
            return -v if sign == "-" else v
    except (ValueError, ZeroDivisionError):
        return None
    return None


# ------------------------------------------------------------- extraction
def _boxed_contents(text: str) -> list[str]:
    out, i = [], 0
    while True:
        j = text.find("\\boxed", i)
        if j < 0:
            return out
        k = text.find("{", j)
        if k < 0 or text[j + 6:k].strip():
            i = j + 6
            continue
        depth, m = 0, k
        while m < len(text):
            if text[m] == "{":
                depth += 1
            elif text[m] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[k + 1:m])
                    break
            m += 1
        i = m + 1


_ANSWER_LINE = re.compile(r"^\s*(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE)
_FINAL_PHRASE = re.compile(r"final answer is\s*[:：]?\s*([^\n]+?)\s*(?:\.\s*)?$",
                           re.IGNORECASE | re.MULTILINE)
_NUMBER_TOKEN = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?|[+-]?\.\d+")
_HEDGE = re.compile(r"\b(or|either|possibly|maybe|alternatively)\b", re.IGNORECASE)


def _marker_candidates(text: str) -> dict[str, list[str]]:
    return {
        "boxed": _boxed_contents(text),
        "answer_line": [m.group(1).strip("*").strip() for m in _ANSWER_LINE.finditer(text)],
        "final_answer_phrase": [m.group(1) for m in _FINAL_PHRASE.finditer(text)],
    }


def extract_final(text: str) -> tuple[str, list[Fraction], str | None]:
    """Return (status, distinct parsed values, raw string used).

    Marker candidates from every recognised marker are pooled. More than one
    distinct value among them -> multiple_answers. If no marker exists, fall
    back to the last number on the last non-empty line, unless that line
    hedges between values.
    """
    cands = _marker_candidates(text)
    raw = [c for lst in cands.values() for c in lst]
    if raw:
        vals = [v for v in (parse_number(r) for r in raw) if v is not None]
        distinct = sorted(set(vals))
        if not distinct:
            return "parse_error", [], raw[-1]
        if len(distinct) > 1:
            return "multiple_answers", distinct, raw[-1]
        return "ok", distinct, raw[-1]
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "no_answer", [], None
    last = lines[-1]
    toks = _NUMBER_TOKEN.findall(last)
    if not toks:
        return "no_answer", [], None
    vals = {v for v in (parse_number(t) for t in toks) if v is not None}
    if len(vals) > 1 and _HEDGE.search(last):
        return "multiple_answers", sorted(vals), last
    v = parse_number(toks[-1])
    return ("ok", [v], toks[-1]) if v is not None else ("parse_error", [], toks[-1])


# ------------------------------------------------------------------ API
def score_content(text: str, reference: str, truncated: bool = False) -> NumericVerdict:
    """Format-agnostic correctness against an integer/rational reference."""
    ref = parse_number(reference)
    if ref is None:
        raise ValueError(f"reference is not integer/rational: {reference!r}")
    if truncated:
        return NumericVerdict(False, "truncated", None)
    if len(text) > MAX_CHARS:
        return NumericVerdict(False, "parse_error", None)
    status, vals, raw = extract_final(text)
    if status != "ok":
        return NumericVerdict(False, status, raw)
    ok = vals[0] == ref
    return NumericVerdict(ok, "ok" if ok else "wrong", raw)


def format_compliant(text: str, marker: Marker) -> bool:
    """Exactly one occurrence of the instructed marker, and no other marker
    carrying a different value. Used by the format-compliance guardrail."""
    cands = _marker_candidates(text)
    if len(cands[marker]) != 1:
        return False
    if marker == "answer_line":
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines or not _ANSWER_LINE.match(lines[-1]):
            return False
    status, _, _ = extract_final(text)
    return status == "ok"
