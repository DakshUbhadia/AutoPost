"""Independent measurement properties, malformed markers and exploit cases."""

from fractions import Fraction
from random import Random

import pytest

from autopost.verify.exact_scorer import MAX_CHARS, format_compliant, score_content


@pytest.mark.parametrize(
    "text,status",
    [
        (r"\boxed{2}" + "\nAnswer: invalid", "parse_error"),
        (r"\boxed{2}\boxed{", "parse_error"),
        (r"\boxed{2}\boxed", "parse_error"),
        (r"\boxed{2}\boxed{1/0}", "parse_error"),
        ("Answer: __import__('os').system('echo 2')", "parse_error"),
        ("Answer: 2 or 3", "parse_error"),
        (r"\boxed{2}\boxed{3}", "multiple_answers"),
        ("", "no_answer"),
        ("x" * (MAX_CHARS + 1), "parse_error"),
    ],
    ids=[
        "invalid-extra",
        "unclosed",
        "missing-brace",
        "division-zero",
        "code",
        "hedge",
        "conflict",
        "empty",
        "oversize",
    ],
)
def test_exploits_fail_closed(text, status):
    verdict = score_content(text, "2")
    assert verdict.status == status
    assert not verdict.correct


def test_rational_properties_and_marker_guardrails():
    rng = Random(400)
    for _ in range(1000):
        p, q = rng.randint(-100_000, 100_000), rng.randint(1, 1000)
        reference = str(Fraction(p, q))
        for text in (f"Answer: {p}/{q}", rf"\boxed{{\frac{{{p}}}{{{q}}}}}"):
            assert score_content(text, reference).correct
            assert not score_content(text, reference, truncated=True).correct
        assert not score_content(f"Answer: {(p + q)}/{q}", reference).correct
    assert format_compliant("Answer: 2", "answer_line")
    assert not format_compliant("Answer: 2\nmore work", "answer_line")
    assert not format_compliant(r"\boxed{2}\boxed{2}", "boxed")
    assert not format_compliant(r"\boxed{2}\boxed{", "boxed")
