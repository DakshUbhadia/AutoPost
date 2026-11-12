"""Golden training-verifier behavior through real isolated worker processes."""

from dataclasses import replace

import pytest

from autopost.contracts.tasks import Task
from autopost.interfaces import BinaryReward, Completion, Verdict
from autopost.verify.diagnostics import checker_disagreement
from autopost.verify.train import CHECKER_ID, MathVerifyPool


def task(answer="2", answer_type="int"):
    return Task(
        "t",
        "c",
        "s",
        "arithmetic",
        "Compute the answer",
        answer,
        answer_type,
        False,
        "train",
    )


def completion(text, *, truncated=False):
    return Completion("completion", "t", "base", "train", 0, 0, text, 10, truncated)


@pytest.mark.parametrize(
    "text,reference,kind,status",
    [
        (r"\boxed{2}", "2", "int", "ok"),
        ("Answer: 3", "2", "int", "wrong"),
        (r"Work done.\n\boxed{\frac{1}{2}}", "1/2", "rational", "ok"),
        (r"\boxed{\sqrt{2}}", r"\sqrt{2}", "expr", "ok"),
        (r"\boxed{\sqrt{2}}", "sqrt(2)", "expr", "ok"),
        (r"\boxed{(x+1)^2}", "x^2+2x+1", "expr", "ok"),
        ("2", "2", "int", "ok"),
        ("The final answer is 2.", "2", "int", "ok"),
        (r"\boxed{2}" + "\nAnswer: 2", "2", "int", "ok"),
        (r"\boxed{2}" + "\nAnswer: 3", "2", "int", "multiple_answers"),
        ("Answer: 2 or 3", "2", "int", "multiple_answers"),
        (r"\boxed{2", "2", "int", "parse_error"),
        (r"\boxed{}", "2", "int", "parse_error"),
        (r"\boxed{2}" + "\nAnswer: garbage", "2", "int", "parse_error"),
        ("Answer: __import__('os').system('whoami')", "2", "int", "parse_error"),
        ("", "2", "int", "no_answer"),
    ],
)
def test_golden(text, reference, kind, status):
    verifier = MathVerifyPool()
    verdict = verifier.check(task(reference, kind), completion(text))
    assert verdict.status == status
    assert verdict.correct == (status == "ok")
    assert verdict.checker_id == CHECKER_ID


def test_timeout_truncation_and_next_worker_is_healthy():
    assert MathVerifyPool(timeout_seconds=1.0).checker_id != CHECKER_ID
    assert (
        MathVerifyPool(timeout_seconds=0.001)
        .check(task(), completion("Answer: 2"))
        .status
        == "timeout"
    )
    verifier = MathVerifyPool()
    assert (
        verifier.check(task(), completion("Answer: 2", truncated=True)).status
        == "truncated"
    )
    assert verifier.check(task(), completion("x" * 200_001)).status == "parse_error"
    results = verifier.check_many(
        [(task(), completion("Answer: 2")), (task(), completion("Answer: 3"))]
    )
    assert [v.correct for v in results] == [True, False]
    with pytest.raises(ValueError, match="different task"):
        verifier.check(task(), replace(completion("2"), task_id="other"))


def test_binary_rewards_and_explicit_shaping():
    reward = BinaryReward(100)
    good = Verdict(True, "ok", "2", CHECKER_ID)
    assert reward(task(), completion("2"), good).value == 1
    assert reward(task(), completion("2", truncated=True), good).value == 0
    for status in ("wrong", "timeout", "parse_error", "multiple_answers", "no_answer"):
        assert (
            reward(
                task(), completion("2"), Verdict(False, status, None, CHECKER_ID)
            ).value
            == 0
        )
    shaped = BinaryReward(100, 20, 0.5)
    assert shaped(task(), replace(completion("2"), n_tokens=90), good).value == 0.75
    with pytest.raises(ValueError, match="measurement"):
        reward(task(), completion("2"), replace(good, checker_id="exact_numeric@3"))
    with pytest.raises(ValueError, match="inconsistent"):
        reward(task(), completion("2"), replace(good, status="wrong"))
    for args in (
        (0,),
        (100, -1),
        (100, 0, 0.5),
        (100, 20, -1),
        (100, 20, float("nan")),
    ):
        with pytest.raises(ValueError):
            BinaryReward(*args)


def test_disagreement_is_monitoring_only():
    from autopost.verify.exact_scorer import score_content

    train = MathVerifyPool().check(task(), completion("Answer: 1+1"))
    evaluation = score_content("Answer: 1+1", "2")
    metrics = checker_disagreement([train], [evaluation])
    assert metrics["disagreement_rate"] == 1
    assert BinaryReward(100)(task(), completion("Answer: 1+1"), train).value == 1
    with pytest.raises(ValueError, match="paired"):
        checker_disagreement([train], [])


def test_empty_extra_markers_fail_closed_in_both_channels():
    from autopost.verify.exact_scorer import score_content

    pool = MathVerifyPool()
    for suffix in ("Answer:", "The final answer is "):
        text = "\\boxed{2}\n" + suffix
        assert pool.check(task(), completion(text)).status == "parse_error"
        assert score_content(text, "2").status == "parse_error"
