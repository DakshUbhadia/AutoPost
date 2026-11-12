"""Checker agreement is monitoring evidence; it cannot produce rewards."""


def checker_disagreement(training_verdicts, evaluation_verdicts):
    training, evaluation = tuple(training_verdicts), tuple(evaluation_verdicts)
    if len(training) != len(evaluation):
        raise ValueError("checker comparison requires paired samples")
    n = len(training)
    return {
        "n": n,
        "correctness_disagreements": sum(
            a.correct != b.correct for a, b in zip(training, evaluation)
        ),
        "status_disagreements": sum(
            a.status != b.status for a, b in zip(training, evaluation)
        ),
        "disagreement_rate": sum(
            a.correct != b.correct for a, b in zip(training, evaluation)
        )
        / n
        if n
        else 0.0,
        "training_checkers": sorted({v.checker_id for v in training}),
        "evaluation_checkers": sorted({v.checker_id for v in evaluation}),
    }
