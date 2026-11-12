from fractions import Fraction

import pytest

from autopost.contracts.serialization import content_hash
from autopost.data.hidden import (
    HiddenInstance,
    HiddenProgram,
    HiddenTestReport,
    HumanReview,
    Parameter,
    evaluate,
    generate,
    partition_programs,
    verify_review,
)
from autopost.data.hidden import test_program as validate_program
from autopost.store.errors import ConflictError, StoreError


def review_for(program, report, decision="approved"):
    return HumanReview(
        program_digest=program.digest,
        report_digest=content_hash(report),
        reviewer="synthetic-test-reviewer",
        decision=decision,
        sampled_instance_ids=tuple(item.instance_id for item in report.sample),
        notes="Synthetic review record used only to test the human-review workflow.",
    )


def test_every_program_passes_many_generated_instances_and_known_math(
    development_result,
):
    programs = development_result[3]
    for seed in (0, 1, 17):
        for program in programs:
            report = validate_program(program, seed=seed, trials=256)
            assert report.passed and len(report.sample) == 8
            assert HiddenTestReport.from_bytes(report.to_bytes()) == report
            instances, _ = generate(program, seed=seed, count=100)
            for instance in instances:
                values = instance.values
                expected = (
                    values["count"] * values["price"]
                    if program.program_id == "hidden-price"
                    else 2 * (values["width"] + values["length"])
                    if program.program_id == "hidden-perimeter"
                    else Fraction(values["numerator"], values["denominator"])
                )
                assert Fraction(instance.answer) == expected
                assert HiddenInstance.from_bytes(instance.to_bytes()) == instance


def test_generation_and_program_partitioning_are_deterministic(development_result):
    programs = development_result[3]
    assert partition_programs(programs, seed=5) == partition_programs(
        tuple(reversed(programs)), seed=5
    )
    partitions = partition_programs(programs, seed=5)
    assert not set(partitions[0]) & set(partitions[1])
    assert set().union(*map(set, partitions)) == {
        program.program_id for program in programs
    }
    for program in programs:
        assert generate(program, seed=0, count=20) == generate(
            program, seed=0, count=20
        )
        assert generate(program, seed=0, count=20) != generate(
            program, seed=1, count=20
        )
        assert HiddenProgram.from_bytes(program.to_bytes()) == program
        for parameter in program.parameters:
            assert Parameter.from_bytes(parameter.to_bytes()) == parameter


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo bad')",
        "open('file')",
        "x.__class__",
        "[x for x in range(3)]",
        "(lambda: 1)()",
        "values[0]",
    ],
)
def test_model_drafted_code_never_executes(expression, development_result):
    program = development_result[3][0]
    with pytest.raises(ValueError):
        HiddenProgram.from_bytes(
            program.model_copy(update={"answer_expression": expression}).to_bytes()
        )
    with pytest.raises(ValueError):
        evaluate(expression, {"x": 1})


def test_rejection_regeneration_limits_and_answer_leakage_are_checked(
    development_result,
):
    program = development_result[3][0]
    impossible = program.model_copy(update={"invariants": ("count < 0",)})
    report = validate_program(impossible, seed=0, trials=100)
    assert not report.passed and report.rejections["invariant_failed"] == 2000
    with pytest.raises(ValueError):
        evaluate("2 ** 1000", {})
    for template in (
        "The answer is {answer}",
        "Question {count.real}",
        "Question {count[0]}",
    ):
        with pytest.raises(ValueError):
            program.model_copy(update={"template": template}).to_bytes()
    with pytest.raises(ValueError):
        Parameter(name="x", minimum=5, maximum=4)


def test_pending_hidden_data_remains_inspectable_but_cannot_be_evaluated(catalog):
    ref = catalog.refs["hidden_dev"]
    tasks = catalog.repo.inspect(ref)
    assert len(tasks) == 24 and len({task.cluster_id for task in tasks}) == 3
    assert all(task.cluster_id == task.program_id for task in tasks)
    with pytest.raises(StoreError, match="quarantined"):
        catalog.repo.evaluation_tasks(ref, fixture_only=True)


def test_reviews_bind_exact_program_property_report_and_generated_sample(catalog):
    ref = catalog.refs["hidden_dev"]
    packets = catalog.repo.report(ref)["hidden_programs"]
    for packet in packets:
        import json

        program = HiddenProgram.model_validate_json(json.dumps(packet["program"]))
        report = HiddenTestReport.model_validate_json(json.dumps(packet["report"]))
        review = review_for(program, report)
        assert HumanReview.from_bytes(review.to_bytes()) == review
        catalog.repo.record_review(program, report, review)
        catalog.repo.record_review(program, report, review)
        with pytest.raises(ConflictError):
            catalog.repo.record_review(
                program, report, review.model_copy(update={"notes": "overwrite"})
            )
        with pytest.raises(ValueError):
            verify_review(
                program, report, review.model_copy(update={"sampled_instance_ids": ()})
            )
        with pytest.raises(ValueError):
            verify_review(
                program.model_copy(update={"template": program.template + " Explain."}),
                report,
                review,
            )
    assert len(catalog.repo.evaluation_tasks(ref, fixture_only=True)) == 24
    storage = catalog.storage
    handle = storage.metadata.reserve_job(
        storage.evaluation(dataset=ref, split="hidden_dev")
    )
    assert handle.job_id
    with pytest.raises(StoreError):
        catalog.repo.training_tasks(ref, fixture_only=True)


def test_rejected_or_forged_property_reports_never_unlock_hidden_evaluation(
    catalog, development_result
):
    program = development_result[3][0]
    report = validate_program(program, seed=0, trials=256)
    forged = report.model_copy(update={"instance_digest": "0" * 64})
    with pytest.raises(StoreError, match="reconstructed"):
        catalog.repo.record_review(program, forged, review_for(program, forged))
    catalog.repo.record_review(program, report, review_for(program, report, "rejected"))
    with pytest.raises(StoreError, match="quarantined"):
        catalog.repo.evaluation_tasks(catalog.refs["hidden_dev"], fixture_only=True)
