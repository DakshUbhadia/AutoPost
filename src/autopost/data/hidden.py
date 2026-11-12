"""Declarative hidden arithmetic, execution labels and explicit review packets.

No supplied Python, imports, calls, attributes, loops or filesystem operations
are executed. All examples are disposable development data.
"""

from __future__ import annotations

import ast
import random
import re
import string
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autopost.contracts.base import DurableModel
from autopost.contracts.refs import Digest
from autopost.contracts.serialization import content_hash


def evaluate(expression: str, variables: dict[str, Fraction | int]):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid hidden arithmetic expression") from exc
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("hidden expression exceeds complexity limit")

    def visit(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and abs(node.value) <= 1000000
        ):
            return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return Fraction(variables[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (
                -visit(node.operand)
                if isinstance(node.op, ast.USub)
                else visit(node.operand)
            )
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.FloorDiv):
                value = Fraction(left // right)
            elif isinstance(node.op, ast.Mod):
                value = left % right
            elif (
                isinstance(node.op, ast.Pow)
                and right.denominator == 1
                and 0 <= right <= 6
            ):
                value = left ** int(right)
            else:
                raise ValueError("forbidden arithmetic operator")
            if value.numerator.bit_length() + value.denominator.bit_length() > 4096:
                raise ValueError("hidden arithmetic exceeds size limit")
            return value
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for operator, item in zip(node.ops, node.comparators, strict=True):
                right = visit(item)
                if isinstance(operator, ast.Eq):
                    valid = left == right
                elif isinstance(operator, ast.NotEq):
                    valid = left != right
                elif isinstance(operator, ast.Lt):
                    valid = left < right
                elif isinstance(operator, ast.LtE):
                    valid = left <= right
                elif isinstance(operator, ast.Gt):
                    valid = left > right
                elif isinstance(operator, ast.GtE):
                    valid = left >= right
                else:
                    raise ValueError("forbidden comparison")  # noqa: TRY004
                if not valid:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [visit(value) for value in node.values]
            if any(type(value) is not bool for value in values):
                raise ValueError("boolean invariants require comparisons")
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError("forbidden hidden-program syntax")

    return visit(tree.body)


class Parameter(DurableModel):
    name: str
    minimum: Annotated[int, Field(ge=-1000000, le=1000000)]
    maximum: Annotated[int, Field(ge=-1000000, le=1000000)]

    @model_validator(mode="after")
    def bounds(self):
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", self.name)
            or self.name == "answer"
            or self.minimum > self.maximum
        ):
            raise ValueError("invalid hidden parameter")
        return self


class HiddenProgram(DurableModel):
    program_id: str
    origin: Literal["model_drafted", "human_authored"] = "model_drafted"
    template: str
    skill: str
    parent_lineage_keys: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    answer_expression: str
    invariants: tuple[str, ...]

    @property
    def digest(self):
        return content_hash(self)

    @model_validator(mode="after")
    def grammar(self):
        names = {parameter.name for parameter in self.parameters}
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", self.program_id):
            raise ValueError("invalid program identity")
        if (
            len(names) != len(self.parameters)
            or not 1 <= len(names) <= 12
            or not self.parent_lineage_keys
            or not self.invariants
        ):
            raise ValueError(
                "programs need unique parameters, held-out provenance and invariants"
            )
        for _, field, spec, conversion in string.Formatter().parse(self.template):
            if field is not None and (field not in names or spec or conversion):
                raise ValueError(
                    "template fields must be plain parameter names; labels cannot enter prompts"
                )
        # Validate every syntax node even if a short-circuiting comparison
        # would otherwise hide an unsafe branch.
        allowed = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.BoolOp,
            ast.And,
            ast.Or,
        )
        for expression in (self.answer_expression, *self.invariants):
            if len(expression) > 1000:
                raise ValueError("oversized hidden expression")
            try:
                nodes = list(ast.walk(ast.parse(expression, mode="eval")))
            except SyntaxError as exc:
                raise ValueError("invalid hidden arithmetic expression") from exc
            if len(nodes) > 64 or any(not isinstance(node, allowed) for node in nodes):
                raise ValueError("forbidden hidden-program syntax")
            if any(
                isinstance(node, ast.Name) and node.id not in names | {"answer"}
                for node in nodes
            ):
                raise ValueError("unknown hidden variable")
        if any(
            isinstance(node, ast.Name) and node.id == "answer"
            for node in ast.walk(ast.parse(self.answer_expression, mode="eval"))
        ):
            raise ValueError("answer expression cannot depend on its own label")
        return self


class HiddenInstance(DurableModel):
    instance_id: str
    program_id: str
    values: dict[str, int]
    problem: str
    answer: str


class HiddenTestReport(DurableModel):
    program_id: str
    program_digest: Digest
    seed: int
    trials: int
    passed: bool
    instance_digest: Digest
    rejections: dict[str, int]
    sample: tuple[HiddenInstance, ...]


class HumanReview(DurableModel):
    program_digest: Digest
    report_digest: Digest
    reviewer: Annotated[str, Field(min_length=1)]
    decision: Literal["approved", "rejected"]
    sampled_instance_ids: tuple[str, ...]
    notes: Annotated[str, Field(min_length=1)]


def generate(program: HiddenProgram, *, seed: int, count: int):
    program = HiddenProgram.from_bytes(program.to_bytes())
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("invalid hidden sample count")
    rng = random.Random(content_hash({"program": program.digest, "seed": seed}))
    accepted, rejected = [], {}
    for _ in range(count * 20):
        values = {
            parameter.name: rng.randint(parameter.minimum, parameter.maximum)
            for parameter in program.parameters
        }
        try:
            answer = evaluate(program.answer_expression, values)
            if type(answer) is not Fraction:
                raise ValueError("answer must be one exact numeric value")
            checked = {**values, "answer": answer}
            if not all(
                evaluate(invariant, checked) is True for invariant in program.invariants
            ):
                raise ValueError("invariant_failed")
            text = program.template.format_map(values)
            if not text.strip():
                raise ValueError("empty_problem")
        except (ValueError, ArithmeticError) as exc:
            reason = (
                str(exc)
                if str(exc) in {"invariant_failed", "empty_problem"}
                else "invalid_arithmetic"
            )
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        identity = "hidden-" + content_hash(
            {
                "program": program.digest,
                "seed": seed,
                "index": len(accepted),
                "values": values,
            }
        )
        accepted.append(
            HiddenInstance(
                instance_id=identity,
                program_id=program.program_id,
                values=values,
                problem=text,
                answer=str(answer),
            )
        )
        if len(accepted) == count:
            break
    return tuple(accepted), rejected


def test_program(
    program: HiddenProgram, *, seed: int, trials: int = 256
) -> HiddenTestReport:
    if not 100 <= trials <= 10000:
        raise ValueError("hidden property tests require 100–10000 instances")
    instances, rejections = generate(program, seed=seed, count=trials)
    sampled = tuple(
        sorted(
            instances,
            key=lambda item: content_hash(
                {"review_seed": seed, "instance": item.instance_id}
            ),
        )[:8]
    )
    return HiddenTestReport(
        program_id=program.program_id,
        program_digest=program.digest,
        seed=seed,
        trials=trials,
        passed=len(instances) == trials,
        instance_digest=content_hash(
            [instance.model_dump(mode="json") for instance in instances]
        ),
        rejections=rejections,
        sample=sampled,
    )


def verify_review(
    program: HiddenProgram, report: HiddenTestReport, review: HumanReview
):
    if (
        not report.passed
        or report.program_digest != program.digest
        or review.program_digest != program.digest
        or review.report_digest != content_hash(report)
        or review.sampled_instance_ids
        != tuple(item.instance_id for item in report.sample)
    ):
        raise ValueError(
            "review must bind the tested program, report and exact generated-instance sample"
        )


def partition_programs(
    programs: tuple[HiddenProgram, ...], *, seed: int, groups: int = 2
):
    if groups < 1:
        raise ValueError("invalid program partition count")
    partitions = [[] for _ in range(groups)]
    for program in sorted(programs, key=lambda item: item.program_id):
        index = (
            int(content_hash({"seed": seed, "program": program.digest}), 16) % groups
        )
        partitions[index].append(program.program_id)
    return tuple(tuple(partition) for partition in partitions)
