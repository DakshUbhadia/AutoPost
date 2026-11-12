"""Isolated one-request math-verify worker. Never imported by measurement."""

import json
import logging
import re
import sys


def candidates(text):
    """Accept explicit final markers, or one bounded final math line."""
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth, end = match.end(), 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError("unbalanced answer marker")
        values.append(text[start : end - 1])
    values.extend(
        match.group(1)
        for match in re.finditer(
            r"^[ \t]*(?:\*\*)?(?:final[ \t]+answer|answer)(?:\*\*)?[ \t]*[:：][ \t]*([^\n]*)$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    values.extend(
        match.group(1)
        for match in re.finditer(
            r"final answer is[ \t]*[:：]?[ \t]*([^\n]*)", text, re.IGNORECASE
        )
    )
    if not values:
        if "\\boxed" in text:
            raise ValueError("malformed answer marker")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        values = lines[-1:]
    if len(values) > 8:
        raise ValueError("too many answer markers")
    return values


def safe_math(value):
    value = value.strip().strip("$").strip().rstrip(".").strip()
    # Accept the simple sqrt(...) spelling used by normalized source labels
    # without passing it to a Python/SymPy expression evaluator.
    for _ in range(8):
        converted = re.sub(r"\bsqrt\(([^()]*)\)", r"\\sqrt{\1}", value)
        if converted == value:
            break
        value = converted
    if (
        not value
        or len(value) > 512
        or "__" in value
        or re.search(r"(?<![A-Za-z\\])[A-Za-z]{4,}", value)
        or re.search(
            r"[\"'`;\x00-\x1f]|\b(?:import|eval|exec|open|lambda|compile|globals|getattr)\b",
            value,
        )
        or re.search(r"[A-Za-z_]\.[A-Za-z_]", value)
    ):
        raise ValueError("unsupported mathematical answer")
    return value


def check(request):
    # Unix memory/CPU limits supplement the parent's cross-platform deadline.
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from math_verify import LatexExtractionConfig, parse, verify

    logging.disable(logging.CRITICAL)

    def parsed(value):
        # No Python expression parser or string-equality fallback. The outer
        # process deadline owns timeouts, avoiding nested Windows processes.
        result = parse(
            "$" + safe_math(value) + "$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=None,
        )
        if not result or isinstance(result[0], str):
            raise ValueError("unparseable answer")
        return result

    values = candidates(request["text"])
    if not values:
        return {"correct": False, "status": "no_answer", "extracted": None}
    if any(
        re.search(
            r"\b(?:or|either|maybe|possibly|alternatively)\b", value, re.IGNORECASE
        )
        for value in values
    ):
        return {"correct": False, "status": "multiple_answers", "extracted": None}
    gold = parsed(request["reference"])
    predictions = [parsed(value) for value in values]
    if any(
        not verify(predictions[0], pred, timeout_seconds=None)
        for pred in predictions[1:]
    ):
        return {"correct": False, "status": "multiple_answers", "extracted": None}
    correct = bool(verify(gold, predictions[0], timeout_seconds=None))
    return {
        "correct": correct,
        "status": "ok" if correct else "wrong",
        "extracted": str(predictions[0][0]),
    }


if __name__ == "__main__":
    try:
        result = check(json.loads(sys.stdin.buffer.read(240_000)))
    except Exception:  # noqa: BLE001 -- untrusted parser failures are zero-reward results
        result = {"correct": False, "status": "parse_error", "extracted": None}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
