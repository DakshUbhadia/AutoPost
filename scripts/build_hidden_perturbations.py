"""Generate inspectable hidden_dev preview/review packets only; never research assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autopost.data.fixtures import development_inputs
from autopost.data.hidden import HiddenProgram, generate, test_program


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/hidden_dev/previews.json")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--instances", type=int, default=8)
    args = parser.parse_args(argv)
    programs = (
        tuple(
            HiddenProgram.model_validate_json(json.dumps(value))
            for value in json.loads(args.programs.read_text(encoding="utf-8"))
        )
        if args.programs
        else development_inputs(args.seed)[3]
    )
    packets = []
    for program in programs:
        report = test_program(program, seed=args.seed, trials=args.trials)
        if not report.passed:
            raise ValueError("program failed its property campaign")
        instances, rejections = generate(program, seed=args.seed, count=args.instances)
        packets.append(
            {
                "partition": "hidden_dev",
                "review_required": True,
                "program": program.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "instances": [item.model_dump(mode="json") for item in instances],
                "rejections": rejections,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packets, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{len(packets)} hidden_dev programs; generated instances require human review"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
