"""Command line interface for repository setup and preregistration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from autopost.config import load_config, unresolved_fields
from autopost.prereg import freeze, make_draft, validate, verify, warnings
from autopost.store.db import init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autopost")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config").add_subparsers(dest="action", required=True)
    config.add_parser("placeholders")
    db = commands.add_parser("db").add_subparsers(dest="action", required=True)
    init = db.add_parser("init"); init.add_argument("--path", default="runs/autopost.sqlite")
    prereg = commands.add_parser("prereg").add_subparsers(dest="action", required=True)
    draft = prereg.add_parser("init"); draft.add_argument("--version", default="stageB-p1-v1")
    draft.add_argument("--phase", default="phase1"); draft.add_argument("--base-yaml", default="configs/base.yaml")
    draft.add_argument("--lock", default="requirements.lock")
    valid = prereg.add_parser("validate"); valid.add_argument("path"); valid.add_argument("--base-yaml", default="configs/base.yaml")
    fr = prereg.add_parser("freeze"); fr.add_argument("path"); fr.add_argument("--db", default="runs/autopost.sqlite")
    fr.add_argument("--base-yaml", default="configs/base.yaml"); fr.add_argument("--output-dir", default="configs/prereg")
    ver = prereg.add_parser("verify"); ver.add_argument("--db", default="runs/autopost.sqlite")
    ver.add_argument("--prereg-dir", default="configs/prereg")
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            for path, kind in unresolved_fields(load_config()):
                print(f"{path}\t{kind}")
        elif args.command == "db":
            init_db(args.path)
        elif args.command == "prereg" and args.action == "init":
            if args.phase != "phase1":
                parser.error("Phase II pre-registration is built in a later step")
            path = Path("configs/prereg") / f"{args.version}.draft.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(__import__("autopost.prereg", fromlist=["canonical_bytes"]).canonical_bytes(
                make_draft(args.version, Path(args.base_yaml), Path(args.lock))))
            print(path)
        elif args.command == "prereg" and args.action == "validate":
            doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
            problems = validate(doc, Path(args.base_yaml))
            for warning in warnings(doc): print(f"WARNING: {warning}")
            for problem in problems: print(f"ERROR: {problem}")
            if problems: return 1
            print("valid")
        elif args.command == "prereg" and args.action == "freeze":
            doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
            output = Path(args.output_dir) / f"{doc['meta']['version']}.json"
            print(f"sha256 {freeze(doc, Path(args.base_yaml), output, Path(args.db))}")
            print("Create the prereg-stageB-p1-v1 git tag after review.")
        elif args.command == "prereg" and args.action == "verify":
            results = verify(Path(args.db), Path(args.prereg_dir))
            for version, ok in results: print(f"{'OK' if ok else 'MISMATCH'} {version}")
            if not results: print("no pre-registrations")
            return 0 if all(ok for _, ok in results) else 1
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
