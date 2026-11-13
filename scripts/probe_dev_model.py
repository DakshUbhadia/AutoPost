"""Run preflight by default; --run-gpu explicitly enables the GPU campaign."""

import sys

from autopost.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["execution", "spike", *sys.argv[1:]]))
