import json
from pathlib import Path

from autopost.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_profile_selection_and_hashes(capsys):
    base = str(ROOT / "configs/base.yaml")
    lock = str(ROOT / "requirements.lock")
    assert main(["config", "hashes", "--base-yaml", base, "--lock", lock]) == 0
    dev = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "config",
                "hashes",
                "--base-yaml",
                base,
                "--lock",
                lock,
                "--execution",
                "execution/smoke.yaml",
            ]
        )
        == 0
    )
    smoke = json.loads(capsys.readouterr().out)
    assert dev["config_id"] == smoke["config_id"]
    assert dev["stack_id"] != smoke["stack_id"]
    assert (
        main(
            [
                "config",
                "show",
                "--base-yaml",
                base,
                "--execution",
                "execution/smoke.yaml",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["execution"]["validation_mode"] == "smoke"
    )
