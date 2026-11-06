import json

import pytest

from autopost.cli import main


@pytest.mark.parametrize(
    "scenario,outcome,budget",
    [
        ("success", "accepted", 9),
        ("duplicate_success", "duplicate", 9),
        ("conflict", "conflict", 9),
        ("stale_parent", "stale_parent", 10),
        ("incomplete_artifact", "missing_artifact", 10),
        ("executor_failure", None, 10),
    ],
)
def test_fixture_cli_demonstrates_expected_semantics(
    tmp_path, capsys, scenario, outcome, budget
):
    assert (
        main(["fixture", "run", "--root", str(tmp_path), "--scenario", scenario]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["fixture_only"] is True
    assert report["remaining_budget"] == budget
    assert (report["dispositions"][-1] if report["dispositions"] else None) == outcome
    assert json.loads((tmp_path / scenario / "report.json").read_text()) == report
    if scenario == "success":
        assert main(["fixture", "run", "--root", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out)["dispositions"] == ["duplicate"]


def test_db_cli_reports_and_migrates_schema(tmp_path, capsys):
    path = str(tmp_path / "metadata.sqlite")
    assert main(["db", "migrate", "--path", path]) == 0
    assert capsys.readouterr().out.strip() == "metadata schema 2"
    assert main(["db", "status", "--path", path]) == 0
    assert capsys.readouterr().out.strip() == "metadata schema 2"
