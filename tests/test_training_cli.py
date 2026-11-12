import json
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml

from autopost.cli import main
from autopost.contracts.signals import LabelReview
from autopost.data.commands import build_command
from autopost.interfaces import Completion
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.signals import TrainingSignalRepository, task_digest


def test_cli_verifies_retains_samples_reserves_and_reviews_with_real_workers(
    tmp_path, capsys
):
    root = tmp_path / "data"
    index = build_command(root)
    reference = tmp_path / "train-ref.json"
    reference.write_text(json.dumps(index["splits"]["train"]), encoding="utf-8")
    repo = TrainingSignalRepository(
        root / "datasets.sqlite", LocalArtifactStore(root / "artifacts")
    )
    from autopost.contracts import DatasetRef

    ref = DatasetRef.from_bytes(reference.read_bytes())
    task = next(
        t for t in repo.datasets.training_tasks(ref, fixture_only=True) if t.measurable
    )
    completions = tmp_path / "completions.json"
    completions.write_text(
        json.dumps(
            [
                asdict(
                    Completion(
                        f"cli-sample-{i}",
                        task.task_id,
                        "fixture-base",
                        "T_train",
                        0,
                        i,
                        f"Answer: {answer}",
                        10,
                        False,
                    )
                )
                for i, answer in enumerate((task.answer, "999999"))
            ]
        ),
        encoding="utf-8",
    )
    # A small single-source fixture policy, never a research configuration.
    config_root = tmp_path / "configs"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", config_root)
    base = config_root / "base.yaml"
    value = yaml.safe_load(base.read_text(encoding="utf-8"))
    value["curriculum"]["estimate_k"] = 2
    value["curriculum"]["quotas"]["max_share"] = 1.0
    value["curriculum"]["quotas"]["low_entropy_max_share"] = 1.0
    base.write_text(yaml.safe_dump(value), encoding="utf-8")
    common = [
        "--root",
        str(root),
        "--dataset-ref",
        str(reference),
        "--base-yaml",
        str(base),
        "--fixture-only",
    ]
    verify = [
        "training",
        "verify",
        *common,
        "--task-id",
        task.task_id,
        "--completions",
        str(completions),
        "--group-id",
        "cli-group",
        "--max-new-tokens",
        "100",
    ]
    assert main(verify) == 0
    report = json.loads(capsys.readouterr().out)
    assert [r["value"] for r in report["rewards"]] == [1, 0]
    assert report["observation"]["n_correct"] == 1
    assert main(verify) == 0
    assert json.loads(capsys.readouterr().out) == report
    sample = [
        "training",
        "sample",
        *common,
        "--slice-id",
        "cli-slice",
        "--n-prompts",
        "1",
        "--seed",
        "1",
    ]
    assert main(sample) == 0
    selection = json.loads(capsys.readouterr().out)
    assert selection["task_ids"] == [task.task_id]
    assert main(sample) == 0
    assert json.loads(capsys.readouterr().out) == selection
    assert (
        main(["training", "refresh", *common, "--promotion-id", "fixture-promotion"])
        == 0
    )
    capsys.readouterr()
    evidence = repo.retain_evidence(
        {
            "synthetic_test_only": True,
            "task": task.task_id,
            "recomputation": "test derivation",
        }
    )
    review = LabelReview(
        review_id="cli-review",
        task_id=task.task_id,
        task_digest=task_digest(task),
        method="deterministic",
        checker_id="synthetic-independent@1",
        reviewer="test-harness",
        decision="confirmed_bad",
        reason="Synthetic test exclusion",
        independence_basis="Synthetic derivation independent of sampled answers",
        evidence=evidence,
    )
    review_file = tmp_path / "review.json"
    review_file.write_bytes(review.to_bytes())
    assert main(["training", "review", *common, "--review", str(review_file)]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "confirmed_bad"
    assert main(sample) == 1
    assert "excluded" in capsys.readouterr().err
