import hashlib
from pathlib import Path

import pytest
import yaml

from autopost.prereg import canonical_bytes, freeze, make_draft, validate, verify
from autopost.store.db import init_db


def _fixture(base_path: Path):
    cfg = {
        "model_profiles": {
            "m": {
                "hf_repo": "example/model",
                "revision": "a" * 40,
                "tokenizer_revision": "a" * 40,
            }
        },
        "data": {
            "dedup": {"embed_model": "example/embed", "embed_model_revision": "b" * 40}
        },
        "gate": {
            "floor": 0.03,
            "primary": {"alpha_a": 0.05, "alpha_b": 0.05},
            "guardrails": [
                {"name": name, "margin": 0.02, "alpha": 0.05}
                for name in (
                    "transfer",
                    "instruction_following",
                    "multilingual_math",
                    "format_compliance",
                )
            ],
        },
    }
    base_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    doc = {
        "schema": "autopost.prereg/1",
        "meta": {
            "version": "fixture-v1",
            "phase": "phase1",
            "stage": "B",
            "created_at": "2000-01-01T00:00:00Z",
            "code_commit": "fixture",
        },
        "stack": {
            "base_yaml_sha256": "0" * 64,
            "lock_sha256": "1" * 64,
            "models": {
                "m": {
                    "hf_repo": "example/model",
                    "revision": "a" * 40,
                    "tokenizer_revision": "a" * 40,
                }
            },
            "embedding_model": "example/embed",
            "embedding_revision": "b" * 40,
        },
        "judgements": {
            "floor": 0.03,
            "guardrail_margins": {
                name: 0.02
                for name in (
                    "transfer",
                    "instruction_following",
                    "multilingual_math",
                    "format_compliance",
                )
            },
            "phase1_compute_budget": {
                "hardware": "fixture GPU",
                "gpu_hours_total": 1,
                "calendar_limit_days": 1,
            },
            "e1a_learning_rates": [2e-5, 1e-5],
        },
        "statistics": {
            "alpha_a": 0.05,
            "alpha_b": 0.05,
            "guardrail_alpha": 0.05,
            "n_boot": 4000,
            "bootstrap_unit": "cluster",
            "k_ref_multiplier": 2,
            "per_set_power_target": 0.9,
            "guardrail_power_target": 0.97,
            "target_joint_power": 0.8,
            "nominal_false_promotion": 0.0025,
        },
        "phase1_plan": {
            "e1a": {"rates": [2e-5, 1e-5], "steps_per_rate": 200, "split": "dev"},
            "e1": {"steps": 1000, "checkpoint_every_steps": 50, "split": "pilot"},
            "e2_compute_matched_to": "e1",
            "e9_compute_matched_to": "e1",
        },
        "go_no_go": {
            "signal": {
                "comparison": "fixture",
                "interval": "fixture",
                "minimum_gain": "floor",
                "within_step_budget": True,
            },
            "power": {
                "promote_at_floor_at_least": 0.8,
                "promote_under_no_effect_at_most": 0.0025,
            },
            "controller": {
                "expected_false_halts_at_most": 1.0,
                "halt_improving_run": False,
            },
            "budget": {"main_run_plus_replication_and_evaluation_within_budget": True},
        },
    }
    return cfg, doc


def test_draft_requires_human_values(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("{}", encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture", encoding="utf-8")
    draft = make_draft("fixture-v1", base, lock)
    assert validate(draft, base)


def test_fixture_validation_canonicalization_and_freeze(tmp_path):
    base, lock = tmp_path / "base.yaml", tmp_path / "requirements.lock"
    _, doc = _fixture(base)
    lock.write_text("fixture", encoding="utf-8")
    assert validate(doc, base) == []
    assert (
        hashlib.sha256(canonical_bytes(doc)).digest()
        == hashlib.sha256(canonical_bytes(dict(reversed(list(doc.items()))))).digest()
    )
    db = tmp_path / "runs/test.sqlite"
    init_db(db)
    dest = tmp_path / "prereg/fixture-v1.json"
    digest = freeze(doc, base, dest, db)
    assert dest.stat().st_mode & 0o222 == 0
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    assert verify(db, dest.parent) == [("fixture-v1", True)]
    import sqlite3

    with sqlite3.connect(db) as connection:
        assert (
            connection.execute(
                "SELECT sha256 FROM prereg WHERE version='fixture-v1'"
            ).fetchone()[0]
            == digest
        )
        assert (
            connection.execute(
                "SELECT kind FROM events WHERE kind='prereg_frozen'"
            ).fetchone()[0]
            == "prereg_frozen"
        )
    try:
        freeze(doc, base, dest, db)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate version accepted")
    dest.chmod(0o644)
    dest.write_bytes(dest.read_bytes() + b" ")
    assert verify(db, dest.parent) == [("fixture-v1", False)]


def test_freeze_rejects_floor_mismatch(tmp_path):
    base, doc = tmp_path / "base.yaml", None
    _, doc = _fixture(base)
    doc["judgements"]["floor"] = 0.04
    assert any("gate.floor" in problem for problem in validate(doc, base))


def _split_fixture(tmp_path):
    import shutil

    source = Path(__file__).resolve().parents[1] / "configs"
    configs = tmp_path / "configs"
    shutil.copytree(source, configs)
    base = configs / "base.yaml"
    experiment = yaml.safe_load(base.read_text(encoding="utf-8"))
    experiment["profiles"]["execution"] = "execution/research-qwen4b-local80g.yaml"
    experiment["profiles"]["model"] = "models/qwen3_4b_instruct_2507.yaml"
    experiment["gate"]["floor"] = 0.03
    for guardrail in experiment["gate"]["guardrails"]:
        guardrail["margin"] = 0.02
    base.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    for relative in (
        experiment["profiles"]["model"],
        experiment["profiles"]["execution"],
        "data/math.yaml",
    ):
        path = configs / relative
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "tokenizer_revision" in cfg:
            cfg.update(
                revision="a" * 40, tokenizer_revision="a" * 40, max_context_tokens=2048
            )
        if "rollout_sampling" in cfg:
            cfg["rollout_sampling"]["max_new_tokens"] = 128
            cfg["eval_sampling"]["max_new_tokens"] = 128
        if "dedup" in cfg:
            cfg["dedup"]["embed_model_revision"] = "b" * 40
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    lock = tmp_path / "fixture.lock"
    lock.write_text("example==1.0\n", encoding="utf-8")
    draft = make_draft("split-v2", base, lock)
    draft["judgements"].update(
        floor=0.03,
        guardrail_margins={
            name: 0.02 for name in draft["judgements"]["guardrail_margins"]
        },
        phase1_compute_budget={
            "hardware": "fixture",
            "gpu_hours_total": 1,
            "calendar_limit_days": 1,
        },
        e1a_learning_rates=[2e-5, 1e-5],
    )
    draft["phase1_plan"]["e1a"]["rates"] = [2e-5, 1e-5]
    return base, lock, draft


def test_split_prereg_freezes_canonical_snapshot_and_preserves_v1(tmp_path):
    base, lock, draft = _split_fixture(tmp_path)
    assert draft["schema"] == "autopost.prereg/2"
    assert validate(draft, base, lock=lock) == []
    db = tmp_path / "prereg.sqlite"
    init_db(db)
    destination = tmp_path / "prereg/split-v2.json"
    digest = freeze(draft, base, destination, db, lock=lock)
    assert len(digest) == 64
    assert verify(db, destination.parent) == [("split-v2", True)]
    destination.chmod(0o644)


@pytest.mark.parametrize(
    "section", ["model", "execution", "data", "experiment", "lock"]
)
def test_split_prereg_detects_changed_composition_before_side_effects(
    tmp_path, section
):
    base, lock, draft = _split_fixture(tmp_path)
    assert validate(draft, base, lock=lock) == []
    if section == "lock":
        lock.write_text("example==2.0\n", encoding="utf-8")
    else:
        selection = yaml.safe_load(base.read_text(encoding="utf-8"))["profiles"]
        path = base if section == "experiment" else base.parent / selection[section]
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if section == "model":
            cfg["revision"] = "c" * 40
        elif section == "execution":
            cfg["tensor_parallel_size"] = 2
        elif section == "data":
            cfg["dedup"]["embed_cos"] = 0.93
        else:
            cfg["train"]["learning_rate"] = 1e-5
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert validate(draft, base, lock=lock)
    output, db = tmp_path / "prereg/frozen.json", tmp_path / "absent.sqlite"
    with pytest.raises(ValueError):
        freeze(draft, base, output, db, lock=lock)
    assert not output.exists() and not db.exists()


def test_split_prereg_draft_does_not_allow_dev_evidence_to_be_frozen_as_research(
    tmp_path,
):
    base, lock, draft = _split_fixture(tmp_path)
    execution_path = base.parent / "execution/research-qwen4b-local80g.yaml"
    cfg = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    cfg["validation_mode"] = "dev"
    execution_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    draft = make_draft("dev-draft", base, lock)
    assert draft["meta"]["mode"] == "dev"
    assert any(
        "research execution profile" in issue
        for issue in validate(draft, base, lock=lock)
    )
