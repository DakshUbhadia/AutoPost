import hashlib
from pathlib import Path

import yaml

from autopost.prereg import canonical_bytes, freeze, make_draft, validate, verify
from autopost.store.db import init_db


def _fixture(base_path: Path):
    cfg = {
        "model_profiles": {"m": {"hf_repo": "example/model", "revision": "a" * 40,
                                  "tokenizer_revision": "a" * 40}},
        "data": {"dedup": {"embed_model": "example/embed", "embed_model_revision": "b" * 40}},
        "gate": {"floor": 0.03, "primary": {"alpha_a": 0.05, "alpha_b": 0.05},
                 "guardrails": [{"name": name, "margin": 0.02, "alpha": 0.05} for name in
                                ("transfer", "instruction_following", "multilingual_math", "format_compliance")]},
    }
    base_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    doc = {
        "schema": "autopost.prereg/1",
        "meta": {"version": "fixture-v1", "phase": "phase1", "stage": "B", "created_at": "2000-01-01T00:00:00Z", "code_commit": "fixture"},
        "stack": {"base_yaml_sha256": "0" * 64, "lock_sha256": "1" * 64,
                  "models": {"m": {"hf_repo": "example/model", "revision": "a" * 40,
                                     "tokenizer_revision": "a" * 40}},
                  "embedding_model": "example/embed", "embedding_revision": "b" * 40},
        "judgements": {"floor": 0.03, "guardrail_margins": {name: 0.02 for name in
                          ("transfer", "instruction_following", "multilingual_math", "format_compliance")},
                        "phase1_compute_budget": {"hardware": "fixture GPU", "gpu_hours_total": 1,
                                                  "calendar_limit_days": 1}, "e1a_learning_rates": [2e-5, 1e-5]},
        "statistics": {"alpha_a": 0.05, "alpha_b": 0.05, "guardrail_alpha": 0.05,
                       "n_boot": 4000, "bootstrap_unit": "cluster", "k_ref_multiplier": 2,
                       "per_set_power_target": 0.9, "guardrail_power_target": 0.97,
                       "target_joint_power": 0.8, "nominal_false_promotion": 0.0025},
        "phase1_plan": {"e1a": {"rates": [2e-5, 1e-5], "steps_per_rate": 200, "split": "dev"},
                        "e1": {"steps": 1000, "checkpoint_every_steps": 50, "split": "pilot"},
                        "e2_compute_matched_to": "e1", "e9_compute_matched_to": "e1"},
        "go_no_go": {"signal": {"comparison": "fixture", "interval": "fixture", "minimum_gain": "floor", "within_step_budget": True},
                      "power": {"promote_at_floor_at_least": 0.8, "promote_under_no_effect_at_most": 0.0025},
                      "controller": {"expected_false_halts_at_most": 1.0, "halt_improving_run": False},
                      "budget": {"main_run_plus_replication_and_evaluation_within_budget": True}},
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
    assert hashlib.sha256(canonical_bytes(doc)).digest() == hashlib.sha256(canonical_bytes(dict(reversed(list(doc.items()))))).digest()
    db = tmp_path / "runs/test.sqlite"
    init_db(db)
    dest = tmp_path / "prereg/fixture-v1.json"
    digest = freeze(doc, base, dest, db)
    assert dest.stat().st_mode & 0o222 == 0
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    assert verify(db, dest.parent) == [("fixture-v1", True)]
    import sqlite3
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT sha256 FROM prereg WHERE version='fixture-v1'").fetchone()[0] == digest
        assert connection.execute("SELECT kind FROM events WHERE kind='prereg_frozen'").fetchone()[0] == "prereg_frozen"
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
