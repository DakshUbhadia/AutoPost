from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from autopost.config import (
    config_hash,
    load_config,
    merged_config_hash,
    stack_hash,
    unresolved_fields,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_is_key_order_independent():
    assert config_hash({"a": 1, "b": {"x": 2}}) == config_hash({"b": {"x": 2}, "a": 1})


def test_base_config_loads_and_reports_known_pin_placeholders():
    cfg = load_config(ROOT / "configs/base.yaml")
    pins = {p for p, kind in unresolved_fields(cfg) if kind == "pin"}
    assert pins == {
        "model.revision",
        "model.tokenizer_revision",
        "data.dedup.embed_model_revision",
    }


def test_toolchain_controls_are_explicit():
    cfg = load_config(ROOT / "configs/base.yaml")
    train = cfg["experiment"]["train"]
    trl = train["trl"]
    assert cfg["execution"]["training_backend"] == "trl@1.17.1"
    assert trl["loss_type"] == "dapo"
    assert trl["scale_rewards"] is False
    execution = cfg["execution"]
    assert execution["backend_options"]["vllm_mode"] == "colocate"
    assert execution["importance_sampling"]["correction"] is True
    assert execution["importance_sampling"]["mode"] == "token_truncate"
    assert execution["importance_sampling"]["clip_max"] == 2.0
    assert cfg["experiment"]["curriculum"]["quarantine"]["action"] == "flag"


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("cpu_offload", True),
        ("dtype", "float16"),
        ("gradient_checkpointing", False),
    ],
)
def test_execution_changes_stack_but_not_scientific_identity(field, value):
    original = load_config(ROOT / "configs/base.yaml")
    changed = deepcopy(original)
    changed["execution"][field] = value
    assert config_hash(original) == config_hash(changed)
    assert stack_hash(original, lock_sha256="a" * 64) != stack_hash(
        changed, lock_sha256="a" * 64
    )
    assert merged_config_hash(original) != merged_config_hash(changed)


def test_all_profiles_compose_and_sampling_and_lock_change_stack():
    original = load_config(ROOT / "configs/base.yaml")
    for profile in (ROOT / "configs/execution").glob("*.yaml"):
        cfg = load_config(ROOT / "configs/base.yaml", execution=profile)
        assert "profiles" not in cfg["experiment"]
    changed = deepcopy(original)
    changed["execution"]["importance_sampling"]["clip_max"] = 3.0
    assert stack_hash(original, lock_sha256="a" * 64) != stack_hash(
        changed, lock_sha256="a" * 64
    )
    assert stack_hash(original, lock_sha256="a" * 64) != stack_hash(
        original, lock_sha256="b" * 64
    )
    changed = deepcopy(original)
    changed["model"]["revision"] = "a" * 40
    assert config_hash(original) != config_hash(changed)


def test_profile_paths_do_not_enter_hashes(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "relocated")
    assert load_config(tmp_path / "relocated/base.yaml") == load_config(
        ROOT / "configs/base.yaml"
    )
    base = tmp_path / "relocated/base.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["schema_version"] = 99
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_config(base)
