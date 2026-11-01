from pathlib import Path

from autopost.config import config_hash, load_config, unresolved_fields

ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_is_key_order_independent():
    assert config_hash({"a": 1, "b": {"x": 2}}) == config_hash({"b": {"x": 2}, "a": 1})


def test_base_config_loads_and_reports_known_pin_placeholders():
    cfg = load_config(ROOT / "configs/base.yaml")
    pins = {p for p, kind in unresolved_fields(cfg) if kind == "pin"}
    # T4/T5 must replace these with verified PyPI/Hugging Face values. Until
    # then, fail only if an unexpected placeholder is introduced.
    assert pins == {
        "model_profiles.qwen3_0p6b.revision",
        "model_profiles.qwen3_4b_instruct_2507.revision",
    }


def test_toolchain_controls_are_explicit():
    cfg = load_config(ROOT / "configs/base.yaml")
    train = cfg["train"]
    trl = train["trl"]
    assert train["trl_version"] == "1.17.1"
    assert trl["loss_type"] == "dapo"
    assert trl["scale_rewards"] is False
    assert trl["vllm_mode"] == "colocate"
    assert trl["vllm_importance_sampling_correction"] is True
    assert trl["vllm_importance_sampling_mode"] == "token_truncate"
    assert trl["vllm_importance_sampling_clip_max"] == 2.0
