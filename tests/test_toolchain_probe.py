import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_toolchain_api.py"
SPEC = importlib.util.spec_from_file_location("autopost_toolchain_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_probe_requires_architecture_controls():
    required = PROBE.REQUIRED_GRPO_FIELDS
    assert {
        "loss_type",
        "scale_rewards",
        "vllm_mode",
        "vllm_importance_sampling_correction",
        "vllm_importance_sampling_mode",
        "vllm_importance_sampling_clip_max",
    } <= required


def test_missing_names_is_deterministic():
    assert PROBE.missing_names({"b", "c"}, {"a", "b"}) == ["a"]


def test_probe_requires_callbacks_peft_and_mismatch_metrics():
    assert {"callbacks", "peft_config"} <= PROBE.REQUIRED_TRAINER_PARAMETERS
    assert "importance_sampling_ratio" in PROBE.REQUIRED_METRIC_TOKENS
    assert "sampling_logp_difference" in PROBE.REQUIRED_METRIC_TOKENS
