import importlib.util
from pathlib import Path

import pytest

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


def test_load_locked_versions_requires_exact_marker_free_entries(tmp_path):
    lock = tmp_path / "probe.lock"
    lock.write_text("# generated\ntrl==1.17.1\npeft==0.21.2\n", encoding="utf-8")
    assert PROBE.load_locked_versions(lock) == {"trl": "1.17.1", "peft": "0.21.2"}

    lock.write_text("trl>=1.17.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported lock entry"):
        PROBE.load_locked_versions(lock)


def test_runtime_statuses_do_not_claim_gpu_or_vllm_execution():
    assert PROBE.runtime_statuses(False) == ("PENDING_HARDWARE", "NOT_TESTED", [])
    gpu, vllm, blockers = PROBE.runtime_statuses(True)
    assert gpu == "REQUIRED_NOT_RUN"
    assert vllm == "NOT_TESTED"
    assert blockers


def test_probe_reads_split_config_without_changing_explicit_algorithm_controls():
    version, configured = PROBE._load_train_config(PATH.parents[1] / "configs/base.yaml")
    assert version == "1.17.1"
    assert PROBE.REQUIRED_GRPO_FIELDS <= configured.keys()
    assert configured["vllm_importance_sampling_correction"] is True
    assert configured["vllm_importance_sampling_mode"] == "token_truncate"
    assert configured["vllm_importance_sampling_clip_max"] == 2.0
    assert configured["vllm_mode"] == "colocate"
    assert configured["loss_type"] == "dapo"
    assert configured["scale_rewards"] is False
