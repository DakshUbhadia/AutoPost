"""Inspect the pinned TRL/PEFT API without loading a model or starting vLLM."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import inspect
import json
import platform
import sys
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

SCHEMA = "autopost.toolchain-probe/1"
REQUIRED_GRPO_FIELDS = {
    "beta",
    "epsilon",
    "epsilon_high",
    "loss_type",
    "mask_truncated_completions",
    "num_generations",
    "num_iterations",
    "scale_rewards",
    "use_vllm",
    "vllm_gpu_memory_utilization",
    "vllm_importance_sampling_clip_max",
    "vllm_importance_sampling_clip_min",
    "vllm_importance_sampling_correction",
    "vllm_importance_sampling_mode",
    "vllm_mode",
}
REQUIRED_TRAINER_PARAMETERS = {
    "args",
    "callbacks",
    "model",
    "peft_config",
    "processing_class",
    "reward_funcs",
    "train_dataset",
}
REQUIRED_METRIC_TOKENS = {
    "importance_sampling_ratio",
    "sampling_logp_difference",
    "rewards/",
}
REQUIRED_LORA_PARAMETERS = {"lora_alpha", "lora_dropout", "r", "target_modules"}


def missing_names(available: Iterable[str], required: Iterable[str]) -> list[str]:
    """Return required names absent from an API surface."""
    return sorted(set(required) - set(available))


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_train_config(path: Path) -> tuple[str, dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("train"), dict):
        raise TypeError(f"missing train mapping in {path}")
    train = document["train"]
    if not isinstance(train.get("trl"), dict):
        raise TypeError(f"missing train.trl mapping in {path}")
    return str(train["trl_version"]), train["trl"]


def _cuda_summary() -> dict:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        return {
            "torch": _version("torch"),
            "available": available,
            "device_count": count,
            "devices": names,
        }
    except Exception as exc:  # noqa: BLE001  # optional runtime must become a report
        return {"torch": _version("torch"), "available": False, "device_count": 0, "error": str(exc)}


def run_probe(config_path: Path) -> tuple[dict, int]:
    errors: list[str] = []
    checks: dict[str, object] = {}
    versions = {
        name: _version(name)
        for name in ("trl", "transformers", "peft", "vllm", "torch")
    }
    cuda = _cuda_summary()

    try:
        expected_trl, configured = _load_train_config(config_path)
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        expected_trl, configured = "unknown", {}
        errors.append(str(exc))

    if versions["trl"] != expected_trl:
        errors.append(f"TRL version mismatch: configured {expected_trl}, installed {versions['trl'] or 'missing'}")

    expected_values = {
        "loss_type": "dapo",
        "scale_rewards": False,
        "vllm_mode": "colocate",
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "token_truncate",
    }
    for name, expected in expected_values.items():
        if configured.get(name) != expected:
            errors.append(f"train.trl.{name} must be {expected!r}; found {configured.get(name)!r}")
    cap = configured.get("vllm_importance_sampling_clip_max")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
        errors.append("train.trl.vllm_importance_sampling_clip_max must be an explicit positive number")
    checks["configured_controls"] = not any(error.startswith("train.trl.") for error in errors)

    try:
        from trl import GRPOConfig, GRPOTrainer

        fields = {field.name for field in dataclasses.fields(GRPOConfig)}
        missing_fields = missing_names(fields, REQUIRED_GRPO_FIELDS)
        checks["grpo_config_fields"] = {"ok": not missing_fields, "missing": missing_fields}
        if missing_fields:
            errors.append(f"GRPOConfig missing fields: {', '.join(missing_fields)}")

        constructor_values = {
            key: value
            for key, value in configured.items()
            if key in fields and value is not None
        }
        constructor_values.update({"use_vllm": False, "bf16": False, "fp16": False, "report_to": "none"})
        with tempfile.TemporaryDirectory(prefix="autopost-probe-") as output_dir:
            instance = GRPOConfig(output_dir=output_dir, **constructor_values)
        constructor_ok = (
            instance.loss_type == "dapo"
            # TRL 1.17.1 normalizes False to its internal literal "none".
            and instance.scale_rewards in (False, "none")
            and instance.vllm_importance_sampling_mode == "token_truncate"
        )
        checks["grpo_config_construction"] = constructor_ok
        if not constructor_ok:
            errors.append("GRPOConfig changed one or more explicit algorithm controls")

        trainer_parameters = set(inspect.signature(GRPOTrainer.__init__).parameters)
        missing_trainer = missing_names(trainer_parameters, REQUIRED_TRAINER_PARAMETERS)
        checks["trainer_hooks"] = {"ok": not missing_trainer, "missing": missing_trainer}
        if missing_trainer:
            errors.append(f"GRPOTrainer missing hooks: {', '.join(missing_trainer)}")

        trainer_source = inspect.getsource(GRPOTrainer)
        missing_metrics = sorted(token for token in REQUIRED_METRIC_TOKENS if token not in trainer_source)
        checks["expected_metrics"] = {"ok": not missing_metrics, "missing_tokens": missing_metrics}
        if missing_metrics:
            errors.append(f"GRPOTrainer missing metric tokens: {', '.join(missing_metrics)}")
    except Exception as exc:  # noqa: BLE001  # probe failures must not hide the report
        checks["trl_import_and_construction"] = False
        errors.append(f"TRL API inspection failed: {type(exc).__name__}: {exc}")

    try:
        from peft import LoraConfig, get_peft_model

        lora_parameters = set(inspect.signature(LoraConfig).parameters)
        missing_lora = missing_names(lora_parameters, REQUIRED_LORA_PARAMETERS)
        peft_ok = not missing_lora and callable(get_peft_model)
        checks["peft_lora_hooks"] = {"ok": peft_ok, "missing": missing_lora}
        if not peft_ok:
            errors.append(f"PEFT LoRA hooks missing: {', '.join(missing_lora)}")
    except Exception as exc:  # noqa: BLE001  # probe failures must not hide the report
        checks["peft_lora_hooks"] = False
        errors.append(f"PEFT API inspection failed: {type(exc).__name__}: {exc}")

    if cuda.get("available"):
        gpu_smoke = "REQUIRED_NOT_RUN"
        errors.append("CUDA GPU detected: add and run the one-to-two-step dev smoke before Sprint 0 can close")
    else:
        gpu_smoke = "SKIPPED_NO_GPU"

    result = {
        "schema": SCHEMA,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if not errors else "BLOCKED",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": str(config_path),
        "versions": versions,
        "cuda": cuda,
        "gpu_smoke": gpu_smoke,
        "checks": checks,
        "errors": errors,
    }
    return result, 0 if not errors else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/toolchain_probe.json"))
    args = parser.parse_args(argv)
    result, return_code = run_probe(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
