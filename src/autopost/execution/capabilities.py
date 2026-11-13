"""Runtime capability assertions, with no optional GPU imports on import."""

import math
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version

from autopost.contracts import CapabilityReport, CapabilitySpec, CompatibilityError

API_PINS = {"trl": "1.17.1", "transformers": "5.20.1", "peft": "0.21.2"}
GPU_PINS = {**API_PINS, "torch": "2.14.0", "vllm": "0.33.0"}


def installed_versions():
    result = {}
    for name in (*API_PINS, "torch", "vllm"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def runtime_snapshot():
    snapshot = {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "versions": installed_versions(),
        "cuda_available": False,
        "devices": [],
    }
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_version"] = torch.version.cuda
        if snapshot["cuda_available"]:
            snapshot["devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "memory_bytes": int(
                        torch.cuda.get_device_properties(i).total_memory
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except (ImportError, OSError, RuntimeError) as exc:
        snapshot["runtime_error"] = type(exc).__name__
    return snapshot


def algorithm_controls(experiment, execution):
    trl = experiment["train"]["trl"]
    sampling = execution.importance_sampling
    required = {
        "loss_type": "dapo",
        "scale_rewards": False,
        "num_iterations": 1,
        "mask_truncated_completions": False,
    }
    reasons = [
        f"train.trl.{key} must be {expected!r}"
        for key, expected in required.items()
        if trl.get(key) != expected
    ]
    if (
        sampling.get("correction") is not True
        or sampling.get("mode") != "token_truncate"
    ):
        reasons.append("token-level truncated importance sampling must be explicit")
    cap = sampling.get("clip_max")
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        reasons.append("importance-sampling cap must be finite and positive")
    if sampling.get("clip_min") is not None:
        reasons.append("this backend requires clip_min=None")
    if set(sampling.get("required_metrics", [])) != {
        "importance_sampling_ratio",
        "sampling_logp_difference",
    }:
        reasons.append("both importance-sampling health metrics must be required")
    if (
        execution.backend_options.get("vllm_mode") != "colocate"
        or execution.backend_options.get("use_vllm") is not True
    ):
        reasons.append("colocated vLLM must be explicitly enabled")
    if execution.training_backend != "trl@" + API_PINS["trl"]:
        reasons.append("training backend disagrees with the pinned TRL version")
    if reasons:
        raise CompatibilityError("; ".join(reasons))
    return {
        **trl,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "token_truncate",
        "vllm_importance_sampling_clip_min": None,
        "vllm_importance_sampling_clip_max": cap,
    }


def discover(model, execution, stack_id, *, snapshot=None):
    snapshot = runtime_snapshot() if snapshot is None else snapshot
    reasons = []
    if snapshot.get("python", "3.13.7") != "3.13.7":
        reasons.append("small-gpu-dev requires Python 3.13.7")
    if snapshot["platform"] != "Linux":
        reasons.append(
            "vLLM execution requires Linux (native or a compatible GPU environment)"
        )
    if not snapshot["cuda_available"] or not snapshot["devices"]:
        reasons.append("CUDA GPU is unavailable")
    for name, pin in GPU_PINS.items():
        if snapshot["versions"].get(name) != pin:
            reasons.append(f"{name} must be {pin}")
    if not snapshot["versions"].get("vllm"):
        reasons.append("vllm is not installed")
    devices = snapshot["devices"][: execution.tensor_parallel_size]
    capability = min(
        (tuple(device["capability"]) for device in devices), default=(0, 0)
    )
    if devices and capability < (7, 5):
        reasons.append(
            "selected GPU is below this backend's compute capability 7.5 floor"
        )
    dtypes = (
        ("float16", "float32", "bfloat16")
        if capability >= (8, 0)
        else (("float16", "float32") if capability >= (7, 5) else ())
    )
    if execution.dtype not in dtypes:
        reasons.append(
            "requested dtype is unsupported; there is no automatic precision fallback"
        )
    if execution.tensor_parallel_size > len(snapshot["devices"]):
        reasons.append("requested tensor parallelism exceeds visible devices")
    if execution.tensor_parallel_size != 1:
        reasons.append("Sprint 5 reference backend validates one GPU only")
    if execution.quantization is not None or execution.cpu_offload:
        reasons.append(
            "quantization and CPU offload are not validated by this reference backend"
        )
    if execution.generation_backend != "vllm-colocate":
        reasons.append("generation backend must be vllm-colocate")
    if execution.validation_mode != "dev":
        reasons.append("this compatibility spike is development-only")
    for field in ("revision", "tokenizer_revision"):
        if re.fullmatch(r"[0-9a-f]{40}", getattr(model, field)) is None:
            reasons.append(f"model.{field} requires a pinned commit SHA")
    if model.max_context_tokens is None:
        reasons.append("model context limit is unresolved")
    for name in ("rollout_sampling", "eval_sampling"):
        sampling = getattr(execution, name)
        if sampling.max_new_tokens is None:
            reasons.append(f"{name}.max_new_tokens is unresolved")
        elif (
            model.max_context_tokens is not None
            and sampling.max_new_tokens >= model.max_context_tokens
        ):
            reasons.append(f"{name} leaves no room for the prompt")
    memory = execution.backend_options.get("vllm_gpu_memory_utilization")
    if type(memory) not in (int, float) or not 0 < memory < 1:
        reasons.append("vLLM GPU memory fraction must be explicit and in (0,1)")
    lora = execution.lora
    if (
        type(lora.get("r")) is not int
        or lora["r"] not in {8, 16, 32, 64}
        or not model.lora_targets
        or lora.get("dropout") != 0.0
    ):
        reasons.append(
            "LoRA targets/rank or deterministic dropout setting are unsupported"
        )
    runtime_present = (
        snapshot["platform"] == "Linux"
        and snapshot["cuda_available"]
        and all(snapshot["versions"].get(name) == pin for name, pin in GPU_PINS.items())
    )
    capabilities = CapabilitySpec(
        dtypes=dtypes,
        quantizations=(),
        max_context_tokens=model.max_context_tokens or 1,
        lora=bool(model.lora_targets) and runtime_present,
        training_backends=(execution.training_backend,) if not reasons else (),
        generation_backends=("vllm-colocate",)
        if runtime_present and capability >= (7, 5)
        else (),
        max_tensor_parallel_size=1,
        importance_sampling_modes=("token_truncate",) if runtime_present else (),
        colocated_generation=bool(runtime_present),
    )
    return CapabilityReport(
        backend_id="local-vllm@1",
        stack_id=stack_id,
        capabilities=capabilities,
        compatible=not reasons,
        reasons=tuple(reasons),
    )


def require_compatible(report):
    report.to_bytes()
    if not report.compatible:
        raise CompatibilityError("; ".join(report.reasons))
