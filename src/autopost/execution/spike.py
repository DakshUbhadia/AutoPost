"""Opt-in development compatibility campaign; never a research experiment."""

import gc
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path

from autopost.config import load_config, stack_hash
from autopost.contracts import (
    ArtifactComponent,
    ArtifactManifest,
    EvalGenerationRequest,
    ExecutionProfile,
    GenerationPrompt,
    ModelSpec,
    RolloutRequest,
    SamplingConfig,
    StateRef,
    content_hash,
)

from .capabilities import algorithm_controls, discover, runtime_snapshot


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def run_spike(root, *, base_yaml="configs/base.yaml", execution=None, run_gpu=False):
    """Preflight never downloads models. Each GPU phase owns a fresh process."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(base_yaml, execution=execution)
    model = ModelSpec.model_validate_json(json.dumps(cfg["model"]))
    profile = ExecutionProfile.model_validate_json(json.dumps(cfg["execution"]))
    snapshot = runtime_snapshot()
    # Exact observed versions enter identity; a successful campaign exports its
    # full installed package lock, rather than calling an input candidate a lock.
    observed_lock = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        check=True,
    ).stdout
    lock_digest = hashlib.sha256(observed_lock).hexdigest()
    # pip freeze may contain an editable checkout path. Keep those raw bytes
    # for audit, but derive identity from versions and relative source hashes.
    inventory = sorted(
        (item.metadata["Name"], item.version) for item in distributions()
    )
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(package_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sql"}
    }
    backend_digest = content_hash(sources)
    stack = stack_hash(
        cfg,
        lock_sha256=content_hash(
            {
                "packages": inventory,
                "runtime": snapshot,
                "backend_source": backend_digest,
            }
        ),
    )
    report = discover(model, profile, stack, snapshot=snapshot)
    campaign = root / ("campaign-" + uuid.uuid4().hex)
    campaign.mkdir()
    (campaign / "environment.observed.lock").write_bytes(observed_lock)
    controls = algorithm_controls(cfg["experiment"], profile)
    result = {
        "schema": "autopost.dev-compatibility/1",
        "scientific_execution": False,
        "status": "BLOCKED" if not report.compatible else "READY_NOT_RUN",
        "capabilities": report.model_dump(mode="json"),
        "runtime": snapshot,
        "model_spec_id": model.spec_id,
        "execution_profile_id": profile.profile_id,
        "algorithm_controls": controls,
        "checks": {},
        "campaign": str(campaign),
    }
    result["observed_environment_lock_sha256"] = lock_digest
    result["backend_source_sha256"] = backend_digest
    _write(campaign / "config.json", cfg)
    _write(
        campaign / "identity.json",
        {
            "stack_id": stack,
            "model_spec_id": model.spec_id,
            "execution_profile_id": profile.profile_id,
        },
    )
    if run_gpu and report.compatible:
        for phase in ("adapter", "trainer"):
            with (
                (campaign / f"{phase}.stdout.log").open("wb") as stdout,
                (campaign / f"{phase}.stderr.log").open("wb") as stderr,
            ):
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-I",
                            "-m",
                            "autopost.execution.spike",
                            str(campaign.resolve()),
                            phase,
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=1200)
                    except subprocess.TimeoutExpired:
                        from .subprocess import LocalSubprocessExecutor

                        LocalSubprocessExecutor._terminate(process)
                        raise
                    finally:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    path = campaign / f"{phase}.json"
                    result["checks"][phase] = (
                        json.loads(path.read_bytes())
                        if path.is_file()
                        else {"passed": False}
                    )
                    if code != 0 or not result["checks"][phase].get("passed"):
                        result["status"] = "FAIL"
                        break
                except subprocess.TimeoutExpired:
                    result["checks"][phase] = {
                        "passed": False,
                        "reason": "phase deadline exceeded",
                    }
                    result["status"] = "FAIL"
                    break
        else:
            lock = subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                capture_output=True,
                check=True,
            )
            (campaign / "environment.lock").write_bytes(lock.stdout)
            checked = subprocess.run(
                [sys.executable, "-m", "pip", "check"], capture_output=True, check=False
            )
            (campaign / "pip-check.txt").write_bytes(checked.stdout + checked.stderr)
            result["status"] = "PASS" if checked.returncode == 0 else "FAIL"
            result["environment_lock_sha256"] = hashlib.sha256(lock.stdout).hexdigest()
    _write(campaign / "report.json", result)
    _write(root / "latest-report.json", result)
    return result


def _adapter_phase(root, cfg, model_spec, profile):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from autopost.store.artifacts import LocalArtifactStore

    from .generation import (
        ArtifactAdapterResolver,
        LocalEvalGenerationProvider,
        LocalVLLMRolloutProvider,
        VLLMBackend,
    )

    torch.manual_seed(17)
    dtype = getattr(torch, profile.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.tokenizer_id,
        revision=model_spec.tokenizer_revision,
        trust_remote_code=False,
    )
    prompt = "What is 1 + 1? Reply with only the number."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **model_spec.chat_template_args,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to("cuda")

    def base():
        return AutoModelForCausalLM.from_pretrained(
            model_spec.model_id,
            revision=model_spec.revision,
            dtype=dtype,
            trust_remote_code=False,
            use_safetensors=True,
        ).to("cuda")

    model = base()
    if model.config.max_position_embeddings < model_spec.max_context_tokens:
        raise ValueError("declared context exceeds model context")
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(model_spec.lora_targets) <= suffixes:
        raise ValueError("LoRA targets do not exist in model")
    model = get_peft_model(
        model,
        LoraConfig(
            r=profile.lora["r"],
            lora_alpha=profile.lora["alpha"],
            lora_dropout=profile.lora["dropout"],
            target_modules=list(model_spec.lora_targets),
            task_type="CAUSAL_LM",
        ),
    )
    # A nonzero adapter makes parity a meaningful test of weight application.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.001)
    model.eval()
    with torch.no_grad():
        before_logits = model(**encoded).logits[:, -1].float().cpu()
        before_tokens = model.generate(**encoded, max_new_tokens=16, do_sample=False)[
            0, encoded.input_ids.shape[1] :
        ].tolist()
    export = root / "adapter"
    model.save_pretrained(export, safe_serialization=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base(), export).eval()
    with torch.no_grad():
        after_logits = model(**encoded).logits[:, -1].float().cpu()
        after_tokens = model.generate(**encoded, max_new_tokens=16, do_sample=False)[
            0, encoded.input_ids.shape[1] :
        ].tolist()
    tolerance = {"atol": 0.05 if profile.dtype == "bfloat16" else 0.02, "rtol": 0.01}
    parity = (
        bool(torch.allclose(before_logits, after_logits, **tolerance))
        and before_tokens == after_tokens
    )
    max_error = float((before_logits - after_logits).abs().max())
    del model, encoded
    gc.collect()
    torch.cuda.empty_cache()
    artifacts = LocalArtifactStore(root / "artifacts")
    payload = {
        name: (export / name).read_bytes()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    manifest = ArtifactManifest.create(
        artifact_id="probe-adapter",
        kind="state",
        producer_job_id="dev-spike",
        components=tuple(
            ArtifactComponent(
                name=name,
                sha256=hashlib.sha256(blob).hexdigest(),
                size_bytes=len(blob),
                media_type="application/octet-stream",
            )
            for name, blob in payload.items()
        ),
    )
    staged = artifacts.stage(manifest.artifact.artifact_id)
    for name, blob in payload.items():
        artifacts.write(staged, name, blob)
    artifacts.finalize(staged, manifest)
    adapter_state, base_state = (
        StateRef(state_id="probe-adapter"),
        StateRef(state_id="probe-base"),
    )
    resolver = ArtifactAdapterResolver(
        artifacts, base_state, adapters={adapter_state.state_id: manifest.artifact}
    )
    identity = json.loads((root / "identity.json").read_bytes())
    backend = VLLMBackend(model_spec, profile, resolver, identity["stack_id"])
    sampling = SamplingConfig(
        temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, max_new_tokens=16
    )
    common = {
        "job_id": "probe-generation",
        "state": adapter_state,
        "prompts": (
            GenerationPrompt(prompt_id="probe-math", template_id="probe", text=prompt),
        ),
        "sampling": sampling,
        "seed": 17,
        "samples_per_prompt": 1,
    }
    rollout = LocalVLLMRolloutProvider(backend).generate(RolloutRequest(**common))
    evaluation = LocalEvalGenerationProvider(backend).generate(
        EvalGenerationRequest(**common, eval_read_id="probe-read", fresh_reference=True)
    )
    base_rollout = LocalVLLMRolloutProvider(backend).generate(
        RolloutRequest(**{**common, "state": base_state})
    )
    _write(
        root / "raw-generations.json",
        {
            "rollout": rollout.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "base": base_rollout.model_dump(mode="json"),
        },
    )
    cross_engine = list(rollout.completions[0].token_ids) == after_tokens
    result = {
        "passed": parity and cross_engine,
        "adapter_reload_parity": parity,
        "hf_vllm_greedy_token_parity": cross_engine,
        "tolerance": tolerance,
        "max_reload_logit_error": max_error,
        "hf_tokens": after_tokens,
        "chat_template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "adapter_artifact": manifest.artifact.model_dump(mode="json"),
        "context_tokens": model_spec.max_context_tokens,
        "dtype": profile.dtype,
    }
    backend.close()
    return result


def _trainer_phase(root, cfg, model_spec, profile):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from autopost.contracts.tasks import Task
    from autopost.interfaces import BinaryReward, Completion
    from autopost.verify.train import MathVerifyPool

    controls = algorithm_controls(cfg["experiment"], profile)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.tokenizer_id,
        revision=model_spec.tokenizer_revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        dtype=getattr(torch, profile.dtype),
        trust_remote_code=False,
        use_safetensors=True,
    )
    group_size = controls["num_generations"]
    args = GRPOConfig(
        output_dir=str(root / "trainer-output"),
        **controls,
        max_steps=1,
        per_device_train_batch_size=group_size,
        gradient_accumulation_steps=1,
        max_completion_length=profile.rollout_sampling.max_new_tokens,
        vllm_max_model_length=model_spec.max_context_tokens,
        vllm_gpu_memory_utilization=profile.backend_options[
            "vllm_gpu_memory_utilization"
        ],
        vllm_tensor_parallel_size=1,
        bf16=profile.dtype == "bfloat16",
        fp16=profile.dtype == "float16",
        gradient_checkpointing=profile.gradient_checkpointing,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        learning_rate=cfg["experiment"]["train"]["learning_rate"],
        seed=17,
        chat_template_kwargs=model_spec.chat_template_args,
    )
    if (
        args.loss_type != "dapo"
        or args.scale_rewards not in (False, "none")
        or not args.vllm_importance_sampling_correction
        or args.vllm_importance_sampling_mode != "token_truncate"
        or args.vllm_importance_sampling_clip_max
        != controls["vllm_importance_sampling_clip_max"]
    ):
        raise ValueError("TRL changed explicit algorithm controls")
    verifier_options = profile.backend_options["verifier_pool"]
    verifier, reward = (
        MathVerifyPool(
            workers=verifier_options["workers"],
            timeout_seconds=verifier_options["timeout_s_cpu"],
        ),
        BinaryReward(profile.rollout_sampling.max_new_tokens),
    )
    raw = []

    def verified_reward(completions, answer, **_):
        values = []
        for index, (completion, reference) in enumerate(zip(completions, answer)):
            text = (
                completion[0]["content"] if isinstance(completion, list) else completion
            )
            task = Task(
                "spike-" + reference,
                "spike-" + reference,
                "dev-spike",
                "arithmetic",
                "Engineering arithmetic fixture",
                reference,
                "int",
                False,
                "train",
            )
            # The callback lacks finish reasons: conservatively score every
            # cap-length completion as truncated, including an EOS at the cap.
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            value = Completion(
                f"spike-{len(raw)}",
                task.task_id,
                "probe-base",
                "T_train",
                17,
                index,
                text,
                n_tokens,
                n_tokens >= profile.rollout_sampling.max_new_tokens,
            )
            verdict = verifier.check(task, value)
            raw.append({"completion": asdict(value), "verdict": asdict(verdict)})
            values.append(reward(task, value, verdict).value)
        _write(root / "raw-trainer-completions.json", raw)
        return values

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Compute 1+1. Finish with Answer: <number>.",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Compute 2+3. Finish with Answer: <number>.",
                    }
                ],
            ],
            "answer": ["2", "5"],
        }
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=profile.lora["r"],
            lora_alpha=profile.lora["alpha"],
            lora_dropout=0.0,
            target_modules=list(model_spec.lora_targets),
            task_type="CAUSAL_LM",
        ),
        reward_funcs=verified_reward,
        train_dataset=dataset,
    )
    trainer.train()
    history = trainer.state.log_history
    for metric in profile.importance_sampling["required_metrics"]:
        observed = [
            float(value)
            for item in history
            for name, value in item.items()
            if metric in name and type(value) in (int, float)
        ]
        if not observed or not all(math.isfinite(value) for value in observed):
            raise ValueError("missing/non-finite emitted health metric: " + metric)
    _write(root / "trainer-log-history.json", history)
    return {
        "passed": trainer.state.global_step == 1,
        "global_step": trainer.state.global_step,
        "controls": controls,
        "emitted_metrics": sorted({name for row in history for name in row}),
        "production_training_state": False,
    }


if __name__ == "__main__":
    directory, phase = Path(sys.argv[1]), sys.argv[2]
    configuration = json.loads((directory / "config.json").read_bytes())
    model = ModelSpec.model_validate_json(json.dumps(configuration["model"]))
    execution = ExecutionProfile.model_validate_json(
        json.dumps(configuration["execution"])
    )
    implementation = {"adapter": _adapter_phase, "trainer": _trainer_phase}[phase]
    output = implementation(directory, configuration, model, execution)
    _write(directory / f"{phase}.json", output)
    raise SystemExit(0 if output["passed"] else 2)
