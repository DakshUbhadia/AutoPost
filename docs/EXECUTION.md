# Local execution and development GPU compatibility

Sprint 5 implements local subprocess placement, vLLM rollout/evaluation providers,
runtime assertions and an opt-in development compatibility campaign. These are
engineering checks using Qwen3-0.6B, not target-model scientific evidence.

## CPU workflow

```text
python -m autopost.cli execution fixture
python -m autopost.cli execution capabilities
python scripts/probe_dev_model.py
```

The fixture command runs a real child process against the durable stores. The
other commands inspect availability without downloading models. A missing GPU
produces a structured incompatible report and exit code 2. `make spike` runs this
preflight; it no longer points to an unimplemented command.

`LocalSubprocessExecutor` implements `JobExecutor` and accepts a trusted installed
handler mapping, for example `{"training": "autopost.execution.fixture_worker:execute"}`
for synthetic conformance only. Handlers receive the versioned spec and backend
store objects and return `(JobResult, AcceptanceEffects | None)`. Production
training handlers belong to Sprint 6; evaluation suites/scoring belong to Sprint 7.
An unregistered job kind is rejected before launch. Job data cannot select a
handler or inject shell commands. No serialized job contains an executable,
physical checkpoint locator, callable or library runtime object.

The executor validates specs, reserves a job/attempt and atomically claims pending
work in metadata. A second executor cannot run the same attempt, even with a
different local work directory. Every attempt has independent request/response,
stdout/stderr and timeout data under the backend root. Children run Python with
`-I`; subprocesses use argument arrays without shell interpretation. Responses
are published atomically. A monotonic submission deadline is checked against the
worker's completion time as well as by the watchdog, so delayed monitoring cannot
accept a late response. Cancellation/timeouts terminate process trees; Linux
workers own process groups and Windows uses `taskkill /T` with a direct-child
fallback. Closing the executor cancels owned work and joins its monitor threads.

`status`, `result`, `logs` and `effects` retrieve journaled work. Retrying the same
attempt does not execute again. A failed/cancelled attempt needs a new attempt
number. Results do not accept state, pointer or budget effects: the controller
must explicitly call `MetadataStore.accept_result`. Completed responses can be
collected after executor restart. An active process belongs to its original
executor; a replacement cannot assume ownership or relaunch its claimed attempt.
Full interrupted-controller recovery is a later release gate.

## Generation boundary

`LocalVLLMRolloutProvider` and `LocalEvalGenerationProvider` wrap `VLLMBackend`.
The engine/tokenizer load lazily after state resolution and capability assertions.
Only request/batch contracts cross the boundary. Every sample carries token IDs,
aligned finite sampled-token log probabilities, explicit truncation, prompt,
template, state/job and evaluation-read identities. Requests require a generation
cap and must fit the declared context after chat-template rendering.

The model and tokenizer use the same pinned repository revision; remote model
code is disabled. Generation parameters override model repository defaults.
Per-sample seeds derive from stable job/state/prompt/template/sample/read inputs.
The evaluation provider has no reference-output cache: every call reaches the
engine. Library results missing samples or log probabilities fail closed.

`ArtifactAdapterResolver` resolves the base model explicitly and materializes
other immutable adapter artifacts in a backend-private temporary directory.
Production state lookup checks the execution stack; the compatibility probe
registers its own disposable StateRef-to-ArtifactRef map. Physical adapter paths
are passed only to vLLM. Models/engines should live in an isolated job process,
which owns GPU cleanup.

Generation text is opaque evidence: slash-prefixed LaTeX such as `\boxed{2}`
round-trips through nested batch contracts. Path validation still applies to
identities/settings/manifests; this is not a general exemption for JSON metadata.

## Linux GPU campaign (including Kaggle)

For ordered copyable notebook cells and optional GitHub evidence upload, use
[the Kaggle walkthrough](KAGGLE.md) and
[ready-to-import notebook](../notebooks/sprint5_kaggle.ipynb).

Use a separate CPython 3.13.7 Linux environment on the actual GPU. First make
sure the Sprint 5 source or built wheel is available there. Do not copy the
Windows CPU toolchain lock onto a GPU host.

```text
python -m pip install --requirement requirements/small-gpu-dev.in
python -m pip install --no-deps --no-build-isolation .
python -m pip check
python scripts/probe_dev_model.py --execution execution/dev-qwen06b-t4.yaml --root runs/dev-spike --run-gpu
```

For a bf16-capable GPU, select `execution/dev-qwen06b-local24g.yaml` instead.
The T4 profile explicitly selects float16; there is no automatic precision change.
Only one GPU is validated; two visible T4s are not treated as pooled memory.
The pinned model is `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`, with a deliberately bounded 4,096-token
development context and 128-token completion caps.

`requirements/small-gpu-dev.in` is an **unvalidated direct-pin candidate**, not a
resolved lock or a claim that its CUDA/PyTorch/vLLM combination runs on Kaggle.
Its candidate vLLM 0.33.0 requires torch 2.14.0 according to its published
[package metadata](https://pypi.org/pypi/vllm/0.33.0/json); Transformers 5.20.1
satisfies that release's declared range. Dependency resolution, driver/kernel
support and memory fit still need validation on the allocated machine. The
capability probe rejects mismatches instead of silently upgrading or falling back.

`--run-gpu` explicitly allows model downloads and two isolated GPU phases:

1. Attach a **nonzero** LoRA adapter; generate with Hugging Face; export/reload;
   compare greedy tokens and logits; publish immutable adapter bytes; generate
   through both project providers and compare Hugging Face/vLLM greedy tokens.
2. Run one disposable TRL GRPO step with explicit DAPO, no reward scaling,
   token-level truncated IS, a finite cap and colocated vLLM. Rewards use the
   pinned training verifier. Require finite emitted `importance_sampling_ratio`
   and `sampling_logp_difference` metrics. This produces no accepted production
   training state. The reward callback conservatively treats cap-length text as
   truncated because that library callback does not expose finish reasons.

Adapter reload tolerance is declared before execution: atol 0.02 for float16 or
0.05 for bfloat16, rtol 0.01, plus exact greedy-token agreement for this probe.
Cross-engine greedy tokens must also agree. A mismatch is a failed report;
parameters/tolerances are not adjusted after viewing its result.

Each campaign retains composed config, stack/model/profile identities, observed
package inventory, phase logs, raw generations/verdicts, adapter manifests,
training metrics and a machine-readable report. `PASS` requires both GPU phases
and `pip check`; only then is `environment.lock` exported. Preserve the entire
campaign directory as notebook output/downloads before ending a GPU session.
Stack identity uses package names/versions, hardware/runtime settings and hashes
of relative backend source files; editable installation paths stay in the audit
inventory and do not define execution identity.
To run the same gate through pytest, set `AUTOPOST_RUN_GPU_SPIKE=1` and optionally
`AUTOPOST_EXECUTION_PROFILE=execution/dev-qwen06b-t4.yaml`, then run
`python -m pytest -q -m gpu`. Ordinary CPU CI skips that opt-in test.

After a successful real run, review/export the resolved dependency lock, retain
the hardware/runtime report and update `environments/small-gpu-dev.toml` from
`PENDING_HARDWARE`. The CPU implementation alone cannot pass Sprint 5's GPU exit
gate, and Sprint 6 depends on that gate.
