# Sprint 5 implementation audit

Scope: roadmap 1.1, Sprint 5. Dependencies are available through the Sprint 2
stores; Sprints 3 and 4 are implemented. **The engineering implementation is
CPU-validated; Sprint 5's required real-GPU exit gate remains unpassed.**

| Requirement | Implementation / evidence |
|---|---|
| Local subprocess executor | Versioned specs/results, isolated installed worker, captured logs, atomic response publication and stable failures |
| Attempts and cancellation | Atomic metadata claim prevents duplicate execution; deadline, cancellation, spawn failure, backend failure, duplicate delivery and attempt retry tests use real child processes |
| Safe finalization | Worker outputs are journaled independently of state/budget acceptance; explicit store acceptance remains idempotent |
| Restart behavior | Completed responses are collectible after restart; ownership of active work is never guessed or duplicated |
| Runtime capability checks | Platform/Python/package pins, GPU capability/dtype, context/caps, LoRA, single-device layout, memory fraction, quantization/offload and explicit algorithm controls |
| Rollout provider | Lazy colocated vLLM backend, model/tokenizer pins, chat-template application, per-sample seeds and sampled-token log probabilities |
| Evaluation provider | Separate typed boundary, durable read identity and fresh engine calls without reference-output reuse |
| Stable adapter refs | Immutable state-artifact lookup/materialization inside backend; execution-stack checks when resolving metadata states |
| Runtime object isolation | Runtime imports/objects and physical adapter locators remain within execution backends; batch contracts round-trip with opaque LaTeX text |
| Dev configuration | Pinned Qwen3-0.6B revision/tokenizer, bounded context and generation caps, explicit vLLM memory fraction; separate bfloat16 and float16/T4 profiles |
| Reproducibility | Unvalidated direct-pin Linux GPU candidate; real campaign records package inventory, runtime and stack identities; resolved lock exported only after GPU phases and dependency consistency pass |
| Actual GPU checks | Opt-in adapter/save/export/reload/HF-vLLM generation phase and one-step TRL phase requiring emitted IS/mismatch metrics; implemented but not run here |
| Public commands | `execution fixture`, `execution capabilities`, `execution spike`, `scripts/probe_dev_model.py`, and functional `make spike` preflight |

## Validation

- Initial executor/provider/contract suite: **70 passed**.
- Full CPU suite after the deadline race correction: **257 passed, 1 slow test
  deselected**, in 143.88 seconds.
- Final wheel/CPU acceptance, including the additional artifact resolver test:
  **258 passed, 1 GPU test skipped, 1 slow test deselected**, in 155.75 seconds.
  The wheel was installed into the isolated CPU acceptance environment.
- After the final stack-identity normalization, all **16 targeted execution tests
  passed, 1 GPU test skipped**, both from source and from a rebuilt wheel
  (13.00 seconds for the wheel run). Final `pip check` and Ruff passed.
- Ruff: passed for new execution modules and changed contract/store/test code.
- Wheel build and dependency consistency: passed; no GPU packages or models were
  installed into the CPU environment.
- Live subprocess fixture: passed with a finalized state and a single budget
  charge. CPU-only preflight reports the unavailable Linux/CUDA/runtime stack.
  Its machine-readable summary is retained in `sprint5_capability_report.json`.

The timeout regression was found by running the full suite: delayed scheduling
could let a completed late response win against the watchdog. The executor now
records the deadline at submission and compares the worker's monotonic finish
time before journaling its result; a dedicated delayed-watchdog test covers it.

## Remaining hardware acceptance

`environments/small-gpu-dev.toml` remains `PENDING_HARDWARE`. This Windows CPU
machine has no detected NVIDIA runtime. No bf16/fp16 GPU behavior, vLLM kernel,
adapter parity, emitted training metrics or memory-fit result is claimed.
The candidate dependency file is not a validated lock.

On the actual Linux GPU, install/resolve the candidate in an isolated environment,
run the opt-in [campaign](EXECUTION.md), retain the full report/artifact/lock
directory and review/update the GPU environment identity only after `PASS`.
The T4 profile explicitly uses float16. If a dependency, driver, kernel or parity
check fails, preserve that failed report and revise the versioned development
stack explicitly. Sprint 6 must wait for Sprint 5's real compatibility exit gate.

The one-step training phase is a disposable API/metrics spike, not a production
trainer or a scientific result. Complete resumable training/state acceptance is
Sprint 6; evaluation suite/scorer orchestration is Sprint 7. Active-executor
ownership recovery is not silently implemented as replaying already claimed work.
