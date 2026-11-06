# Sprint 1 audit

Audit date: 2026-11-06. Scope: roadmap version 1.1, Sprint 1; authoritative architecture Revision 3. Sprint 0's local exit gate and CPU/toolchain evidence are recorded in the repository's existing Sprint 0 audit and environment manifests.

| Requirement | Implementation / acceptance evidence |
|---|---|
| Durable IDs and location-independent refs | `contracts/refs.py`; ID constraints and ref round-trips |
| Versioned training/evaluation specs, results, statuses and errors | `contracts/jobs.py`, `errors.py`; union decoding, result invariants, tampered-ID rejection |
| Manifests and raw evaluation evidence | `contracts/manifests.py`, `evidence.py`; complete state component refs, independent raw-completion artifact refs |
| Canonical serialization and compatibility rules | `contracts/base.py`, `serialization.py`; full SHA-256, strict JSON, nested-version rejection, documented supported/rejected/migration policy |
| Stable job IDs exclude attempt metadata | Retry/timestamp identity tests for training and evaluation; seed and stack changes produce different IDs |
| No runtime/path leakage | Absolute-path, arbitrary-object, callable, nonfinite-number and unknown-field rejection; path-free `TrainState` |
| Split model and execution configuration | `ModelSpec`, `ExecutionProfile`; model, execution and data profiles composed by `config.py` |
| Scientific identity separate from topology | Config relocation/key-order tests; execution/topology/IS/lock changes invalidate stack identity |
| Executor, providers and store protocols | `contracts/protocols.py`; generic acceptance effects, immutable artifact lifecycle, distinct generation values |
| Capability descriptors only | `CapabilitySpec`, `CapabilityReport`, `CompatibilityError`; no runtime negotiation framework |
| Preregistration hashing migration | Canonical v2 snapshot/hash/lock validation; historical v1 freeze/verify tests; every component mutation rejected before side effects |
| Preserve scorer/statistics/gate | Existing tests and synthetic self-check unchanged |
| Preserve Sprint 0 toolchain probe | Probe translates split config to pinned TRL controls; regression test and real API inspection pass |

## Local acceptance results

- Final fast acceptance suite: **90 passed, 1 slow test deselected** with CPython 3.13.7 and the pinned CPU environment.
- Full CPU suite, including the unchanged long synthetic self-check: **79 passed in 704.84 seconds**. This ran while additional Sprint 1 integration tests were being added; those additions passed in the final 90-test fast run. The full check uses 1,500-run coverage/false-positive simulations and 400 trials per joint-power effect.
- Ruff: all checks passed on new contract modules and changed configuration, preregistration, CLI, probe and acceptance-test code.
- `pip check`: no broken requirements.
- Real pinned toolchain API probe: **PASS** for explicit configured controls, dependency lock, GRPO construction/fields, callbacks, metrics and PEFT hooks. Evidence: `docs/sprint1_toolchain_probe.json`. No CUDA device; GPU remains `PENDING_HARDWARE` and vLLM runtime remains `NOT_TESTED`.
- CLI profile/hash commands and root configuration compatibility: passed.
- `git diff --check`: passed.

Sprint 1's local exit gate passes. No dependency pins, calibration thresholds or scientific claims were inferred from these engineering results. Sprint 2 persistence implementations, Sprint 5 runtime capability checks, GPU execution, target model pins and scientific calibration remain outside this sprint's gate.
