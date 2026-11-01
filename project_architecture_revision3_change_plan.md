# Autonomous Post-Training System — Revision 3 Documentation Change Plan

## Purpose

This document is a **documentation and architecture revision plan** for `post_training_documentation.md` Revision 2. It is written for the project's actual constraint: the system must be **fully useful on small models and limited compute today**, while the architecture should be capable of moving to larger models and distributed execution later **without rewriting the control plane, measurement semantics, lineage semantics, or decision logic**.

The central change is not to add distributed infrastructure now. The central change is to make the current single-node implementation the **reference execution backend** of a larger architecture whose resource-dependent parts are replaceable.

The target architectural claim should become:

> **The project implements model-improvement semantics independently from compute topology. V1 is a small-compute reference implementation. Larger-model and distributed execution replace execution-plane implementations while preserving the controller, measurement, provenance, gating, lineage, and experiment contracts.**

This document also includes the experimental-design corrections identified during review: run-level variance, curriculum quarantine, LoRA-scaling calibration, explicit importance-sampling configuration, resume semantics, and several documentation inconsistencies.

---

# 1. Executive recommendation

The existing design is already strong in the areas that matter most for long-term engineering value:

- training and measurement are deliberately separated;
- the champion is immutable;
- the controller is deterministic;
- full training state is persisted;
- claim-relevant reads are budgeted;
- raw completions are preserved;
- SQLite + Parquet are used instead of premature distributed infrastructure;
- future features enter through explicit interfaces;
- statistical power is validated before trusting the gate.

The main weakness is that **model semantics and execution topology are not separated strongly enough**. `ModelProfile`, `OnPolicyTrainer`, `EvalRunner`, local paths, vLLM colocate mode, subprocess launching, and storage assumptions are still close enough to the core architecture that scaling from one GPU to multiple GPUs or remote workers could become a structural rewrite.

Revision 3 should therefore introduce a clear distinction between:

1. **Control plane — durable semantics that should remain stable at every scale.**
2. **Execution plane — replaceable implementations selected according to available hardware.**

Revision 3 should also distinguish:

1. **Engineering validation on small models**, which can be completed now.
2. **Scientific validation on the target model**, which must be repeated when sufficient compute becomes available.

The small-model implementation should be treated as a **canonical reference implementation**, not a disposable prototype.

---

# 2. Priority list

## P0 — change before treating Revision 3 as authoritative

| ID | Required change | Why |
|---|---|---|
| P0-1 | Explicitly split the architecture into **control plane** and **execution plane** | Makes the scalability claim concrete rather than aspirational |
| P0-2 | Split `ModelProfile` into `ModelSpec` and `ExecutionProfile` | Model identity and hardware topology are different concerns |
| P0-3 | Add a `JobExecutor` boundary | Controller must not depend directly on local subprocess execution |
| P0-4 | Add `ArtifactStore` and `MetadataStore` abstractions | Prevent filesystem and SQLite assumptions from leaking into core logic |
| P0-5 | Add a `RolloutProvider` boundary | Colocated vLLM must be replaceable by remote rollout workers later |
| P0-6 | Define versioned, serializable `JobSpec` / `JobResult` types | Future remote execution should not require redesigning job semantics |
| P0-7 | Use durable IDs / refs rather than filesystem paths across interfaces | Storage location must not become part of system semantics |
| P0-8 | Make expensive jobs idempotent and document commit semantics | Required for safe retries locally and essential for distributed execution |
| P0-9 | Split validation into **reference-architecture validation** and **target-model scientific validation** | Small models can validate engineering but not target-model statistical sensitivity |
| P0-10 | Fix E1/E2 run-level variance | Item/cluster bootstrap does not capture training-seed variance |
| P0-11 | Change consensus quarantine from automatic exclusion to **flag + verify before exclusion** | A model can be confidently wrong on exactly the frontier examples worth training |
| P0-12 | Treat E3 LoRA interpolation as supplementary; include actual E1 checkpoints | Scaled adapters need not behave like real intermediate training states |
| P0-13 | Make importance-sampling behavior explicit configuration, never “library default” | Defaults are version-sensitive and part of algorithm semantics |
| P0-14 | Replace “exact resume” claims with “complete resumable state”; reserve bit-exactness separately | GPU generation and kernel scheduling may not reproduce bit-for-bit |
| P0-15 | Fix the statement “No task, label or test is generated by a model” | Hidden perturbation programs are model-authored but execution-labelled and human-reviewed |

## P1 — strongly recommended for the current implementation

| ID | Change | Why |
|---|---|---|
| P1-1 | Add `smoke`, `dev`, and `research` execution modes | Makes limited-compute development a first-class workflow |
| P1-2 | Add a replay/simulation execution backend | Controller and gate can be exhaustively tested without GPU use |
| P1-3 | Add contract tests for every replaceable backend | Demonstrates real substitutability |
| P1-4 | Add failure-injection tests | Proves crash/retry semantics before distribution makes failures harder |
| P1-5 | Add schema versioning and migrations | Required before SQLite metadata becomes durable across revisions |
| P1-6 | Add backend capability descriptors | Avoid hard-coding assumptions such as colocate, LoRA, TIS, quantization |
| P1-7 | Add transactional artifact-finalization semantics | Avoid DB state pointing at partial checkpoints and vice versa |
| P1-8 | Reclassify evaluation-cost risk from “closed” to “verified/partially controlled” | E1/E3 must prove the actual eval/train ratio |
| P1-9 | Add engineering success criteria separate from model-improvement success criteria | The project can succeed as an architecture before expensive scientific validation |

## P2 — document now, implement only when triggered

| ID | Deferred implementation | Trigger |
|---|---|---|
| P2-1 | PostgreSQL metadata backend | Multiple writers / remote execution |
| P2-2 | S3/MinIO/object-storage backend | Checkpoints/completions no longer fit comfortably on local NVMe |
| P2-3 | Remote/Slurm/Ray executor | Jobs must run outside one process/node |
| P2-4 | Remote vLLM rollout pool | Colocated rollout generation becomes the bottleneck |
| P2-5 | Formal alpha-spending/sequential testing | Gate cadence/read count grows enough that read budgets are no longer adequate |
| P2-6 | FSDP/DeepSpeed/tensor-parallel training backend | A target model does not fit or train efficiently with the local backend |

---

# 3. New architectural framing to add near the beginning

Add a new section immediately after the Executive Summary.

## Proposed new section: “Architecture boundary: control plane vs execution plane”

Use language close to the following:

> The system deliberately separates **model-improvement semantics** from **resource topology**. V1 runs on one machine and one GPU because that is the cheapest environment that exercises the architecture. This is a reference execution backend, not a claim that larger models will use the same physical topology.
>
> The **control plane** owns experiment semantics: controller state, promotion policy, measurement definitions, read budgets, lineage, provenance, pre-registration, tripwires, and durable identities. These semantics must not change merely because the model is larger or work runs on another machine.
>
> The **execution plane** owns resource-dependent mechanisms: model loading, rollout generation, optimizer execution, evaluation generation, verifier parallelism, job placement, checkpoint transport, and storage implementations. These implementations may change from local subprocesses to remote workers without changing control-plane semantics.
>
> Scalability in V1 therefore means **replaceable edges**, not pre-built distributed infrastructure. No core interface may require a local filesystem path, a CUDA device object, a live Python callback, or process-local mutable state to cross an execution boundary.

Add this diagram:

```text
                         CONTROL PLANE
┌───────────────────────────────────────────────────────────┐
│ Controller / state machine                                │
│ Gate / statistics / read budgets                         │
│ Lineage / champion semantics                             │
│ Pre-registration / provenance / tripwires                │
│ Data policy / prompt policy / experiment definitions      │
└──────────────────────────┬────────────────────────────────┘
                           │ versioned JobSpec / JobResult
                           │ durable StateRef / ArtifactRef
                           ▼
                         EXECUTION PLANE
┌───────────────────────────────────────────────────────────┐
│ JobExecutor                                               │
│ RolloutProvider / evaluation generation                  │
│ Trainer implementation / verifier workers                │
│ Model loading / device topology / quantization           │
│ MetadataStore / ArtifactStore implementations            │
└──────────────────────────┬────────────────────────────────┘
                           │
             ┌─────────────┴────────────────┐
             ▼                              ▼
       V1 reference backend            Future backend
       local subprocesses              remote workers
       one GPU                         multi-GPU / multi-node
       colocated vLLM                  rollout service
       SQLite                          PostgreSQL
       local NVMe                      object storage
```

---

# 4. Change the project identity from “small model vs target model” to three validation levels

The present documentation makes Qwen3-0.6B “plumbing only” and Qwen3-4B the point where almost all meaningful validation begins. That is scientifically correct but undersells what can be rigorously validated with limited compute.

Introduce these three modes throughout the document.

| Mode | Purpose | Typical environment | What it can establish |
|---|---|---|---|
| `smoke` | CI, schema, controller branches, storage, failure handling | CPU or tiny model / prerecorded results | Software correctness only |
| `dev` | Real end-to-end RL and evaluation | 0.6B–1.5B, 16–24 GB GPU | Integration correctness and operational behavior |
| `research` | Scientific claim | target model / larger compute | Target-model effect, gate calibration, reproducibility |

Replace wording that implies the small model is only useful for catching crashes. It can validate much more:

- data split invariants;
- measurement/training separation;
- scorer/verifier integration;
- state persistence;
- checkpoint/restart semantics;
- ledger and read budgets;
- promotion/continue/restore/halt branches;
- artifact lineage;
- job idempotency;
- replay;
- failure recovery;
- reports;
- contract compliance.

It still **cannot** validate:

- target-model learning effect size;
- target-model completion lengths;
- gate power at the target-model floor;
- target-model false-halt behavior;
- target-model throughput or memory;
- target-model seed variance.

This distinction should be repeated in §§1, 4, 11, 25, 28, 29, 35, and 36.

---

# 5. Required interface changes

## 5.1 Replace `ModelProfile` with `ModelSpec` + `ExecutionProfile`

### Problem

The current `ModelProfile` mixes model semantics with execution choices. As model size grows, it is likely to accumulate device maps, quantization, tensor parallelism, memory utilization, offloading, attention backends, and distributed settings.

### Add

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    tokenizer_id: str | None
    chat_template_args: dict[str, object]
    max_context_tokens: int
    lora_targets: tuple[str, ...]
    mode: str

@dataclass(frozen=True)
class ExecutionProfile:
    backend: str
    dtype: str
    quantization: str | None
    generation_backend: str
    training_backend: str
    max_concurrency: int | None
    tensor_parallel_size: int
    gradient_checkpointing: bool
    cpu_offload: bool
    backend_options: dict[str, object]
```

### Documentation invariant

> `ModelSpec` says **what model is being experimented on**. `ExecutionProfile` says **how that model is executed on the available hardware**. Scientific identity must not depend on incidental hardware topology, but execution parameters that can change numerical behavior are included in the stack hash.

---

## 5.2 Add `JobExecutor`

### Purpose

The controller should schedule typed work; it should not know that V1 uses `subprocess.run`.

### Contract

```python
class JobExecutor(Protocol):
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def result(self, handle: JobHandle) -> JobResult: ...
    def cancel(self, handle: JobHandle) -> None: ...
```

V1:

```text
LocalSubprocessExecutor
```

Future:

```text
SlurmExecutor
RayExecutor
RemoteQueueExecutor
```

Do not implement the future executors yet.

---

## 5.3 Add versioned, serializable job schemas

Every expensive operation should be representable as a durable specification rather than a graph of live Python objects.

Recommended types:

```text
TrainingJobSpec
EvaluationJobSpec
VerificationJobSpec (optional in V1)
TrainingJobResult
EvaluationJobResult
```

Every spec/result should contain:

- `schema_version`;
- deterministic `job_id` or inputs from which one is derived;
- config hash;
- stack hash;
- model spec ID;
- execution profile ID;
- input artifact/state refs;
- output artifact refs;
- timestamps and attempt number;
- code revision;
- error classification on failure.

### Important rule

Do not put these across execution boundaries:

- local absolute filesystem paths;
- CUDA objects;
- tokenizer/model instances;
- callables/lambdas;
- open database connections;
- process-local locks.

---

## 5.4 Add `RolloutProvider`

The current colocated-vLLM path should become one implementation of a rollout contract.

```python
class RolloutProvider(Protocol):
    def generate(self, request: RolloutRequest) -> RolloutBatch: ...
```

The result should be a stable project-owned type, not a vLLM-owned object:

```python
@dataclass
class RolloutBatch:
    batch_id: str
    prompt_ids: list[str]
    completion_texts: list[str]
    token_ids: object
    sampled_logprobs: object
    truncated: list[bool]
    generation_metadata: dict[str, object]
```

V1 implementation:

```text
ColocatedVLLMRolloutProvider
```

Future:

```text
RemoteVLLMRolloutProvider
```

The on-policy trainer may still own policy updates, but it must not depend directly on vLLM-specific return types throughout the codebase.

---

## 5.5 Add a separate evaluation-generation boundary

Evaluation generation does not have the same correctness requirements as on-policy rollout generation. Avoid forcing both through an overly generic abstraction.

Recommended:

```text
EvalGenerationProvider
  ├── LocalVLLMEvalProvider
  └── RemoteEvalProvider   # future
```

Both rollout and evaluation providers may share internal vLLM utilities, but their contracts remain semantically distinct.

---

## 5.6 Split metadata and artifact persistence

Add two contracts.

```text
MetadataStore
  └── SQLiteMetadataStore       # V1
  └── PostgresMetadataStore     # future

ArtifactStore
  └── LocalArtifactStore        # V1
  └── ObjectArtifactStore       # future
```

### MetadataStore owns

- runs;
- state metadata;
- lineage edges;
- budgets;
- decisions;
- looks;
- prompt statistics;
- events;
- job records;
- preregistration hashes.

### ArtifactStore owns

- model/adaptor states;
- optimizer states;
- rollout files;
- evaluation completions;
- datasets/snapshots;
- reports;
- manifests.

### Core invariant

The rest of the project receives `StateRef`, `ArtifactRef`, `DatasetRef`, etc.; it does not construct physical paths.

---

## 5.7 Add stable references

Introduce project-owned identifiers:

```text
RunId
JobId
StateId / StateRef
ArtifactId / ArtifactRef
DatasetId / DatasetRef
PromptId
ClusterId
EvalReadId
ConfigId
StackId
VerifierVersion
ScorerVersion
PreregistrationId
```

Prefer content hashes where appropriate, especially for immutable manifests and datasets.

A logical checkpoint should not be identified by `runs/states/<path>` outside the storage implementation.

---

## 5.8 Add backend capabilities

Do not encode assumptions such as “all backends support colocated vLLM” in controller logic.

A simple capability record is enough:

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_lora: bool
    supports_quantization: bool
    supports_tis: bool
    supports_colocate: bool
    supports_remote_generation: bool
    supports_full_state_resume: bool
    max_context_tokens: int | None
```

The compatibility spike should check required capabilities before a run.

---

# 6. Idempotency and commit semantics — add as a first-class architectural principle

This is important even on one machine and becomes mandatory under distributed execution.

Add a new principle:

> **Expensive work is idempotent.** Every training segment, evaluation read, and artifact-producing job has a stable identity derived from its immutable inputs. Retrying a job after a crash must not create a second logical training segment, spend a read budget twice, or produce two conflicting lineage transitions.

Document a commit protocol such as:

1. Create/claim job record as `RUNNING`.
2. Write outputs to temporary artifact names.
3. Validate hashes/manifests.
4. Atomically finalize artifacts in `ArtifactStore`.
5. Commit `JobResult` and artifact refs in `MetadataStore`.
6. Only then permit controller state transition / budget consumption.

For claim-relevant evaluation, define exactly when a **look is spent**. Recommended: the look is reserved before generation and finalized once a valid result exists; crashes should not allow unlimited retries to fish for a favorable sample.

Add tests for:

- job submitted twice;
- result delivered twice;
- process killed after artifact write but before DB commit;
- DB job record written but artifact incomplete;
- retry after timeout;
- stale result arrives after retry has completed.

---

# 7. Add a replay/simulation backend

The controller is important enough that it should not require a GPU to test.

Add:

```text
ReplayJobExecutor
```

or

```text
SimulationBackend
```

It should consume recorded or synthetic `TrainingJobResult` / `EvaluationJobResult` objects and exercise the same controller path as real execution.

Use it to cover:

- `PROMOTE`;
- `CONTINUE`;
- `RESTORE`;
- `HALT`;
- repeated confident regressions;
- read-budget exhaustion;
- human-change refresh;
- stack change;
- checker bug invalidation;
- pool exhaustion;
- tripwire false halt;
- crash/retry.

This should become part of ordinary CI.

---

# 8. Add contract tests

Do not merely state that backends are replaceable. Prove it.

Recommended test suites:

```text
ModelSpecContract
ExecutionProfileValidation
MetadataStoreContract
ArtifactStoreContract
JobExecutorContract
RolloutProviderContract
EvalGenerationProviderContract
LineageStoreContract
```

Examples:

```text
SQLiteMetadataStore passes MetadataStoreContract
LocalArtifactStore passes ArtifactStoreContract
LocalSubprocessExecutor passes JobExecutorContract
ColocatedVLLMRolloutProvider passes RolloutProviderContract
```

Future replacements must pass the same suites.

This is one of the strongest ways to demonstrate architecture quality with limited compute.

---

# 9. Experimental-design corrections

## 9.1 E1/E2 do not currently capture run-level variance

### Problem

E1 is one GRPO training trajectory and E2 is one random-reward trajectory. A cluster bootstrap over evaluation items estimates evaluation uncertainty, but it does not estimate variation caused by:

- rollout randomness;
- training data order;
- optimization stochasticity;
- generation seed;
- stochastic training trajectory.

Therefore a very precise item-level difference between one GRPO run and one control run can still be a seed-specific result.

### Change

For the `research` profile, require at least **two independent training seeds per E1/E2 condition**, or use a staged confirmation rule:

1. one full E1 and one full E2 trajectory;
2. if the effect is below the floor or clearly absent, stop;
3. if it is promising, run at least one additional independent seed per condition to confirm direction and rough magnitude.

If compute permits, three seeds per condition is preferable.

If this cannot be afforded, change Level 1 wording from:

> “A reward-driven learning signal exists.”

To:

> “One target-model run provides evidence of a reward-associated learning signal; run-level variance remains unestimated.”

Do not let an item bootstrap imply training-run replication.

---

## 9.2 Change the consensus quarantine rule

Current behavior automatically excludes a prompt when model consensus repeatedly disagrees with the reference answer.

Change it to:

> Consensus disagreement **flags** an item as a suspected label error. It is excluded only after the reference can be independently recomputed, verified from provenance, or human-audited. Until then, it remains eligible subject to ordinary band/exposure rules, or is placed in a temporary “suspect” pool that cannot silently disappear from the dataset.

Reason: a confidently wrong model can identify a hard systematic failure, not only a bad label.

Update D10, §14.4, Appendix A, failure mode F10, and the final decisions table.

---

## 9.3 E3 LoRA scaling should be supplementary calibration

Keep adapter scaling because it is compute-efficient, but do not treat a scaled adapter as equivalent to a real intermediate training state.

Change E3 to combine:

- actual E1 checkpoints at multiple steps;
- scaled versions of one or more adapters for denser interpolation;
- comparison of gate behavior at similar measured effect sizes from real checkpoints vs scaled adapters.

Add a pass condition:

> The scaled-adapter power curve must not materially disagree with the power observed on available real checkpoints at comparable pilot-set effect sizes. If it does, set sizes are derived conservatively from real checkpoints.

---

## 9.4 Make importance-sampling configuration explicit

Remove wording that says TIS is enabled because it is a default.

Record the exact pinned configuration in `base.yaml` / `ExecutionProfile`, including:

- importance-sampling mode;
- granularity if configurable;
- truncation/cap value;
- whether correction is enabled for rollout generation;
- any library option that changes its behavior.

Add a compatibility-spike assertion that the runtime configuration matches the preregistered value.

The rule should be:

> **Algorithmically relevant behavior is explicit configuration, never an assumed library default.**

---

## 9.5 Soften “exact resume”

Replace claims such as “exact resume” with:

> **Complete resumable state:** all project-controlled state needed to continue the run without manual reconstruction is persisted: model/adaptor state, optimizer, scheduler, RNG states that are available, sampler/data cursor, config, stack identity, and lineage metadata.

Retain a separate statement:

> Bit-exact continuation of future generated tokens is not guaranteed because GPU kernels, generation scheduling, and external libraries may be nondeterministic. Reproducibility is established through independent seeded runs and recorded manifests, not by assuming bit-for-bit replay.

Update §§3, 7, 20, 23, 31, and tests named `state_roundtrip`.

---

## 9.6 Fix the model-generated-content contradiction

The Executive Summary currently says no task, label, or test is generated by a model, while hidden perturbation programs are written with a strong external model and reviewed by a human.

Replace the broad statement with:

> **No unverified model-generated task, label, or test enters training or measurement.** Public training tasks use curated references. Hidden perturbation programs may be drafted by an external model, but every program is human-reviewed, constrained by executable invariants, and its labels are produced by deterministic execution rather than accepted from the generator.

Use the same terminology in §6, D07, and §12.2.

---

## 9.7 Reclassify evaluation-cost risk

Failure mode F17 currently marks “evaluation cost exceeds training cost” as closed.

Change status to `V` or `P/V` until the measured E1/E3 evaluation/train token and GPU-time ratios are available.

Add a pre-registered budget metric:

```text
eval_gpu_seconds / total_gpu_seconds
```

and optionally:

```text
eval_generated_tokens / train_generated_tokens
```

Define a threshold at which the design is rescaled before Phase II.

---

## 9.8 Keep repeated-look accounting as V1, but document the upgrade path

The current read-budget approach is acceptable for a bounded V1. Add one sentence:

> If the number of adaptive gate decisions grows beyond the preregistered bounded regime, replace the union-bound/read-budget approximation with a formal sequential-testing or alpha-spending design before increasing autonomy.

Do not implement this now.

---

# 10. Section-by-section documentation changes

## Header / metadata block

### Change

Replace:

```text
Development model: Qwen3-0.6B — plumbing validation only
Target model: Qwen3-4B-Instruct-2507 — ... all instrument validation happens on it
Future ceiling: ~9–12B
```

With something like:

```text
Reference development profile: Qwen3-0.6B — full architecture/integration validation on limited compute
Scientific target profile: Qwen3-4B-Instruct-2507 — target-model statistical and learning validation when compute is available
Future scale envelope: ~9–12B initially; execution topology may change without changing control-plane semantics
Execution modes: smoke | dev | research
```

Also change decision numbering from D01–D30 to the new final range after adding Revision 3 decisions.

Add `execution/` schemas and backend contracts to Companion Files.

---

## §1 Executive Summary

Add these points explicitly:

1. V1 is a **reference implementation on limited compute**.
2. Small-model results validate architecture but do not calibrate target-model statistical power.
3. The architecture is divided into stable control-plane semantics and replaceable execution-plane implementations.
4. Distributed infrastructure is deliberately absent, but local execution, storage, generation, and model loading are behind contracts.
5. Scientific claims are made only under the `research` profile after target-model Phase I.
6. Replace the “no model-generated task/test” sentence as described above.

Recommended paragraph:

> The V1 implementation is intentionally local and small-compute. Its purpose is to make the complete control plane executable and testable now: data invariants, measurement independence, state transitions, lineage, budgets, rollback, job replay, crash recovery, and reports. Larger-model runs do not require a new controller; they replace execution-plane implementations and repeat target-model calibration before scientific claims are made.

---

## §2 Problem Definition

Add new subsections:

### 2.8 Small-compute development must not hard-code small-compute assumptions

State that the current hardware constraint creates a second engineering problem: prove the full semantics on cheap hardware while preventing local process, local filesystem, one-GPU, or vLLM-colocate assumptions from becoming architecture contracts.

### 2.9 Training-run variance is distinct from evaluation uncertainty

Explain that bootstrap uncertainty over items does not substitute for multiple independent training trajectories.

---

## §3 Requirements

Add requirements:

| ID | Requirement |
|---|---|
| R15 | The control plane must be independent of whether work executes locally or remotely. |
| R16 | Expensive jobs must be serializable, versioned, durable, and idempotent. |
| R17 | Persistent entities are referenced by stable IDs/refs, not physical filesystem paths. |
| R18 | Model identity must be separate from hardware/execution configuration. |
| R19 | Every replaceable execution/storage backend must pass a common contract suite. |
| R20 | The complete controller must be testable using replay/synthetic results without GPU execution. |

Change R5 wording from “exact resume” to “resume without manual reconstruction from complete persisted state.”

---

## §4 Constraints

Rewrite the compute constraint so it describes **current implementation**, not architectural ceiling:

> Current development must run on one node and one affordable GPU. This constrains the V1 execution backend, not the control-plane architecture.

Replace the single model-size row with:

```text
Reference model: 0.6B–1.5B depending available hardware
Scientific target: 4B class when compute is available
Initial scale envelope: 9–12B with replaceable execution topology
```

Move the 80 GB assumption out of a general architecture assumption and label it specifically as a **research execution profile prior**.

Add assumptions:

- local execution is sufficient for V1 reference validation;
- all job types are serializable before remote execution is introduced;
- state/artifact refs remain stable across storage backend migration.

---

## §5 Design Goals

Add after simplicity:

> **Scale by replacement, not conditional complexity.** Resource-specific behavior belongs behind execution interfaces. The single-GPU path remains simple; future backends replace implementations rather than adding distributed branches throughout controller logic.

Consider moving Extensibility above Autonomy if architecture quality is a primary project objective.

---

## §6 Non-Goals

Clarify:

> Distributed infrastructure is not a V1 implementation goal. **Distributed-compatible contracts are a V1 architecture goal.**

Add non-goals:

- implementing Kubernetes/Ray/Slurm before a real resource trigger;
- generic “universal trainer/evaluator” abstractions with no second implementation;
- bit-exact replay of GPU generation;
- proving large-model learning behavior from the small-model reference run.

Update “model-generated tasks” language to “unverified model-generated tasks/labels/tests.”

---

## §7 Architectural Principles

Keep P1–P10 and add:

### P11. Separate control plane from execution plane

Controller/gate/lineage semantics cannot depend on local process topology.

### P12. Cross execution boundaries only with versioned serializable contracts

No live Python/runtime objects cross a job boundary.

### P13. Expensive work is idempotent

Retries do not double-spend budgets or create duplicate logical states.

### P14. Persistent identities are location-independent

State and artifact identity is not a local path.

### P15. The small-model implementation is the reference semantics

Future distributed implementations must match its contract behavior; they do not redefine the controller.

### P16. Algorithmically relevant library behavior is explicit

No scientific decision depends on an undocumented/default library setting.

Also change P2 language from “checkpoints are full, resumable states” if it says “exact” anywhere.

---

## §8 Final Architecture

Add components:

| Component | Responsibility | V1 implementation | Future implementation |
|---|---|---|---|
| `ModelSpec` | Model identity and semantic rendering | YAML | same |
| `ExecutionProfile` | Hardware/backend execution policy | local single-GPU YAML | multi-GPU/remote profiles |
| `JobExecutor` | Execute durable job specs | local subprocess | Slurm/Ray/remote queue |
| `RolloutProvider` | On-policy generation contract | colocated vLLM | remote rollout workers |
| `EvalGenerationProvider` | Evaluation generation | local vLLM | remote inference service |
| `MetadataStore` | Durable metadata and ledger | SQLite | PostgreSQL |
| `ArtifactStore` | Immutable large artifacts | local filesystem | object storage |
| Replay backend | Controller simulation | recorded/synthetic results | same |

Change `ModelProfile` references accordingly.

---

## §9 Architecture Diagram

Replace the existing diagram with a two-layer diagram or add a second diagram showing:

```text
CONTROL PLANE
  orchestrator/controller
  gate/stats
  ledger/lineage/budgets
  prompt/data policy
          |
     JobSpec / refs
          |
EXECUTION PLANE
  JobExecutor
  trainer
  rollout provider
  eval provider
  verifier pool
  stores
```

Then show two execution implementations:

```text
Reference: local subprocess + one GPU + SQLite + local NVMe
Future: remote jobs + multi-GPU + Postgres + object storage
```

The current detailed training/measurement diagram can remain as the **reference backend data flow**.

---

## §10 Core Learning Loop

Replace direct calls conceptually like:

```python
trainer.train(...)
```

with orchestration through a durable job spec:

```python
job = TrainingJobSpec(...)
handle = executor.submit(job)
result = executor.result(handle)
```

Do the same for monitor/gate evaluation.

The controller still decides the same outcomes; only execution is abstracted.

Add idempotency behavior around each segment.

The pseudocode should use `StateRef` rather than raw path/state objects where possible.

---

## §11 Model Strategy

### D01

Reframe the small model as:

> **Reference development model:** validates architecture and end-to-end execution but not target-model effect size.

Reframe 4B as:

> **Scientific target:** target-model calibration and claims are deferred until sufficient compute is available; no architecture milestone depends on already owning that compute.

### Replace current `ModelProfile` subsection

Create:

```text
11.1 ModelSpec
11.2 ExecutionProfile
11.3 Compatibility/capability spike
```

The compatibility spike should verify:

- model can load under the chosen execution profile;
- required backend capabilities exist;
- LoRA attach/serve works if applicable;
- generation/training compatibility;
- explicit IS configuration matches preregistration;
- peak memory and throughput;
- export parity.

---

## §12 Data and Task Generation

Fix the model-generated-content sentence.

Keep the existing data architecture.

Add stable IDs for:

- source revision;
- normalized task;
- cluster;
- split assignment;
- hidden program;
- generated hidden instance.

Ensure split membership is metadata, not inferred from physical directory location.

---

## §13 Replay

The current “No replay” decision is about **training-data replay**. Avoid confusion with the new **controller execution replay backend**.

Rename this section or add a note:

> “Replay” here means training replay/curriculum replay. Controller/job-result replay for testing is a separate infrastructure capability and is part of V1.

---

## §14 Curriculum

Change D10 and §14.4:

- consensus disagreement becomes `suspected_label_error`;
- exclusion requires independent confirmation;
- preserve audit trail for all suspected items;
- never silently remove a hard item solely because the model agrees with itself.

Update Appendix A quarantine config.

---

## §15 Rollout

Change D11 so that “colocated vLLM” is explicitly the **V1 `RolloutProvider` implementation**, not the architecture.

Add:

> The trainer consumes project-owned `RolloutBatch` structures. vLLM-specific objects are converted at the provider boundary.

Make IS/TIS settings explicit, not default-derived.

Change future extension wording from “separate rollout workers behind `OnPolicyTrainer`” to “replace `RolloutProvider` while preserving `OnPolicyTrainer` semantics.”

---

## §16 Verification

Keep `TrainVerifier` and `EvalScorer` semantic interfaces.

Clarify that concurrency/execution is separate from the verifier semantics:

```text
TrainVerifier = correctness contract
VerifierPool / execution backend = where/how it runs
```

Do not turn `TrainVerifier` itself into a remote-service interface yet.

---

## §17 Reward

No major architecture change.

Ensure `RewardFunction` receives durable task/completion/verdict values and is deterministic for a given version.

Persist `reward_version` or include it in the stack/config hash.

---

## §18 RL Training

Add the run-level variance limitation and research-profile replication rule.

Replace any library-default assumptions with explicit configuration.

Document that the local trainer may use colocated execution, but the controller sees only `TrainingJobSpec` / `TrainingJobResult`.

Add `execution_profile_id` to every training state manifest.

---

## §19 Continual Learning

No major structural change.

Add one caveat:

> Monitor thresholds calibrated on the small-model development profile are not automatically carried to a new model size. Model/domain changes repeat the required calibration subset.

---

## §20 Champion/Candidate Management

Replace filesystem-oriented state language with logical state references.

State manifest should include at minimum:

```text
state_id
parent_state_id
artifact_ref
model_spec_id
execution_profile_id
config_hash
stack_hash
verifier_version
scorer_version
training_slice_id
job_id
```

Do not require the physical checkpoint layout to be part of lineage semantics.

---

## §21 Evaluation

Add explicit separation of uncertainty sources:

1. evaluation-sample/item uncertainty — cluster bootstrap;
2. training-run uncertainty — independent training seeds.

Update E3 as described above.

Reclassify evaluation-cost risk.

Add a note that target-model gate sizes are not inferred from Stage A.

Keep small-model gate runs for software validation only.

---

## §22 Promotion

No change to the mathematical decision logic unless target-model Phase I changes the calibration.

Add:

> The gate accepts durable evaluation results; it must not call generation backends directly. Therefore promotion semantics are invariant to local vs distributed evaluation execution.

---

## §23 Rollback

Replace “exact resume” terminology.

Add atomicity requirements:

- a state is eligible for lineage transition only after artifact finalization and manifest validation;
- incomplete states are garbage-collectable but never selectable as champion;
- retries with the same logical job ID must not create a second lineage edge.

Keep raw-completion rescoring procedure.

---

## §24 Recovery

Add execution failures:

| Failure | Automatic behavior |
|---|---|
| Worker/process dies before finalized result | retry same logical job ID according to retry policy |
| Duplicate result | ignore if identical; halt on conflicting result for same job ID |
| Partial artifact | never expose as finalized `ArtifactRef` |
| Stale result from superseded attempt | record but do not apply |

Keep human-change budget semantics.

---

## §25 Compute Strategy

Reframe this section around **execution profiles**, not one universal hardware floor.

Recommended table:

| Profile | Model class | Hardware prior | Goal |
|---|---|---|---|
| smoke | synthetic/tiny | CPU or minimal GPU | software/contract tests |
| dev | 0.6B–1.5B | 16–24 GB | real end-to-end system validation |
| research-4b | ~4B | ~80 GB-class prior until measured | scientific target calibration |
| future-large | 9–12B | execution-profile dependent; likely multi-GPU or larger-memory GPU | future scientific runs |

Keep the throughput model, but make all throughput measurements fields of the `ExecutionProfile`/run manifest.

Add compute reporting by job type and backend.

Do not state that 80 GB is the floor for the **system**; state that it is the current prior for the **4B research execution profile**.

---

## §26 Infrastructure

This section requires the largest rewrite.

### Replace D20

Rename to:

> **D20 — Local reference execution via `JobExecutor`**

Chosen:

> V1 uses `LocalSubprocessExecutor` on one node. The controller communicates only through versioned job specs/results. Remote execution is deferred.

### Replace/extend D21

Rename to:

> **D21 — MetadataStore + ArtifactStore; SQLite/local filesystem in V1**

Chosen:

> V1 uses SQLite and local Parquet/filesystem artifacts behind storage contracts. Logical IDs are location-independent.

Add schema migrations and `schema_version`.

### Add new decisions

- `D31 — Control plane / execution plane separation`
- `D32 — ModelSpec separate from ExecutionProfile`
- `D33 — Versioned serializable JobSpec / JobResult`
- `D34 — Idempotent jobs and atomic artifact finalization`
- `D35 — RolloutProvider and EvalGenerationProvider boundaries`
- `D36 — Location-independent MetadataStore / ArtifactStore refs`
- `D37 — Replay backend and contract/failure-injection tests`

### Security additions

When remote execution is introduced:

- credentials are passed through the execution environment, not serialized into job specs;
- job specs may reference secret names/roles but never contain raw credentials;
- remote workers validate schema and allowed job type before execution.

Document this now; implement only when remote execution exists.

---

## §27 Interfaces

Replace the table with the expanded interface map:

| Interface | Meaning | V1 | Future |
|---|---|---|---|
| `ModelSpec` | model semantic identity | Qwen small + 4B spec | larger model specs |
| `ExecutionProfile` | resource/backend configuration | one-GPU local | multi-GPU/remote |
| `JobExecutor` | job placement/lifecycle | subprocess | Slurm/Ray/remote |
| `RolloutProvider` | on-policy generation | colocated vLLM | rollout service |
| `EvalGenerationProvider` | evaluation generation | local vLLM | inference workers |
| `MetadataStore` | structured durable state | SQLite | PostgreSQL |
| `ArtifactStore` | large immutable artifacts | local filesystem | object storage |
| `TaskSource` / `IngestStage` | data | existing | teacher/procedural |
| `TrainVerifier` | reward correctness semantics | math-verify | code verifier |
| `EvalScorer` | measurement semantics | exact scorer | code scorer |
| `RewardFunction` | reward semantics | binary | future variants |
| `OnPolicyTrainer` | learning algorithm | TRL GRPO | other RL |
| `OfflineTrainer` | offline learning | RFT | distillation |
| `EvalSuite` | measurement definition | existing | more domains |
| `PromotionPolicy` | gate semantics | `DefaultGate` | multi-domain |
| `PromptSampler` | curriculum policy | existing | future samplers |
| `LineageStore` | logical model lineage | metadata + state refs | same semantics |
| `Tripwire` | health/capability stop condition | existing | domain-specific |

Add enforced invariants:

6. Core interfaces exchange project-owned serializable values or durable refs, not backend-native runtime objects.
7. Controller logic does not branch on “local vs remote.”
8. Every finalized expensive job has one logical job ID and one accepted result.
9. Physical paths do not cross the storage boundary.
10. Every backend implementation passes its contract suite.

---

## §28 V1 Scope

Rewrite V1 as **reference architecture V1**, not “4B or nothing.”

### V1 must include now

- control-plane/execution-plane separation;
- small-model real end-to-end run;
- local executor;
- local metadata/artifact stores;
- rollout/eval provider boundaries;
- typed job specs/results;
- stable IDs;
- replay backend;
- contract tests;
- failure injection;
- existing measurement and controller logic.

### Research validation when compute is available

- target model E1–E9 or revised subset;
- target-model set sizing;
- target-model memory/throughput;
- target-model seed replication;
- sealed claim runs.

This prevents the project from being described as incomplete merely because the expensive target-model experiment is deferred.

---

## §29 Implementation Plan

### 29.1 Repository tree

Change to something closer to:

```text
autopost/
├── configs/
│   ├── experiment/
│   ├── model/                 # ModelSpec
│   ├── execution/             # ExecutionProfile
│   ├── data/
│   └── prereg/
├── src/autopost/
│   ├── contracts/
│   │   ├── jobs.py
│   │   ├── refs.py
│   │   ├── results.py
│   │   └── capabilities.py
│   ├── execution/
│   │   ├── executor.py
│   │   ├── local_subprocess.py
│   │   ├── replay.py
│   │   ├── rollout.py
│   │   └── eval_generation.py
│   ├── storage/
│   │   ├── metadata.py
│   │   ├── sqlite.py
│   │   ├── artifacts.py
│   │   └── local_artifacts.py
│   ├── models/
│   │   ├── spec.py
│   │   ├── loading.py
│   │   └── export.py
│   ├── controller/
│   ├── train/
│   ├── eval/
│   ├── verify/
│   ├── tasks/
│   ├── gate/
│   └── obs/
├── tests/
│   ├── contracts/
│   ├── failure_injection/
│   ├── replay/
│   └── ...existing tests...
```

Do not implement remote backends in this repository yet unless needed.

### Split the implementation plan

#### Phase 0 — architecture/reference implementation

Can be completed on small compute.

Acceptance:

- contracts defined;
- local implementations pass contract suites;
- job specs round-trip through serialization;
- no core component requires local path types;
- replay backend covers all controller outcomes;
- failure-injection suite passes;
- end-to-end small-model segment succeeds;
- crash/resume succeeds;
- duplicate job result cannot double-apply.

#### Phase I-A — small-model systems validation

Run real end-to-end training with the dev profile. Validate plumbing and operational semantics only.

#### Phase I-B — target-model scientific validation

When compute is available, run revised E1–E9 on the research profile. Only this phase calibrates the gate for scientific claims.

#### Phase II — gated target-model loop

Unchanged in scientific purpose, but executes through the stable job/storage interfaces.

---

## §29.2 Experiments E1–E9

Change:

- E1/E2: add run-level seed replication/confirmation rule;
- E3: include actual checkpoints plus adapter scaling;
- E5: run through the same replay/controller path used in CI;
- E9: retain the explicit caveat that one arm/seed does not estimate run-level variance; preferably apply the same staged replication rule if trainer choice is close.

Add a cheap **E0 architecture validation** before target-model E1:

```text
E0 — reference-backend validation
- local executor job round-trip
- checkpoint crash/resume
- artifact/metadata atomicity
- duplicate job idempotency
- replay all controller branches
- backend contract suites
```

E0 is not evidence about model improvement.

---

## §30 Failure Modes

Add the following rows.

| ID | Failure mode | Mitigation | Status initially |
|---|---|---|---|
| F44 | Controller coupled to local subprocess semantics | `JobExecutor`; controller accepts job results only | V |
| F45 | Local paths leak into durable state | `ArtifactRef` / `StateRef`; contract tests | V |
| F46 | Retry applies one logical job twice | deterministic job IDs; idempotent result application | V |
| F47 | DB and artifact store disagree after crash | finalize-then-commit protocol; reconciliation test | V |
| F48 | Model identity mixed with hardware config | `ModelSpec` / `ExecutionProfile` split | C after refactor |
| F49 | Evaluation bootstrap ignores training-run variance | replicated/staged E1/E2 seeds; explicit limitation | V |
| F50 | LoRA scaling gives misleading E3 power curve | compare with real checkpoints | V |
| F51 | Consensus quarantine removes valid hard examples | flag then independent verification | C after rule change |
| F52 | Backend/library default silently changes algorithm | explicit algorithmic configuration; stack hash | C/V |
| F53 | Duplicate/out-of-order remote result later changes lineage | job attempt/version rules; stale-result rejection | D/V |
| F54 | Storage migration changes logical identities | location-independent refs; contract tests | D/V |
| F55 | Small-model success mistaken for target-model validation | separate dev/research success levels | C |

Change F17 from `C` to `V` or `P/V`.

Change F10 mitigation after quarantine revision.

---

## §31 Solved / Partially Solved / Unsolved

Change:

```text
Exact resume — Solved
```

to:

```text
Complete operational resume — Solved when round-trip and crash tests pass
Bit-exact future token reproduction — Unsolved by design / not required
```

Add:

```text
Execution-topology independence — Partially solved until contracts and replay tests pass
Backend substitutability — Demonstrated when two implementations or contract harnesses pass
Target-model scientific validity — Deferred until research-profile Phase I
```

---

## §32 Trade-offs

Add:

| Trade-off | Chosen side | Cost | Reason |
|---|---|---|---|
| Premature distribution vs scalable boundaries | Local implementations + distributed-compatible contracts | Future backend work still required | Avoid infrastructure with no current operational need |
| Generic framework vs evidence-driven interfaces | Narrow contracts only | Some interfaces may evolve | Avoid speculative abstraction |
| Path convenience vs location-independent refs | refs/IDs | slightly more plumbing | Enables storage migration and remote jobs |
| One-off scripts vs idempotent jobs | idempotent jobs | bookkeeping complexity | Safe retry and future distribution |

---

## §33 Deferred Components

Add:

```text
PostgreSQL implementation — interface exists now; implement at multiple-writer trigger
Object-store implementation — interface exists now; implement when local storage is limiting
Remote executor — contract exists now; implement when jobs leave one node
Remote rollout service — provider contract exists now; implement when colocate is limiting
Formal sequential alpha spending — bounded read budgets remain V1
```

Make a distinction between **deferred implementation** and **interface that must exist now**.

---

## §34 Future Extensions

Rewrite M5 and M10.

### M5 Larger model

Replace:

> “Larger model → `ModelProfile`; spike; QLoRA if needed.”

With:

> “Larger model → new `ModelSpec` plus an appropriate `ExecutionProfile`. The control plane remains unchanged. The execution profile may select quantization, larger-memory hardware, tensor parallelism, or a different trainer/generation backend. A reduced target-model Phase I is repeated.”

### M10 Distributed execution

Replace:

> “Rollout workers / second node → `OnPolicyTrainer`, `EvalRunner` internals; Postgres and object storage.”

With:

> “Distributed execution replaces `JobExecutor`, `RolloutProvider`, `EvalGenerationProvider`, `MetadataStore`, and `ArtifactStore` implementations. `Controller`, `PromotionPolicy`, measurement definitions, lineage semantics, and job/result schemas remain unchanged.”

Add a future `M11` only if useful:

```text
M11 — distributed/sharded trainer backend
Trigger: target model cannot be trained efficiently within one execution worker
Boundary: OnPolicyTrainer implementation / ExecutionProfile; control plane unchanged
```

---

## §35 Scaling Strategy

This section should be rewritten substantially.

Recommended text:

> **Scaling is defined as preservation of semantics across execution backends, not preservation of physical topology.** The small-model V1 is the reference implementation of the control plane. Larger models may require different generation, training, storage, and job-placement implementations.

Use this table:

| Transition | Control-plane change | Execution-plane change | Required revalidation |
|---|---|---|---|
| smoke → dev | none | real model/GPU backend | integration tests |
| 0.6B → 1.5B | model spec/profile | possibly batching/memory | short compatibility spike |
| dev → 4B research | none | research execution profile; possibly 80GB-class GPU | target E1/E3/E4/E5 + scientific Phase I |
| 4B → 9–12B | none by default | quantization, multi-GPU or remote generation may change | reduced target Phase I |
| one node → multi-node | none | executor/providers/stores replaced | contract suite + failure/recovery tests + calibration |

Retain the statement that durable data/checker/ledger assets carry across scales, but add:

> Checkpoint *identity and lineage metadata* carry across storage topologies; checkpoint binary formats may remain backend/model specific.

---

## §36 Success Metrics

Add a new subsection **before** scientific Level 0.

### Engineering success levels

#### E-Level A — contract-complete

- all versioned job schemas round-trip serialization;
- all local backends pass contract tests;
- no core interface leaks physical path/device/runtime objects;
- stable IDs/refs recorded in metadata.

#### E-Level B — reference backend end-to-end

- small-model real training segment runs through the same controller/job/storage boundaries;
- checkpoint/resume works;
- evaluation/gate result is durable and replayable;
- reports reconstruct from stored metadata/artifacts.

#### E-Level C — failure-resilient

- crash injection, partial artifact, duplicate result, retry, and stale-result tests pass;
- no duplicate budget consumption or lineage transition is possible.

These levels can be achieved with limited compute and are legitimate architecture milestones.

### Scientific success levels

Keep the current Level 0–4 structure, but:

- rename them “Scientific Level 0–4”;
- update Level 1 for run-level seed variance;
- ensure small-model engineering success is never described as target-model gate validation.

### Level 1 recommended stronger wording

Preferred when ≥2 seeds/condition are run:

> A reward-driven learning signal exists if the reward-conditioned runs outperform matched random-reward runs with consistent direction across independent training seeds and the aggregate effect satisfies the preregistered criterion.

Fallback when only one seed is affordable:

> A single-run reward-associated effect is observed; it is not promoted to a reproducibility claim and run-level variance remains unestimated.

---

## §37 Risks

Add/revise risks:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Architecture accidentally hard-codes current one-GPU topology | Medium | High | control/execution split; contract tests |
| Small-model behavior mistaken for larger-model evidence | High | High | separate engineering and scientific validation tracks |
| Distributed migration reveals nonserializable/local state assumptions | Medium | High | versioned job specs and replay now |
| Duplicate/retried jobs corrupt lineage or budgets | Medium | High | idempotent job IDs and atomic commit protocol |
| Run-level stochasticity dominates E1/E2 result | Medium–High | High | independent seeds/staged confirmation |
| Evaluation becomes dominant compute at small floor | Medium | Medium–High | measure eval/train ratio; E3 sizing; budget threshold |
| Backend/library update changes algorithmic behavior | Medium | High | explicit settings + stack hash + compatibility tests |

The existing “compute insufficient” risk should be split into:

1. **current architecture development compute insufficient** — low for small model;
2. **target scientific experiment compute insufficient** — medium/high but deferrable without invalidating the architecture milestone.

---

## §38 Final Architectural Decisions

Keep D01–D30 after updating affected wording and append:

| ID | Decision | Selected approach |
|---|---|---|
| D31 | Architecture planes | Stable control plane; replaceable execution plane |
| D32 | Model/execution config | `ModelSpec` separate from `ExecutionProfile` |
| D33 | Job boundary | Versioned serializable `JobSpec` / `JobResult` through `JobExecutor` |
| D34 | Job correctness | Stable job IDs, idempotent retries, atomic artifact/result commit |
| D35 | Generation boundary | `RolloutProvider` and `EvalGenerationProvider`; local vLLM in V1 |
| D36 | Persistence boundary | `MetadataStore` + `ArtifactStore`; SQLite/local filesystem in V1 |
| D37 | Reference validation | replay backend, contract tests, failure injection; small model is canonical reference semantics |
| D38 | Validation tracks | engineering validation on small compute; scientific calibration on target model |
| D39 | Run-level uncertainty | target E1/E2 requires replicated/staged training seeds or explicit single-run limitation |
| D40 | Curriculum disagreement | model consensus flags suspected labels; exclusion requires independent confirmation |

Update the header convention to D01–D40.

---

# 11. Appendix changes

## Appendix A — Starting configuration

Split the current table into four blocks.

### A.1 Experiment configuration

- group size;
- prompts/step;
- gate settings;
- curriculum settings;
- reward;
- evaluation statistics.

### A.2 ModelSpec

- model ID/revision;
- tokenizer;
- chat template;
- context limit;
- LoRA target modules;
- inference mode.

### A.3 ExecutionProfile

- backend name/version;
- dtype;
- quantization;
- vLLM mode;
- explicit importance-sampling options;
- GPU topology;
- concurrency/memory utilization;
- training backend;
- generation backend.

Create separate example profiles:

```text
smoke.yaml
dev-qwen06b-local24g.yaml
research-qwen4b-local80g.yaml
```

A future multi-GPU profile can be added without changing experiment semantics.

### A.4 Persistence/execution settings

- metadata backend;
- artifact backend;
- job executor;
- retry policy;
- schema versions.

---

## Appendix B — Glossary

Add:

| Term | Meaning |
|---|---|
| Control plane | Durable experiment semantics: controller, gate, lineage, budgets, provenance |
| Execution plane | Resource-dependent implementations of training/generation/jobs/storage |
| `ModelSpec` | Semantic identity/configuration of the model |
| `ExecutionProfile` | How that model is run on available hardware |
| `JobSpec` | Versioned serializable definition of one expensive operation |
| `JobResult` | Durable result accepted by the control plane |
| `StateRef` | Logical reference to a finalized training state |
| `ArtifactRef` | Location-independent reference to an immutable artifact |
| Idempotent job | Retrying the same logical job cannot apply its effects twice |
| Reference backend | Local single-node implementation used to define/test semantics |
| Research profile | Execution profile used for target-model scientific validation |

Change “exact resume” glossary language if present.

---

## Appendix C — References

No mandatory new academic references are required for the architecture refactor.

However, revise library-reference wording so algorithmically relevant behavior is tied to the **pinned version and explicit configuration**, not generalized claims about “the default.”

---

## Appendix D — Design review response

Keep the Revision 2 review mapping.

Add a new Appendix E rather than overwriting it.

---

## New Appendix E — Revision 3 small-compute/scalability review response

Recommended table:

| Review item | Finding | Revision 3 response |
|---|---|---|
| S1 | Small-model implementation can be mistaken for a large-model scientific proxy | Separate engineering and scientific validation tracks |
| S2 | `ModelProfile` mixes model semantics and hardware topology | `ModelSpec` + `ExecutionProfile` |
| S3 | Controller is too close to local subprocess execution | `JobExecutor` and versioned job contracts |
| S4 | vLLM colocate is treated as trainer architecture | `RolloutProvider` boundary |
| S5 | SQLite/filesystem may leak into core semantics | `MetadataStore` + `ArtifactStore` + durable refs |
| S6 | Retry/distribution semantics are undefined | idempotent jobs + atomic finalization |
| S7 | Scalability is documented but not testable | backend contract suites and replay executor |
| S8 | Controller branch coverage would otherwise require GPU runs | replay/simulation backend |
| S9 | E1/E2 bootstrap ignores training-run variance | independent seed/staged confirmation |
| S10 | Consensus quarantine can remove valid hard prompts | flag + independent verification |
| S11 | E3 scaled adapter may not represent true training trajectory | real checkpoints + scaling cross-check |
| S12 | Algorithm behavior relies on library defaults | explicit config + compatibility assertions |
| S13 | “Exact resume” overstates determinism | complete operational resume; bit-exactness not required |
| S14 | “No model-generated test” conflicts with hidden program construction | distinguish unverified model labels from model-drafted executable programs |

---

# 12. Recommended revised architecture invariants

The documentation should list these in one place. They provide a concise test of whether future changes preserve the design.

1. **Measurement independence:** the gate never uses the training verifier.
2. **No measurement leakage:** no trainer receives measurement-split items.
3. **Immutable champion:** unaccepted states cannot mutate the champion.
4. **Durable lineage:** every accepted/rejected candidate has stable provenance.
5. **Control/execution separation:** controller semantics do not depend on local/remote topology.
6. **Serializable jobs:** expensive work is specified by versioned durable values.
7. **Location-independent state:** core logic never identifies states by physical path.
8. **Idempotent effects:** one logical job can affect lineage/budgets at most once.
9. **Explicit algorithm configuration:** no preregistered scientific behavior relies on library defaults.
10. **Small-model claims stay engineering-only:** target-model effect/power is recalibrated.
11. **Independent run uncertainty is acknowledged:** evaluation bootstrap does not stand in for training seeds.
12. **Backend substitution is tested:** every implementation passes the same contract suite.
13. **Failure before finalization is non-semantic:** partial work cannot become a champion, spend a claim result twice, or create a valid state.
14. **Human intervention remains budgeted:** architecture changes do not bypass the existing search-accounting rules.

---

# 13. Recommended revised repository ownership boundaries

The most important rule is not the exact filenames; it is dependency direction.

Recommended dependency flow:

```text
controller / gate / stats / domain policy
          |
          v
contracts (refs, job specs, results, capabilities)
          |
          v
execution + storage interfaces
          |
          v
local implementations / third-party libraries
```

Do not allow:

```text
controller -> vllm
controller -> sqlite3
controller -> pathlib state paths
controller -> subprocess
promotion policy -> model loading
statistics -> storage backend
```

Allow:

```text
controller -> JobExecutor
controller -> MetadataStore / LineageStore
trainer implementation -> RolloutProvider
local rollout provider -> vLLM
SQLiteMetadataStore -> sqlite3
LocalArtifactStore -> pathlib/filesystem
```

This dependency rule is more important than adding many abstract base classes.

---

# 14. Documentation language changes to make consistently

Search the original document and update these phrases consistently.

| Current idea | Revision 3 wording |
|---|---|
| “0.6B plumbing validation only” | “0.6B reference architecture and end-to-end integration validation; not target-model statistical validation” |
| “all instrument validation happens on 4B” | “target-model statistical calibration happens on the research target; software/architecture validation happens on the reference profile” |
| “one GPU architecture” | “one-GPU V1 execution backend” |
| “80 GB floor” | “current prior for the 4B research execution profile” |
| “exact resume” | “complete operational resume / full persisted state” |
| “TIS default” | exact explicit pinned setting |
| “No task, label or test generated by model” | “No unverified model-generated label/test enters training or measurement” |
| “future larger model is a new ModelProfile” | “new ModelSpec + appropriate ExecutionProfile; execution backend may change” |
| “distributed execution changes trainer internals” | “distributed execution replaces execution/provider/store implementations” |
| “evaluation cost risk closed” | “evaluation cost controlled by tiering but verified empirically before Phase II” |

---

# 15. What should NOT be added now

The revised documentation should explicitly resist overengineering.

Do **not** build now solely for future appearance:

- Kubernetes;
- Ray cluster management;
- microservices;
- PostgreSQL if SQLite has one writer;
- S3/MinIO if local disk is sufficient;
- distributed locks;
- message queues;
- multi-node trainer;
- candidate populations;
- generic plugin registries with dynamic discovery;
- a universal model-serving abstraction;
- a “universal evaluator” with dozens of optional fields.

The engineering goal is **stable contracts with simple local implementations**.

---

# 16. Definition of done for Revision 3 documentation

Revision 3 should not be marked authoritative until the document itself satisfies all of these:

- [ ] The project is explicitly described as a small-compute reference architecture with deferred target-model scientific validation.
- [ ] Control plane and execution plane are defined and shown in a diagram.
- [ ] `ModelSpec` and `ExecutionProfile` are separate concepts.
- [ ] `JobExecutor`, `RolloutProvider`, `EvalGenerationProvider`, `MetadataStore`, and `ArtifactStore` are documented.
- [ ] Job specs/results are versioned and serializable.
- [ ] Physical paths are removed from cross-component contracts.
- [ ] Idempotency and artifact/result commit semantics are specified.
- [ ] Replay/simulation backend is part of V1 scope.
- [ ] Backend contract tests and failure injection are part of V1 acceptance.
- [ ] `smoke`, `dev`, and `research` validation modes are documented.
- [ ] E1/E2 run-level variance is addressed or explicitly limited.
- [ ] Consensus quarantine requires independent label confirmation before exclusion.
- [ ] E3 combines real checkpoints with scaled-adapter interpolation.
- [ ] Importance-sampling configuration is explicit and pinned.
- [ ] “Exact resume” wording is corrected.
- [ ] Model-generated hidden-program wording is internally consistent.
- [ ] Evaluation-cost risk is marked as empirical/verified rather than closed by design.
- [ ] M5 and M10 describe execution-backend replacement rather than a simple profile switch.
- [ ] Engineering success metrics exist separately from scientific success metrics.
- [ ] New failure modes F44+ and decisions D31+ are added.
- [ ] Appendices/config examples reflect the new model/execution/storage separation.

---

# 17. Suggested implementation order corresponding to the documentation

This order preserves your limited-compute goal and produces visible engineering value early.

1. **Introduce stable refs and job/result schemas.**
2. **Split `ModelProfile` into `ModelSpec` and `ExecutionProfile`.**
3. **Wrap existing subprocess behavior in `LocalSubprocessExecutor`.**
4. **Wrap SQLite and local files in metadata/artifact interfaces without changing behavior.**
5. **Wrap colocated vLLM behind rollout/eval providers.**
6. **Move controller code to consume only job specs/results and refs.**
7. **Add idempotency and finalize/commit semantics.**
8. **Add replay backend.**
9. **Add contract and failure-injection tests.**
10. **Run end-to-end `dev` profile on the small model.**
11. **Make the experimental fixes: quarantine, explicit IS config, E3 real-checkpoint comparison.**
12. **When compute becomes available, run the research-profile target-model Phase I with run-level replication.**

This sequence allows the architecture to become strong and demonstrable before expensive compute is available.

---

# 18. Final architectural position after these changes

After Revision 3, the defensible project description becomes:

> This project is a local-first reference implementation of an auditable autonomous post-training control plane. It can run end-to-end on small models and limited hardware today. The controller, measurement channel, lineage, provenance, budgets, and statistical decision semantics are independent of resource topology. Training, rollout generation, evaluation generation, job placement, and persistence are execution-plane backends behind versioned contracts. Larger-model or distributed runs replace those backends and repeat target-model calibration rather than rewriting the system. Small-model runs validate the architecture; target-model runs validate the scientific claim.

That is a stronger and more realistic engineering claim than saying the current single-GPU implementation itself “scales” to 12B. It also gives the project value even before expensive research runs are affordable.
