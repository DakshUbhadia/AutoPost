# AutoPost Implementation Sprint Roadmap

**Architecture baseline:** Revision 3
**Roadmap version:** 1.1
**Live project status:** tracked in GitHub Issues/Projects and the versioned environment manifests under `environments/`.

**Purpose:** Define the dependency-bounded work packages that take AutoPost from its initial repository skeleton to a complete reference-architecture release, and then—only when research compute is available—to a calibrated target-model experiment and replication.

---

## 1. Direction and source of truth

AutoPost is a **local-first reference implementation of an auditable post-training control plane**. Small-model runs prove engineering behavior; only target-model runs can establish scientific claims.

Use the public documents in this order when decisions conflict:

1. [`post_training_documentation.md`](post_training_documentation.md) — authoritative Revision 3 architecture and scientific protocol.
2. This roadmap — implementation dependencies, sprint boundaries, and release gates.

Historical planning notes may exist locally, but they are intentionally excluded from the public repository and are not normative.

The non-negotiable dependency direction is:

```text
controller / gate / statistics / domain policy
                    |
                    v
       versioned contracts and stable refs
                    |
                    v
       execution and persistence interfaces
                    |
                    v
 local implementations and third-party libraries
```

Core code must not depend directly on vLLM, `sqlite3`, subprocess placement, CUDA objects, or physical checkpoint paths. Third-party runtime objects stop at backend implementations.

---

## 2. Repository baseline at roadmap creation

This section is a **dated snapshot from 2026-11-01**, retained only to explain the roadmap's starting point. It is not live status; use the repository's GitHub Issues/Projects and versioned environment manifests for current progress.

### 2.1 Existing foundation

| Area | Baseline snapshot |
|---|---|
| Package skeleton | `src/autopost`, CLI, tests, configuration, and architecture documentation exist |
| Domain contracts | Revision 2 task, completion, reward, trainer, evaluation, gate, and ledger interfaces exist in `interfaces.py` |
| Measurement | Standard-library exact numeric scorer and cluster-bootstrap/statistical helpers exist |
| Gate | Pure `PROMOTE`, `CONTINUE`, `RESTORE`, and `HALT` policy logic exists |
| Configuration | A combined `configs/base.yaml` and placeholder inventory exist |
| Pre-registration | Draft, validation, freeze, and verification foundations exist |
| Persistence | Initial SQLite schema and initialization command exist |
| Validation | Fast unit tests and a synthetic self-check exist |

### 2.2 Baseline gaps that determine the sprint order

| Priority | Gap at roadmap creation | Consequence |
|---|---|---|
| P0 | Code/config still use Revision 2 `ModelProfile` | Sprint 1 must split `ModelSpec` from `ExecutionProfile` |
| P0 | `TrainState` exposes a physical path | Stable refs must precede stores, trainers, and controller work |
| P0 | No versioned jobs/results or executor boundary | Retry, replay, and remote substitution cannot yet be proven |
| P0 | No metadata/artifact store contracts or finalization protocol | Partial work and duplicate result semantics are undefined |
| P0 | No ingest, deduplication, decontamination, split, or hidden-program pipeline | Training/measurement separation is not end-to-end |
| P0 | No real verifier pool, execution providers, trainer, or evaluation runner | No real segment can run |
| P0 | No durable controller, lineage semantics, ledger, or replay executor | Gate logic is not yet an operational system |
| P1 | Dependency/model pins and environment identities are unresolved | Development and research reproducibility must be separated |
| P1 | Database schema has no version/migration mechanism | Durable metadata cannot safely evolve |
| P1 | `Task` and `tasks` disagree on `source_ver`/`measurable` | Data-contract migration is required |
| P1 | Quarantine still implies consensus exclusion in places | Revision 3 flag-plus-independent-verification rule must be enforced |
| P1 | Importance-sampling behavior is described partly as a default | Scientific algorithm behavior must be explicit and asserted |

---

## 3. Delivery strategy

The work is divided into four release horizons.

| Horizon | Sprints | Compute | Outcome |
|---|---:|---|---|
| Foundation | 0–4 | CPU/ordinary development machine | Revision 3 contracts, durable stores, safe data and verification foundations |
| Reference backend | 5–9 | CPU plus a small/dev GPU when available | Complete R0 reference architecture with replay and failure resilience |
| Scientific calibration | 10–12 | Target research GPU | Selected trainer, frozen measurement instrument, calibrated gate/controller, and go/no-go |
| Gated research loop | 13–15 | Target research GPU | Main run, sealed evaluation, replication, and final audit package |

Sprint numbers represent **dependency-bounded work packages**, not equal calendar durations. A sprint may be split into multiple pull-request milestones without changing its exit gate. Scientific sprints are bounded by preregistered GPU budgets rather than calendar duration.

### 3.1 Reference Architecture Release R0

Sprint 9 produces the first complete release of the project.

R0 demonstrates that:

- production contracts are versioned and serializable;
- local backends satisfy their contracts;
- small-model training and evaluation run through production boundaries;
- controller decisions are durable and replayable;
- restart and retry semantics are proven;
- failure injection does not create duplicate semantic effects;
- the system can be audited from durable records;
- no claim is made about target-model scientific improvement.

R0 is a valid stopping point when target-model compute is unavailable. Waiting at R0 is an intentional project state, not unfinished engineering.

### 3.2 Global definition of done

A sprint is complete only when:

- its acceptance tests pass in a clean compatible environment;
- configuration, schema, protocol, and contract versions are recorded where applicable;
- measurement items cannot reach any trainer;
- raw evidence and manifests needed for audit are retained;
- new public behavior has tests and concise documentation;
- unresolved choices are recorded in GitHub Issues with an owner and trigger;
- no scientific threshold is selected after looking at the result it will judge;
- all declared dependencies have passed their exit gates.

**No sprint may begin until all of its declared dependencies have passed their exit gates. Independent branches may proceed in parallel when the dependency graph explicitly permits it.**

---

## 4. Sprint overview

| Sprint | Objective | Primary exit gate |
|---:|---|---|
| 0 | Stabilize CPU/dev workflow and retire early toolchain API risk | Reproducible local development stack |
| 1 | Introduce Revision 3 contracts and split configuration | Serializable contracts with no runtime/path leakage |
| 2 | Build stores, artifacts, job journal, and idempotent commits | Retry/crash cannot double-apply semantic effects |
| 3 | Build data pipeline and disposable hidden-data framework | Isolation/decontamination invariants pass |
| 4 | Build verification, reward, curriculum, and safe quarantine | Trusted training-signal path passes golden tests |
| 5 | Implement local execution providers and dev-model spike | Dev execution backend satisfies contracts |
| 6 | Implement GRPO/RFT backends and state round-trip | Real resumable small-model segment succeeds |
| 7 | Implement evaluation generation and durable evidence | Pilot result reconstructs from stored raw evidence |
| 8 | Implement controller, lineage semantics, budgets, and replay | Every state transition is durable and replayable |
| 9 | Release R0 and pass E0/failure-injection/auditability | Reference architecture release complete |
| 10 | Activate research stack, target spike, preregister, and run short feasibility campaign | Target Phase I is technically viable and affordable |
| 11 | Run E1/E2 and E9; select trainer with run-level uncertainty | Trainer selected; selected-trainer trajectory available |
| 12 | Freeze scorer/templates; calibrate instrument/controller in dependency order; issue go/no-go | Final instrument/controller validated or project stops |
| 13 | Freeze Phase II and perform production-readiness drill | Immutable Phase II protocol |
| 14 | Execute gated main experiment | Scientific Level 2 or interpretable negative result |
| 15 | Replicate and release | Reproducibility conclusion and audit package |

---

## 5. Foundation and reference-backend sprints

### Sprint 0 — Repository health and early toolchain feasibility

**Dependencies:** None.

**Goal:** Establish reproducible development environments without making target research hardware a prerequisite for R0.

**Work:**

- Define three environment identities: `cpu-dev`, `small-gpu-dev`, and `research-target`.
- Make the CPU-safe package/test workflow reproducible in a fresh checkout.
- Make `small-gpu-dev` reproducible when a local dev GPU is available; otherwise record its availability clearly.
- Permit `research-target` to remain `PENDING_HARDWARE` through Sprint 9.
- Add supported commands for install, fast tests, full self-check, DB init, and preregistration verification.
- Configure CPU CI without downloading models.
- Add `scripts/probe_toolchain_api.py` to verify, without a full run where possible:
  - required TRL trainer arguments;
  - explicit loss-type support;
  - explicit importance-sampling configuration;
  - PEFT/LoRA hooks;
  - supported vLLM integration mode;
  - expected metrics and callback hooks.
- If a dev GPU is available, add a one-to-two-step smoke run to the feasibility probe.
- Keep environment status in `environments/`, machine-readable probe results in `docs/toolchain_probe.json`, and mutable progress in GitHub Issues/Projects rather than this roadmap.

**Deliverables:** Reproducible CPU environment, optional dev-GPU lock, environment identity records, developer commands, CPU CI, and toolchain API probe.

**Acceptance:**

- CPU-safe tests run with one documented command in a fresh compatible checkout.
- The dev-GPU environment is reproducible when such hardware is available.
- `research-target` may remain `PENDING_HARDWARE`.
- No target-hardware placeholder blocks Sprints 1–9.
- Scientific execution is blocked while the research environment is unresolved.
- The API probe proves the selected development stack exposes required controls or records a blocking incompatibility before architectural implementation continues.

### Sprint 1 — Revision 3 contracts and configuration migration

**Dependencies:** Sprint 0.

**Goal:** Make execution and persistence boundaries explicit before expensive backend work.

**Work:**

- Create contract modules for refs, jobs/results, capabilities, manifests, evidence, and canonical serialization.
- Add durable IDs/refs including `RunId`, `JobId`, `StateRef`, `ArtifactRef`, `DatasetRef`, `EvalReadId`, `MeasurementInstrumentId`, `ConfigId`, and `StackId`.
- Define versioned training/evaluation job specs, results, statuses, and stable error classes.
- Require explicit or implied versions at minimum for `JobSpec`, `JobResult`, `StateManifest`, `ArtifactManifest`, and `EvalEvidence`.
- Define deterministic compatibility rules for supported, migratable, and rejected protocol versions.
- Split `ModelProfile` into `ModelSpec` and `ExecutionProfile`.
- Split experiment, model, execution, and data config while retaining canonical merged config/stack hashes.
- Remove physical path identity from control-plane state.
- Define `JobExecutor`, `MetadataStore`, `ArtifactStore`, `RolloutProvider`, and `EvalGenerationProvider` protocols.
- Define only `CapabilitySpec`, `CapabilityReport`, and `CompatibilityError`; defer runtime discovery/assertion to Sprint 5 and do not build a generic negotiation framework.
- Migrate pre-registration hashing to the new config structure.

**Deliverables:** Revision 3 contracts/configs, protocol compatibility rules, and transitional adapters only where required.

**Acceptance:**

- Every durable spec, result, ref, manifest, and evidence object round-trips through canonical serialization.
- Stable job IDs derive from semantic inputs and exclude attempt number.
- Unsupported versions fail deterministically before side effects.
- Contracts contain no absolute path, open connection, callable, CUDA/model/tokenizer object, or third-party runtime object.
- Scientific identity is separate from topology while numerically relevant execution settings enter the stack hash.
- Existing scorer/statistics/gate behavior remains green.

### Sprint 2 — Durable persistence, fixture execution, and idempotent commits

**Dependencies:** Sprint 1.

**Goal:** Prove safe retry and recovery semantics before real training exists.

**Work:**

- Implement `SQLiteMetadataStore` behind its contract.
- Add schema versioning and forward migrations without rewriting historical scientific records.
- Implement `LocalArtifactStore` with staging, manifest validation, content hashes, and atomic finalization.
- Add durable job/attempt records and state/result acceptance transactions.
- Enforce: reserve job → stage outputs → validate → finalize artifacts → atomically accept generic metadata/budget effects.
- Implement state records, parent relationships, immutable artifacts, generic pointer/reference transactions, and job-result acceptance.
- Do **not** encode promotion, champion, restore, or candidate-pruning policy in the store.
- Detect duplicate identical results, conflicting results, stale parents, missing artifacts, and corrupt manifests.
- Keep raw-completion manifests independent of aggregate scores.
- Add a deterministic `FixtureExecutor` capable of success, duplicate success, conflict, stale parent, incomplete artifact, and executor failure.

**Deliverables:** Local stores, migrations, job journal, atomic commit primitives, and deterministic fixture executor.

**Acceptance:**

- Retrying one logical job cannot create a second state, consume a second budget item, or double-apply metadata effects.
- Crash before finalization has no semantic effect.
- Crash after finalization but before metadata commit is safely recoverable.
- Conflicting/stale results are quarantined and audited.
- Store contract and fixture failure tests pass without a trainer.
- Persistence exposes generic atomic primitives and contains no gate/promotion policy.

### Sprint 3 — Data pipeline and disposable hidden-data framework

**Dependencies:** Sprint 2.

**Goal:** Produce auditable datasets with cluster-level isolation while keeping claim-bearing hidden data out of development.

**Work:**

- Resolve the `Task`/`tasks` mismatch through a versioned schema migration.
- Implement source adapters, normalization, answer typing, rejection logs, and immutable dataset manifests.
- Implement MinHash plus embedding-assisted deduplication, source-lineage decontamination, and cluster-level splits.
- Enforce exact-numeric measurable items for measurement splits.
- Build development train/dev/pilot/screen/gate/reservoir/transfer fixtures with stable prompt IDs.
- Build the hidden-program schema, invariant/property tests, deterministic generation, program-level split mechanics, and human-review workflow.
- Create only a disposable, inspectable `hidden_dev` pool during engineering development.
- Do **not** create the final research `hidden_audit` or `hidden_sealed` assets in this sprint.
- Add reachability tests proving no measurement item or cluster can reach an on-policy or offline trainer.
- Treat model-drafted programs as untrusted until execution labeling and review succeed.

**Deliverables:** Versioned dataset builder, development split manifests, `hidden_dev`, contamination report, and hidden-program test report.

**Acceptance:**

- Zero forbidden cluster overlap.
- Development measurement splits contain only allowed answer types.
- Every hidden program passes invariant/property tests over many generated instances.
- Same inputs/seed yield identical dataset IDs and manifests.
- Every rejection/exclusion has a reason.
- Final research hidden partitions do not yet exist and cannot be accidentally consumed.

### Sprint 4 — Verification, reward, curriculum, and safe quarantine

**Dependencies:** Sprint 3.

**Goal:** Complete the trusted training-signal path while preserving scorer independence.

**Work:**

- Implement the sandboxed `math-verify` train-verifier pool with timeouts and stable checker versions.
- Expand exact-scorer fixed, property, exploit, multiple-answer, truncation, and format tests.
- Keep the evaluation scorer standard-library-only and independent of training/verifier modules.
- Implement binary reward and explicit truncation behavior.
- Implement pass-rate estimates, derived signal band, quotas, global cluster exposure caps, and deterministic sampling.
- Treat consensus disagreement as a flag requiring independent verification before exclusion.
- Record pass-rate observations, verifier versions, exposures, quarantine decisions, audit flags, and supporting evidence.
- Add checker-disagreement metrics without allowing the evaluation scorer to affect training rewards.

**Deliverables:** Train-verifier pool, golden suites, curriculum sampler, and quarantine review workflow.

**Acceptance:**

- Train-verifier golden suite is 100%.
- Exact-scorer property and independence tests pass.
- Timeout, malformed output, multiple answers, and truncation resolve deterministically.
- Quotas/exposure caps survive restart tests.
- Model consensus alone never excludes an item.
- Every suspected-label exclusion cites an independent verification result or human audit record.

### Sprint 5 — Local execution providers and development-model compatibility

**Dependencies:** Sprint 2. May proceed in parallel with Sprints 3–4.

**Goal:** Validate local backend contracts using only the small/dev model; target hardware must not block R0.

**Work:**

- Implement `LocalSubprocessExecutor` with process isolation, log capture, cancellation, attempts, and stable results.
- Implement colocated vLLM rollout and local evaluation-generation providers using project-owned request/result types.
- Keep vLLM/TRL runtime objects inside backend implementations.
- Implement runtime capability discovery/assertion against Sprint 1's `CapabilitySpec`/`CapabilityReport`.
- Run the compatibility spike only for the small/dev model as the required exit gate.
- Validate tokenizer/chat template, LoRA targets, context limits, bf16 support, save/export/reload parity, and short generation.
- Explicitly configure and assert importance-sampling implementation, granularity, cap, and emitted metrics.
- An optional early target probe may be recorded if hardware exists, but it cannot block or qualify R0.

**Deliverables:** Local executor, rollout/eval providers, dev spike, and capability report.

**Acceptance:**

- Executor/provider contracts pass.
- A local job survives submit/status/result, cancellation, and failure paths.
- Dev model loads, attaches LoRA, generates, saves/exports, reloads, and behaves within documented tolerance.
- Required algorithm controls are runtime-asserted.
- No research-target compatibility or performance result is required.

### Sprint 6 — Training backends and complete operational state

**Dependencies:** Sprints 4 and 5.

**Goal:** Run real small-model training without compromising lineage or resumability.

**Work:**

- Implement the TRL GRPO backend with explicit DAPO/TIS configuration and mismatch telemetry.
- Adapt `RolloutProvider` to the real pinned trainer/library seam. If TRL owns generation internally, wrap that behavior behind the backend boundary rather than reimplementing GRPO to force an artificial call shape.
- Preserve adapter, optimizer, scheduler, available RNG state, trainer state, sampler/data cursor, hashes, and verifier version at segment boundaries.
- Implement coarse adapter-sync validation.
- Implement RFT/offline training using only verified, provenance-bearing examples.
- Enforce measurement isolation at API, job-validation, and database boundaries.
- Add state round-trip, resume-within-tolerance, and export-parity tests.
- Store raw rollouts and metadata before accepting a state result.

**Deliverables:** GRPO/RFT backends, state manifests, training telemetry, and resume/export tests.

**Acceptance:**

- A short dev-model segment creates a finalized state through job/store boundaries.
- Resume continues from the correct step/data cursor within declared tolerance.
- Measurement tasks are rejected before model execution.
- Incomplete resumable-state metadata invalidates the result.
- Required metrics are durable.
- The invariant is that runtime objects do not escape into control-plane contracts—not that every library literally calls a project-owned rollout function.

### Sprint 7 — Evaluation generation and durable evidence

**Dependencies:** Sprints 4 and 6.

**Goal:** Convert raw candidate/reference generations into durable evidence while separating development reads from scientific reads.

**Work:**

- Implement suites, template rotation, deterministic seeds, stratification, and separate evaluation generation.
- Generate base/champion/candidate samples and require fresh references for Gate-B semantics.
- Store raw completions, versions, per-sample outcomes, cluster IDs, and aggregates.
- Give every `EvalReadId` a purpose: `dev`, `pilot`, `gate`, `audit`, or `sealed`.
- Keep development reads outside future scientific read budgets; only preregistered scientific purposes consume them.
- Implement durable read identities, reservation, acceptance, and double-spend prevention.
- Build `GateContext` from stored evidence using cluster-level statistics.
- Add primary and guardrail evidence with explicit test directions.
- Support re-scoring raw completions under a new scorer version without rewriting history.

**Deliverables:** Evaluation runner, suite registry, read-purpose accounting, evidence builder, and re-scoring command.

**Acceptance:**

- Pilot result reconstructs from stored raw completions.
- Gate-B semantics refuse cached reference samples where fresh draws are required.
- Cluster—not item—is the resampling unit.
- Scientific reads require the correct budget; `dev` reads cannot consume or masquerade as scientific reads.
- Re-scoring creates new versioned evidence while preserving the original.

### Sprint 8 — Control plane, lineage semantics, budgets, and replay

**Dependencies:** Sprint 7.

**Goal:** Prove correct durable state transitions and replayability; defer user-facing reporting polish to Sprint 9.

**Work:**

- Implement the persisted controller state machine and scheduling.
- Implement champion/candidate semantics, valid promotion transitions, restore, archive/pruning, and milestones using Sprint 2's generic transactions.
- Implement read budgets, human-change budgets, forced Gate-A refresh, and sealed-read enforcement.
- Integrate gate policy and tripwires with explicit metric classes and preregistered alpha allocation.
- Implement `ReplayExecutor` over stored results using the real controller, ledger, and lineage behavior.
- Reconstruct controller state from durable records after restart.
- Add exhaustive branch, stale-state, duplicate-result, and recovery tests.

**Deliverables:** Durable controller, lineage semantics, ledger, replay backend, and branch/restart tests.

**Acceptance:**

- Replay covers `PROMOTE`, `CONTINUE`, `RESTORE`, and every `HALT` reason deterministically.
- Champion moves only through one valid committed promotion transition.
- Restore/archive/pruning behavior follows controller policy rather than store policy.
- Budget exhaustion prevents scheduling before an external read.
- Restart reconstructs state solely from durable metadata/artifacts.
- Every state transition and replay result is durable and reproducible.

### Sprint 9 — R0 reference-architecture release, failure injection, and auditability

**Dependencies:** Sprint 8.

**Goal:** Release the complete low-compute reference architecture and establish a safe indefinite pause point.

**Work:**

- Run a real dev-model train/evaluate/decide/replay/resume flow through production boundaries.
- Inject crash-before-finalize, partial artifact, crash-after-finalize-before-commit, duplicate, conflict, stale result, timeout, cancellation, and corrupt-manifest cases.
- Run restart tests at every durable transition.
- Measure dev evaluation/training cost ratio without treating it as target-model evidence.
- Finish Markdown/HTML reports, audit export, operational status/halt commands, and provenance traversal moved from Sprint 8.
- Publish backend contract conformance, user-facing failure matrix, E0 report, and a single end-to-end architecture manifest.
- Add one command that produces the R0 audit bundle.
- Generate an architecture-conformance report.
- Tag the release, for example `r0-reference`.

**Deliverables:** R0 release, E0 report, failure matrix, conformance report, architecture manifest, reference artifacts, operational CLI/reporting, and audit bundle.

**Acceptance — all required:**

- **Engineering Level A:** contracts serialize, local backends conform, and no core boundary leaks physical/runtime objects.
- **Engineering Level B:** real dev-model train/eval/decision/resume/replay/report flow succeeds.
- **Engineering Level C:** required failure cases pass with no duplicate semantic effect.
- Every controller decision is traversable to evidence, configs, code revision, state manifest, and raw artifacts.
- R0 is tagged and reproducible from its recorded environment identities.

**Release outcome:** After Sprint 9, the project may pause indefinitely while waiting for research compute without accumulating architectural debt. Small-model results are engineering evidence only.

---

## 6. Research-horizon activation prerequisites

The project may remain at R0 indefinitely. Sprints 10–15 begin only when all of the following are available:

1. target-model-compatible compute;
2. a preregistered Phase I GPU-hour budget;
3. target dataset and license availability;
4. a reproducible research stack can be locked on the actual hardware;
5. the R0 architecture release remains green;
6. no unresolved architecture change would alter experiment semantics.

Waiting for research compute is not considered an incomplete engineering state.

### 6.1 Architecture review before Sprint 10

Every answer below must be **yes**:

- Can a second execution backend be added without changing controller code?
- Can artifact location change without changing `StateRef`?
- Can jobs be retried without double-spending budgets?
- Can every controller decision be reconstructed from durable evidence?
- Can a small-model run be replayed without GPU work?
- Are all core contracts free of local filesystem paths and runtime objects?
- Are scientific and development read budgets separate?
- Is the final research hidden set still untouched?

If any answer is no, remain in the R0 horizon.

---

## 7. Scientific calibration and gated-research sprints

### Sprint 10 — Research environment activation, target spike, preregistration, and short feasibility run

**Dependencies:** R0 release and all Section 6 prerequisites/review gates.

**Goal:** Prove that the target campaign is technically viable and affordable before full E1/E2 spending.

**Work:**

1. Acquire/identify target hardware and create the exact `research-target` dependency lock on it.
2. Run the target-model compatibility spike and validate actual memory/runtime behavior.
3. Freeze Phase I inputs that must precede results:
   - practical floor and four guardrail margins;
   - seed/confirmation rule;
   - candidate learning rates;
   - Phase I GPU-hour budget;
   - maximum acceptable evaluation/train cost ratio;
   - short-feasibility abort rule.
4. Freeze hidden-program construction code.
5. Build/review the research program pool, run invariants, and review a separate instance sample.
6. Freeze approved program IDs; randomly split by program into `hidden_audit` and `hidden_sealed`.
7. Record the hidden pool/split manifest and lock sealed access behind the read ledger.
8. Run E1a under the preregistered rule.
9. Run a preregistered short E1 feasibility segment, typically 50–100 steps.
10. Measure mean/p95 completion length, memory, throughput, verifier bottleneck, truncation, and projected E1/E2/evaluation cost.

**Deliverables:** Research lock/profile, target capability report, Phase I preregistration, hidden research manifests, E1a result, and feasibility/cost report.

**Mandatory abort gate:** If projected Phase I cost exceeds the frozen budget or target runtime violates a frozen feasibility limit, stop here. Code correctness is not sufficient reason to continue.

**Acceptance:**

- Target stack and model behavior are compatible with the required algorithm controls.
- Final hidden assets are frozen before target scientific results are inspected.
- Sealed outputs remain unread.
- The projected signal campaign and evaluation ratio fit frozen budgets.
- The target Phase I campaign is technically viable and affordable.

### Sprint 11 — Target learning signal, run-level confirmation, and trainer selection

**Dependencies:** Sprint 10 feasibility gate.

**Goal:** Establish the target learning signal, quantify run-level limitations, and select the trainer before final instrument/controller calibration.

**Work order:**

1. Complete GRPO E1 using the preregistered seed/confirmation rule.
2. Run matched random-reward E2 for required GRPO seeds.
3. Run compute-matched E9 RFT comparison.
4. Select GRPO or RFT using the frozen gain-per-GPU-hour rule.
5. Preserve cluster bootstrap as within-evaluation uncertainty only; training seeds address run-level uncertainty.

**Trainer-selection branch:**

- **Branch A — GRPO selected:** use the E1 GRPO trajectory as the selected-trainer source for learning-curve, cadence, and later gate calibration.
- **Branch B — RFT selected:**
  1. record the trainer-selection decision;
  2. create a versioned amendment/new preregistration for selected-trainer calibration;
  3. run an RFT learning-curve campaign sufficient to derive checkpoint cadence, evaluation cadence, training horizon, and expected effect trajectory;
  4. remove GRPO-only TIS/mismatch health metrics from decision semantics where they do not apply;
  5. proceed only when a selected-trainer trajectory exists.

**Deliverables:** E1/E2/E9 reports, trainer decision, run-level limitation statement, and selected-trainer checkpoint trajectory.

**Acceptance:**

- Trainer selection follows the frozen rule.
- Required independent seeds or staged confirmation are complete.
- A one-pair result, if that is all the budget allows, is labeled a single-run association rather than reproducibility evidence.
- No GRPO-derived cadence or health rule silently carries into an RFT-selected Phase II plan.
- A selected-trainer learning trajectory is available for Sprint 12.

### Sprint 12 — Measurement-instrument freeze, calibration, and Phase I go/no-go

**Dependencies:** Sprint 11 and its applicable trainer branch.

**Goal:** Freeze one instrument, calibrate it on the selected trainer, validate the final controller, and decide whether Phase II may begin.

**Required internal order:**

1. **E6 — checker study first.** Finalize scorer scope, disagreement categories, checker baseline, and permitted parser/scorer corrections. Freeze the scorer version after acceptance.
2. **E8 — template study second.** Select/freeze the final 3–5 evaluation-only templates and primary endpoint rotation.
3. **Freeze `MeasurementInstrumentId`.** Its durable manifest must contain scorer version, template set, sampling/reference policies, stratification rule, cluster definition, primary metric, and guardrails.
4. **E3 — power.** Use the selected trainer's real checkpoints. LoRA interpolation is supplementary only.
5. **E4 — error/coverage.** Use the frozen instrument and final planned sizes. If preregistered alpha adjustment is triggered, create a new instrument-calibration version before E5.
6. **E5 — controller shadow run.** Replay the selected trainer's trajectory through final calibrated gate/controller rules; measure promotions, restores, false halts, and budgets. E5 must not precede final power/error calibration.
7. **E7 — contamination probe.** May run in parallel after scorer/template freeze; it informs interpretation and claim wording.
8. Issue the Phase I go/no-go.

E3, E4, and E5 must reference the same frozen `MeasurementInstrumentId`, except for an explicitly versioned preregistered E4-triggered calibration amendment.

**Signal criterion:**

- **If GRPO is selected:** require the preregistered E1-vs-E2 reward-conditioned criterion.
- **If RFT is selected:** require a separately preregistered selected-trainer effect criterion. E2 remains evidence about GRPO reward causality but does not decide RFT viability. Do not claim RLVR reward-driven improvement for an RFT-selected project.

**Go requires all four:**

1. The selected-trainer signal criterion passes.
2. Instrument power/error criteria pass.
3. Controller/false-halt criteria pass.
4. Main run, replication, and evaluation fit the compute budget.

**Deliverables:** Frozen measurement-instrument manifest, E3–E8 reports in dependency order, selected-trainer gate/cadence parameters, controller shadow trace, and signed Phase I go/no-go report.

**If no-go:** Stop the gated run and classify the reason as absent selected-trainer signal, underpowered instrument, unsafe controller, or insufficient budget. Redesign requires a new versioned plan and applicable preregistration.

### Sprint 13 — Phase II immutable freeze and production readiness

**Dependencies:** Sprint 12 go decision.

**Goal:** Freeze the calibrated selected-trainer system and prove that production execution cannot drift silently.

**Work:**

- Freeze set sizes, templates, `N/G/S`, generation caps, drift limits, tripwire alpha, budgets, trainer, run budget, and success definitions.
- Record selected trainer identity, `MeasurementInstrumentId`, selected-trainer calibration run IDs, exact research `ExecutionProfile` ID, and hidden program-pool/split manifest ID.
- Generate/verify Phase II preregistration and code/config/data/stack identities.
- Dry-run exact production commands without reading sealed sets.
- Validate storage, monitoring, retention, interruption recovery, audit export, and the sealed-read ledger.
- Fail readiness if any Phase II decision is still derived from a non-selected trainer.

**Deliverables:** Frozen Phase II preregistration, readiness report, and immutable run manifest.

**Acceptance:**

- No unresolved value affects execution or a decision.
- Sealed data are inaccessible to routine monitoring and limited to one allowed read per run.
- Production identities match validated/calibrated identities.
- Recovery drill succeeds without manual database edits.
- Every cadence, health, gate, and controller parameter is traceable to the selected trainer and frozen instrument.

### Sprint 14 — Gated main experiment

**Dependencies:** Sprint 13 readiness gate.

**Goal:** Execute the autonomous loop under the immutable Phase II protocol.

**Work:**

- Train segments from immutable base/champion state.
- Apply monitor, Gate-A, guardrails, Gate-B, promotion, restore, halt, and milestone rules exactly as frozen.
- Record every attempt, look, decision, state transition, intervention, and artifact.
- Treat tripwire/budget exhaustion as `HALT`; do not self-repair silently.
- At run end, read `hidden_sealed` and `transfer_sealed` once for final champion against base.

**Deliverables:** Main-run lineage, final champion or halt state, sealed evidence, and audit report.

**Acceptance:**

- At least two confirmed promotions plus frozen sealed/guardrail conditions are required for Scientific Level 2.
- Halt or failed sealed result remains a valid reported negative outcome.
- No threshold/result is retroactively changed or discarded.

### Sprint 15 — Independent replication and release

**Dependencies:** Sprint 14 complete, regardless of positive or negative outcome when replication is required by protocol.

**Goal:** Test reproducibility and publish a bounded, auditable conclusion.

**Work:**

- Start from base with a new seed/data order under the same scientific protocol.
- Use the same measurement-instrument version and allowed sealed reads.
- Compute preregistered per-run and pooled conclusions.
- Reconstruct reports from durable stores and verify claims against artifacts.
- Publish scope, run-level uncertainty, measurement limits, compute cost, and deviations.
- Archive code/config/data/environment identities, preregistrations, reports, and checksums.

**Replication rule:** Operational fixes that change numerical behavior require a documented deviation or a new replication protocol; they may not be silently substituted.

**Deliverables:** Replication report, pooled analysis, audit bundle, and final release notes.

**Acceptance:**

- Scientific Level 3 requires the replication's own sealed lower bound above zero and pooled lower bound at least the frozen floor.
- Replication failure is reported as non-reproducibility rather than pooled away.
- Every public claim maps to exact run, state, read, instrument, and preregistration identities.

---

## 8. Cross-sprint dependency map

```text
Sprint 0: reproducible dev baseline + API probe
    |
    v
Sprint 1: contracts/config split
    |
    v
Sprint 2: stores + artifacts + idempotent job semantics
    |--------------------------------------|
    v                                      v
Sprint 3: data pipeline              Sprint 5: local execution/dev spike
    |                                      |
    v                                      v
Sprint 4: verification/curriculum    Sprint 6: training/state
    |--------------------------------------|
                       v
                 Sprint 7: evaluation
                       |
                       v
                 Sprint 8: control plane/replay
                       |
                       v
                 Sprint 9: R0 / E0 release
                       |
               ===== OPTIONAL PAUSE =====
                       |
         target compute + research budget available
                       |
                       v
        Sprint 10: research activation + feasibility
                       |
                       v
        Sprint 11: E1/E2 + E9 trainer selection
                       |
             +---------+----------+
             |                    |
          GRPO wins             RFT wins
             |                    |
             |              RFT learning-curve
             |              calibration amendment
             +---------+----------+
                       |
                       v
    Sprint 12: E6 -> E8 -> instrument freeze
               -> E3 -> E4 -> E5
               + E7
               -> Phase I go/no-go
                       |
                       v
        Sprint 13: Phase II immutable freeze
                       |
                       v
        Sprint 14: gated main experiment
                       |
                       v
        Sprint 15: independent replication/release
```

Data and local-execution branches may proceed in parallel only after Sprint 2. Training, evaluation, and controller integration must not be developed against ad hoc paths or transient runtime objects.

---

## 9. Backlog policy and scope control

### 9.1 Implement for R0

- Local subprocess and replay execution.
- SQLite metadata and local immutable artifact storage.
- Stable refs, versioned jobs/results/manifests/evidence, finalization, and idempotent acceptance.
- Deterministic fixture executor for store/idempotency tests.
- `hidden_dev` fixtures and hidden-program infrastructure.
- Measurement-instrument identity/versioning.
- Environment identities: `cpu-dev`, `small-gpu-dev`, and `research-target`.
- Dev-model end-to-end execution through production interfaces.
- Contract, restart, replay, and failure-injection tests.
- Independent training/measurement channels.
- R0 release, architecture-conformance report, and audit bundle.

### 9.2 Defer until target compute exists

- Target-model vLLM/TRL performance tuning.
- Final research GPU lock file.
- Final `hidden_audit`/`hidden_sealed` generation and split.
- Target-model set sizing.
- Target-specific concurrency and memory tuning.

None of these may block R0.

### 9.3 Defer until a measured scaling trigger

| Deferred item | Trigger |
|---|---|
| PostgreSQL metadata store | Multiple writers or remote execution are required |
| Object storage | Local NVMe is insufficient or workers do not share a filesystem |
| Slurm/Ray/remote executor | Jobs must run outside the local process/node |
| Remote rollout pool | Colocated generation is a measured bottleneck |
| FSDP/DeepSpeed/tensor parallel training | Selected target no longer fits or trains efficiently locally |
| Sequential alpha-spending machinery | Read cadence exceeds the validated fixed-budget protocol |
| Training-data replay/population methods | A measured failure mode justifies added complexity |

Do not implement distributed infrastructure merely to claim scalability. Demonstrate scalability first through clean contracts, serializable jobs, stable identities, and backend conformance.

---

## 10. Release success model

| Release level | Required evidence | Claim boundary |
|---|---|---|
| R0 / Engineering A | Versioned contracts and conforming local backends | Contract-complete architecture |
| R0 / Engineering B | Real dev-model end-to-end segment, resume, replay, and report | Reference backend works operationally |
| R0 / Engineering C | Failure injection with no duplicate semantic effects | Reference backend is failure-resilient |
| Scientific 0 | Frozen instrument passes power, error/coverage, and controller validation | Target instrument is calibrated |
| Scientific 1 | Selected-trainer signal criterion passes with stated run-level limits | Target learning signal exists at tested scope |
| Scientific 2 | Main gated run meets promotion, sealed, and guardrail rules | Repeated target improvement in one run |
| Scientific 3 | Independent replication and pooled floor rule pass | Result is reproducible under frozen protocol |

R0 is complete without Scientific Levels 0–3. Scientific success cannot be inferred from dev-model behavior.

---

## 11. Roadmap maintenance

- Keep mutable progress, owners, test counts, hardware availability, and latest gates in GitHub Issues/Projects; keep reproducible environment state in `environments/`.
- Change this roadmap only when dependency/order semantics, exit gates, or release definitions change.
- Version every roadmap change and record why the prior dependency model was insufficient.
- Any research-stage amendment must preserve the original preregistration and create a new version rather than rewriting history.
