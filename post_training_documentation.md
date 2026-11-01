# Autonomous Post-Training System — Architecture

| | |
|---|---|
| **Status** | **Revision 3 — authoritative.** Supersedes Revision 2. Appendix D preserves the Revision 2 review mapping; Appendix E maps the small-compute/scalability review to this revision. |
| **Reader** | Engineers building and operating the system; the project owner deciding scope and budget. |
| **Purpose** | Define what to build, in what order, why each decision was made, what is deliberately not built, and what must be demonstrated before the system's verdicts are trusted. |
| **Reference model** | Qwen3-0.6B — reference-architecture and end-to-end integration validation; not target-model statistical validation |
| **Research target** | Qwen3-4B-Instruct-2507 — target-model scientific calibration and model-improvement claims |
| **Future scale envelope** | ~9–12B initially; execution topology may change without changing control-plane semantics |
| **Execution modes** | `smoke` · `dev` · `research` |
| **Companion files** | `interfaces.py` plus execution/ref/job schemas (contracts), `stats.py`, `exact_scorer.py`, `selfcheck.py`, `schema.sql`, and experiment/model/execution configuration |

**Conventions.** A value marked *start* is an initial setting. A value marked *E1*, *E3*, etc. is set from the named Phase I experiment (§29.2). All such values, the success criteria and the compute budget are frozen in `prereg.json` before the gated loop runs; changing any of them requires a new, logged pre-registration. Decisions are numbered `D01`–`D40` (§38). Claims attributed to published work are summaries; see Appendix C. Results labelled *synthetic* come from `selfcheck.py` and validate code and reasoning, not model behaviour. `smoke` and `dev` results are engineering evidence; only `research` results support target-model scientific claims.

### What changed in Revision 3

| Area | Revision 2 | Revision 3 | Section |
|---|---|---|---|
| Architecture | Single-node components mixed semantics and topology | Stable control plane with replaceable execution-plane backends | §1.1, §8, D31–D38 |
| Model configuration | `ModelProfile` mixed identity and hardware | `ModelSpec` + `ExecutionProfile` | §11, §27, Appendix A |
| Jobs and storage | Local subprocesses, SQLite and paths appeared in core contracts | Versioned jobs, durable refs, `JobExecutor`, `MetadataStore`, `ArtifactStore` | §8, §26–§27 |
| Validation | 0.6B plumbing; scientific validation at 4B | `smoke`, `dev`, and `research`; engineering and scientific success tracks | §4, §28–§29, §36 |
| Reliability | Resume and retry semantics implicit | Complete operational resume, idempotent jobs and atomic finalization | §20, §23–§24 |
| Experimental design | One E1/E2 trajectory per condition; scaled LoRA as E3 path | Run-level replication or explicit limitation; real checkpoints plus scaling cross-check | §29.2, §36 |
| Curriculum | Consensus disagreement could exclude examples | Flag, independently verify, then exclude only on confirmed label error | §14 |
| Algorithm configuration | Some TIS behavior described as a library default | Explicit pinned importance-sampling semantics and compatibility assertions | §15, §18, Appendix A |

### What changed in Revision 2

| Area | Revision 1 | Revision 2 | Section |
|---|---|---|---|
| Order of work | Build the whole loop, then run it | **Phase I first:** build the measurement channel, run experiments E1–E9 on the target model, pass a pre-registered go/no-go, then build the controller | §29 |
| Gate test | Lower confidence bound > `min_gain` (the detectable effect) | Lower one-sided bound > 0 on Gate-A **and** Gate-B, with sets sized so the pre-registered floor is promoted with ≥ 80% joint probability; verified by simulation (E3) | §21.7, §22 |
| Gated quantities | ~16 jointly required checks | One primary endpoint plus four non-inferiority guardrails with explicit direction; everything else monitored | §21.1 |
| Statistics | Item bootstrap with inner resampling of samples | Paired **cluster** bootstrap, no inner resampling; fresh reference samples for Gate-B | §21.4–21.5 |
| Training horizon | Gate every 50 steps; restore after 2 inconclusive retests; ≤150 steps to show a gain | Gate cadence and step budget set from the E1 learning curve; restore only on a **confident regression** | §20 |
| Tripwires | Trends on KL and entropy; reward/measurement divergence | Drift metrics have absolute limits only; divergence replaced by dual-checker disagreement on the same outputs; false-halt rate pre-registered and measured (E5) | §24 |
| Templates | Worst of T_train and one unseen template, strict format | Format-agnostic accuracy averaged over 3–5 evaluation-only templates; format compliance is a guardrail | §21.3 |
| Measurement scorer | From-scratch symbolic scorer | Exact-arithmetic module; measurement sets restricted to integer/rational answers | §16 |
| Hidden set | One set, read at milestones and for the final claim | Budgeted **audit** part for milestones; **sealed** part read exactly once per run for the claim; sized by program | §12.2, §36 |
| Hardware | 24 GB minimum for Stage B | 80 GB treated as the floor until measured; throughput model from E1 | §25 |
| Mismatch handling | Parity tripwire | TRL truncated importance sampling on; mismatch logged; coarse sync-failure tripwire only | §15 |
| Deferred from V1 | — | Regression replay, lineage-scoped accounting, quarantine-by-persistence, β and LoRA-vs-full ablations, MLflow, thinking-mode probe, retroactive-rollback automation, reservoir automation | §33 |

---

## 1. Executive Summary

This project builds a system that improves an open-source language model through automated post-training, and that decides — with auditable evidence — whether each change actually improved the model.

The system trains one model lineage with on-policy reinforcement learning from verifiable rewards (RLVR). The V1 reference backend uses TRL, bf16 LoRA, colocated vLLM and one GPU; those are execution choices, not control-plane semantics. V1 covers one domain, mathematics with checkable final answers. Training tasks come from curated public datasets, deduplicated by cluster and decontaminated against every evaluation set. No unverified model-generated label or test enters training or measurement. Hidden perturbation programs may be model-drafted only when their labels are execution-derived, their invariants pass, and a human reviews a sample.

The core of the design is the **measurement channel**, which is independent of training: a separately written exact-arithmetic scorer, evaluation-only prompt templates, disjoint data splits, a transfer set from a different source, and a private hidden set whose labels are computed by executing programs. A candidate replaces the champion only when one pre-registered primary endpoint — format-agnostic accuracy averaged over several evaluation-only templates — shows a gain whose one-sided lower bound exceeds zero on two disjoint gate sets (the second with freshly drawn reference samples), and four guardrails demonstrate non-inferiority against the original base model. Intervals come from a paired bootstrap over clusters, the actual unit of independence. When anything looks wrong the system halts and asks for a human; it never repairs itself.

Revision 1 built that instrument on statistical choices that would almost never have promoted a modest gain. Revision 2 corrected the statistics and reordered target-model work around E1–E9 and a scientific go/no-go. Revision 3 retains those experiments but first builds the reference control plane, replay path, durable job/storage contracts and failure semantics on limited compute. The target gated loop is still built/run only after research-profile Phase I shows a reward-driven effect, adequate power and coverage, acceptable false halts, and an affordable measured budget.

Qwen3-0.6B is the canonical `dev` reference profile. It can validate end-to-end integration, separation invariants, durable state, budgets, controller branches, replay, idempotency, failure recovery, lineage and reports. Its effects still say nothing about the instrument's sensitivity, effect size, seed variance, throughput or memory on the 4B research target.

The claim the system can support is deliberately narrow: **the model became more accurate on integer- and rational-answer mathematics problems of the kind in its training sources, robustly to numeric perturbation and to prompt wording, without measured regression on the guardrails, and reproducibly from a second seed.** Nothing in V1 supports a broader "better at mathematics" claim. A negative result is acceptable and, unlike in Revision 1, interpretable, because the gate's power at the floor is demonstrated before any verdict is trusted.

V1 excludes unverified model-generated labels/tests, LLM judges in reward or gating, reward models, candidate populations, adversarial training, distributed infrastructure, automatic tuning and multi-domain training. It defines stable boundaries for future backends without pre-building them; each deferred implementation has a measurable trigger.

### 1.1 Architecture boundary: control plane vs execution plane

The system deliberately separates **model-improvement semantics** from **resource topology**. V1 runs on one machine and one GPU because that is the cheapest environment that exercises the architecture. This is a reference execution backend, not a claim that larger models will use the same physical topology.

The **control plane** owns experiment semantics: controller state, promotion policy, measurement definitions, read budgets, lineage, provenance, pre-registration, tripwires, data policy and durable identities. These semantics must not change merely because the model is larger or work runs on another machine.

The **execution plane** owns resource-dependent mechanisms: model loading, rollout generation, optimizer execution, evaluation generation, verifier parallelism, job placement, checkpoint transport, and storage implementations. These implementations may change from local subprocesses to remote workers without changing control-plane semantics.

Scalability in V1 therefore means **replaceable edges**, not pre-built distributed infrastructure. No core interface may require a local filesystem path, CUDA device object, live Python callback, open database connection, process-local lock or mutable process-local state to cross an execution boundary.

```text
                         CONTROL PLANE
┌──────────────────────────────────────────────────────────────┐
│ Controller / state machine                                  │
│ Gate / statistics / read budgets                           │
│ Lineage / champion semantics                               │
│ Pre-registration / provenance / tripwires                  │
│ Data policy / prompt policy / experiment definitions       │
└──────────────────────────────┬───────────────────────────────┘
                               │ versioned JobSpec / JobResult
                               │ durable StateRef / ArtifactRef
                               ▼
                         EXECUTION PLANE
┌──────────────────────────────────────────────────────────────┐
│ JobExecutor                                                 │
│ RolloutProvider / EvalGenerationProvider                    │
│ Trainer implementation / verifier workers                  │
│ Model loading / device topology / quantization             │
│ MetadataStore / ArtifactStore implementations              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
       V1 reference backend             Future backend
       local subprocesses               remote workers
       one GPU                          multi-GPU / multi-node
       colocated vLLM                   rollout service
       SQLite + local NVMe              PostgreSQL + object storage
```

### 1.2 Validation modes

| Mode | Purpose | Typical environment | What it can establish |
|---|---|---|---|
| `smoke` | CI, schemas, controller branches, storage and failure handling | CPU, tiny model or prerecorded results | Software correctness only |
| `dev` | Real end-to-end RL and evaluation | 0.6B–1.5B, 16–24 GB GPU | Integration correctness and operational behaviour |
| `research` | Scientific claim | Research target / larger compute | Target-model effect, gate calibration, run-level variance and reproducibility |

Small-model runs validate architecture, not target-model statistical sensitivity. Any new research target repeats the scientific calibration required by §29 and §36.

---

## 2. Problem Definition

**The question.** Can an automated pipeline take a small open model, train it on automatically verified tasks, and produce successive versions that are measurably and reproducibly better, without unacceptable regressions and with minimal human input?

The difficulty is not running a training loop. Off-the-shelf libraries do that. The difficulty is that every part of an automated loop can produce the appearance of improvement without the substance. The architecture is organised around six specific problems.

**2.1 Measurement noise is comparable to the effect.** Reinforcement learning on a model that is already post-trained typically moves benchmark accuracy by a few points. Sampling seeds, temperature, prompt wording and software versions can move small-model reasoning scores by a similar amount (Hochlehnert et al., 2025). A loop that promotes on raw score changes will promote noise, and over many cycles it will select for evaluation-set quirks.

**2.2 Training and measurement errors can be correlated.** If the same answer extractor computes the reward and the evaluation score, a bug that accepts wrong answers inflates both. Statistical rigour does not help against a bias shared by the thing being measured and the instrument measuring it.

**2.3 Rewards can be exploited.** A policy optimised against a checker will find the checker's weaknesses: answers that parse leniently, multiple hedged answers, formats that trigger lenient matching, and — in code — harness tricks.

**2.4 Damage can occur outside what is measured.** Training on mathematics can degrade instruction following, multilingual ability, refusal behaviour, calibration, or system-prompt adherence. If these are not measured, the system reports success while the model gets worse as a general assistant. Gradual narrowing can also pass every per-cycle check.

**2.5 Contamination distorts both baselines and gains.** Public training sets and public benchmarks share sources, and the base model has likely seen public benchmarks in pretraining. Gains on contaminated sets can reflect memorisation or template matching.

**2.6 On-policy RL is expensive under limited compute.** Rollout generation dominates cost. Prompts the model always solves or never solves produce no learning signal in group-relative methods and waste most of the budget unless filtered out.

**2.7 The instrument itself must be validated.** A gate that is too conservative produces a stream of "no improvement" verdicts that look like evidence but are only a statement about the gate. A gate that is too liberal produces promotions that look like progress. Neither failure is visible from the gate's own output. The gate's power and false-positive rate must therefore be measured at the effect size the project actually expects, on the target model, before its verdicts are used — which Revision 1 did not do.

**2.8 Small-compute development must not hard-code small-compute assumptions.** A local reference implementation is useful only if controller, measurement, lineage and provenance semantics cross execution boundaries through durable, versioned values. Otherwise moving to remote or multi-GPU execution becomes a control-plane rewrite.

**2.9 Training-run variance is distinct from evaluation uncertainty.** A cluster bootstrap over evaluation items estimates uncertainty in that evaluation. It does not estimate variation between independently trained policies caused by initialization, data order, rollout sampling or nondeterministic kernels. Research conclusions therefore require independent training seeds or an explicit single-run limitation.

**What the system must therefore do:** produce learning signal efficiently; keep measurement independent of training; decide promotions with calibrated statistics; measure broadly enough to see collateral damage; stop rather than proceed when its own signals become untrustworthy; and demonstrate its own power before its verdicts are used.

---

---

## 3. Requirements

| ID | Requirement | Addressed in |
|---|---|---|
| R1 | Improve measurable accuracy on a target capability through automated post-training. | §10, §18 |
| R2 | Verify model outputs and compute rewards automatically, without continuous human labelling. | §16, §17 |
| R3 | Decide automatically whether a candidate becomes the new champion. | §22 |
| R4 | Protect against catastrophic forgetting and regression, including in capabilities not targeted by training. | §19, §21 |
| R5 | Iterate continuously; resume after interruption without manual reconstruction from complete persisted project-controlled state. | §10, §20, §23–§24 |
| R6 | Make improvement reproducible: a second independent run must reproduce it. | §36.1 |
| R7 | Make every decision traceable to its inputs (data, code, config, software stack). | §26 |
| R8 | Allow later addition of task sources, trainers, verifiers, domains and larger models without replacing the core loop. | §27, §34 |
| R9 | Minimise human intervention; confine it to scheduled, small actions. | §24, §28 |
| R10 | Run usefully on limited compute through a one-node/one-GPU reference backend. | §25 |
| R11 | Execute untrusted model output and untrusted data safely. | §26 |
| R12 | Be debuggable: any outcome must be attributable to a specific cause. | §7, §24 |

---
| R13 | Demonstrate, before relying on any verdict, the gate's power at the pre-registered effect size and its false-positive rate, on the target model. | §21.7, §29.2, §36 |
| R14 | State exactly what a promotion and a final success do and do not show. | §1, §21.1, §36.3 |
| R15 | Preserve control-plane semantics when execution, generation or persistence backends change. | §1.1, §8, §27 |
| R16 | Make expensive jobs serializable, versioned, durable and idempotent. | §20, §26–§27 |
| R17 | Reference persistent entities through stable IDs/refs, never physical filesystem paths. | §8.1, §20, §27 |
| R18 | Separate model identity from hardware and execution configuration. | §11, §27, Appendix A |
| R19 | Require every replaceable execution/storage backend to pass a common contract suite. | §8.3, §27, §36.1 |
| R20 | Make the complete controller testable with replay/synthetic results without GPU execution. | §8.3, §28–§29 |
| R21 | Support useful `smoke` and `dev` validation without presenting it as target-model scientific evidence. | §1.2, §28–§29, §36 |

---

## 4. Constraints

| Constraint | Detail | Consequence for the design |
|---|---|---|
| Compute | Limited today; the V1 reference backend is one node and one GPU. | Keep local implementations simple while ensuring control-plane contracts do not encode topology; every run-equivalent is budgeted (§25). |
| Model size | Reference 0.6B, research target 4B, future ceiling ~9–12B. | Model identity lives in `ModelSpec`; resource topology and numerical execution choices live in `ExecutionProfile`. |
| Human time | Minutes per decision; no per-example labelling. | Deterministic checking; tripwires halt; human actions limited and budgeted. |
| Data | Public datasets plus one-time curation. | Curated verified pools; private hidden set built once. |
| Stack | Python; PyTorch, Transformers, PEFT, TRL, vLLM. | Wrap mature libraries; pin exact versions. |

**Assumptions** (if one is false, revisit the listed section):

| ID | Assumption | If false |
|---|---|---|
| A1 | The current prior for the 4B `research` execution profile is one 80 GB-class GPU until E1 measures memory at realistic output lengths. The `dev` profile targets 16–24 GB. | §25: change the research `ExecutionProfile`, reduce generation cap or batch only if E1 supports it, or use QLoRA as a last resort (D06). |
| A2 | The pinned TRL release supports GRPO with LoRA, explicit vLLM colocate mode, `loss_type="dapo"`, explicit reward scaling off, and explicitly configured truncated importance sampling. TRL does not implement DAPO's dynamic sampling. | §18: pin a different release or implement another trainer backend; never inherit algorithmic behaviour from an undocumented default. |
| A3 | Enough decontaminated, in-band problems with exact numeric answers exist for Stage B measurement sets at the sizes E3 requires. | §12: reduce the number of gate sets read per decision, or accept a higher floor. |
| A4 | A strong external model or a human is available for one-time template writing, instance spot checks and occasional label audits. | §12.2: hidden set falls back to post-release competition problems only, which are too small to gate on. |
| A5 | Public training sources were plausibly part of the vendor's own post-training data. | Part of any gain on same-source sets may be sharpened recall; E7 measures this and the claim is narrowed accordingly (§36.3). |
| A6 | Local execution is sufficient for V1 reference validation. | Implement the smallest triggered remote executor, preserving job contracts. |
| A7 | Every expensive job can be represented as serializable project-owned values. | Redesign that job boundary before remote execution; do not pass live runtime objects. |
| A8 | Logical state/artifact refs remain stable across storage migration. | Add a migration manifest before changing backends; never expose physical locations as identity. |

**Open questions**, all answered in Phase I or the pre-registration: the floor (smallest gain worth claiming); the Phase I compute budget and the main-run budget; the human-change budget; the gate cadence and step budget (from E1).

---

## 5. Design Goals

When goals conflict, the earlier goal wins.

| Order | Goal | Meaning |
|---|---|---|
| G1 | Reliability of the improvement signal | A promotion must mean the model improved. False promotions are worse than missed ones. |
| G2 | Genuine improvement | Gains must transfer beyond the training distribution and pipeline format. |
| G3 | Compute efficiency | Maximise confirmed gain per GPU-hour; spend evaluation compute in proportion to a candidate's promise. |
| G4 | Simplicity | Fewest moving parts that achieve G1–G3; prefer configuration of mature libraries over new code. |
| G5 | Autonomy | Run unattended between scheduled human touchpoints. Autonomy never overrides G1: the system halts rather than continuing on untrusted signals. |
| G6 | Extensibility | New capabilities enter through existing interfaces. |
| G7 | Scale by replacement | Resource-specific behaviour stays behind execution interfaces; future backends replace implementations rather than adding distributed branches throughout controller logic. |

---

---

## 6. Non-Goals

| Non-goal | Reason | Revisit when |
|---|---|---|
| General self-improving AI | Out of scope; the project tests a narrow, measurable claim. | Not planned |
| Multi-domain training in V1 | Interference cannot be measured until single-domain measurement is trusted. | Scientific Level 2 reached (§34, M1) |
| Unverified model-generated tasks, labels, or tests | Model assertions cannot supply trusted labels. Model-drafted executable programs are allowed only when labels come from execution, invariants pass and samples are human-reviewed. | Pool exhaustion, in an executor-grounded domain (M3) |
| LLM judges or reward models in reward or gating | Learned judges can be exploited and add correlated error. | Not planned for gating; audits only |
| Candidate populations | Multiply compute; selecting the best of N on finite evaluation sets inflates measured gains. | ~20× compute and evaluation sets large enough to separate candidates (M7) |
| Adversarial training and self-play | Unstable, expensive, and weak with small generators. | Executor-grounded domain and spare compute (M8) |
| Distributed infrastructure in V1 | Current work fits one node; stable contracts are sufficient now. Do not add Kubernetes, Ray clusters, queues, distributed locks, PostgreSQL or object storage for appearance. | Replace execution/provider/store implementations when M10 fires |
| Premature cluster managers | Kubernetes, Ray/Slurm orchestration and message queues are not justified without a resource trigger. | M10 |
| Universal trainer/evaluator abstraction | A universal API with dozens of optional fields hides semantics and has no proven second implementation. | Add only the narrow method needed by a real backend |
| Bit-exact GPU replay | Kernel and generation scheduling may be nondeterministic. | Not planned; operational resume and seed replication are required |
| Large-model evidence from the reference run | Small-model behavior is engineering evidence only. | Run the `research` profile and repeat target calibration |
| Automatic hyperparameter tuning in the loop | Repeated tuning against evaluation sets is overfitting. | Never inside the gated loop; offline on the dev split only |
| Thinking-mode training | Rollout lengths are unaffordable under current compute. | ~5× compute and truncation evidence (M6) |
| Eliminating all forgetting or all reward hacking | Not achievable; the goal is detection and containment. | — |

---
| Claims about mathematics in general | Measurement sets contain only exact-numeric, training-source-style problems; the supportable claim is narrow (§36.3). | A broader independent-source endpoint with adequate power exists |
| Symbolic-answer measurement | Keeping the independent scorer small enough to test exhaustively (D27). | A second independent symbolic checker is justified by evidence |

---

## 7. Architectural Principles

**P1. Separate the training channel from the measurement channel.** Everything the gate relies on differs from training in at least one important way: a separately written scorer with no shared code, evaluation-only templates, disjoint splits, a different-source transfer set, and hidden-set labels computed by executing programs. *Enforced by* separate `TrainVerifier`/`EvalScorer` types, an import test, cluster-level splits, a schema constraint that only exact-numeric items enter measurement splits, and `assert_not_measurement` on every trainer input. *Limitation:* reference labels shared between training sources and same-source gate sets remain a correlated-error channel (§21.1); the transfer set and the hidden set exist to bound it.

**P2. The champion is immutable; checkpoints are full, resumable states.**

**P3. Deterministic checking; no learned judge decides anything.**

**P4. Nothing uncalibrated decides.** Every threshold is a pre-registered choice (floor, margins, α) or is derived from Phase I measurements. A threshold is never chosen after seeing the result it decides.

**P5. Halt, do not self-repair — and treat human repair as a budgeted search.** The controller never changes hyperparameters, data or thresholds. A human who resumes after a halt with a changed configuration is performing the same search more slowly, so each such change consumes a pre-registered human-change budget and forces a Gate-A refresh.

**P6. Count and budget every look.** Each read of Gate-A, Gate-B, the transfer sealed part and both hidden parts is logged against a budget. The claim set is read exactly once per run.

**P7. Data assets are durable; checkpoints are not.**

**P8. Extensions enter through existing interfaces, only when a measurable trigger fires.**

**P9. Validate the instrument before trusting its verdicts.** The gate's power at the floor, its false-promotion rate, the bootstrap's coverage and the controller's false-halt rate are measured on the target model (E3–E5) before the gated loop runs. A verdict from an unvalidated instrument is not evidence.

**P10. Measurement cadence follows the learning curve, not the other way round.** How often the gate runs and how long a lineage may train without promotion are set from the observed learning curve (E1). Inconclusive results continue training; only confident regressions restore.

**P11. Separate the control plane from the execution plane.** Controller, gate, lineage, budgets and provenance do not depend on local or remote topology.

**P12. Cross execution boundaries only with versioned serializable contracts.** Expensive work is described by project-owned `JobSpec` and `JobResult` values, never live runtime objects.

**P13. Expensive work is idempotent.** A stable logical job ID can affect artifacts, budgets and lineage at most once, regardless of retries.

**P14. Persistent identities are location-independent.** Core logic uses `StateRef`, `ArtifactRef` and `DatasetRef`; only storage implementations resolve physical locations.

**P15. The small-model implementation defines reference semantics.** `smoke` and `dev` must exercise the same control/job/storage boundaries as `research`, while making engineering-only claims.

**P16. Algorithmically relevant library behaviour is explicit.** No scientific decision depends on an undocumented or default library setting; pinned values and runtime compatibility assertions enter the stack hash.

---

## 8. Final Architecture

A single Python control-plane process owns state-machine and decision semantics. In V1 it uses a `LocalSubprocessExecutor`, `SQLiteMetadataStore` and `LocalArtifactStore`; these are reference execution-plane implementations rather than assumptions embedded in the controller.

| Component | Responsibility | V1 reference implementation |
|---|---|---|
| `ModelSpec` | Scientific model identity: pinned model/tokenizer revisions, chat-template arguments, context limit, LoRA targets and inference mode. | Versioned YAML model spec |
| `ExecutionProfile` | Backend/resource configuration: dtype, quantization, generation/training backends, topology and backend options. | `smoke`, `dev-qwen06b-local24g`, `research-qwen4b-local80g` YAML profiles |
| `JobExecutor` | Submit, observe, retrieve and cancel versioned jobs without exposing placement to the controller. | `LocalSubprocessExecutor` |
| Job schemas | Serializable training/evaluation/verification requests and results with stable IDs, refs, hashes, attempts and errors. | Project-owned dataclasses/validated JSON |
| `RolloutProvider` | Produce project-owned on-policy rollout batches without leaking vLLM return types. | `ColocatedVLLMRolloutProvider` |
| `EvalGenerationProvider` | Generate evaluation completions under evaluation-specific correctness rules. | `LocalVLLMEvalProvider` |
| `MetadataStore` | Runs, jobs, lineage, budgets, decisions, looks, prompt statistics, events and preregistration hashes. | `SQLiteMetadataStore` |
| `ArtifactStore` | Immutable states, optimizer data, rollout/evaluation files, datasets, reports and manifests. | `LocalArtifactStore` on local NVMe |
| Ingest pipeline | Normalise, filter, cluster, decontaminate, split; mark items `measurable`; log rejections. | `tasks/` modules |
| Prompt table | Pass rates (by verifier version), global exposure, quarantine, audit flags; serves training slices. | SQLite + `PromptSampler` |
| On-policy trainer | GRPO-family policy updates through `RolloutProvider`; stores raw rollouts; coarse sync check. | TRL `GRPOTrainer` wrapper with explicit pinned IS/TIS settings |
| Offline trainer | SFT on verified examples with provenance. | RFT (Phase I control, E9) |
| Train verifier | Binary correctness for reward, all answer types. | `math-verify` in sandboxed pool |
| Eval scorer | Content accuracy and format compliance on exact-numeric items. | `exact_scorer.py` (stdlib only) |
| Eval runner and suites | Request generation, score completions, store raw text and draw fresh Gate-B references. | `EvalGenerationProvider` + exact scorer |
| Statistics | Paired cluster bootstrap, sizing, simulation of gate power and coverage. | `stats.py` |
| Gate | Primary endpoint + guardrails with explicit directions. | `DefaultGate` |
| Controller | States, segment scheduling, monitor/gate cadence, step budget, tripwires, milestones. | Branch-complete with replay in Phase 0; used for the research gated loop only after go/no-go |
| Lineage store | Base, champion pointer, stable state refs, lineage transitions and pruning. | Metadata + artifact stores |
| Ledger | Looks, read budgets, human changes, decisions, events. | SQLite |
| Phase I harness | Runs E1–E9 from stored completions where possible. | `scripts/phase1/` |
| Reporting | Metrics and decision reports. | SQLite + generated Markdown/HTML reports (MLflow deferred) |

### 8.1 Stable references and serializable jobs

Project-owned identities include `RunId`, `JobId`, `StateId`/`StateRef`, `ArtifactId`/`ArtifactRef`, `DatasetId`/`DatasetRef`, `PromptId`, `ClusterId`, `EvalReadId`, `ConfigId`, `StackId`, `VerifierVersion`, `ScorerVersion` and `PreregistrationId`. Immutable manifests and datasets use content hashes where practical. A logical checkpoint is never identified by a path outside `ArtifactStore`.

`TrainingJobSpec`, `EvaluationJobSpec` and optional `VerificationJobSpec`, with corresponding results, are versioned and JSON-serializable. Every job value contains `schema_version`, stable `job_id`, config and stack hashes, model-spec and execution-profile IDs, input refs, expected output refs, code revision, attempt number and timestamps. Failures carry a stable error classification. Cross-boundary values never contain absolute paths, CUDA objects, model/tokenizer instances, callables, open connections or process-local locks.

```python
class JobExecutor(Protocol):
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def result(self, handle: JobHandle) -> JobResult: ...
    def cancel(self, handle: JobHandle) -> None: ...

class RolloutProvider(Protocol):
    def generate(self, request: RolloutRequest) -> RolloutBatch: ...

class EvalGenerationProvider(Protocol):
    def generate(self, request: EvalGenerationRequest) -> EvalGenerationBatch: ...
```

`RolloutBatch` is project-owned and contains stable prompt IDs, completion text, token IDs, sampled log probabilities, truncation flags and generation metadata. Evaluation generation remains a distinct contract because it has different correctness requirements from on-policy rollouts. Metadata schemas are versioned and upgraded through explicit, tested migrations before durable V1 state is relied on across releases.

Backends publish capability descriptors such as supported dtypes/quantization, maximum context, LoRA support, training/generation backends, tensor-parallel limits, TIS support and whether generation is colocated. Compatibility is checked before submission, not discovered halfway through a job.

### 8.2 Idempotency and atomic finalization

Every expensive operation has a stable `job_id` derived from immutable inputs. Retrying a job cannot create a second logical segment, spend a read budget twice, or apply two lineage transitions.

1. The control plane records a pending job with its expected inputs and output refs.
2. The executor may make any number of attempts under that logical ID; attempts are recorded separately.
3. Artifacts are written to a temporary identity and validated before finalization.
4. `ArtifactStore.finalize()` publishes an immutable manifest and content hash atomically within that backend.
5. One metadata transaction accepts the finalized refs, records the result, consumes any budget and applies the lineage transition.
6. Duplicate results return the already accepted logical result. Conflicting results for one job ID halt the controller.
7. Failed or abandoned temporary artifacts are non-semantic and garbage-collectable; metadata never points to them.

Local SQLite/filesystem code must implement these semantics before remote execution is introduced. Contract and failure-injection tests cover crash-before-finalize, crash-after-finalize-before-commit, duplicate delivery, conflicting delivery, stale results and retries.

### 8.3 Replay backend and backend contracts

V1 includes a `ReplayExecutor` that consumes stored job results and drives the real controller, gate, budgets and lineage logic without GPU work. It is used for exhaustive decision branches, deterministic regression tests, historical reconstruction and controller shadow runs; it never supplies scientific model evidence.

Every replaceable backend passes the same contract suite for serialization, stable IDs, capabilities, idempotency, atomic finalization, error classification and recovery. Local implementations are the reference behavior. Future Slurm/Ray/remote executors, PostgreSQL metadata, object storage and remote rollout/evaluation providers are deferred until their triggers fire.

---

## 9. Architecture Diagram

### 9.1 System

```
                    ORCHESTRATOR (one Python process · SQLite · generated reports)
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ DATA  curated sources ─► normalize ─► filter ─► cluster ─► decontaminate ─► split    │
 │       (measurement splits: exact-numeric answers only)          │                    │
 │                         PROMPT TABLE: pass rates (by verifier version) · exposure    │
 │                              suspected-label flags · confirmed quarantine · audits   │
 │ TRAINING CHANNEL                                                ▼                    │
 │ ┌───────────────────────────────────────────────────────────────────────────────┐  │
 │ │ GPU segment (N steps): TRL GRPO-variant · bf16 LoRA · vLLM colocate · TIS on  │  │
 │ │   reward ◄── TrainVerifier (math-verify, sandboxed pool) · raw rollouts saved │  │
 │ │   coarse sync check · mismatch, zero-variance share and lengths logged        │  │
 │ └──────────────────────────────────┬────────────────────────────────────────────┘  │
 │                                    │ checkpoint (full state) every N steps          │
 │ MEASUREMENT CHANNEL (independent)  ▼                                                │
 │ every N:  monitor  — screen set, unstratified: confident regression ─► RESTORE     │
 │ every G:  gate     — Gate-A primary ─► guardrails vs base ─► Gate-B (fresh refs)    │
 │ milestones: hidden_audit (budgeted)       run end: hidden_sealed + transfer_sealed  │
 │ EvalScorer = exact_scorer (stdlib) · eval-only templates · paired cluster bootstrap │
 │                                    │                                                │
 │ GATE ─► PROMOTE │ CONTINUE (until step budget S) │ RESTORE (confident regression) │ HALT
 │ LINEAGE (base + adapter states)    LEDGER (looks, budgets, human changes)           │
 │ TRIPWIRES (pre-registered false-halt budget) ──► HALT + notify human                │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Data flow

```
pinned public datasets (Parquet/JSON only)
 ─► normalize answers, assign answer_type, mark measurable (int / rational / terminating decimal)
 ─► drop multiple-choice, yes/no; mark low-entropy
 ─► cluster near-duplicates (MinHash + embeddings + source lineage); hidden set: cluster = program
 ─► decontaminate training clusters against every measurement split and public benchmarks
 ─► split by cluster: train | dev | pilot | screen | gate_a (+reservoir) | gate_b (+reservoir)
                      transfer_guard | transfer_sealed (different source)
                      hidden_audit | hidden_sealed (private programs, split by program)
 ─► stratification samples (never reused): base pass rate per gate item; drop saturated
 ─► initial training pass rates (base, k=8, once)
 ─► per segment: band-filtered, quota-balanced, exposure-capped slice ─► trainer
 ─► raw rollouts (Parquet) + per-task counts (tagged with verifier version) ─► prompt table
```

### 9.3 Evaluation and decision flow

```
every N steps:  screen (unstratified, small)
                  upper bound < 0 ────────────────────────────────► RESTORE
every G steps:  Gate-A primary: content accuracy, eval-template rotation, stratified, vs champion
                  upper < 0 ──► RESTORE      lower ≤ 0 ──► CONTINUE (HALT if steps ≥ S)
                guardrails vs BASE (transfer, instruction following, multilingual math, format)
                  upper < −margin ──► RESTORE    lower ≤ −margin ──► CONTINUE (HALT if steps ≥ S)
                Gate-B primary with FRESH reference samples
                  upper < 0 ──► RESTORE      lower ≤ 0 ──► CONTINUE (HALT if steps ≥ S)
                ──► PROMOTE
every P promotions: hidden_audit read (budgeted) + human sign-off
end of run:         hidden_sealed and transfer_sealed, read exactly once ──► final claim (§36)
```

### 9.4 Controller states

```
            ┌──────────── resume (human; config change consumes budget) ───────┐
            ▼                                                                   │
       ┌─────────┐  tripwire · R restores · step budget S · read budget ── ┌────────┐
       │ RUNNING │ ──────────────────────────────────────────────────────► │ HALTED │
       └────┬────┘                                                          └────────┘
            │ P promotions                                                      ▲
            ▼                                                                   │
   ┌──────────────────┐ ── sign-off ──► RUNNING     audit contradiction ───────┘
   │ MILESTONE_REVIEW │
   └──────────────────┘
```

---

## 10. Core Learning Loop

Training proceeds in **segments** of `N` steps (*start* 50), each ending in a full-state checkpoint. Every checkpoint gets a cheap **monitor** evaluation that can only detect a confident regression. Every `G` steps (a multiple of `N`, set from the E1 learning curve) the checkpoint goes to the **gate**. A lineage may train for at most `S` steps (the step budget, set from where E1's learning curve flattens) without a promotion before the system halts. This decouples how often the system measures from how long it is allowed to learn (P10).

```python
state, since = lineage.champion_state(), 0
while controller.status == RUNNING and budget.remaining():
    prompts = sampler.next_slice(N * prompts_per_step, seed=next_seed())
    assert_not_measurement(prompts)
    spec = TrainingJobSpec.from_refs(state, prompts, model_spec, execution_profile,
                                     config_hash, stack_hash, steps=N)
    handle = executor.submit(spec)
    result = executor.result(handle)                       # project-owned JobResult
    metadata.accept_result_once(result, artifacts)         # idempotent + transactional
    if not result.sync_ok:                                   # orders-of-magnitude mismatch only
        controller.sync_failure()                            # retry segment once, then HALT
        continue
    ckpt = lineage.save(result); prompt_table.update(result.rollout_stats)
    since += N

    if monitor_regression(evaluator.screen(ckpt, lineage.champion())):   # confident regression
        lineage.archive(ckpt); state, since = lineage.champion_state(), 0
        controller.restores += 1
        if controller.restores >= R: controller.halt("repeated confident regressions")
        continue
    state = ckpt
    if tripwires.any_fired(): controller.halt(tripwires.reasons()); continue

    if since % G == 0:
        d = gate.decide(gate_context(ckpt, steps_since_promotion=since), ckpt, champion, base)
        if d.outcome is PROMOTE:
            lineage.promote(ckpt); since = 0; controller.restores = 0
            prompt_table.refresh_band_edges(); label_audit.flag_consensus_disagreements()
            if controller.promotions_since_milestone() >= P: controller.enter(MILESTONE_REVIEW)
        elif d.outcome is RESTORE:
            lineage.archive(ckpt, d); state, since = lineage.champion_state(), 0
            controller.restores += 1
            if controller.restores >= R: controller.halt("repeated confident regressions")
        elif d.outcome is HALT:
            controller.halt(d.reasons)          # step budget, read budget, or tripwire
        # CONTINUE: keep training from ckpt; the champion is unchanged
```

| Outcome | Trigger | Next segment starts from | Champion |
|---|---|---|---|
| PROMOTE | Primary confirmed on Gate-A and Gate-B; guardrails demonstrated | Candidate | Becomes candidate |
| CONTINUE | Anything inconclusive | Candidate (same lineage) | Unchanged |
| RESTORE | Confident regression (monitor, primary, or guardrail violation) | Champion state, next data slice | Unchanged |
| HALT | Tripwire; step budget `S` spent; read budget spent; `R` restores | — | Unchanged |

**What happens at the end of a run.** A run ends when its pre-registered compute budget is spent or the controller halts on the step budget. The final champion is then evaluated once on `hidden_sealed` and `transfer_sealed` against the base; those single reads produce the run's claim (§36).

---

## 11. Model Strategy

### D01 — Reference model and research target

| | |
|---|---|
| **Chosen** | `dev`: **Qwen3-0.6B** as the canonical reference architecture and end-to-end integration profile. `research`: **Qwen3-4B-Instruct-2507** as the target for scientific calibration and improvement claims. |
| **Alternatives considered** | Start directly at 4B with no small model; use Stage A to validate the measurement system; Qwen3-4B-Base; the hybrid Qwen3-4B; Qwen3.5 small models (2026). |
| **Why rejected** | *No small model:* pipeline bugs found at ~7× the cost. *Stage A as instrument validation (Revision 1):* effects at 0.6B are large and fast, mostly format and answer commitment, so they say nothing about sensitivity to a few-point gain at 4B. *Base model:* rebuilding instruction-following is unaffordable. *Hybrid Qwen3-4B:* mode interference. *Qwen3.5:* hybrid linear-attention architecture; tooling must be verified first (M5). |
| **Why it fits** | The reference profile validates real training, storage, restart, jobs, budgets, lineage, controller branches, replay, failure recovery and reports cheaply. The research profile answers the model-improvement question. Both are dense transformers with Apache-2.0 licences. |
| **Limitations** | The target was heavily post-trained by its developers, including on mathematics; headroom is limited and gains may be small or partly recall (A5). |
| **Replace or extend** | New `ModelSpec` plus an appropriate `ExecutionProfile` after `scripts/spike_model.py` passes; repeat reduced target-model calibration. |

### D02 — Inference mode: non-thinking, but not "short"

| | |
|---|---|
| **Chosen** | Non-thinking only. Qwen3-4B-Instruct-2507 supports only non-thinking mode; Qwen3-0.6B is rendered with `enable_thinking=False`. |
| **Alternatives considered** | Thinking mode (Qwen3-4B-Thinking-2507); mixing modes. |
| **Why rejected** | Thinking-mode rollouts are longer still; mixing modes makes training and evaluation ambiguous. |
| **Why it fits** | One mode, one distribution. |
| **Limitations** | **Non-thinking does not mean short.** The vendor reports AIME25 of 47.4 for Instruct-2507 against 19.1 for the original Qwen3-4B in non-thinking mode, and recommends a 16,384-token output length for most queries. A model at that level writes long reasoning on hard problems, and the band filter selects exactly the hard problems. Revision 1 assumed short rollouts; Revision 2 measures them (E1) and sizes the hardware and cap from the measurement. |
| **Replace or extend** | Thinking profile at ~5× compute (M6). |

### 11.1 Model specification and execution profile

`ModelSpec` says **what model is being experimented on**: model and tokenizer identity/revisions, chat-template arguments, context limit, LoRA targets and inference mode. `ExecutionProfile` says **how it is run**: backend, dtype, quantization, generation/training implementations, concurrency, tensor parallelism, checkpointing, offload and backend options. Scientific identity does not depend on incidental topology, but execution settings that can change numerical behaviour enter the stack hash.

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

| `ModelSpec` field | Dev (Qwen3-0.6B) | Research (Qwen3-4B-Instruct-2507) |
|---|---|---|
| Revision | Pinned commit | Pinned commit |
| Tokenizer | Pinned ID/revision | Pinned ID/revision |
| Chat template arguments | `enable_thinking=False` | Explicit empty/non-thinking arguments |
| Context limit | Pinned model limit | Pinned model limit |
| LoRA targets | q, k, v, o, gate, up, down | same |

| `ExecutionProfile` field | Dev local 24G | Research local 80G prior |
|---|---|---|
| Backend | local reference | local reference; replaceable if measurements require |
| Dtype / quantization | bf16 / none | bf16 / none initially |
| Training / generation | TRL / colocated vLLM | TRL / colocated vLLM initially |
| Rollout sampling | T 1.0, top_p 1.0, top_k off, min_p 0 | same |
| Evaluation sampling | T 0.7, top_p 0.8, top_k 20, min_p 0 | same (vendor guidance) |
| Generation cap | From a short Stage A run | *E1*: smallest cap at which truncation on band-filtered prompts is low **and** accuracy on the hardest in-band prompts no longer rises with length; the vendor's 16,384 is the starting point |

### 11.2 Compatibility spike

`scripts/spike_model.py` must pass before a model/profile pair is used: capability compatibility; LoRA attach; vLLM LoRA serving; two GRPO steps in the selected mode with the exact preregistered IS/TIS settings; runtime assertions for those settings; logging of mismatch metrics; peak memory at the planned cap; and export parity. The spike no longer sets a mismatch threshold; mismatch is monitored on real rollouts (§15).

---

## 12. Data and Task Generation

### D03 — First domain: mathematics with final-answer verification

| | |
|---|---|
| **Chosen** | One domain: mathematics problems with a single checkable final answer. |
| **Alternatives considered** | Code with unit tests; tool use; several domains. |
| **Why rejected** | Code needs a hardened execution harness and has a larger exploit surface; tool use needs environments; multiple domains add interference that cannot be measured yet. |
| **Why it fits** | Cheapest reliable checking; no execution of model-written code; plenty of labelled data. |
| **Limitations** | Reasoning is not checked; mathematics gains need not transfer elsewhere. |
| **Replace or extend** | Code first (M1). |

### D07 — Task source: curated, verified datasets only

| | |
|---|---|
| **Chosen** | Public mathematics datasets with known answers and pinned revisions. No unverified model-generated label or test enters training or measurement. Model-drafted hidden programs are allowed only under §12.2's execution-label and human-review rules. |
| **Alternatives considered** | LLM-generated problems; failure rewriting; procedural generators; teacher data. |
| **Why rejected** | Generated problems come with unverified labels; procedural generators teach template behaviour; teacher data adds a dependency and licence question. |
| **Why it fits** | Correct labels are the scarce resource. |
| **Limitations** | Finite pool; label errors; overlap with what the vendor trained on (A5). |
| **Replace or extend** | New `TaskSource` through the same ingest pipeline (M2, M3). |

### D27 — Measurement sets contain only exact-numeric answers

| | |
|---|---|
| **Chosen** | Every measurement split (pilot, screen, gate sets, transfer, hidden) contains only items whose reference answer is an integer, a rational number or a terminating decimal. Training may still use symbolic answers. Enforced by a schema constraint. |
| **Alternatives considered** | Full symbolic measurement with an independently written symbolic scorer (Revision 1). |
| **Why rejected** | A from-scratch symbolic scorer re-implements a large amount of LaTeX and equivalence handling, produces more false negatives than `math-verify`, and as RL pushes outputs toward forms the train verifier accepts, scorer bugs would shrink measured gains and trigger halts that are not hacks. It could absorb months. |
| **Why it fits** | The scorer becomes a small exact-arithmetic module that can be tested exhaustively, independence is preserved, and no parser that can execute code runs on the measurement side. |
| **Limitations** | **The claim narrows** to exact-numeric problems. Problems with symbolic answers are trained on but not measured. |
| **Replace or extend** | A second independent symbolic checker later, if evidence justifies it. |

### D08 — Ingestion: cluster-level deduplication, decontamination and splitting

| | |
|---|---|
| **Chosen** | Normalise → filter → cluster → decontaminate → split by cluster, all logged. For the hidden set the cluster is the perturbation **program**. |
| **Alternatives considered** | Exact-match dedup; item-level splits; decontaminating only against public benchmarks. |
| **Why rejected** | Near-copies of evaluation items would enter training. |
| **Why it fits** | Prevents the most common way a gate set becomes a memorisation test, and defines the unit of independence used by the bootstrap. |
| **Limitations** | Structurally identical problems with new wording and numbers can still escape; E7 measures how much gate-set accuracy reflects recall. |
| **Replace or extend** | Additional `IngestStage`s. |

**Pipeline:** (1) normalise answers, assign `answer_type`, mark `measurable`; (2) drop multiple-choice and yes/no, cap low-entropy integers (|value| ≤ 10) at 10% of a training slice; (3) cluster with MinHash on 5-grams, embedding similarity and source lineage; (4) remove any training cluster that touches any measurement item or public benchmark; (5) split by cluster.

### 12.1 Splits, sources and read rules

| Split | Source | Unit | Purpose | Read rule |
|---|---|---|---|---|
| train | Training sources | — | RL prompts | Continuous |
| dev | Training sources | cluster | Human debugging and the E1a learning-rate check | Any time; never a gate input |
| pilot | Training sources, held-out clusters | cluster | Phase I only: large disposable set for E1–E9 | Unlimited in Phase I; **retired** afterwards, never used for gating |
| screen | Training sources | cluster | Monitor (confident regression only), unstratified | Every checkpoint |
| gate_a / gate_a_reservoir | Training sources | cluster | Primary endpoint, stratified | Budgeted; refreshed from reservoir after the budget or any human change |
| gate_b / gate_b_reservoir | Training sources | cluster | Confirmation, fresh reference samples, stratified | Budgeted; refreshed from reservoir |
| transfer_guard | **Different source** | cluster | Non-inferiority guardrail | Every gate decision reaching guardrails |
| transfer_sealed | Same different source | cluster | Secondary endpoint at run end | Once per run |
| hidden_audit | Private programs | program | Milestone audit | Budgeted |
| hidden_sealed | Private programs, disjoint | program | **Primary claim** at run end | Exactly once per run |

**Stage B instrument, defined now** (Revision 1 left it open):

| Role | Stage A | Stage B |
|---|---|---|
| Training | GSM8K train; MATH train levels 1–3; easy Big-Math subset | MATH train (all levels); filtered Big-Math / other open RL math sets |
| Transfer (guard + sealed) | SVAMP; ASDiv numeric | Omni-MATH numeric-answer subset, provided ingest confirms that no training source contains it (source lineage plus decontamination); otherwise the next candidate olympiad-style source with numeric answers is chosen before Phase I |
| Hidden (audit + sealed) | Private perturbations | Private perturbations of held-out training-source problems |
| Secondary, descriptive | — | Competition problems with numeric answers published after the checkpoint's release; reported with intervals, never gated (sets of AIME size cannot resolve a few points) |
| Public benchmarks | GSM8K test; MATH-500 | MATH-500 numeric subset — reported only |

Sizes of every gated or claimed set are given in clusters (programs for hidden sets) and come from E1 variance and the E3 power simulation (§21.7).

### 12.2 Hidden set construction

Built once by `scripts/build_hidden_perturbations.py` from held-out training-source problems. Each problem becomes a small program (written once by a strong external model, reviewed by a human) that samples numbers and computes the answer by execution.

- **Invariant assertions are mandatory.** Each program declares constraints its numbers must satisfy (integer counts, positive quantities, valid probabilities, plausible ranges, a unique answer). Instances that fail are rejected and regenerated. A human reviews a random sample of *generated instances*, not only the template, because resampled numbers can produce nonsensical problems a template review will not reveal.
- **Programs are the unit.** Instances of one program share difficulty, so the bootstrap resamples programs, and sizes are set in programs.
- **Split by program** into `hidden_audit` and `hidden_sealed`.

*Limitation:* perturbations test robustness to memorisation of numbers, not breadth: they reuse the reasoning templates of training-source problems.

---

## 13. Replay

### D09 — No training-data replay in V1; execution replay is required

| | |
|---|---|
| **Chosen** | No prompt/response replay for training in V1. A `ReplayExecutor` for stored `JobResult`s is required to validate controller, budget, lineage and recovery semantics without repeating GPU work. |
| **Alternatives considered** | Prompt-level regression replay (Revision 1); fixed replay mixtures; replay of stored responses. |
| **Why rejected** | Regression replay detected drops from pass-rate estimates built on a few groups of 8; it would trigger rarely or on noise, and added accounting. Mastered prompts give zero gradient in group-relative RL; stored responses are off-policy. |
| **Why it fits** | Forgetting is controlled by measurement (guardrails against the base, capability monitors) and by on-policy RL's comparatively low drift (Shenfeld et al., 2025). |
| **Limitations** | No mechanism actively retrains regressed training prompts. |
| **Replace or extend** | Reintroduce a `ReplayPolicy` if E1 or the main run shows confident regression on training-source items that the gate does not already catch, or with SFT/distillation trainers (M2). |

---

## 14. Curriculum

### D10 — Rule-based selection: derived band, quotas, global exposure caps, verified quarantine

| | |
|---|---|
| **Chosen** | Select training prompts whose estimated pass rate lies in a band derived from the group size; balance by source and skill; cap exposure per cluster (global counts). Model consensus against a label creates an audit flag; exclusion occurs only after an independent verifier, deterministic derivation or human confirms the label problem. |
| **Alternatives considered** | Evaluation-driven reallocation; learned curriculum; generated tasks; lineage-scoped accounting and quarantine-by-persistence (Revision 1). |
| **Why rejected** | Evaluation-driven reallocation leaks measurement into training. Lineage-scoped accounting mattered when restores were frequent; with restores limited to confident regressions, global counts are adequate and simpler. Quarantine-by-persistence removed correctly labelled frontier prompts that are improving slowly — the most valuable ones — while mislabelled prompts mostly sit near p ≈ 0, out of band, where they do no harm. |
| **Why it fits** | Without selection, most rollouts go to prompts that produce no gradient; TRL does not implement dynamic sampling, so the band filter is the only defence against zero-variance groups. |
| **Limitations** | Pool saturation; never-solved skills are never trained; possible narrowing (monitored). |
| **Replace or extend** | `PromptSampler` implementations; generative sources via `TaskSource` (M3). |

### 14.1 Band derivation

For pass rate *p* and group size *G*, P(a group has any reward variance) = 1 − p^G − (1 − p)^G. The band is where this is ≥ 0.5; for G = 8 that is approximately [0.083, 0.917] (`band_for` in `interfaces.py`). The share of groups that still have zero variance is measured every segment (§15) and reported; it is the direct measure of how well the filter works.

### 14.2 Pass-rate estimation

Initial: k = 8 samples per training prompt from the base, once. Ongoing: every rollout group updates its prompt's estimate for free. After a promotion, only prompts within 0.05 of a band edge are re-estimated. Estimates are Beta(1,1) posterior means. Every observation carries the **verifier version** that produced it, so observations from a verifier later found buggy can be excluded and rebuilt from stored rollouts (§23).

### 14.3 Quotas and caps

No source or skill above 50% of a slice; low-entropy items ≤ 10%; at most 4 rollout groups per cluster (global count, *start*).

### 14.4 Quarantine and audit flags

- **Flag then verify:** at least 60% of samples agreeing on one answer that differs from the label across two estimates creates `suspected_label_error`; the prompt remains eligible unless an immediately safe temporary hold is explicitly configured.
- **Exclusion:** `quarantined_confirmed` requires independent confirmation by deterministic derivation, a second verifier with a different failure mode, or human review. The evidence, verifier/version and reviewer are durable metadata. Model consensus alone never excludes a frontier prompt.
- **Resolution:** confirmed-good labels return to normal eligibility; fixed labels invalidate dependent pass-rate observations; irreparable items are dropped with an auditable reason.
- **Flag only (former rule a):** a prompt in the band across 3 consecutive promotions is flagged for audit but **stays in training**.

---

## 15. Rollout

### D11 — On-policy rollouts inside the trainer, with importance-sampling correction

| | |
|---|---|
| **Chosen** | The V1 `ColocatedVLLMRolloutProvider` generates from the current LoRA weights at temperature 1.0 and top_p 1.0. Truncated importance sampling is explicitly enabled with the pinned implementation, granularity, cap 2.0 and logged ratio/truncation metrics; none of these semantics are inherited from a library default. |
| **Alternatives considered** | Pre-generated batches; separate rollout servers; truncated sampling (top_p < 1); a halting parity tripwire (Revision 1). |
| **Why rejected** | Pre-generated batches are off-policy after the first step. Truncated sampling makes the sampling distribution differ systematically from the model's log-probabilities. A parity tripwire mis-specifies a persistent, known property of this setup as a failure: the gap between vLLM and trainer log-probabilities is normal, can drift during training, and is what TIS corrects; a threshold set from a two-step spike on a fixed probe is not a stable reference. |
| **Why it fits** | Correct algorithm, one model copy in the reference backend, mismatch handled by an explicit correction whose runtime configuration is asserted against preregistration. |
| **Limitations** | TIS reduces but does not remove mismatch bias; the cap trades bias for variance. |
| **Replace or extend** | Replace `RolloutProvider` with `RemoteVLLMRolloutProvider` at M10; trainer and control-plane contracts remain unchanged. |

**Per step (start):** 16 prompts × 8 samples = 128 completions under T_train. `num_iterations = 1` (one optimizer update per generation batch); see §18 for what that implies for clipping.

**Logged on every step, monitored, never gated:** TRL's importance-sampling statistics (mean ratio, fraction truncated at the cap), completion length (mean, p95), truncation rate, share of zero-variance groups, token entropy.

**Coarse sync-failure tripwire.** An adapter that fails to sync to vLLM produces a log-probability gap orders of magnitude above normal. After each weight sync a fixed probe is scored by both engines; the tripwire fires only if the gap exceeds 100× its running median over the last 50 syncs. Normal drift does not fire it. On firing, the segment is aborted and retried once, then the system halts.

**Raw rollouts are stored** (Parquet, with prompt id, completion text, verdict, reward and verifier version) for every step. This is what makes re-scoring after a verifier bug possible without regeneration (§23).

---

## 16. Verification

### D12 — Two independent checkers: math-verify for reward, exact arithmetic for measurement

| | |
|---|---|
| **Chosen** | `TrainVerifier`: `math-verify`, all answer types, in a sandboxed worker pool. `EvalScorer`: `exact_scorer.py`, a stdlib-only exact-arithmetic module for integer/rational/terminating-decimal answers. No shared code. |
| **Alternatives considered** | One shared extractor; a from-scratch symbolic eval scorer (Revision 1); an LLM judge; a reward model. |
| **Why rejected** | A shared extractor makes reward and measurement errors identical. The symbolic scorer was an engineering sink and a liveness risk (D27). Judges and reward models are exploitable and unnecessary. |
| **Why it fits** | Independence turns checker bugs into measurable disagreement; the measurement scorer is small enough to test exhaustively and cannot execute code. |
| **Limitations** | Neither checker sees reasoning or detects wrong reference labels shared by training and same-source gate sets. |
| **Replace or extend** | Code checker pair at M1. |

### 16.1 Train verifier

Extracts the final boxed answer and tests equivalence with the reference using `math-verify`. Two or more distinct boxed answers are wrong. Runs in worker processes in a container with no network, read-only filesystem, non-root user, resource limits and a 5-second CPU-time limit per check, because SymPy parsing can execute code and simplification can hang.

### 16.2 Eval scorer (`exact_scorer.py`)

- **`score_content`** (the primary metric): candidates are pooled from every recognised marker (`\boxed{}`, an `Answer:` line, "the final answer is …"). Parsed exactly with `fractions.Fraction`: integers, thousands separators, decimals, `a/b`, `\frac`/`\dfrac`/`\tfrac`, leading "x =", trailing units in `\text{}`, percent signs. More than one distinct value among markers → `multiple_answers` (wrong). No marker → the last number on the last non-empty line, unless that line hedges between values. Truncated → wrong.
- **`format_compliant`**: exactly one occurrence of the template's instructed marker, consistent with the content answer. Feeds the format-compliance guardrail, not the capability metric.
- **Validation (synthetic):** `selfcheck.py` runs 21 fixed cases including hedging and malformed answers, and 4,000 randomized fractions rendered in up to eight formats each, plus wrong-value and two-answer variants; all pass. An import check confirms the module imports nothing beyond the standard library.

### 16.3 Testing and monitoring

- Golden and exploit suites for the train verifier; 100% required.
- Property tests for the eval scorer (above), run in CI.
- `tests/test_independence.py` — no shared imports.
- **E6 checker study (Phase I):** `math-verify` against `exact_scorer` on base and E1 outputs for exact-numeric items, with disagreements categorised. Its result fixes the scorer's scope and the baseline disagreement rate.
- **Dual-checker disagreement tripwire:** on training rollouts for *measurable* training items, the rate at which the train verifier accepts and the eval scorer rejects the **same output**. This is the actual reward-hacking signature and replaces Revision 1's reward/measurement divergence tripwire (§24).

---

## 17. Reward

### D13 — Binary reward; truncation counts as wrong

| | |
|---|---|
| **Chosen** | 1 if correct and not truncated, otherwise 0. Overlong shaping is implemented but **off**; it is switched on only by a pre-registered rule if E1 shows a truncation problem. |
| **Alternatives considered** | Revision 1's −1 for truncation with a soft penalty; partial credit; format or length bonuses; masking truncated outputs. |
| **Why rejected** | −1 for truncation, with no reward for abstaining, strengthens pressure to answer early and to guess on hard problems — the failure Revision 1 itself listed. Partial credit and bonuses invite exploitation. Masking makes rambling free. |
| **Why it fits** | Simple, hard to exploit, aligned with the measured objective. The generation cap is set high enough (E1) that truncation is rare, so the choice of truncation reward matters little. |
| **Limitations** | Sparse on hard problems; rewards right answers reached by flawed reasoning; no reward for appropriate abstention. |
| **Replace or extend** | `RewardFunction` implementations; abstention and process rewards deferred. |

**Monitored:** accuracy as a function of completion length on the hardest in-band prompts (bottom pass-rate quartile), per segment. A falling accuracy at long lengths, or length collapsing on hard prompts, is the signature of premature answering.

---

## 18. RL Training

### D05 — Algorithm: GRPO-family through a pinned TRL release

| | |
|---|---|
| **Chosen** | TRL `GRPOTrainer`, exact release pinned: `loss_type="dapo"` (token-level aggregation), reward scaling off (group-mean baseline, no standard-deviation scaling), ε = 0.2 / ε_high = 0.28, `num_iterations = 1`, β = 0, vLLM colocate, TIS on. Parameter names and accepted values are checked against the pinned release in the spike. |
| **Alternatives considered** | PPO; offline DPO; vanilla GRPO; a custom implementation; RFT as the main method. |
| **Why rejected** | PPO needs a value model; offline DPO is generally weaker for reasoning; vanilla GRPO has documented length and difficulty biases (Liu et al., 2025) and entropy collapse (Yu et al., 2025); custom code adds risk. RFT is the Phase I comparison (E9) and becomes the trainer if GRPO is not confidently better. |
| **Why it fits** | On-policy, cheap with LoRA, maintained library. |
| **Limitations** | **With `num_iterations = 1` the clipping settings are largely inert**: the policy that generated a batch is the one being updated, so ratios start at 1 and clipping only acts through TIS and any within-batch optimizer steps. ε values are kept for the case μ > 1. TRL does not implement DAPO dynamic sampling. Published RLVR gains on some models are partly elicitation (Shao et al., 2025; Yue et al., 2025) — tested directly by E2. |
| **Replace or extend** | Another `OnPolicyTrainer`. μ = 2 (two updates per rollout batch) is an optional efficiency variant of E1 if budget allows, since rollouts dominate cost. |

### D06 — Parameter-efficient training: bf16 LoRA, one continuing adapter

| | |
|---|---|
| **Chosen** | LoRA rank 32, alpha 64, all linear layers, bf16; one adapter continues across the lineage; merged only for export. |
| **Alternatives considered** | QLoRA; full fine-tuning; new adapter per cycle; merging at each promotion. |
| **Why rejected** | QLoRA slows training and widens generator/trainer mismatch; full fine-tuning has larger state and was reported to bring no advantage over LoRA for RL (Thinking Machines, 2025); per-cycle adapters and repeated merges complicate the reference model. |
| **Why it fits** | Disabling the adapter gives the base model for free; states are small; rollback is cheap. |
| **Limitations** | The rank caps capacity, but **it does not bound how far behaviour drifts**; Revision 1 wrongly listed it as a forgetting defence. LoRA for RL needs a higher learning rate than full fine-tuning. |
| **Replace or extend** | LoRA versus full fine-tuning is **deferred**: a 0.6B comparison says little about 4B; it is run on the target only if E1 shows no learning signal. QLoRA at 9–12B (M5). |

### D04 — No SFT stage; RFT as Phase I comparison

| | |
|---|---|
| **Chosen** | No SFT in the main loop. RFT (sample, keep verified-correct, SFT) runs once in Phase I at matched compute (E9). A format-only SFT is not planned: Instruct-2507 follows answer-format instructions, and the primary metric is format-agnostic. |
| **Alternatives considered** | SFT warm-up; SFT on external traces; a separate format-only control run. |
| **Why rejected** | Warm-up unnecessary for a post-trained model; external traces damage existing behaviour. A format-only control is redundant now that the primary metric ignores format and E8 measures template sensitivity directly. |
| **Why it fits** | E9 answers whether the RL machinery is worth its cost before it is built. |
| **Limitations** | One RFT run and one GRPO run include seed variance not captured by item-level intervals (§36). |
| **Replace or extend** | Teacher distillation via `OfflineTrainer` (M2). |

### D14 — Reference model and KL

| | |
|---|---|
| **Chosen** | "Base" always means the **original vendor checkpoint**; all guardrails, KL and the final claim are measured against it. KL to base is **monitored** with an absolute ceiling set from E1, never trend-tested and never budgeted. β = 0. |
| **Alternatives considered** | KL as a gated budget plus a trend tripwire (Revision 1); KL to champion; fixed β > 0. |
| **Why rejected** | With β = 0, KL to base grows with every promotion by construction, so a trend test would trip in any successful run, and "noise floor plus acceptable loss" has no meaning for a divergence. If a cap on total drift is wanted, a nonzero β enforces it far more cheaply than rejecting whole evaluations. |
| **Why it fits** | Damage is measured directly by guardrails and capability monitors; KL is context, not a verdict. |
| **Limitations** | Nothing in the loss restrains drift. |
| **Replace or extend** | Pre-registered switch: if E1 shows guardrail damage rising with KL, β > 0 is set before the main run. The β ablation itself is deferred. |

### 18.1 Training procedure

The orchestrator launches `autopost train --start <state> --slice <hash> --steps N` as a GPU subprocess: load base in bf16, attach the lineage adapter, restore optimizer, scheduler, RNG and cursor, run N steps, store raw rollouts, save the full state, exit. Learning rate 2e-5 (*start*). **E1a** (budgeted, §25) runs two learning rates for 200 steps each on the dev split before E1 proper; the rate with the better dev-split reward slope is used. No further tuning happens inside the gated loop.

---

## 19. Continual Learning

Forgetting is limited and detected by layered mechanisms, now classified so that expected drift is never treated as damage.

| Layer | Mechanism | Class | What it does | Limitation |
|---|---|---|---|---|
| 1 | Four guardrails, non-inferiority against the **base**, cumulative | Gated | A promotion must *demonstrate* that transfer accuracy, instruction following, multilingual mathematics and format compliance have not dropped by more than their margins since the base | Only what the guardrails measure |
| 2 | Capability monitors (general knowledge, small code set, refusal and over-refusal, abstention, language consistency, MATH-500 numeric subset) | Monitored; confident-negative trend across promotions is a tripwire | Alerts on damage outside the guardrails | Crude scoring for some (refusal string matching), so never gated |
| 3 | Drift monitors (KL to base, entropy, length) | Absolute limits only | Catch pathological drift (collapse, runaway length) | Expected to move; not evidence of damage by themselves |
| 4 | On-policy RL | — | Lower drift than SFT (Shenfeld et al., 2025) | Not a guarantee |

Revision 1 also listed the LoRA rank and regression replay; neither is a forgetting defence in Revision 2 (D06, D09).

**Narrowing.** Diversity is monitored through entropy and pass@4 on a fixed probe set; RLVR is known to raise pass@1 while lowering pass@k at larger k (Yue et al., 2025). These are drift monitors with absolute limits, because some narrowing is the expected effect of RL.

**Honest limits.** Damage to capabilities neither gated nor monitored is not detected.

---

## 20. Champion/Candidate Management

### D15 — One lineage, one candidate, full training states

| | |
|---|---|
| **Chosen** | One lineage. Full-state checkpoints every N steps. One candidate evaluated at a time. |
| **Alternatives considered** | Independent runs per cycle; candidate populations; weight-only states. |
| **Why rejected** | Repeated warmups waste compute; selecting the best of N inflates gains; weight-only states cannot resume. |
| **Why it fits** | Cheapest; one comparison per decision; complete operational resume from persisted project-controlled state. |
| **Limitations** | No exploration over strategies. |
| **Replace or extend** | Candidate pool at M7. |

### D28 — Decouple gate cadence from training horizon

| | |
|---|---|
| **Chosen** | Three separate quantities, all set from the E1 learning curve and pre-registered: **N** (checkpoint and monitor interval, *start* 50 steps), **G** (gate interval, a multiple of N), and **S** (step budget: maximum steps since the last promotion before HALT). RESTORE only on confident regression. Inconclusive results CONTINUE without a retest count. |
| **Alternatives considered** | Revision 1: gate every 50 steps, restore after two inconclusive retests, halt after three restores. |
| **Why rejected** | It required a confirmed gain within 150 steps (~19k completions), which is implausible for a heavily post-trained 4B model. The expected trajectory was CONTINUE, CONTINUE, RESTORE three times, then HALT — discarding up to 450 steps per attempt and then repeating. Measurement cadence was wired directly to the training horizon. |
| **Why it fits** | The system learns for as long as E1 shows learning is plausible, measures at a cadence where a gain could have accumulated, and only abandons a trajectory on evidence of harm or when the pre-registered horizon is spent. |
| **Limitations** | Depends on E1 being representative of the main run. |
| **Replace or extend** | Re-derive N, G, S at each new pre-registration. |

**How G and S are set.** E1 evaluates checkpoints every 50 steps on the large pilot set, giving a learning curve with intervals. G is the smallest multiple of N at which E1's expected gain since the last promotion reaches the effect that the gate promotes with ≥ 80% joint probability (from E3). S is the step at which E1's curve has flattened (no confident gain over the preceding window), with a margin.

**State statuses:** `base`, `checkpoint`, `champion`, `promoted`, `rejected`, `pruned`. Core logic stores a `StateRef`; `LocalArtifactStore` may resolve it to `runs/states/<state_id>/`, whose manifest covers adapter, optimizer, scheduler, available RNG states, trainer state, sampler/data cursor, parent, config hash, slice hash, stack hash and verifier version. Physical layout is not part of state identity.

**Complete resumable state, not bit-exact replay.** The project persists all project-controlled state needed to continue without manual reconstruction. GPU kernels, generation engines and scheduling may still be nondeterministic; bit-exact continuation is neither promised nor required. State round-trip tests use documented numerical/behavioral tolerances.

**Exposure** is counted globally (lineage-scoped accounting deferred).

---

## 21. Evaluation

### D16 — Independent measurement channel with one primary endpoint, cluster-level statistics, and validated power

| | |
|---|---|
| **Chosen** | One pre-registered primary endpoint, four gated guardrails, and monitored metrics; paired cluster bootstrap with one-sided bounds; fresh reference samples for confirmation; stratified gate sets; budgets on every claim-relevant read; power and coverage verified by simulation on target-model outputs (E3, E4). |
| **Alternatives considered** | Revision 1: ~16 jointly required checks each sized separately; item bootstrap with inner resampling; cached reference samples for all comparisons; worst-of-two-templates strict scoring; the hidden set read at every milestone and for the claim. |
| **Why rejected** | See C1, C4, M4–M6 in Appendix D: the conjunction of separately sized checks has unknown and low joint power; the bootstrap was simultaneously too narrow (ignoring clusters) and too wide (counting sampling noise twice); a shared reference draw correlates Gate-A and Gate-B; the strict unseen template mixed format bleed into the capability metric; repeated reads of the claim set inflate spurious success. |
| **Why it fits** | Every gated quantity has an explicit direction and error rate, and the full gate's behaviour at the floor is measured, not assumed. |
| **Limitations** | Same-source gate sets share artefacts and possibly vendor-training exposure with the training data; the claim is narrow (§36.3). |
| **Replace or extend** | New suites via `EvalSuite`; thresholds re-derived per pre-registration. |

### 21.1 Endpoints

| Quantity | Class | Set(s) | Comparison | Rule |
|---|---|---|---|---|
| **Primary:** content accuracy, averaged over the evaluation-template rotation | Gated | Gate-A, then Gate-B | Candidate vs champion | PROMOTE needs lower one-sided bound > 0 on both; RESTORE if upper < 0 on either |
| Transfer accuracy | Guardrail | transfer_guard | Candidate vs **base** | Non-inferiority: PROMOTE needs lower > −margin; RESTORE if upper < −margin |
| Instruction following (verifiable-instruction subset) | Guardrail | battery | vs base | same |
| Multilingual mathematics (content accuracy) | Guardrail | battery | vs base | same |
| Format compliance (instructed marker, from gate completions — no extra generation) | Guardrail | Gate-A completions | vs base | same |
| General knowledge, small code set, refusal / over-refusal, abstention, language consistency, MATH-500 numeric | Capability monitor | battery | vs base | Alert if upper < −margin; tripwire on confident negative trend |
| KL to base, entropy, length, pass@4 | Drift monitor | probes | absolute | Absolute limits from E1 only |
| **Final claim:** content accuracy | Claim | hidden_sealed (once) | Final champion vs base | §36.3 |
| **Secondary:** transfer accuracy | Claim | transfer_sealed (once) | Final champion vs base | Reported with interval |

**Correlated-label caveat.** Screen, Gate-A and Gate-B are held-out clusters of the training sources, so they share label conventions and errors with training; checker independence does not address that. The guardrail on a different-source transfer set, the hidden set with execution-computed labels, and E7 (recall versus perturbation) bound this; they do not remove it.

### 21.2 Margins and error allocation

The floor and guardrail margins are pre-registered judgements of what is worth claiming and what loss is acceptable. Error rates:

- **False promotion** requires the primary test to pass on Gate-A and on Gate-B. With one-sided α = 0.05 on each, disjoint items, and fresh reference samples for Gate-B, the nominal false-promotion probability per decision is about 0.05 × 0.05 = 0.0025. A run with D gate decisions has family-wise false-promotion probability at most about D × 0.0025. E3 measures the actual rate at α-scale 0 (the base model as candidate).
- **Guardrail false pass** (declaring non-inferiority when the candidate is worse by more than the margin) is bounded by that guardrail's α = 0.05; a conjunction does not inflate it.
- **False block** (a real gain blocked) is a power question: the conjunction lowers joint power, so guardrail sets are sized for high individual power at zero true difference, and joint power is verified by simulation (§21.7).

### 21.3 Templates

T_train is never used for measurement. The primary metric rotates over 3–5 **evaluation-only** templates (final set chosen in E8): sample *j* of an item uses template *j* mod T, so averaging is free. Content accuracy ignores the answer marker. Compliance with each template's instructed marker is measured separately as the format-compliance guardrail — a real instruction-following regression (RL strengthens the `\boxed{}` habit), but no longer one that masquerades as a mathematics regression.

### 21.4 Statistics: paired cluster bootstrap

### D30 — Paired cluster bootstrap

| | |
|---|---|
| **Chosen** | Resample clusters (dedup clusters; programs for hidden sets), keep each item's samples attached, no inner resampling; one-sided bounds; Wilson intervals for any rate estimated by simulation. |
| **Alternatives considered** | Revision 1's item bootstrap with inner resampling of the k samples; a pure item bootstrap; closed-form normal intervals only. |
| **Why rejected** | Item resampling treats correlated items as independent (too narrow); inner resampling counts within-item noise twice (too wide); the net coverage of Revision 1's method was unknown. Closed-form intervals are kept as a cross-check and for sizing. |
| **Why it fits** | Matches the actual sampling structure and supports different k for candidate and reference. |
| **Limitations** | Slightly liberal at moderate cluster counts in synthetic tests; corrected by E4 if it holds on real data. |
| **Replace or extend** | Studentized or BCa variants if E4 shows a need. |

`paired_cluster_bootstrap` (`stats.py`):

- computes each item's mean score per model (candidate and reference may have different *k*), takes the paired item difference, sums differences within each cluster, and resamples **clusters** with replacement; the statistic is total difference over total items;
- does **not** resample samples within items — resampling clusters with their samples attached already carries within-item sampling noise;
- reports one-sided bounds at the pre-registered α;
- `cluster_se` gives the closed-form cluster-robust standard error for sizing and as a cross-check.

Units: dedup clusters for training-source sets; programs for hidden sets; source-level clusters for the transfer set.

**Synthetic validation** (`selfcheck.py`, 1,500 simulated comparisons each; Wilson 95% intervals):

| Scenario | Revision 1 bootstrap | Item-level, no inner | Cluster bootstrap | Nominal |
|---|---|---|---|---|
| A/A false-positive rate (80 clusters × 3 items, k = 4) | 0.012 (0.008–0.019): too wide | — | 0.061 (0.050–0.074) | 0.050 |
| Coverage of the population gain when the effect varies by program (60 programs × 8 instances) | 0.899 (0.882–0.913): too narrow | 0.845 (0.825–0.862) | 0.935 (0.921–0.946) | 0.950 |

The Revision 1 method was miscalibrated in both directions, as the review predicted. The cluster bootstrap is close to nominal but slightly liberal at these sizes. **E4 measures coverage on real target-model outputs at the planned sizes**; if the empirical one-sided error exceeds α, the pre-registered rule is to reduce the decision α until the empirical rate is at or below the nominal one.

### 21.5 Reference samples

- **Gate-A:** reference (champion) samples may be reused across comparisons within one stack hash, with `k_ref = 2k` (*start*) because the cost is shared.
- **Gate-B:** **fresh reference samples** are drawn in the same evaluation job as the candidate's, so an unlucky champion draw cannot make a candidate look better on both gate sets.
- **Base references for guardrails:** generated once per stack with large k. Their sampling error is shared across all guardrail comparisons; large k keeps it small. This is a documented residual.

### 21.6 Stratification

Gate-A and Gate-B drop items the base solves almost always (base pass rate ≥ 0.95 from 8 *stratification samples that are never used in any comparison*, *start*). In a paired design, items both models always solve carry almost no information; dropping a saturated fraction f gives roughly the same power at about (1 − f) of the evaluation compute. The **screen** set and the **sealed** claim sets are **not** stratified, so regressions on easy items remain visible and the final claim refers to the unselected population.

### 21.7 Sizing and joint power

1. **From E1:** the cluster-robust standard error of the primary difference at a realistic effect, per set (`cluster_se`).
2. **Per-set sizes:** `clusters_needed` gives the clusters for each of Gate-A and Gate-B so that P(lower bound > 0 | true gain = floor) ≈ 0.90; joint ≈ 0.81. Example (synthetic pilot SE of 0.0129 at 200 clusters): 711 clusters for a 2-point floor, 316 for 3 points, 114 for 5 points. The floor choice drives evaluation cost quadratically.
3. **Guardrail sets:** sized for ≈ 0.97 power each at zero true difference against their margins.
4. **Verification (E3):** run the complete gate — Gate-A, guardrails, Gate-B with fresh references — repeatedly at each graded true effect by subsampling stored completions. Required: P(promote | floor) ≥ 0.80 and P(promote | no effect) ≤ nominal.

*Synthetic illustration* (two gate sets of 150 clusters × 2 items, k = 4, k_ref = 8, one-sided α = 0.05 each): P(promote) = 0.000 (0.000–0.010) with no effect; 0.34 at +3.0 accuracy points; 0.97 at +6.0; 1.00 at +8.9. Under Revision 1's margin rule, a true gain equal to the detectable effect passed about 2.5% of the time (normal approximation: requiring the lower bound to exceed 2.8σ needs a point estimate above about 4.8σ; 80% pass needs a true gain near 5.6σ).

### 21.8 Behavioural battery

| Probe | Class | Scoring |
|---|---|---|
| Verifiable-instruction subset (IFEval-style) | Guardrail | Deterministic checks |
| Multilingual mathematics subset (MGSM-style, numeric) | Guardrail | Exact scorer |
| Format compliance on evaluation templates | Guardrail | Exact scorer (`format_compliant`) |
| System-prompt adherence | Capability monitor | Deterministic checks |
| General-knowledge subset | Capability monitor | Strict letter extraction |
| Small code subset | Capability monitor | Locked-down execution container |
| Refusal (unsafe) and over-refusal (safe) | Capability monitor | String matching — too crude to gate |
| Unanswerable-problem probe | Capability monitor | Deterministic patterns |
| Language consistency | Capability monitor | Language identification |
| KL to base, entropy, length, pass@4 | Drift monitor | Absolute limits |

The Stage A thinking-mode probe is removed.

### 21.9 Measurement hygiene

- Results are cached only under an identical stack hash; any stack change forces re-evaluation of base and champion.
- **Raw completions are stored for every evaluation read** (Parquet), with scorer version.
- Every claim-relevant read is a look, counted against a pre-registered budget (`budgets` table): Gate-A, Gate-B, hidden_audit, and exactly one read each of hidden_sealed and transfer_sealed per run.
- Gate-A and Gate-B are refreshed from their reservoirs when their budgets run out (a script, run at a halt), and Gate-A is refreshed after **any** human configuration change. When a reservoir is exhausted, a new pre-registration is required.
- Humans see aggregates with intervals, never items, except on dev and pilot.

---

## 22. Promotion

### D17 — Deterministic gate with explicit test directions

| | |
|---|---|
| **Chosen** | `DefaultGate` (`interfaces.py`). The primary endpoint is tested against **zero**; sets are sized so that the pre-registered floor passes with ≥ 80% joint probability. Guardrails are non-inferiority tests. Every test states which bound decides which outcome. |
| **Alternatives considered** | Revision 1: lower bound > `min_gain` where `min_gain` = max(minimum detectable effect, floor); point thresholds; LLM or learned gates. |
| **Why rejected** | Revision 1 confused a minimum detectable effect with a margin on the lower bound. Sets were sized so the effect was detectable with 80% power against zero, and then the lower bound had to exceed that effect: a true gain equal to the detectable effect passed about 2.5% of the time, and 80% passage needed roughly twice the calibrated effect — before losses from the worst-template rule and the conjunction of ~16 checks. If the floor bound instead, a gain equal to the floor passed about 2.5% of the time at any sample size. Point thresholds promote noise; learned gates are exploitable. |
| **Why it fits** | The decision rule and the sizing rule now refer to the same test, and joint power is verified (E3). |
| **Limitations** | Gains below the floor are not promoted (by design). Same-source gate sets (§21.1). |
| **Replace or extend** | Re-derive sizes per pre-registration; new domains add guardrails. |

### 22.1 Decision logic

| Step | Test | PROMOTE path | CONTINUE (HALT if steps ≥ S) | RESTORE |
|---|---|---|---|---|
| 0 | Tripwires; Gate-A budget | none fired; budget left | — | (HALT if fired or budget spent) |
| 1 | Primary on Gate-A vs champion | lower > 0 | lower ≤ 0 and upper ≥ 0 | upper < 0 |
| 2 | Each guardrail vs base | lower > −margin | lower ≤ −margin ≤ upper | upper < −margin |
| 3 | Gate-B budget | budget left | — | (HALT if spent) |
| 4 | Primary on Gate-B vs champion, fresh references | lower > 0 → **PROMOTE** | lower ≤ 0 and upper ≥ 0 | upper < 0 |

Guardrails run before Gate-B so that Gate-B reads are spent only on candidates that could be promoted.

### 22.2 Threshold sources

| Threshold | Source |
|---|---|
| α for Gate-A and Gate-B | Pre-registered (0.05 one-sided each); reduced if E4 shows the bootstrap is liberal at the planned sizes |
| Floor | Pre-registered practical gain |
| Gate-A / Gate-B sizes | E1 standard error + `clusters_needed` at ~0.90 per set; verified by E3 |
| Guardrail margins | Pre-registered acceptable loss per guardrail |
| Guardrail set sizes | ~0.97 power each at zero difference; verified by E3 |
| G, S | E1 learning curve and E3 (§20) |
| Read budgets | Pre-registered; family-wise false promotion ≤ budgeted decisions × 0.0025 |

---

## 23. Rollback

### D18 — Full-state rollback; re-scoring from stored completions

| | |
|---|---|
| **Chosen** | RESTORE moves the champion pointer back and resumes from the champion's full state. All promoted states are kept, plus the last 5 rejected (*start*). Raw completions are stored for every rollout and evaluation read, and all derived prompt-table state is tagged with the verifier version that produced it. |
| **Alternatives considered** | Weight-only rollback; Revision 1's retroactive rollback that re-scored old champions from stored per-sample scores. |
| **Why rejected** | Weight-only rollback cannot provide complete operational resume. Re-scoring needs the completions, not scores; regenerating them costs a full evaluation and does not reproduce the original outputs. Revision 1 also left prompt-table state derived from a buggy verifier (quarantines, pass rates, exposure) outside the rollback. |
| **Why it fits** | Every verdict can be recomputed from stored text with a corrected checker. |
| **Limitations** | Storage grows with rollouts (estimated in §26). Retroactive rollback is a documented manual procedure in V1; automation is deferred. Training already done with a buggy reward cannot be undone except by rolling back to an earlier state. |
| **Replace or extend** | Automate the procedure once it has been run manually. |

**Procedure after a verifier or scorer bug (manual in V1):**

1. Mark all states trained with the affected verifier version as suspect.
2. Re-score stored **evaluation** completions with the fixed scorer; recompute every affected gate decision.
3. Move the champion pointer to the last state whose decisions still hold.
4. Exclude prompt-table rows produced by the affected verifier version; rebuild pass rates and suspected-label flags from stored **rollouts** re-scored with the fixed verifier. Confirmed quarantine decisions retain or renew independent evidence.
5. Log a human change (consumes budget; forces a Gate-A refresh).

**Tests:** state round-trip (save, restore, continue: same loss within tolerance); export parity (merged export scores the same as the served adapter on a fixed probe).

---

## 24. Recovery

### D19 — Deterministic controller; tripwires halt; human changes are budgeted

| | |
|---|---|
| **Chosen** | States RUNNING, HALTED, MILESTONE_REVIEW. Actions: promote, continue, restore, quarantine, halt. No automatic changes to hyperparameters, data or thresholds. Every tripwire receives an α allocated from a pre-registered **expected false-halt budget**. Any configuration change by a human after a halt consumes the human-change budget and forces a Gate-A refresh. |
| **Alternatives considered** | Automatic recovery; learned controller; LLM agent; unbudgeted human changes (Revision 1). |
| **Why rejected** | Automatic recovery is search against the evaluation sets. A human resuming frequent halts with changes informed by gate results performs the same search more slowly; Revision 1 said such changes "count as looks" but attached no consequence. |
| **Why it fits** | Halts are rare by construction (expected count pre-registered) and diagnosable. |
| **Limitations** | A human must restart after a halt. The accept/reject loop itself selects which data slices' training survives — inherent to any gated loop, and a further reason Gate-B is budgeted. |
| **Replace or extend** | Rule-based recovery after enough logged halts (M9). |

### 24.1 Tripwires

| Tripwire | Class | Fires when |
|---|---|---|
| Dual-checker disagreement | Health | On the **same** rollouts of measurable training items, the rate at which the train verifier accepts and the exact scorer rejects exceeds its E6 baseline at the allocated α |
| Capability trend | Capability | A guardrail or capability monitor shows a confidently negative slope across promoted champions (Bonferroni across metrics and checks) |
| Drift limits | Drift | KL to base above its absolute ceiling; entropy below its floor; length p95 above its limit — all set from E1's trajectory, not from trends |
| Sync failure | Health | Probe log-probability gap > 100× its running median (after one retry) |
| Pool exhaustion | Health | In-band fraction below floor |
| Budgets | Health | Step budget S, Gate-A / Gate-B / hidden_audit read budget, human-change budget |
| Repeated restores | Health | R (*start* 3) confident regressions in a row |
| Audit contradiction | Capability | Upper bound of (champion − base) on hidden_audit < 0 — a *confident* contradiction. An audit gain below the floor is reported but does not halt, so an underpowered audit cannot cause spurious halts |

**Removed from Revision 1:** reward/measurement divergence (it fires in exactly the expected state of an underpowered or in-distribution gain and cannot distinguish hacking — dual-checker disagreement on the same outputs can); KL and entropy trend tests (both are expected to move in successful RL); the parity tripwire (§15).

**Expected false-halt rate.** `expected_false_halts` sums (α per check × number of checks) over all tripwires for the planned run length. It is pre-registered (*start* target: at most 1 expected false halt per main run) and validated by replaying E1's checkpoints through the controller (E5).

### 24.2 Failure handling

| Failure | Automatic action | Human action |
|---|---|---|
| Crash or preemption | Resume from the last segment phase and full state | None |
| Confident regression | Restore champion; next slice | None |
| Tripwire | Halt; event; notify | Diagnose; resume. A config change consumes budget and refreshes Gate-A |
| Step budget S spent | Halt | Accept the run's result, or pre-register a new run |
| Read budget spent | Halt | Run the refresh script; reservoir exhausted → new pre-registration |
| Checker bug | Mark affected states suspect | Procedure in §23 |
| Stack change | Re-evaluate base and champion | None |

---

## 25. Compute Strategy

### 25.1 Throughput model

Revision 1 gave hardware minimums with no throughput model while planning several full-run equivalents per stage. Revision 2 budgets from measurements. Let *P* be prompts per step, *G* group size, *L̄* mean completion length on band-filtered prompts, *R_gen* sustained generation throughput (tokens/s) in colocate mode, and *T_upd(P·G, L̄)* the optimizer-step time:

- RL step: *T_step ≈ P·G·L̄ / R_gen + T_upd*
- Evaluation read of a suite: *T_eval ≈ (Σ items × (k + k_ref) × L̄_eval) / R_gen,eval* (reference samples omitted when cached)
- Run: *S_run · T_step + (S_run/N)·T_screen + (S_run/G)·T_gate + T_milestones + T_sealed*

All terms are measured during the first 50 steps of E1 at the real cap and real band. Cost scales roughly linearly with *L̄*. Evaluation cost is controlled by tiering but not assumed solved: the eval/train token and GPU-time ratios are acceptance measurements from E1/E3. If their preregistered threshold or the Phase I budget is exceeded, E1 stops and the plan is revised before more compute is spent.

**Order of magnitude.** The design review estimated a few GPU-days for E1 and E2 together on one 80 GB H100-class card at about 3k-token average completions, including evaluation. Treat this as a prior to be replaced by the E1 measurement.

### 25.2 Run-equivalents per stage

| Item | Stage A (0.6B) | Stage B (4B) |
|---|---|---|
| E1a learning-rate check | Short | 2 × 200 steps (dev split) |
| E1 open-loop GRPO pilot | Short end-to-end reference run | ~1,000 steps/checkpoints every 50; ≥2 seeds preferred |
| E2 random-reward control | Optional systems path | Matched to E1 per seed; ≥2 seeds preferred |
| E9 RFT at matched compute | — | One run |
| E3–E5 simulations | — | Mostly CPU, from stored completions; E3 also generates on the pilot set for ~10 scaled adapters |
| E6–E8 | — | Generation on base and selected E1 checkpoints |
| Phase II main run | Optional | One run |
| Phase II replication | — | One run (second seed) |
| Deferred unless E1 shows no learning | — | β ablation; LoRA vs full fine-tuning |
| Removed | Format-only control; thinking-mode probe | Format-only control |

### 25.3 Cadence

| When | Work |
|---|---|
| Every step | Rollouts through the selected provider; batched verification; update; weight sync; raw rollouts finalized; mismatch, lengths and zero-variance share logged |
| After each weight sync | Coarse sync probe |
| Every N steps | Full-state checkpoint; screen monitor (unstratified, small); tripwire checks |
| Every G steps | Gate (stops at the first failing tier) |
| After each promotion | Re-estimate near-band-edge prompts; flag consensus disagreements for independent label verification |
| Every P promotions | hidden_audit read; human sign-off |
| Run end | hidden_sealed and transfer_sealed, once each |
| Only on failure | Re-baselining on stack change; re-scoring after checker bugs; reservoir refresh |

### 25.4 Hardware

| Stage | Assumed | Binding constraint |
|---|---|---|
| A (0.6B) | One 16–24 GB GPU | Rollout throughput |
| Research target (4B) | **One 80 GB-class GPU is the current profile prior until E1 measures otherwise** | KV cache and activations at multi-thousand-token outputs, next to a vLLM copy of the weights |

If E1 shows memory pressure: lower `vllm_gpu_memory_utilization` and prompts per step first; lower the cap only if truncation and accuracy-versus-length on the hardest prompts stay acceptable; QLoRA last (D06). The CPU verifier pool is sized from measured GPU idle time waiting on verification.

---

## 26. Infrastructure

### D20 — Stable control plane; local reference executor

| | |
|---|---|
| **Chosen** | The controller submits versioned jobs through `JobExecutor`. V1 uses one CPU control-plane process and `LocalSubprocessExecutor` for sequential GPU jobs. |
| **Alternatives considered** | Ray; Kubernetes; microservices; concurrent GPU jobs. |
| **Why rejected** | Everything fits one node; concurrency contends for memory. |
| **Why it fits** | Local subprocesses release GPU memory and contain crashes, while stable job/result contracts keep topology out of controller semantics. |
| **Limitations** | GPU idle during CPU-only phases. |
| **Replace or extend** | Replace the executor/providers at M10; the controller and job schemas remain unchanged. |

### D21 — Persistence contracts with local reference stores

| | |
|---|---|
| **Chosen** | `MetadataStore` owns runs, jobs, state/lineage metadata, budgets, decisions, looks, prompt statistics, events and preregistration hashes. `ArtifactStore` owns immutable state, optimizer data, datasets, raw rollouts/evaluation completions, reports and manifests. V1 implements these with SQLite plus local Parquet/files. |
| **Alternatives considered** | PostgreSQL; object storage; storing only scores (Revision 1). |
| **Why rejected** | One writer, one node; scores alone cannot be re-scored. |
| **Why it fits** | Inspectable, transactional and re-scorable locally; stable refs prevent paths and SQLite APIs from leaking into core logic. |
| **Limitations** | Raw text is the largest storage item: roughly P·G·L̄ tokens per step for rollouts (≈ 128 × 3,000 tokens ≈ 1.5 MB of text per step at the review's 3k-token estimate, before compression), plus evaluation reads. Budget disk accordingly. |
| **Replace or extend** | `PostgresMetadataStore` at multiple writers/remote execution; `ObjectArtifactStore` when local NVMe is insufficient. |

### D22 — Tracking and reporting

| | |
|---|---|
| **Chosen** | `MetadataStore` and finalized artifact manifests are the source of truth; V1 implements them with SQLite/local files. `obs/report.py` generates Markdown/HTML reports. MLflow is deferred. |
| **Alternatives considered** | MLflow (Revision 1); hosted trackers. |
| **Why rejected** | Not needed for one operator; adds a component. |
| **Why it fits** | Reports are reproducible from the database. |
| **Limitations** | Less interactive. |
| **Replace or extend** | MLflow or a hosted tracker as a sink when collaborators join. |

### D23 — Security

| Threat | Control |
|---|---|
| Code execution through SymPy parsing in the **train** verifier | Sandboxed worker processes: no network, read-only filesystem, non-root, CPU/memory limits, 5 s CPU timeout |
| Code execution in the **eval** scorer | None possible: stdlib `re` and `fractions` only, input length capped; runs in-process |
| Model-written code | Not executed in training; the code-monitor probe runs in a separate locked-down container |
| Malicious model or dataset files | `trust_remote_code=False`; revisions pinned by hash; Parquet/JSON only |
| Credentials | None in sandbox environments |
| Claim-set leakage | Restricted permissions on hidden data; read budgets; aggregate-only visibility |

Containers are not strong isolation; acceptable while no model-written code runs in training. gVisor or VM isolation is required before M1.

---

## 27. Interfaces

### D24 — Plugin interfaces and enforced invariants

| | |
|---|---|
| **Chosen** | Stable refs; versioned job/result schemas; `ModelSpec`, `ExecutionProfile`, `JobExecutor`, `RolloutProvider`, `EvalGenerationProvider`, `MetadataStore`, `ArtifactStore`; plus domain/training/evaluation/gate interfaces. Definitions live in project-owned contracts modules; third-party types stop at backend implementations. |
| **Alternatives considered** | One `Trainer` that "receives prompts, never completions" (Revision 1); a generic plugin framework; stubs. |
| **Why rejected** | The single-trainer invariant conflicted with planned distillation and with RFT. Stubs encode guesses. |
| **Why it fits** | Each interface has a V1 implementation and a named future one. |
| **Limitations** | Future modules may need small additions. |
| **Replace or extend** | Add methods when a second implementation is about to be written. |

| Interface | Key method | V1 implementation | Planned |
|---|---|---|---|
| `ModelSpec` | semantic model identity and rendering inputs | Qwen3-0.6B; Qwen3-4B-Instruct-2507 | Qwen3.5 / ~12B |
| `ExecutionProfile` | backend/resource configuration and capabilities | local 24G / local 80G prior | quantized, multi-GPU, remote |
| `JobExecutor` | `submit`, `status`, `result`, `cancel` | local subprocess; replay | Slurm, Ray, remote queue |
| `RolloutProvider` | `generate(request)` | colocated vLLM | remote vLLM pool |
| `EvalGenerationProvider` | `generate(request)` | local vLLM eval | remote evaluation workers |
| `MetadataStore` | transactional metadata/result acceptance | SQLite | PostgreSQL |
| `ArtifactStore` | stage, validate, finalize and resolve refs | local filesystem | object storage |
| `TaskSource`, `IngestStage` | `raw_tasks()`, `apply(tasks)` | Curated datasets; five stages | Teacher, procedural, generative |
| `TrainVerifier` | `check(task, completion)` | math-verify | Code harness |
| `EvalScorer` | `score_content`, `format_compliant` | `exact_scorer.py` | Code scorer |
| `RewardFunction` | `__call__(task, completion, verdict)` | `BinaryReward` | Abstention-aware |
| `OnPolicyTrainer` | `train(start, prompts, verifier, reward, profile, steps)` | TRL GRPO wrapper | Other RL methods |
| `OfflineTrainer` | `train(start, examples, profile, steps)` | RFT | Teacher distillation |
| `EvalSuite`, `EvalRunner` | `items`, `score`; `run(..., fresh_reference)` | All suites | Code suites |
| `PromotionPolicy` | `decide(ctx, candidate, champion, base)` | `DefaultGate` | Multi-domain |
| `Tripwire` | `check(history)`, `alpha_per_check` | §24.1 | Domain-specific |

**Enforced invariants:**

1. The gate accepts only the `EvalScorer`.
2. `exact_scorer.py` imports only the standard library (import test).
3. **On-policy trainers receive prompts only.** Offline trainers receive `OfflineExample`s carrying provenance and a correct train-verifier verdict.
4. **No trainer ever receives an item from a measurement split** (`assert_not_measurement`, plus a schema constraint that measurement splits hold only measurable items).
5. Every gate comparison writes a look and decrements its budget.
6. Every backend passes the shared serialization, capability, idempotency, finalization and failure-injection contract suite.
7. Core interfaces contain no physical paths, device/runtime objects or third-party result types.

---

## 28. V1 Scope

**Phase 0 — reference architecture (built first, limited compute):**

- Stable refs, versioned serializable job/result schemas and backend capability descriptors.
- `LocalSubprocessExecutor`, `ReplayExecutor`, `SQLiteMetadataStore` and `LocalArtifactStore` behind contracts.
- Idempotent result application and atomic artifact-finalization protocol.
- Contract tests plus crash, partial-artifact, duplicate-result, conflicting-result, retry and stale-result injection.

**Phase I-A — `dev` systems validation:**

- A real small-model segment through the same job, storage, resume, evaluation and controller boundaries.
- Durable/replayable gate result, budget effects, lineage and reports reconstructed from stored data.

**Phase I-B — `research` scientific validation:**

- Ingest pipeline with cluster splits, measurable-only measurement splits, the Stage B instrument (§12.1), and the hidden-set builder with invariant assertions.
- Train verifier (sandboxed) and exact eval scorer with property tests.
- Evaluation runner storing raw completions; fresh-reference mode; stratification.
- `stats.py`: cluster bootstrap, sizing, simulation helpers.
- TRL GRPO configuration with explicit pinned IS/TIS semantics and runtime assertions; RFT trainer.
- Phase I experiment scripts E1–E9 and the go/no-go report.

**Phase II — gated research-target loop (only after scientific go):**

- Controller with segments, monitor, gate cadence G, step budget S, tripwires with allocated α, ledger budgets, human-change accounting.
- Lineage store and rollback procedure; generated reports.
- Main run, then replication run; final sealed reads.

**Human involvement (scheduled, budgeted):** one-time data review and hidden-program review (including generated-instance samples); pre-registration; Phase I go/no-go decision; milestone sign-off; halt diagnosis (configuration changes count against the budget); occasional label audits.

**Not in V1:** production remote executors, remote rollout/evaluation pools, PostgreSQL, object storage, distributed locks/queues, sharded training, and the items in §6/§33 whose triggers have not fired. V1 defines and tests their replacement boundaries; it does not implement them.

---

## 29. Implementation Plan

### 29.1 Repository

```
autopost/
├── pyproject.toml            # exact pins: torch, transformers, peft, trl, vllm, math-verify, sympy,
│                             #   datasketch, sentence-transformers, pydantic, pyyaml, pyarrow
├── Makefile                  # spike | data | hidden | selfcheck | phase1 | gonogo | run | report
├── configs/{base.yaml, prereg/, model/, execution/, data/}
├── docker/{verifier.Dockerfile, codeexec.Dockerfile}
├── src/autopost/
│   ├── contracts/{refs.py, jobs.py, capabilities.py, serialization.py}
│   ├── interfaces.py  config.py  cli.py
│   ├── models/{spec.py, loading.py, stack.py, export.py}
│   ├── execution/{local.py, replay.py, rollout.py, eval_generation.py}
│   ├── tasks/{sources/, normalize.py, filters.py, dedup.py, decontam.py, splits.py, pipeline.py}
│   ├── store/{metadata.py, artifacts.py, schema.sql, db.py, prompt_table.py, lineage.py, ledger.py, completions.py}
│   ├── curriculum/{band.py, sampler.py, estimate.py, quarantine.py}
│   ├── verify/{pool.py, math_train.py, exact_scorer.py, golden/}
│   ├── reward/binary.py
│   ├── train/{grpo.py, rft.py, callbacks.py, state.py}
│   ├── eval/{runner.py, stats.py, context.py, stratify.py, suites/}
│   ├── gate/{policy.py, tripwires.py}
│   ├── controller/{loop.py, state_machine.py, milestone.py, jobs.py}      # Phase II
│   └── obs/{metrics.py, manifest.py, report.py}
├── scripts/
│   ├── spike_model.py  build_hidden_perturbations.py  refresh_gate_sets.py
│   └── phase1/{e1_pilot.py, e2_random_reward.py, e3_gate_power.py, e4_coverage.py,
│               e5_shadow_controller.py, e6_checkers.py, e7_contamination.py,
│               e8_templates.py, e9_rft.py, gonogo_report.py}
├── tests/{test_exact_scorer.py, test_independence.py, test_verifiers_golden.py, test_bootstrap.py,
│          test_gate.py, test_band.py, test_sampler_caps.py, test_decontam.py,
│          test_measurement_invariant.py, test_state_roundtrip.py, test_export_parity.py,
│          test_hidden_invariants.py, test_smoke_segment.py, test_backend_contracts.py,
│          test_job_serialization.py, test_idempotency.py, test_failure_injection.py}
├── data/   (gitignored; hidden/ restricted)
└── runs/   (gitignored; autopost.sqlite, states/, completions/, reports/)
```

### 29.2 Experiments and validation tracks

Before target-model E1, run **E0 architecture validation** in `smoke`/`dev`: job-schema round trips; stable-ref resolution; local backend contract suites; real small-model segment; checkpoint/resume within tolerance; evaluation/gate durability; replay; report reconstruction; and injected crash/partial/duplicate/conflict/stale-result cases. E0 is an engineering milestone, never target-model evidence.

Then validate the instrument and premise on the `research` target:

Build steps (no controller needed):

| Step | Build | Acceptance |
|---|---|---|
| I-1 | Repo, contracts, `ModelSpec`, `ExecutionProfile`, stack hash, research spike | Capability check passes; explicit IS/TIS runtime settings match preregistration; metrics and export parity pass |
| I-2 | Ingest, splits, Stage B instrument, hidden-set builder | Zero cluster overlap; measurable-only constraint holds; every hidden program passes its invariant tests over many generated instances; human review of sampled instances |
| I-3 | Train verifier, exact scorer, tests | Golden 100%; scorer property tests pass; independence test passes |
| I-4 | Evaluation runner (raw completions, fresh references, stratification), `stats.py` | `selfcheck.py` passes; runner round-trip on pilot items |

Experiments, all on Qwen3-4B-Instruct-2507 unless stated:

| ID | Experiment | Measures | Decides |
|---|---|---|---|
| E1a | Two learning rates × 200 steps, dev split | Reward slope | Learning rate for E1 |
| **E1** | Open-loop GRPO at planned settings, preferably ≥2 independent seeds; checkpoints every 50 and pilot evaluation | Gain trajectory; between-run direction/variance; output lengths, throughput, memory, explicit IS/TIS metrics, drift | N, G, S, cap, floor plausibility, set sizes, drift limits, research profile |
| **E2** | Random-reward GRPO matched to each E1 seed | Reward-conditioned minus random-reward effect per seed and aggregate; cluster bootstrap only for within-evaluation uncertainty | Whether the reward signal matters; run-level limitation if replication is unaffordable |
| **E3** | Gate power using actual E1 checkpoints spanning the observed trajectory; supplement with scaled-adapter interpolation only after checking it tracks nearby real checkpoints | P(promote) versus real checkpoint effect; scaling calibration; null false promotion | Set sizes; floor promotability; Scientific Level 0 power test |
| **E4** | Bootstrap coverage: large k per item once (base and one E1 checkpoint), thousands of simulated A/A and planted comparisons by subsampling | Empirical one-sided error and coverage at planned sizes | Decision α adjustment if liberal |
| **E5** | Shadow controller: replay E1 checkpoints through monitor, gate and tripwire logic with subsampled fresh draws | Promotions, restores, false halts, against the pilot-set truth | Whether the controller design halts an improving run; false-halt allocation |
| **E6** | Checker study: math-verify vs exact scorer on base and E1 outputs (measurable items) | Disagreement rates by category | Scorer scope; disagreement baseline |
| **E7** | Contamination probe: base accuracy on originals vs program-generated perturbations of gate-source problems | Recall gap | How much same-source gate sets measure recall; claim wording |
| **E8** | Template sensitivity: ~5 templates on base and E1 checkpoints | Template variance; format bleed | Final evaluation-template set |
| **E9** | GRPO vs RFT at matched compute | Gain per GPU-hour with interval | Trainer choice (simpler method wins unless GRPO is confidently better) |

The `dev` profile runs the same boundaries at small scale for integration and operational validation. It does not calibrate research-target power, false halts, throughput, memory or seed variance.

### 29.3 Pre-registered go/no-go (all four required)

1. **Signal:** preferred: at least two independent seeds per E1/E2 condition show consistent reward-conditioned direction and the preregistered aggregate criterion is met. Staged alternative: run one matched pair, then require an independent confirmation pair before go. If only one pair is affordable, report a single-run reward-associated effect and do not promote it to a reproducibility claim.
2. **Power:** E3 shows the full gate promotes that effect at least ~80% of the time while false promotions stay at or below the nominal rate.
3. **Controller:** E5 shows an expected false-halt rate within the pre-registered target, and the controller does not halt a run the pilot set shows was improving.
4. **Budget:** throughput measured in E1 fits the compute budget for the main run, its replication, and the evaluation reads.

**If no-go:** the outcome is interpretable. Criterion 1 failing with 2 passing means no reward-driven gain was available at this scope and budget — a valid negative result. Criterion 2 failing means the instrument cannot see the effect; options are a higher floor, larger sets if affordable, or stopping. Criterion 3 failing means redesigning tripwires before building the controller. Criterion 4 failing means rescoping (cap, steps) or more compute. E9 may independently replace GRPO with RFT.

### 29.4 Phase II — the gated loop

| Step | Build | Acceptance |
|---|---|---|
| II-1 | Lineage store, ledger budgets, human-change accounting | State round-trip; budgets enforced in tests |
| II-2 | Gate, tripwires (α from the false-halt budget), controller | Gate branch tests; replaying E5 through the real controller reproduces E5's counts |
| II-3 | Reports; freeze `prereg.json` | Pre-registration hash recorded |
| II-4 | Main run; milestones; sealed reads at run end | Scientific Level 2 |
| II-5 | Replication run from base, new seed and data order | Scientific Level 3 |

---

## 30. Failure Modes

P/S: probability/severity (L/M/H). Status: **C** closed by design to the extent described; **P** partially mitigated; **V** to be verified in Phase I; **D** deferred.

| ID | Failure mode | Mechanism | P/S | Mitigation | Status |
|---|---|---|---|---|---|
| F1 | Shared-extractor inflation | One parser for reward and measurement | H/H | Independent exact scorer; dual-checker tripwire | C |
| F2 | Hedged / multiple answers rewarded | Lenient extraction | M/H | Multiple distinct answers = wrong in both checkers; exploit and property tests | C |
| F3 | Guessing on small answer spaces | Band favours guessable prompts | M/M | MC/yes-no removed; low-entropy cap | C |
| F4 | Format learning counted as capability | Extraction fixes look like gains | H/M | Format-agnostic primary metric; format is a guardrail | C |
| F5 | Gains from elicitation, not reward | Some models improve under random rewards | M/H | Matched E1/E2 effects across independent seeds or explicit single-run limitation | V |
| F6 | Premature answering on hard prompts | −1 truncation penalty (Rev. 1) | M/M | Truncation = 0; cap from E1; accuracy-vs-length monitor on hardest prompts | C |
| F7 | Right answer, wrong reasoning | Outcome reward | H/M | Milestone audit sample (strong external model), monitor only | P |
| F8 | Template-level contamination | Re-parameterised problems escape clustering | H/H | Private programs; transfer guardrail; E7 | P |
| F9 | Same-source gate artefacts and shared label errors | Held-out clusters of training sources | H/H | Different-source transfer guardrail; execution-labelled hidden set; narrowed claim | P |
| F10 | Label noise concentrating in training | Mastered prompts leave the band | M/M | Consensus flags; independent confirmation before exclusion; mislabelled prompts mostly sit out of band | C/V |
| F11 | Near-duplicate memorisation | Item-level caps | H/M | Cluster-level caps | C |
| F12 | Narrowing | Band/quota dynamics | M/M | Quotas; pass@4 and entropy limits; guardrails | P |
| F13 | Claim-set leakage | Repeated reads | M/H | Sealed sets read once; audit set budgeted; aggregate-only | C |
| F14 | Damage outside measured capabilities | Limited battery | H/H | Guardrails + capability monitors | P |
| F15 | Silent adapter-sync failure | Stale vLLM weights | M/M | Coarse sync tripwire (100× running median) | C |
| F16 | Software change mistaken for gain | Cached references | M/H | Stack-hash keyed caching | C |
| F17 | Evaluation cost exceeds training cost | Many reads | H/M | Tiering, stratification, guardrails before Gate-B, read budgets; verify eval/train ratios in E1/E3 | P/V |
| F19 | Repeated gate reads while a lineage CONTINUEs | Sequential looks at the same sets | M/M | Gate-A and Gate-B read budgets; family-wise false promotion ≤ budgeted decisions × ≈0.0025; measured in E5 | C/V |
| F20 | Gate-set wear over a long run | Adaptive selection across many decisions | M/M | Budgets; reservoir refresh; forced refresh after human changes; sealed claim sets untouched by the loop | P |
| F26 | Loss of abstention | Binary reward never rewards "cannot determine" | H/M | Unanswerable-problem capability monitor | P |
| F25 | Code-harness exploits | — | — | Not applicable in V1; required before M1 | D |
| **F28** | **Gate cannot promote a real gain (C1)** | Margin on the lower bound; conjunction of separately sized checks | H/H | Test against zero; joint sizing; E3 | V |
| **F29** | **Training horizon capped by gate cadence (C2)** | 150-step window | H/H | N, G, S from E1; restore only on confident regression | C/V |
| **F30** | **Instrument unvalidated at the relevant effect (C3)** | Stage A effects too large; vacuous A/A | H/H | Phase I on target; E3/E4/E5; go/no-go | V |
| **F31** | **Intervals ignore cluster structure (C4)** | Item-level resampling | H/H | Cluster bootstrap; program-level hidden sets | C/V |
| **F32** | **Within-item noise counted twice (C4)** | Inner resampling | H/M | Removed | C |
| **F33** | **Shared reference draw correlates Gate-A and Gate-B (C4)** | Cached champion samples | M/H | Fresh references for Gate-B; larger k_ref | C |
| **F34** | **Repeated reads of the claim set (C4)** | Milestone audits on the claim set | M/H | Audit/sealed split | C |
| **F35** | **Tripwires firing in normal RL (M1)** | Trend tests on KL/entropy; divergence tripwire | H/M | Reclassification; absolute drift limits; dual-checker disagreement; false-halt budget; E5 | C/V |
| **F36** | **Human feedback loop as hidden search (M2)** | Unbudgeted config changes | M/H | Human-change budget; forced Gate-A refresh; Gate-B budget | C |
| **F37** | **Optimistic lengths and hardware (M3)** | "Non-thinking = short" | H/M | 80 GB assumed; E1 throughput model | V |
| **F38** | **Format bleed read as maths regression (M5)** | Strict unseen template in primary metric | H/M | Content metric over eval-template rotation; format guardrail | C |
| **F39** | **Battery rule ambiguity and multiplicity (M6)** | Undefined direction; ~12 gated metrics | H/M | Four guardrails with explicit non-inferiority rule; rest monitored | C |
| **F40** | **Scorer engineering sink (M7)** | From-scratch symbolic scorer | M/M | Exact-numeric measurement sets; exact scorer; E6 | C |
| **F41** | **Parity tripwire on normal mismatch (M8)** | Threshold from a spike | H/L | TIS on; mismatch monitored; coarse sync check | C |
| **F42** | **Retroactive rollback impossible (M9)** | Scores stored, not text | M/M | Raw completions; verifier-versioned prompt table | C |
| **F43** | **Controls judged by gate outcome (M10)** | "Not promoted" passes vacuously | M/H | E2 effect size with interval; E9 decision rule | C |
| **F44** | **Controller coupled to local subprocess semantics** | Placement/process details leak into scheduling | M/H | `JobExecutor`; controller consumes job results only | V |
| **F45** | **Physical paths become state identity** | Filesystem layout leaks into core contracts | M/H | Stable refs; paths resolved only by `ArtifactStore` | V |
| **F46** | **Retry applies one logical job twice** | Duplicate delivery after crash | M/H | Deterministic job IDs; idempotent result application | V |
| **F47** | **Metadata points to a partial artifact** | Crash between write and commit | M/H | Stage/validate/finalize protocol; transactional result acceptance | V |
| **F48** | **Model identity mixed with hardware config** | Topology changes scientific identity | M/M | `ModelSpec` / `ExecutionProfile` split | C after refactor |
| **F49** | **Evaluation bootstrap ignores training-run variance** | One E1/E2 trajectory per condition | M/H | Replicated or staged training seeds; explicit single-run limitation | V |
| **F50** | **Scaled LoRA does not represent real trajectory** | Weight interpolation is nonlinear in behaviour | M/M | Actual E1 checkpoints are primary; scaling is a calibrated cross-check | V |
| **F51** | **Consensus quarantine removes valid hard examples** | Model is confidently wrong on frontier prompts | M/H | Flag then independent verification | C after rule change |
| **F52** | **Backend/library default silently changes algorithm** | Version-sensitive implicit settings | M/H | Explicit pinned configuration, stack hash and compatibility assertions | C/V |
| **F53** | **Small-model evidence is mistaken for target evidence** | Engineering and scientific validation conflated | H/H | Separate modes, phases and success tracks | C |

Revision 1 rows F18 (sub-threshold gains discarded), F21 (screen/gate double-dipping — the screen is now a monitor only), F22 (now F16), F23 (now F31–F32), F24 (budget spent on noise — guardrails now require demonstrated non-inferiority) and F27 (now F17) are merged into the rows above.

---

## 31. Solved, Partially Solved, and Unsolved Problems

| Problem | Status | Basis |
|---|---|---|
| Bad runs damaging the known-good model | **Solved operationally** | Immutable champion; full-state rollback |
| Traceability | **Solved** | Manifests, hashes, ledger, stored completions |
| Complete operational resume | **Solved for project-controlled state** | Full persisted state; segment phases; tolerance-based round-trip test |
| Bit-exact resume | **Unsolved by design** | GPU kernels/generation may be nondeterministic; not required |
| Control/execution separation | **To be verified** | Versioned jobs, refs and backend contract suite |
| Idempotent retries and atomic artifact commit | **To be verified** | Failure-injection tests on local backends |
| Shared-code reward/measurement errors | **Solved for exact-numeric measurement** | Independent stdlib scorer; import test |
| Gate statistics coherent with sizing | **Solved on paper; to be verified** | Test against zero; joint sizing; E3 |
| Promotion on noise | **Largely controlled** | Two-set confirmation with fresh references; cluster bootstrap; E4 |
| Format-only and template-specific gains | **Largely controlled** | Format-agnostic metric over template rotation |
| Knowing whether a negative result means anything | **Solved conditionally** | Go/no-go requires demonstrated power (E3) |
| Catastrophic forgetting | **Partially** | Detected within guardrails and monitors |
| Contamination and vendor-training recall | **Partially** | Cluster decontamination; private programs; E7; narrowed claim |
| Shared label errors in same-source gate sets | **Partially** | Transfer guardrail and hidden set bound it |
| Evaluation-set wear over long runs | **Partially** | Budgets, reservoirs, sealed sets |
| Correct answer, flawed reasoning | **Partially** | Audited, not prevented |
| Pool exhaustion | **Unsolved in V1** | Halts; needs new sources |
| Bit-exact reproducibility | **Unsolved by design** | Replaced by seed replication |
| Gains beyond the narrow claim | **Unsolved** | Not measured with adequate power |

---

## 32. Trade-offs

| Trade-off | Chosen side | Cost accepted | Reason |
|---|---|---|---|
| Build everything vs validate first | Validate first (Phase I) | Controller built later | Avoids months of engineering on an instrument that may not see the effect |
| False promotions vs missed gains | False promotions ≈ 0.0025 per decision | Gains below the floor are missed | G1; floor pre-registered |
| Rigour vs evaluation cost | Rigour, with stratification and tiering | Large gate sets if the floor is small | Cost scales with 1/floor²; the floor is chosen knowingly |
| Breadth of claim vs testability | Exact-numeric measurement | Narrower claim | Scorer can be tested exhaustively |
| Training horizon vs abandonment speed | Long horizons (S from E1) | Slower to give up | A post-trained 4B model learns slowly |
| Autonomy vs safety | Halt on anomalies; budget human changes | Human restarts | Self-repair and unbudgeted tweaks are search |
| Strong start vs headroom | Post-trained Instruct model | Small gains | Best final model |
| LoRA vs full fine-tuning | LoRA | Capacity cap | Memory and rollback; full FT tested only if no signal |
| Simplicity vs replay | No replay | No active repair of regressed training prompts | Revision 1 mechanism triggered on noise |
| Local scripts vs idempotent jobs | Idempotent jobs | Additional manifests and bookkeeping | Safe retry and future distribution |
| Concrete local code vs backend boundaries | Small, project-owned contracts | Contract tests and adapters | Preserve control-plane semantics across scale |

---

## 33. Deferred Components

| Component | Why deferred | Trigger |
|---|---|---|
| Regression replay | Noisy trigger; little value | Confident regression on training-source items not caught by the gate |
| Lineage-scoped pass-rate and exposure accounting | Restores are now rare | Frequent restores in a run |
| Quarantine by persistence (former rule a) | Removed valuable frontier prompts | Audits of flagged prompts show label problems |
| β ablation | KL is monitored, not gated | E1 links guardrail damage to KL (then β > 0 by pre-registered switch) |
| LoRA vs full fine-tuning | 0.6B result uninformative; costly at 4B | E1 shows no learning signal |
| MLflow | Generated reports suffice | Collaborators join |
| Thinking-mode probe (Stage A) | No information about Stage B | — |
| Retroactive-rollback automation | Needs a manual run first | After the first manual execution |
| Gate-set reservoir automation | Gating is infrequent; a script at halt suffices | Frequent budget exhaustion |
| PostgreSQL metadata backend | SQLite has one writer in V1 | Multiple writers or remote execution |
| Object artifact backend | Local NVMe is sufficient | Artifacts no longer fit comfortably locally |
| Slurm/Ray/remote executor | Local executor is sufficient | Jobs must run outside one process/node |
| Remote vLLM rollout/eval pools | Colocation is simplest | Local generation becomes the bottleneck |
| FSDP/DeepSpeed/tensor-parallel trainer | Target fits the current worker prior | Target cannot train efficiently within one worker |
| Formal alpha spending | Read budgets suffice at V1 cadence | Gate cadence/read count grows materially |
| Format-only control | Primary metric is format-agnostic; E8 covers templates | — |
| Code domain, teacher distillation, procedural/generative tasks, per-domain adapters, thinking mode, larger models, candidate pools, adversarial challenger, automatic recovery, process/abstention rewards, distributed execution | As in Revision 1 | §34 triggers |

---

## 34. Future Extensions

Each extension enters through an existing interface, and each new model or domain repeats a reduced Phase I (E1, E3, E4 at minimum) before its gated loop runs. The lineage, measurement channel, gate structure, ledger, tripwires and controller states do not change.

| Step | Extension | Interface | Notes |
|---|---|---|---|
| M1 | Code domain | `TrainVerifier`, `EvalScorer`, `TaskSource`, `EvalSuite` | Solution and tests in separate processes; compare serialised primitive outputs only; tests unreadable by the solution; CPU-time limits; hidden tests; generated extra tests where a reference solution exists; gVisor/VM isolation |
| M2 | Verified teacher distillation | `TaskSource`, `OfflineTrainer` | Teacher outputs pass the same checker; general instruction replay added with SFT |
| M3 | Procedural sources, then code-grounded generative curriculum | `TaskSource` | Labels computed by execution, not by the generator |
| M4 | Per-domain adapters | Lineage | Only if interference is measured |
| M5 | Larger model (Qwen3.5-4B/9B or ~12B) | `ModelSpec` + `ExecutionProfile` | Control plane unchanged; profile may select quantization, larger memory, tensor parallelism or different trainer/generation backends; repeat reduced target Phase I |
| M6 | Thinking mode | `ModelSpec` + `ExecutionProfile` | Longer caps; larger evaluation budget |
| M7 | Candidate pool (2–4) | Wraps `OnPolicyTrainer` | Candidates differ in data source or trainer; same gate |
| M8 | Adversarial challenger | `TaskSource` | Executor-grounded domain only |
| M9 | Rule-based recovery | Controller | Learned from logged halts |
| M10 | Distributed execution | `JobExecutor`, generation providers and stores | Replace executor/provider/store implementations; controller, promotion policy, measurement, lineage and job schemas remain unchanged |
| M11 | Distributed/sharded trainer backend | `OnPolicyTrainer` implementation + `ExecutionProfile` | Trigger only when the target cannot train efficiently within one execution worker |

---

---

## 35. Scaling Strategy

**Scaling is preservation of semantics across execution backends, not preservation of physical topology.** The small-model V1 is the reference implementation of the control plane. Larger models may require different generation, training, storage and placement implementations.

| Transition | Control-plane change | Execution-plane change | Required revalidation |
|---|---|---|---|
| `smoke` → `dev` | none | real model/GPU backend | integration and backend contracts |
| 0.6B → 1.5B | model spec/profile only | possibly batching/memory | short compatibility spike |
| `dev` → 4B `research` | none | research profile; possibly 80 GB-class GPU | target E1/E3/E4/E5 + scientific Phase I |
| 4B → 9–12B | none by default | quantization, multi-GPU or remote generation may change | reduced target Phase I |
| one node → multi-node | none | executor/providers/stores replaced | contract suite, failure/recovery tests and calibration |

Task pools, clusters, pass-rate histories, checkers, golden tests, evaluation sets and ledger semantics carry across scales. Checkpoint identity and lineage metadata carry across storage topologies; checkpoint binary formats may remain backend/model specific.

---

---

## 36. Success Metrics

### D25 — Pre-registration

| | |
|---|---|
| **Chosen** | Before Phase I: freeze the floor, guardrail margins, α values, the Phase I budget and the go/no-go criteria. Before Phase II: freeze set sizes, N, G, S, drift limits, tripwire α allocation, read and human-change budgets, the run budget, and the success definitions below. Hashes stored in the `prereg` table. |
| **Alternatives considered** | Setting criteria as results arrive. |
| **Why rejected** | Thresholds chosen after seeing results move to fit them. |
| **Why it fits** | Makes "it works" a test that can fail. |
| **Limitations** | A poorly chosen floor wastes compute or overclaims; changes need a new logged pre-registration. |
| **Replace or extend** | New pre-registration per stage and extension. |

### D26 — Reference architecture before scientific Phase I; go/no-go before the research loop

| | |
|---|---|
| **Chosen** | First complete and test control-plane/job/storage semantics with `smoke`/`dev`; then run E1–E9 on the research target; proceed to the target gated loop only if all scientific go/no-go criteria pass (§29.3). |
| **Alternatives considered** | Treat small-model results as instrument calibration; build the target gated loop before power/coverage validation. |
| **Why rejected** | The reference profile can validate software but not target sensitivity. Building the expensive target loop first risks a negative result that cannot distinguish no gain from an underpowered instrument. |
| **Why it fits** | Produces a useful, failure-tested architecture milestone on limited compute, then spends research compute only on the scientific questions that require it. |
| **Limitations** | E1 may not represent the main run (different data order, longer horizon). |
| **Replace or extend** | Repeat a reduced Phase I for each new model or domain. |

### D29 — Claim protocol: sealed sets read once

| | |
|---|---|
| **Chosen** | The final claim and the replication claim are read from `hidden_sealed` exactly once per run, on the final champion against the base. The secondary endpoint is read from `transfer_sealed` once per run. Milestones use `hidden_audit` within its budget. |
| **Alternatives considered** | Reading the claim set at every milestone (Revision 1). |
| **Why rejected** | Repeated reads of a success criterion inflate the chance of a spurious success. |
| **Why it fits** | One read, one claim. |
| **Limitations** | Two runs read the same sealed set once each. |
| **Replace or extend** | New sealed sets per pre-registration. |

### 36.1 Engineering success levels

**E-Level A — Contract-complete.** All versioned job schemas round-trip serialization; all local backends pass their contract suites; stable refs are recorded in metadata; and no core interface leaks physical paths, devices, connections, callbacks or third-party runtime objects.

**E-Level B — Reference backend end-to-end.** A real small-model training segment runs through the production controller/job/storage boundaries; complete operational resume works within tolerance; an evaluation/gate result is durable and replayable; and reports reconstruct from stored metadata/artifacts.

**E-Level C — Failure-resilient.** Crash-before-finalize, partial artifact, crash-after-finalize-before-commit, duplicate/conflicting/stale result and retry tests pass. No logical job can consume a budget or apply a lineage transition twice.

These are legitimate limited-compute architecture milestones. They do not establish a target-model effect or calibrated gate.

### 36.2 Scientific success levels (each requires all earlier levels)

**Scientific Level 0 — The instrument is validated on the research target (Phase I-B).**

| Test | Pass condition |
|---|---|
| Train-verifier golden suites | 100% |
| Exact-scorer property tests and independence test | Pass |
| Measurement invariant | No measurement item reachable by any trainer; schema constraint holds |
| Hidden-program invariants | Every program passes over many generated instances; sampled instances reviewed |
| **E4 bootstrap coverage** | Empirical one-sided error ≤ α at planned sizes (after any pre-registered α adjustment) |
| **E3 gate power** | P(promote \| floor) ≥ 0.80; P(promote \| no effect) ≤ nominal, both with Wilson intervals reported |
| **E5 shadow controller** | Expected false halts within target; does not halt a run that the pilot set shows improving |
| State round-trip; export parity | Pass |

**Scientific Level 1 — A reward-driven learning signal exists (E1, E2).** Preferred: reward-conditioned runs outperform matched random-reward runs with consistent direction across at least two independent training seeds per condition, and the aggregate effect meets the preregistered criterion within the step budget. The cluster bootstrap quantifies within-evaluation uncertainty, not training-run variance. If only one matched pair is affordable, report a **single-run reward-associated effect**; it is not a reproducibility claim and run-level variance remains unestimated.

**Scientific Level 2 — Repeated improvement in the gated loop.** At least two confirmed promotions in the main run; at run end, the single `hidden_sealed` read shows final-champion-minus-base gain with one-sided lower bound > 0; all guardrails non-inferior; secondary transfer endpoint reported with its interval.

**Scientific Level 3 — Reproducible.** The replication run (from base, new seed and data order) shows its own `hidden_sealed` lower bound > 0, **and** the pooled two-run gain has a lower bound at least the floor.

**Scientific Level 4 — The method is worth its cost (E9).** GRPO is kept only if its gain per GPU-hour exceeds RFT's under the preregistered replicated or explicitly single-run comparison; otherwise the simpler RFT is the trainer.

### 36.3 What a negative result means

| Observation | Interpretation |
|---|---|
| Scientific Level 1 fails, Scientific Level 0 passes | No replicated reward-driven gain of the floor's size was available at this scope and budget. |
| Scientific Level 0 fails (E3) | The instrument cannot see the floor at affordable sizes. Not a statement about the model. |
| Scientific Level 2 fails after go | The open-loop effect did not survive gating; compare E5's predicted promotion count with the actual. |
| Scientific Level 3 fails | Not reproducible; report all runs. |

### 36.4 What a success shows — and does not

A Scientific Level 3 success supports: **the model became more accurate on integer- and rational-answer problems of the kind in its training sources, robustly to numeric perturbation of those problems and to prompt wording, without demonstrated regression on the four guardrails, reproducibly across independent runs.** The transfer secondary endpoint indicates whether the gain extends to one different source. It does not support: "better at mathematics" in general; better at symbolic-answer problems; better on competition mathematics (sets too small); or that the gain is new capability rather than sharpened recall of vendor-trained material (E7 bounds this).

### 36.5 Metric catalogue

| Category | Metrics | Class |
|---|---|---|
| Primary | Content accuracy, template-rotated, Gate-A and Gate-B deltas | Gated |
| Guardrails | Transfer, instruction following, multilingual mathematics, format compliance (vs base) | Gated |
| Capability monitors | System prompts, general knowledge, code, refusal/over-refusal, abstention, language consistency, MATH-500 numeric | Monitor + trend tripwire |
| Drift | KL to base, entropy, length mean/p95, pass@4, truncation rate | Absolute limits |
| Training | Loss, gradient norm, LR, TIS ratio statistics, zero-variance share, step time | Monitor |
| Reward | Raw mean, fixed-probe accuracy, verdict-status shares, accuracy vs length on hardest prompts | Monitor |
| Health | Dual-checker disagreement, sync probe, pool in-band fraction, quarantine count, budgets used | Tripwires |
| Claims | hidden_sealed gain, transfer_sealed gain, replication pooled gain | Claim |
| Compute | GPU-seconds by job type, eval/train ratio, GPU idle on verifier, gain per GPU-hour | Monitor; milestone |

---

## 37. Risks

| Rank | Risk | Likelihood | Impact | Early warning | Mitigation |
|---|---|---|---|---|---|
| 1 | No reward-driven gain of the floor's size on the post-trained target | High | High | E1 learning curve flat; E2 difference near zero | Discovered in Phase I at low cost; valid negative result; options: higher-difficulty data, teacher distillation (M2), thinking mode at more compute |
| 2 | Gain is mostly recall of vendor-trained material or same-source artefacts | Medium–High | High | E7 gap; Gate-A gains not matched on hidden or transfer | Narrowed claim; transfer guardrail; execution-labelled hidden set |
| 3 | Instrument miscalibrated at real sizes | Medium | High | E4 coverage below nominal; E3 null promotions above nominal | Pre-registered α adjustment; larger sets |
| 4 | Compute insufficient (long outputs) | Medium | Medium–High | E1 throughput projection | Re-scope before Phase II; RFT if more efficient (E9) |
| 5 | Damage outside guardrails and monitors | Medium | High | User-visible changes not reflected in metrics | Extend monitors when new risks appear |
| 6 | E1 unrepresentative of the main run | Medium | Medium | Main-run promotions far below E5's prediction | Re-derive G and S at the next pre-registration |
| 7 | Tooling changes (TRL, vLLM) alter behaviour | Medium | Medium | Mismatch statistics jump; stack-hash change | Exact pins; spike per upgrade; re-baselining |
| 8 | Architecture hard-codes the one-GPU topology | Medium | High | Controller imports executor/storage/runtime details | Control/execution split; dependency checks; contract tests |
| 9 | Small-model behaviour is mistaken for research evidence | High | High | Claims omit profile/validation track | Separate engineering and scientific modes/success levels |
| 10 | Migration exposes nonserializable or local state | Medium | High | Jobs contain paths, callbacks or runtime objects | Versioned schemas and replay now |
| 11 | Duplicate/retried jobs corrupt lineage or budgets | Medium | High | Repeated logical IDs have multiple effects | Idempotent IDs and atomic finalization/commit |
| 12 | Training-run stochasticity dominates E1/E2 | Medium–High | High | Matched conditions disagree across seeds | Independent seeds or staged confirmation; explicit limitation |
| 13 | Evaluation dominates compute at a small floor | Medium | Medium–High | E1/E3 eval/train ratio exceeds threshold | Tiering plus empirical ratio gate before Phase II |

Architecture development compute is low-risk because `smoke`/`dev` milestones use small models and replay. Target scientific compute remains medium/high risk, but insufficient research compute defers scientific validation without invalidating an achieved engineering milestone.

---

## 38. Final Architectural Decisions

| ID | Decision | Selected approach | Rejected alternatives | Rev. 2 change |
|---|---|---|---|---|
| D01 | Models | 0.6B reference architecture/integration; 4B research target/calibration | No reference model; base; hybrid | Rev. 3 separates engineering from scientific validation |
| D02 | Mode | Non-thinking; lengths measured, not assumed | Thinking; mixed | Long outputs acknowledged |
| D03 | Domain | Final-answer mathematics | Code first; multi-domain | — |
| D04 | SFT | None; RFT compared in E9 | Warm-up; format-only control | Format-only control dropped |
| D05 | RL | Pinned TRL GRPO; dapo loss; no reward scaling; μ = 1; explicit pinned IS/TIS configuration | PPO; DPO; vanilla GRPO | Rev. 3 removes default-derived semantics |
| D06 | PEFT | bf16 LoRA r = 32, one adapter | QLoRA; full FT | Rank not a drift bound; full-FT test deferred |
| D07 | Task source | Curated datasets | Generated tasks | — |
| D08 | Ingestion | Cluster dedup/decontam/split; programs for hidden | Item-level | Program as unit |
| D09 | Replay | No training-data replay; `ReplayExecutor` required for control-plane validation | Regression replay | Rev. 3 distinguishes training replay from execution replay |
| D10 | Curriculum | Derived band; quotas; global caps; consensus flag then independent verification | Automatic consensus/persistence quarantine | Corrected in Rev. 3 |
| D11 | Rollout | On-policy provider, T = 1, explicit pinned IS/TIS; coarse sync check | Implicit defaults; parity tripwire | Explicit configuration in Rev. 3 |
| D12 | Checkers | math-verify (reward); exact scorer (measurement) | Symbolic from-scratch scorer | Exact scorer |
| D13 | Reward | Binary; truncation = 0; overlong off | −1 truncation | Changed |
| D14 | Reference/KL | Original base; KL monitored with absolute ceiling; β = 0 | KL budget and trend | Changed |
| D15 | Lineage | One lineage, full states | Populations | — |
| D16 | Evaluation | Primary + 4 guardrails; cluster bootstrap; fresh refs for Gate-B; stratified gate sets; budgets | ~16 checks; item bootstrap | Rewritten |
| D17 | Promotion | Test against zero, joint sizing, explicit directions | Margin on lower bound | Rewritten |
| D18 | Rollback | Full state; re-scoring from stored completions | Scores only | Raw completions stored |
| D19 | Controller | Deterministic; tripwires with false-halt budget; budgeted human changes | Automatic recovery | Budgets added |
| D20 | Compute | Stable `JobExecutor`; local sequential subprocess backend in V1 | Controller-bound subprocesses; premature cluster stack | Refactored in Rev. 3 |
| D21 | Storage | `MetadataStore` + `ArtifactStore`; SQLite/local files in V1 | Paths/SQLite in core semantics; scores only | Refactored in Rev. 3 |
| D22 | Tracking | SQLite + generated reports | MLflow | MLflow deferred |
| D23 | Security | Sandboxed train verifier; stdlib eval scorer | — | Scorer needs no sandbox |
| D24 | Interfaces | Domain/trainer contracts plus stable refs, jobs, providers and stores; measurement invariant | Single trainer; generic plugin registry | Extended in Rev. 3 |
| D25 | Pre-registration | Two freezes (before Phase I and Phase II) | Ad hoc | Extended |
| D26 | Work order | Reference architecture/E0, then scientific Phase I and go/no-go, then research loop | Build-then-run; small-model scientific proxy | Extended in Rev. 3 |
| D27 | Measurement scope | Exact-numeric answers only | Symbolic measurement | New |
| D28 | Cadence | N, G, S from E1; restore only on confident regression | 150-step window | New |
| D29 | Claim protocol | Sealed sets read once per run | Repeated reads | New |
| D30 | Statistics | Paired cluster bootstrap, one-sided, no inner resampling, Wilson intervals for simulated rates | Item bootstrap with inner resampling | New |
| D31 | Architecture planes | Stable control plane; replaceable execution plane | Topology embedded in controller | New in Rev. 3 |
| D32 | Model/execution config | `ModelSpec` separate from `ExecutionProfile` | Mixed `ModelProfile` | New in Rev. 3 |
| D33 | Job boundary | Versioned serializable `JobSpec` / `JobResult` through `JobExecutor` | Direct subprocess calls from controller | New in Rev. 3 |
| D34 | Job correctness | Stable job IDs, idempotent retries, atomic artifact/result commit | Best-effort retry | New in Rev. 3 |
| D35 | Generation boundary | Separate `RolloutProvider` and `EvalGenerationProvider`; local vLLM in V1 | vLLM types throughout code | New in Rev. 3 |
| D36 | Persistence boundary | `MetadataStore` + `ArtifactStore`; SQLite/local files in V1 | Storage paths as semantics | New in Rev. 3 |
| D37 | Reference validation | Replay backend, contract tests and failure injection; small model defines reference semantics | Crash-only plumbing test | New in Rev. 3 |
| D38 | Validation tracks | Engineering validation on small compute; scientific calibration on target | One undifferentiated validation ladder | New in Rev. 3 |
| D39 | Run-level uncertainty | Replicated/staged E1/E2 seeds or explicit single-run limitation | Item bootstrap as substitute for seeds | New in Rev. 3 |
| D40 | Curriculum disagreement | Consensus flags suspected labels; independent confirmation required for exclusion | Automatic consensus quarantine | New in Rev. 3 |

---

## Appendix A — Starting configuration

`base.yaml` contains experiment semantics; model and execution profiles are separate files. *start* = initial, *E-n* = set by an experiment, *prereg* = frozen judgement.

### A.1 Experiment configuration

| Setting | Value |
|---|---|
| Group size / prompts per step | 8 / 16 |
| Checkpoint and monitor interval N | 50 steps (*start*) |
| Gate interval G; step budget S | *E1 + E3* |
| Band | min signal probability 0.5 → ≈ [0.083, 0.917] |
| Learning rate | 2e-5 (*start*); E1a picks between two rates |
| Reward | Binary; truncation 0; overlong shaping off |
| Quarantine | Consensus ≥60% across two estimates flags; independent confirmation required before exclusion |
| Exposure cap | 4 groups per cluster, global |
| Evaluation templates | 3–5 evaluation-only, *E8* |
| Primary α (Gate-A, Gate-B) | 0.05 one-sided each (*prereg*; lowered if E4 requires) |
| Guardrails | transfer, instruction following, multilingual maths, format compliance; margins *prereg*; α 0.05 |
| k; k_ref | *E3/E4*; k_ref = 2k (*start*) |
| Stratification | drop gate items with base pass rate ≥ 0.95 (8 separate samples) |
| Set sizes | *E1 + E3*, in clusters/programs |
| Read budgets | Gate-A, Gate-B, hidden_audit *prereg*; sealed sets 1 per run |
| Restores before halt R; milestone every P | 3; 2 |
| Expected false halts per main run | ≤ 1 (*prereg*), verified by E5 |
| Drift limits | *E1* (absolute) |
| Kept rejected states | 5 |

### A.2 ModelSpec

| Field | Value |
|---|---|
| Model/tokenizer IDs and revisions | Exact pinned identifiers; tokenizer may inherit model ID explicitly |
| Chat template arguments | Explicit non-thinking arguments; never implicit defaults |
| Context limit | Pinned model limit |
| LoRA target modules | q, k, v, o, gate, up, down |
| Inference mode | non-thinking |

### A.3 ExecutionProfile

| Field | Value |
|---|---|
| Profile examples | `smoke.yaml`, `dev-qwen06b-local24g.yaml`, `research-qwen4b-local80g.yaml` |
| Backend / dtype / quantization | local reference / bf16 / none initially |
| Training / generation | pinned TRL / colocated vLLM initially |
| LoRA | r 32, alpha 64, dropout 0 |
| Rollout sampling | T 1.0, top_p 1.0, top_k off, min_p 0 |
| Evaluation sampling | T 0.7, top_p 0.8, top_k 20, min_p 0 |
| Generation cap | *E1*; vendor 16,384 is only a starting prior |
| Importance sampling | Explicit enabled state, implementation/granularity, cap 2.0 and logging; asserted at runtime |
| Topology | concurrency, tensor parallelism, checkpointing, CPU offload and memory utilization explicit |

Execution settings that can affect numerical behavior enter the stack hash. A future multi-GPU profile can be added without changing experiment semantics.

### A.4 Persistence and execution settings

| Setting | V1 |
|---|---|
| Metadata backend | `SQLiteMetadataStore` with schema version/migrations |
| Artifact backend | `LocalArtifactStore` with staged atomic finalization |
| Job executor | `LocalSubprocessExecutor`; `ReplayExecutor` for simulation |
| Retry policy | Stable logical job ID; attempt records; duplicate application prohibited |
| Schema versions | Job specs/results, refs/manifests and metadata schema pinned in stack identity |

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| Base | The original vendor checkpoint; all guardrails, KL and the final claim are measured against it |
| Champion / candidate / lineage | Accepted state / checkpoint awaiting a decision / chain of states from the base |
| Segment | N training steps ending in a full-state checkpoint |
| Monitor | Cheap per-checkpoint screen; can only trigger RESTORE on confident regression |
| Primary endpoint | Format-agnostic accuracy over evaluation-only templates on stratified gate sets |
| Guardrail | Gated non-inferiority test against the base |
| Capability / drift monitor | Watched metric; damage-indicating vs expected-to-move |
| Cluster | Unit of statistical independence: dedup cluster, or perturbation program |
| Look / budget | One read of a claim-relevant split / pre-registered limit on reads |
| Sealed set | Read exactly once per run, for the claim |
| Floor | Pre-registered smallest gain worth claiming |
| Step budget S | Maximum steps since last promotion before HALT |
| TIS | Truncated importance sampling correction for vLLM/trainer mismatch |
| RFT | Rejection-sampling fine-tuning |
| Control plane | Durable experiment semantics: controller, gate, lineage, budgets and provenance |
| Execution plane | Resource-dependent implementations of training, generation, jobs and storage |
| `ModelSpec` | Semantic identity and configuration of the model under experiment |
| `ExecutionProfile` | How that model is run on available hardware |
| `JobSpec` | Versioned serializable definition of one expensive operation |
| `JobResult` | Durable result accepted idempotently by the control plane |
| `StateRef` | Location-independent logical reference to a finalized training state |
| `ArtifactRef` | Location-independent reference to an immutable artifact |
| Idempotent job | Retrying one logical job cannot apply its effects twice |
| Reference backend | Local single-node implementation used to define and test semantics |
| Research profile | Execution profile used for target-model scientific validation |
| Complete operational resume | Persisted project-controlled state sufficient to continue without manual reconstruction; not a bit-exactness promise |

## Appendix C — References

Summaries are paraphrased; consult originals.

- Shao et al., 2024, *DeepSeekMath* — GRPO.
- Yu et al., 2025, *DAPO* — asymmetric clipping, token-level loss, dynamic sampling, overlong shaping.
- Liu et al., 2025, *Understanding R1-Zero-Like Training* (Dr. GRPO).
- Thinking Machines Lab, 2025, *LoRA Without Regret*.
- Shao et al., 2025, *Spurious Rewards: Rethinking Training Signals in RLVR*.
- Yue et al., 2025, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*
- Hochlehnert et al., 2025, *A Sober Look at Progress in Language Model Reasoning*.
- Shenfeld et al., 2025, *RL's Razor*.
- Yao et al., 2025, on training–inference mismatch in RL frameworks and truncated importance sampling.
- Davison & Hinkley, 1997, *Bootstrap Methods and their Application* — resampling for multi-stage data.
- Field & Welsh, 2007, *Bootstrapping clustered data*, JRSS-B.
- Qwen3-4B-Instruct-2507 model card (non-thinking only; reported AIME25; recommended 16,384-token outputs and sampling settings).
- Pinned Hugging Face TRL documentation, GRPO and paper index. Algorithmically relevant IS/TIS behaviour is verified against the pinned release and explicit configuration; dynamic sampling is not assumed available.
- `math-verify`; vLLM LoRA serving documentation.

## Appendix D — Design review response

Every finding in the external review and where it is addressed.

| Review item | Finding (short) | Change | Section |
|---|---|---|---|
| C1 | Margin on lower bound confused with MDE; conjunction; no joint power | Test against zero; one primary + 4 guardrails; joint sizing; E3 | §21.2, §21.7, §22 |
| C2 | Gate cadence caps horizon at 150 steps | N, G, S from E1; restore only on confident regression | §10, §20 (D28) |
| C3 | Instrument never validated at relevant effect | Research Phase I; E3/E4/E5 in Scientific Level 0; real E1 checkpoints primary, scaled adapters supplementary | §29.2, §36.2 (D26, D39) |
| C4a | Correlated items treated as independent | Paired cluster bootstrap; programs as units | §21.4 (D30) |
| C4b | Inner resampling double counts | Removed; synthetic demonstration | §21.4 |
| C4c | Shared reference draws | Fresh references for Gate-B; larger k_ref | §21.5 |
| C4d | Claim set read repeatedly | Audit/sealed split, sealed read once | §12.1 (D29) |
| M1 | Tripwires fire in normal RL; no multiplicity control | Metric classes; absolute drift limits; dual-checker disagreement; false-halt budget; E5 | §19, §24 |
| M2 | Human feedback loop | Human-change budget; forced Gate-A refresh; Gate-B budget | §24 (D19) |
| M3 | Lengths and hardware optimistic | Non-thinking ≠ short; 80 GB assumed; throughput model | §11 (D02), §25 |
| M4 | Narrow evidence; possible recall | Stage B instrument defined; claim narrowed; E7 | §12.1, §21.1, §36.3 |
| M5 | Template rule mixes format with capability | Content metric over eval-template rotation; format guardrail; E8 | §21.3 |
| M6 | Battery rule underspecified | Four guardrails with explicit non-inferiority; others monitored | §21.1–21.2, §21.8 |
| M7 | From-scratch scorer is a sink | Exact-numeric measurement; stdlib scorer; E6 | §16 (D27) |
| M8 | Parity tripwire mis-specified; no dynamic sampling | TIS on; mismatch monitored; coarse sync check; zero-variance share measured | §15 (D11) |
| M9 | Retroactive rollback infeasible | Raw completions stored; verifier-versioned prompt table; manual procedure | §23 (D18) |
| M10 | Controls judged by gate outcome | E2 effect size; E9 decision rule | §29.2, §36.1 |
| Minor | LoRA rank as drift bound | Removed | D06, §19 |
| Minor | −1 truncation pressure | Truncation = 0; accuracy-vs-length monitor | §17 |
| Minor | Quarantine rule (a) | Flag only | §14.4 |
| Minor | Regression replay on noise | Removed from V1 | §13 |
| Minor | `num_iterations` unspecified | μ = 1 explicit; clipping inertness stated | D05 |
| Minor | Eval vs export path | Export parity test | §11.2, §23 |
| Minor | Definition of base | Original vendor checkpoint | D14 |
| Minor | Weak replication criterion | Own lower bound > 0 and pooled lower bound ≥ floor | §36.1 |
| Minor | Unspecified items | Gate-B template (rotation), Gate-B budget, reservoir exhaustion, hidden sizing, audit rule | §12.1, §21.3, §21.9, §24.1 |
| Minor | Trainer invariant vs distillation | On-policy vs offline trainers | §27 |
| Minor | LR sweep unbudgeted | E1a budgeted | §18.1, §25.2 |
| Minor | Perturbation program invariants | Mandatory assertions; instance review | §12.2 |
| §6 | Changes before implementation (13 items) | All adopted | as above; throughput model §25.1 |
| §7 | Changes that can wait | All deferred | §33 |
| §8 | Experiments E1–E9 | Adopted as Phase I | §29.2 |
| §9 | Recommendation and go/no-go | Adopted verbatim in substance | §29.3 |

## Appendix E — Revision 3 small-compute/scalability review response

| Review item | Finding | Revision 3 response |
|---|---|---|
| S1 | Small-model implementation can be mistaken for a large-model scientific proxy | Separate engineering and scientific validation tracks |
| S2 | `ModelProfile` mixes model semantics and hardware topology | `ModelSpec` + `ExecutionProfile` |
| S3 | Controller is too close to local subprocess execution | `JobExecutor` and versioned job contracts |
| S4 | vLLM colocate is treated as trainer architecture | `RolloutProvider` boundary |
| S5 | SQLite/filesystem may leak into core semantics | `MetadataStore` + `ArtifactStore` + durable refs |
| S6 | Retry/distribution semantics are undefined | Idempotent jobs + atomic finalization |
| S7 | Scalability is documented but not testable | Backend contract suites and replay executor |
| S8 | Controller branch coverage would otherwise require GPU runs | Replay/simulation backend |
| S9 | E1/E2 bootstrap ignores training-run variance | Independent seed/staged confirmation |
| S10 | Consensus quarantine can remove valid hard prompts | Flag + independent verification |
| S11 | E3 scaled adapter may not represent true trajectory | Real checkpoints + scaling cross-check |
| S12 | Algorithm behaviour relies on library defaults | Explicit configuration + compatibility assertions |
| S13 | “Exact resume” overstates determinism | Complete operational resume; bit-exactness not required |
| S14 | “No model-generated test” conflicts with hidden-program construction | Distinguish unverified labels from model-drafted executable programs |

### Appendix E.1 Revision 3 architecture invariants

1. **Measurement independence:** the gate never uses the training verifier.
2. **No measurement leakage:** no trainer receives measurement-split items.
3. **Immutable champion:** unaccepted states cannot mutate the champion.
4. **Durable lineage:** every accepted/rejected candidate has stable provenance.
5. **Control/execution separation:** controller semantics do not depend on local/remote topology.
6. **Serializable jobs:** expensive work is specified by versioned durable values.
7. **Location-independent state:** core logic never identifies states by physical path.
8. **Idempotent effects:** one logical job can affect lineage/budgets at most once.
9. **Explicit algorithm configuration:** no preregistered scientific behaviour relies on library defaults.
10. **Small-model claims stay engineering-only:** target-model effect and power are recalibrated.
11. **Independent run uncertainty is acknowledged:** evaluation bootstrap does not stand in for training seeds.
12. **Backend substitution is tested:** every implementation passes the same contract suite.
13. **Failure before finalization is non-semantic:** partial work cannot become champion, spend a claim result twice or create a valid state.
14. **Human intervention remains budgeted:** backend changes do not bypass search-accounting rules.

### Appendix E.2 Ownership and dependency direction

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

Forbidden dependencies include `controller -> vllm`, `controller -> sqlite3`, `controller -> pathlib state paths`, `controller -> subprocess`, `promotion policy -> model loading`, and `statistics -> storage backend`. Allowed dependencies include `controller -> JobExecutor/MetadataStore`, trainer implementations to `RolloutProvider`, local rollout providers to vLLM, SQLite stores to `sqlite3`, and local artifact stores to filesystem APIs.
