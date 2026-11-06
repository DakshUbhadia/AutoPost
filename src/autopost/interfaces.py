"""autopost/interfaces.py  (Revision 3)

Core contracts for the Gated Continual RLVR system.

Revision 2 changes (architecture doc, Appendix D):
  * Gate: one pre-registered primary endpoint tested against zero
    (lower one-sided bound > 0) on Gate-A and on Gate-B, plus a few gated
    non-inferiority guardrails with explicit direction. No "lower bound >
    min_gain" margin (C1). Everything else is monitored, not gated.
  * RESTORE only on confident regression; inconclusive results CONTINUE until
    a pre-registered step budget is spent (C2).
  * Primary metric is format-agnostic accuracy averaged over several
    evaluation-only templates; answer-marker compliance is a guardrail (M5).
  * Gate-B always uses fresh reference samples (C4).
  * Trainers split into OnPolicyTrainer (prompts only) and OfflineTrainer
    (verified examples with provenance); no trainer ever receives items from
    a measurement split.
  * Regression replay removed from V1 (deferred).
  * Reward: truncated = 0 (same as wrong); overlong shaping off by default.
  * Monitored quantities are classified so that expected drift is never
    treated as damage (M1).
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Literal, Mapping, Protocol, Sequence

try:                                    # repository layout
    from autopost.contracts import (ModelSpec, ExecutionProfile, SamplingConfig, StateManifest, StateRef,
                                    JobExecutor, MetadataStore, ArtifactStore,
                                    RolloutProvider, EvalGenerationProvider, content_hash)
    from autopost.eval.stats import BootResult
except ImportError:                     # flat companion-file layout
    from contracts import (ModelSpec, ExecutionProfile, SamplingConfig, StateManifest, StateRef,
                           JobExecutor, MetadataStore, ArtifactStore,
                           RolloutProvider, EvalGenerationProvider, content_hash)
    from stats import BootResult

AnswerType = Literal["int", "rational", "decimal", "expr", "interval", "tuple"]
MEASURABLE_TYPES: frozenset[str] = frozenset({"int", "rational", "decimal"})  # exact numbers only
Split = Literal[
    "train", "dev", "pilot", "screen", "gate_a", "gate_a_reservoir", "gate_b",
    "gate_b_reservoir", "transfer_guard", "transfer_sealed",
    "hidden_dev", "hidden_audit", "hidden_sealed",
]
MEASUREMENT_SPLITS: frozenset[str] = frozenset({
    "pilot", "screen", "gate_a", "gate_a_reservoir", "gate_b", "gate_b_reservoir",
    "transfer_guard", "transfer_sealed", "hidden_dev", "hidden_audit", "hidden_sealed",
})
VerdictStatus = Literal[
    "ok", "wrong", "no_answer", "multiple_answers", "truncated", "parse_error", "timeout"
]


def stable_hash(obj: object) -> str:
    """Compatibility name for full canonical SHA-256 (never default=str)."""
    return content_hash(obj)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RawTask:
    source_id: str
    source_item_id: str
    problem: str
    answer: str
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    task_id: str
    cluster_id: str          # dedup cluster; for perturbation sets, the program id
    source_id: str
    skill: str
    problem: str
    answer: str
    answer_type: AnswerType
    low_entropy: bool
    split: Split

    @property
    def measurable(self) -> bool:
        """May appear in a measurement split (exact numeric answer)."""
        return self.answer_type in MEASURABLE_TYPES


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    role: Literal["train", "eval"]           # eval templates never used in training
    user_format: str                         # contains "{problem}"
    answer_marker: Literal["boxed", "answer_line", "final_answer_phrase"]


@dataclass(frozen=True)
class Completion:
    completion_id: str       # raw text is stored for every rollout and eval read
    task_id: str
    state_id: str
    template_id: str
    seed: int
    sample_idx: int
    text: str
    n_tokens: int
    truncated: bool


@dataclass(frozen=True)
class Verdict:
    correct: bool
    status: VerdictStatus
    extracted: str | None
    checker_id: str


@dataclass(frozen=True)
class RewardResult:
    value: float
    components: Mapping[str, float]


# Transitional name for existing lineage annotations; location-independent.
TrainState = StateManifest


@dataclass
class TaskRolloutStats:
    task_id: str
    n: int
    n_correct: int
    answers: Mapping[str, int]
    verifier_version: str


@dataclass
class TrainResult:
    state: TrainState
    rollout_stats: list[TaskRolloutStats]
    metrics: Mapping[str, float]          # includes TIS / mismatch statistics
    sync_ok: bool                         # coarse adapter-sync check (§15)


# --------------------------------------------------------------------------
# Task sources and ingestion
# --------------------------------------------------------------------------
class TaskSource(ABC):
    source_id: str

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def raw_tasks(self) -> Iterator[RawTask]: ...


@dataclass(frozen=True)
class Rejection:
    key: str
    stage: str
    reason: str


class IngestStage(ABC):
    name: str

    @abstractmethod
    def apply(self, tasks: list[Task]) -> tuple[list[Task], list[Rejection]]: ...


# --------------------------------------------------------------------------
# Verification and reward
# --------------------------------------------------------------------------
class TrainVerifier(ABC):
    """Reward only. math-verify based; runs in the sandboxed pool."""
    checker_id: str

    @abstractmethod
    def check(self, task: Task, completion: Completion) -> Verdict: ...


class EvalScorer(ABC):
    """Measurement only. Exact-arithmetic module (exact_scorer.py).
    Accepts only tasks with task.measurable == True."""
    checker_id: str

    @abstractmethod
    def score_content(self, task: Task, completion: Completion) -> Verdict: ...

    @abstractmethod
    def format_compliant(self, completion: Completion, template: PromptTemplate) -> bool: ...


class RewardFunction(ABC):
    reward_id: str

    @abstractmethod
    def __call__(self, task: Task, completion: Completion, verdict: Verdict) -> RewardResult: ...


class BinaryReward(RewardFunction):
    """1 if correct and not truncated, else 0. Truncation is treated like a
    wrong answer, not worse (Rev. 2). Optional soft overlong shaping, off by
    default; enabled only by pre-registered switch if E1 shows a truncation
    problem."""
    reward_id = "binary@2"

    def __init__(self, max_new_tokens: int, overlong_window: int = 0,
                 overlong_max_penalty: float = 0.0):
        if overlong_window and not 0 < overlong_window < max_new_tokens:
            raise ValueError("overlong_window must be in (0, max_new_tokens)")
        self.max_new_tokens = max_new_tokens
        self.window = overlong_window
        self.max_penalty = overlong_max_penalty

    def __call__(self, task: Task, completion: Completion, verdict: Verdict) -> RewardResult:
        correct = 1.0 if (verdict.correct and not completion.truncated) else 0.0
        penalty = 0.0
        if self.window and self.max_penalty:
            start = self.max_new_tokens - self.window
            excess = max(0, completion.n_tokens - start)
            penalty = self.max_penalty * min(1.0, excess / self.window)
        return RewardResult(correct - penalty, {"correct": correct, "overlong": -penalty})


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------
def signal_probability(p: float, group_size: int) -> float:
    """P(a group of size G has non-zero reward variance) at pass rate p."""
    return 1.0 - p ** group_size - (1.0 - p) ** group_size


def band_for(group_size: int, min_signal_prob: float) -> tuple[float, float]:
    if not 0.0 < min_signal_prob < 1.0:
        raise ValueError("min_signal_prob must be in (0, 1)")
    if signal_probability(0.5, group_size) < min_signal_prob:
        raise ValueError("unreachable: increase group size or lower min_signal_prob")
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if signal_probability(mid, group_size) >= min_signal_prob:
            hi = mid
        else:
            lo = mid
    return hi, 1.0 - hi


def posterior_pass_rate(n_correct: int, n: int) -> float:
    return (n_correct + 1) / (n + 2)


class PromptSampler(ABC):
    """Band filter + quotas + global cluster exposure caps + quarantine skip.
    V1 uses GLOBAL counts (lineage-scoped accounting deferred)."""

    @abstractmethod
    def next_slice(self, n_prompts: int, seed: int) -> list[Task]: ...


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
class OnPolicyTrainer(ABC):
    """Generates its own rollouts. Receives prompts, never completions.
    Implementation: TRL GRPOTrainer wrapper (vLLM colocate, TIS on)."""
    trainer_id: str

    @abstractmethod
    def train(self, start: StateRef, prompts: Sequence[Task], verifier: TrainVerifier,
              reward: RewardFunction, model: ModelSpec, execution: ExecutionProfile, steps: int) -> TrainResult: ...


@dataclass(frozen=True)
class OfflineExample:
    task: Task
    completion_text: str
    provenance: str          # "self_rft:<state_id>" | "teacher:<model>@<rev>"
    verdict: Verdict         # must be correct under the train verifier


class OfflineTrainer(ABC):
    """SFT on verified examples (RFT control; teacher distillation later)."""
    trainer_id: str

    @abstractmethod
    def train(self, start: StateRef, examples: Sequence[OfflineExample],
              model: ModelSpec, execution: ExecutionProfile, steps: int) -> TrainResult: ...


def assert_not_measurement(tasks: Sequence[Task]) -> None:
    """Invariant: no trainer ever receives an item from a measurement split."""
    bad = [t.task_id for t in tasks if t.split in MEASUREMENT_SPLITS]
    if bad:
        raise RuntimeError(f"measurement items passed to a trainer: {bad[:5]}")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalItem:
    item_id: str
    cluster_id: str
    task: Task | None
    payload: Mapping[str, object] = field(default_factory=dict)


class EvalSuite(ABC):
    suite_id: str
    version: str
    split: Split
    k: int                               # candidate samples per item
    k_ref: int                           # reference samples per item (larger; shared cost)
    templates: tuple[str, ...]           # eval-only templates, rotated across samples
    stratified: bool                     # base-saturated items removed (gate sets only)

    @abstractmethod
    def items(self) -> Sequence[EvalItem]: ...

    @abstractmethod
    def score(self, item: EvalItem, completions: Sequence[Completion]) -> Mapping[str, list[float]]: ...


class EvalRunner(ABC):
    """One vLLM engine per job; base/champion/candidate as LoRA requests.
    Stores raw completions for every read. Reference samples may be reused
    only within a stack hash, EXCEPT Gate-B, which always draws fresh
    reference samples in the same job as the candidate."""

    @abstractmethod
    def run(self, suite: EvalSuite, state_ids: Sequence[str], seeds: Sequence[int],
            fresh_reference: bool) -> Mapping[str, Mapping[str, Mapping[str, list[float]]]]:
        """state_id -> metric -> item_id -> per-sample scores."""


# --------------------------------------------------------------------------
# Monitored quantities (M1)
# --------------------------------------------------------------------------
class MetricClass(enum.Enum):
    PRIMARY = "primary"                      # gated: gain must be confirmed
    GUARDRAIL = "guardrail"                  # gated: non-inferiority vs base
    CAPABILITY_MONITOR = "capability"        # negative trend = damage; alerts; trend tripwire
    DRIFT_MONITOR = "drift"                  # expected to move (KL, entropy, length);
                                             # absolute limits only, never trend tests
    HEALTH = "health"                        # checker disagreement, sync, pool, budgets


# --------------------------------------------------------------------------
# Lineage, ledger, tripwires
# --------------------------------------------------------------------------
class LineageStore(ABC):
    @abstractmethod
    def base(self) -> TrainState: ...
    @abstractmethod
    def champion(self) -> TrainState: ...
    @abstractmethod
    def save(self, result: TrainResult) -> TrainState: ...
    @abstractmethod
    def promote(self, state: TrainState) -> None: ...
    @abstractmethod
    def archive(self, state: TrainState, decision: "Decision") -> None: ...
    @abstractmethod
    def prune(self, keep_rejected: int) -> None: ...


class Ledger(ABC):
    @abstractmethod
    def record_look(self, split: Split, suite_version: str, state_ids: Sequence[str]) -> int: ...
    @abstractmethod
    def remaining(self, budget: str) -> int:
        """Remaining reads/changes for: gate_a, gate_b, hidden_audit, hidden_sealed,
        transfer_sealed, human_changes."""
    @abstractmethod
    def record_human_change(self, description: str, informed_by: Sequence[str]) -> None:
        """Consumes the human-change budget and forces a Gate-A refresh."""
    @abstractmethod
    def record_decision(self, decision_id: str, decision: "Decision") -> None: ...
    @abstractmethod
    def record_event(self, kind: str, payload: Mapping[str, object]) -> None: ...


class MetricsHistory(Protocol):
    def series(self, metric: str) -> Sequence[float]: ...


class Tripwire(ABC):
    name: str
    alpha_per_check: float       # allocated from the pre-registered false-halt budget

    @abstractmethod
    def check(self, history: MetricsHistory) -> tuple[bool, str]: ...


def expected_false_halts(checks: Sequence[tuple[float, int]]) -> float:
    """Sum over tripwires of (per-check false-alarm probability x number of
    checks in a run). Pre-registered and validated by the E5 shadow run."""
    return sum(a * n for a, n in checks)


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------
class Outcome(enum.Enum):
    PROMOTE = "promote"
    CONTINUE = "continue"      # inconclusive: keep training; champion unchanged
    RESTORE = "restore"        # confident regression / confident guardrail violation
    HALT = "halt"              # tripwire, or a pre-registered budget is exhausted


@dataclass
class Decision:
    outcome: Outcome
    reasons: list[str]
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailSpec:
    name: str                 # e.g. "transfer", "instruction_following", "multilingual_math", "format_compliance"
    margin: float             # pre-registered non-inferiority margin (accuracy points as a fraction)
    alpha: float              # one-sided


@dataclass(frozen=True)
class GateSpec:
    alpha_a: float                          # one-sided, primary on Gate-A
    alpha_b: float                          # one-sided, primary on Gate-B (fresh reference samples)
    guardrails: tuple[GuardrailSpec, ...]
    step_budget: int                        # max steps since last promotion before HALT


class GateContext(Protocol):
    steps_since_promotion: int

    def primary(self, split: Literal["gate_a", "gate_b"], candidate: str,
                reference: str) -> BootResult: ...
    def guardrail(self, spec: GuardrailSpec, candidate: str, base: str) -> BootResult: ...
    def tripwires_fired(self) -> list[str]: ...
    def budget_left(self, budget: str) -> int: ...


class PromotionPolicy(ABC):
    @abstractmethod
    def decide(self, ctx: GateContext, candidate: str, champion: str, base: str) -> Decision: ...


class DefaultGate(PromotionPolicy):
    """Rev. 2 gate. Each test has an explicit direction:
       primary:   PROMOTE requires lower > 0;  RESTORE requires upper < 0.
       guardrail: PROMOTE requires lower > -margin (demonstrated non-inferiority);
                  RESTORE requires upper < -margin (demonstrated violation).
       Anything in between is inconclusive -> CONTINUE (until the step budget)."""

    def __init__(self, spec: GateSpec):
        self.spec = spec

    def _inconclusive(self, ctx: GateContext, why: str, ev: dict) -> Decision:
        if ctx.steps_since_promotion >= self.spec.step_budget:
            return Decision(Outcome.HALT, [why, "step budget exhausted without confirmed gain"], ev)
        return Decision(Outcome.CONTINUE, [why], ev)

    def decide(self, ctx: GateContext, candidate: str, champion: str, base: str) -> Decision:
        ev: dict = {}
        fired = ctx.tripwires_fired()
        if fired:
            return Decision(Outcome.HALT, [f"tripwire:{f}" for f in fired], ev)
        if ctx.budget_left("gate_a") <= 0:
            return Decision(Outcome.HALT, ["gate_a read budget exhausted"], ev)

        a = ctx.primary("gate_a", candidate, champion)
        ev["gate_a"] = a
        if a.upper < 0:
            return Decision(Outcome.RESTORE, ["confident regression on primary (gate_a)"], ev)
        if a.lower <= 0:
            return self._inconclusive(ctx, "primary gain not confirmed on gate_a", ev)

        for g in self.spec.guardrails:
            r = ctx.guardrail(g, candidate, base)
            ev[f"guard:{g.name}"] = r
            if r.upper < -g.margin:
                return Decision(Outcome.RESTORE, [f"guardrail violated: {g.name}"], ev)
            if r.lower <= -g.margin:
                return self._inconclusive(ctx, f"guardrail not demonstrated: {g.name}", ev)

        if ctx.budget_left("gate_b") <= 0:
            return Decision(Outcome.HALT, ["gate_b read budget exhausted"], ev)
        b = ctx.primary("gate_b", candidate, champion)    # fresh reference samples
        ev["gate_b"] = b
        if b.upper < 0:
            return Decision(Outcome.RESTORE, ["confident regression on primary (gate_b)"], ev)
        if b.lower <= 0:
            return self._inconclusive(ctx, "gain not confirmed on gate_b", ev)
        return Decision(Outcome.PROMOTE, ["primary confirmed on gate_a and gate_b; guardrails hold"], ev)


def monitor_regression(screen: BootResult) -> bool:
    """Every checkpoint: True -> RESTORE. Only a confident regression
    (upper bound < 0) on the unstratified screen set triggers it."""
    return screen.upper < 0
