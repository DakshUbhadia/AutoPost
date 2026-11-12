# Training signals and curriculum

Sprint 4 implements CPU training verification, binary rewards, durable pass-rate
observations, constrained sampling and independent label review. Model execution
and research calibration remain later sprints.

## Installation and verifier

`make install-cpu` installs `requirements/cpu-dev.lock` and the supplemental
`requirements/train-verify.lock`. Existing environments can install the latter
directly. The checker pins math-verify 0.9.0, latex2sympy2_extended 1.11.0, SymPy
1.14.0, ANTLR runtime 4.13.2 and mpmath 1.3.0. Its identity includes the pins and
extraction-policy version and configured deadline. Incompatible installations fail before verification.
See the upstream [Math-Verify API](https://github.com/huggingface/Math-Verify).

Each request runs in a fresh Python `-I` subprocess with a private temporary
directory, restricted environment, bounded JSON input and parent-enforced wall
deadline. A semaphore bounds concurrency; `check_many` preserves order. Unix
workers also have CPU/address-space limits. Internal math-verify timeouts are
disabled to avoid nested Windows processes. This isolates a constrained math
parser; it is not a general sandbox for executing hostile Python. Windows lacks
the Unix address-space limit. Timed-out workers are killed and reaped.

Only bounded final expressions enter the LaTeX parser. Code-like syntax,
malformed boxes and ambiguous answers fail closed. Python expression parsing
and string-equality fallback are disabled. Empty, malformed, over-limit,
conflicting, truncated and timed-out outputs receive non-correct statuses.
Binary rewards are one for correct, non-truncated `ok` verdicts and zero
otherwise. Overlong shaping requires an explicit window and penalty and remains
off by default. Evaluation verdicts cannot produce training rewards or estimates.
`verify/diagnostics.py` computes paired checker disagreement for monitoring only.

## Commands and inputs

Run `python -m autopost.cli data build-dev`. Copy the chosen training split's
DatasetRef from its generated `index.json` into `train-ref.json`. A DatasetRef
contains `schema_version`, `dataset_id` and `sha256`. Prepare `completions.json`
as a JSON array of Completion records: `completion_id`, `task_id`, `state_id`,
`template_id`, integer `seed`, nonnegative integer `sample_idx`, `text`,
nonnegative integer `n_tokens`, and boolean `truncated`.

```text
python -m autopost.cli training verify --dataset-ref train-ref.json --task-id TASK_ID --completions completions.json --group-id base-task-group --max-new-tokens 2048 --fixture-only
python -m autopost.cli training sample --dataset-ref train-ref.json --slice-id segment-1 --n-prompts 16 --seed 17 --fixture-only
python -m autopost.cli training refresh --dataset-ref train-ref.json --promotion-id promotion-1 --fixture-only
python -m autopost.cli training review --dataset-ref train-ref.json --review label-review.json --fixture-only
```

`--root` defaults to `runs/development-data`, using `datasets.sqlite` and
`artifacts/`. Synthetic datasets require `--fixture-only`; curated datasets do
not. Verification supports `--workers` and per-process `--timeout`. Group samples
share task/state/template/seed and have distinct completion IDs/sample indices.
A completion cannot be counted in two groups. Raw text and verdicts are retained
before the metadata transaction. Identical retries preserve observations and
exposures; changed semantic inputs conflict.

`CurriculumConfig.from_experiment` reads existing configuration. Beta(1,1)
posterior estimates require at least `estimate_k` observed samples before a task
can train. Both initial base estimates and ongoing rollout groups use
`record_group`; this sprint does not generate model samples.

Sampling applies the derived band, source/skill quotas, low-entropy quota,
independently confirmed exclusions and global cluster caps together. Quotas
round down: one prompt cannot fit a 50% source quota. The single-source
development fixtures cannot satisfy the default source quota; use balanced
curated data or an explicit disposable fixture configuration. Integration tests
declare their relaxed fixture policy.

Stable ordering, seeded shuffling and bounded deterministic constraint search
produce selections. An unsatisfied slice or exhausted search fails without
relaxing constraints or charging partial exposures. A single write transaction
serializes selection and exposure reservation. Counts are global across states,
restores and builds. Each prompt reservation charges one planned rollout group.
Retry with the same `slice_id`; a new group needs a new ID. `next_slice(n, seed)`
derives this ID from semantic inputs; `reserve(slice_id, n, seed)` accepts one.

Repeat `--exclude-checker CHECKER_ID` on sample/refresh to rebuild estimates
without a known-bad version while retaining history. Promotion refresh records
immutable band snapshots and returns estimated prompts near an edge for
re-estimation. Three consecutive in-band promotions flag audit and preserve
eligibility.

## Label review and isolation

Two estimates with at least 60% agreeing on the same wrong answer create
`suspected_label_error`; the item remains eligible. Exclusion requires a
canonical `LabelReview` with exact task ID/digest, method (`human`, `deterministic`
or `second_verifier`), checker version, reviewer, reason, `independence_basis`,
decision and finalized evidence ArtifactRef. Use the artifact store or
`TrainingSignalRepository.retain_evidence` for the independent derivation,
second-verifier result or actual human audit record.

The reviewer attests independence, including a different failure mode for a
second verifier. The reward checker cannot serve as its own second verifier;
the evaluation scorer cannot decide training eligibility. Consensus is not a
review method. No command invents human approval. Decisions are `confirmed_bad`,
`confirmed_good`, `corrected` and `irreparable`. Confirmed-good labels become
eligible again. Corrected/irreparable immutable labels remain retired; correction
invalidates dependent estimates. Publish corrected labels in a new dataset and
collect fresh observations. On-policy and offline gateways enforce exclusions;
replaying a reservation containing a newly excluded item fails.

Migration 4 adds append-only observation, sample-deduplication, review, slice,
exposure and audit tables. Legacy science and job-result quarantine stay separate.
All training input still passes the Sprint 3 measurement-isolation checks.

## Evaluation scorer

`verify/exact_scorer.py` remains standard-library-only and independent of the
training verifier. `exact_numeric@3` rejects malformed explicit markers even
alongside a valid marker; `@2` could ignore malformed candidates. Numeric scope
and format compliance remain separate. Fixed, randomized rational, exploit,
malformed-marker and truncation tests cover the behavior. This change precedes
the research instrument freeze and does not rewrite historical identities.
