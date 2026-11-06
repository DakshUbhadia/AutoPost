# Revision 3 contracts and configuration

Sprint 1 implements the boundaries defined in architecture §§8, 11.1 and 27. Durable values live in `src/autopost/contracts/`; implementations of stores, executors and generation providers remain later-sprint work.

## Wire rules

Every durable record has `schema_version: 1`. Opaque IDs are bounded strings; config, stack and content digests are full lowercase SHA-256 values. State refs identify logical states and artifact refs identify immutable content, without storage locations. Manifests use logical component names, not file paths. The store owns physical resolution, component-content validation and atomic publication.

Use `value.to_bytes()` and `Type.from_bytes(blob)` for individual records; use `decode_job_spec` / `decode_job_result` for the discriminated training/evaluation unions. Canonical JSON uses sorted string keys, compact separators, UTF-8, and finite numbers. It does not stringify unsupported objects. It rejects duplicate JSON keys on decode. Model constructors reject unknown fields, absolute paths and non-JSON runtime values. Serialization revalidates records, including nested mutable maps.

| Received protocol version | Action |
|---|---|
| Integer `1` | Supported; validate the complete record before backend work |
| Omitted version | Implied `1`; canonical output includes it |
| Any other value, including `0`, `2`, strings and booleans | Reject deterministically with `unsupported_version` |
| Known migratable version | None exist yet; a future migration must be explicit, pure, version-specific and tested before adding it to this table |

Revision 2 Python dataclasses were not a deployed version-0 job protocol. They must not be silently reinterpreted as Revision 3 jobs. Nested records follow the same version rule. `CompatibilityError` describes a backend mismatch; capability discovery and runtime assertions are deferred to Sprint 5.

## Job identity and results

Construct jobs with `TrainingJobSpec.create(...)` or `EvaluationJobSpec.create(...)`. Their `job_id` is `job-` followed by SHA-256 of the complete normalized semantic spec, excluding only `job_id`, `attempt`, and `created_at`. Run, code revision, schema, config, stack, model/profile IDs, seed, input refs, algorithm/measurement parameters and expected output identities participate. Retry metadata changes do not change the logical job. Decoding verifies the ID against the received semantic inputs.

`expected_outputs` names stable artifact output slots before their bytes or content hashes are known. Training also names its expected state. `ArtifactStore.stage` takes an opaque output identity; finalization returns an `ArtifactRef` with the validated content hash. Attempt handles and results retain attempt numbers; timestamps are UTC ISO 8601 strings. Results use stable success/failure/cancellation statuses and explicit stable error codes. Failure cannot publish a usable state or finalized outputs. Training success requires raw rollouts; evaluation success requires raw-completion evidence. Gate-B specs require fresh reference generation.

`MetadataStore.accept_result` takes generic state records, compare-and-set pointer updates and budget charges. Its contract requires atomic acceptance once per logical job, audited conflict/stale/missing-artifact outcomes and duplicate detection. It contains no promotion or gate policy. These semantics are specified here; Sprint 2 supplies implementations and crash/retry tests.

Results expose `result_digest`, a canonical hash of semantic outputs excluding attempt number and delivery timestamps. Redelivery with identical outputs has the same digest; changed outputs conflict. Stores must also verify result identity and inputs against the reserved job before acceptance.

`RolloutProvider` and `EvalGenerationProvider` use distinct project-owned request/batch values. Completions include prompt/template IDs, text, token IDs, sampled log probabilities and truncation flags. Token IDs and probabilities must align. No third-party model, tokenizer, tensor or execution object enters a durable value.

## Configuration identities

`configs/base.yaml` contains experiment semantics and three relative profile selections:

| Component | Checked-in choices |
|---|---|
| Model | `models/qwen3_0p6b.yaml`, `models/qwen3_4b_instruct_2507.yaml` |
| Execution | `execution/smoke.yaml`, `execution/dev-qwen06b-local24g.yaml`, `execution/research-qwen4b-local80g.yaml` |
| Data | `data/math.yaml` |

`load_config` returns a normalized `schema_version`, `experiment`, `model`, `execution`, `data` mapping with no include filenames. `ModelSpec` contains model/tokenizer identity, explicit rendering inputs, context limit, LoRA targets and mode. `ExecutionProfile` contains runtime backends, validation mode, dtype, quantization, sampling, LoRA settings, explicit importance sampling, concurrency, parallelism, checkpointing, offload and backend options. Null contexts/caps and revision placeholders remain visibly unresolved until the model spike/calibration; no model is downloaded in Sprint 1.

| Identity | Inputs |
|---|---|
| `config_id` / `config_hash` | Schema, experiment, model, data; execution topology is separate |
| `merged_config_sha256` | All normalized composed values |
| `stack_id` / `stack_hash` | Config ID, every execution setting, dependency lock digest and contract versions |
| `ModelSpec.spec_id` | Complete canonical model spec |
| `ExecutionProfile.profile_id` | Complete canonical execution profile |

All execution settings enter the stack conservatively: even parallelism, offload and checkpointing can affect numerical behavior. Changing YAML formatting, file locations or mapping order does not change the composed identities. Changing model semantics changes scientific identity; changing execution changes stack identity. Physical storage locations are CLI/backend concerns.

```text
python -m autopost.cli config hashes
python -m autopost.cli config show --execution execution/smoke.yaml
python -m autopost.cli config placeholders --model models/qwen3_4b_instruct_2507.yaml
python -m autopost.cli prereg init --version stageB-p1-v1 --model models/qwen3_4b_instruct_2507.yaml --execution execution/research-qwen4b-local80g.yaml
```

For preregistration validation/freezing, repeat the same profile selections and `--lock` used to create the draft. Version 2 preregistrations retain the full composed snapshot, all three identities, and the lock digest. Validation compares them with the current composed inputs before creating files or opening a database. Unpinned models, uncalibrated context/generation caps, missing human judgments and non-research profiles block scientific freezing. Phase II freezing and scientific execution remain later-sprint work.

## Transition behavior

Historical `autopost.prereg/1` records keep their canonical bytes (including the trailing newline) and file-hash verification. Legacy combined configuration can still be read for those historical records; split config requires `autopost.prereg/2`. Existing frozen records are never rewritten.

`autopost.interfaces` retains domain/scorer/statistics/gate contracts, reexports the new backend protocols, and uses separate model/execution inputs for trainers. `TrainState` is a transitional alias for the path-free `StateManifest`. `ModelProfile` was removed; no backend implementation depended on it. The root `interfaces.py` delegates to the installed package, and root `base.yaml` composes the same defaults for companion-file users. The existing scorer/statistics/gate algorithms are unchanged.
