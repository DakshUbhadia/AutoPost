# Durable local stores and fixture execution

Sprint 2 implements architecture §8.2 and the roadmap's persistence exit gate. All implementation and acceptance tests are CPU-only. These components do not load a model or import training frameworks.

## Lifecycle

1. Register a run and import its finalized root state. Initialize named pointers and immutable, versioned budget definitions explicitly.
2. `MetadataStore.reserve_job(spec)` records one logical job and each attempt separately. Retrying the same attempt is idempotent; a new attempt keeps the logical job ID.
3. `ArtifactStore.stage(output_id)` reserves a private staging identity. Write components, validate the manifest and finalize the artifact. The whole directory becomes visible through one same-filesystem rename.
4. `record_result(result)` journals attempt delivery. A completed executor attempt is not yet an accepted state or a budget charge.
5. `accept_result(result, effects)` validates the reserved inputs, output identities, finalized artifact bytes, state provenance, pointer expectations and budget availability. One SQLite transaction commits all state records, immutable metadata records, pointer updates, budget charges and the accepted result.

Metadata references finalized objects only. Unfinalized files and finalized objects without an accepted metadata transaction are non-semantic. A retry can reuse identical published content after a crash. Abandoned staging directories are backend-local garbage; automatic garbage collection is deferred and is not required for acceptance correctness.

## Artifact integrity and publication

`LocalArtifactStore(root)` owns physical placement. Logical artifact/component names are hashed into portable filenames, avoiding path traversal, Windows device names and case collisions. Extended Windows paths remain private to the backend.

Build a descriptor with `ArtifactManifest.create(artifact_id=..., kind=..., components=..., producer_job_id=...)`. Its digest is canonical SHA-256 of the complete manifest except `artifact.sha256`, avoiding a circular hash. Each component descriptor binds its logical name, byte size, media type and SHA-256. The top-level descriptor additionally binds the producer and artifact identity. The same constructor and digest rule are usable by future backends.

Validation rejects missing, undeclared, corrupt or conflicting components and invalid descriptors. Staged output is never visible to `resolve`. Finalization never overwrites a published identity; identical finalization is idempotent, and conflicting content raises a stable error. Every `resolve`/`read` checks the manifest and component bytes. Raw-completion manifests contain raw content and provenance, independently of score aggregates in evaluation evidence.

Files are flushed and fsynced before publication. POSIX directory metadata is fsynced as well. The tested crash guarantee is abrupt process termination on a local filesystem. On Windows Python cannot portably fsync directories; power-loss and network-filesystem guarantees depend on the deployment filesystem. No stronger guarantee is inferred from process-crash tests.

## Atomic acceptance and audit

`SQLiteMetadataStore(path, artifact_store)` uses WAL, foreign keys, `synchronous=FULL`, short-lived connections and `BEGIN IMMEDIATE` write transactions. It accesses artifact content only through `ArtifactStore`.

| Delivery | Outcome |
|---|---|
| First valid successful result | `accepted`; all effects commit together |
| Same semantic result and same effects, including a new attempt | `duplicate`; original accepted result and counters stay unchanged |
| Different result or effects for an accepted job | `conflict`; incoming result/effects retained in quarantine |
| Pointer expectation or parent fails | `stale_parent`; no state, pointer or budget effect |
| Output is still staged or missing | `missing_artifact`; safe to recover/finalize and retry |
| Manifest or bytes fail integrity checks | `corrupt_artifact`; no semantic effect |
| Total proposed charges exceed a defined budget, or no budget exists | `budget_exhausted`; no semantic effect |
| Failed/cancelled attempt | Journaled for audit; cannot apply acceptance effects |

Semantic result comparison excludes attempt number and delivery timestamps. Effects are hashed independently, so replay cannot add a second charge or change a pointer. Duplicate detection precedes pointer/budget revalidation: an identical accepted result remains a duplicate even after subsequent work advances its pointer. Artifact integrity is still rechecked.

Training acceptance requires exactly the reserved new state, the correct parent and step increment, matching config/stack/slice/verifier provenance and finalized resumable components plus raw rollouts. Evaluation acceptance checks reserved states, dataset, read/instrument/template identities, the declared master seed and fresh-reference requirement. These are consistency rules, not model-quality judgments.

Pointers use compare-and-set expectations. Budget item identities are unique within a run and budget; multiple charges against one budget are summed before checking the limit. Generic records and state/artifact/acceptance history are immutable. Conflict, stale, missing, corrupt and exhausted-budget deliveries retain canonical incoming evidence in `ap_quarantine` and an audit event. `get_result` returns accepted results; `get_attempt` returns execution status and journaled results. Store policy never chooses promotions, restores or pruning.

## Schema evolution

| Version | Meaning |
|---|---|
| Unversioned `0` | Recognized legacy database; table columns and foreign keys must match the known baseline |
| Metadata `1` | Legacy baseline registered in the migration ledger |
| Metadata `2` | Additive Revision 3 runs, jobs, attempts, artifacts, states, records, pointers, budgets, acceptance and audit tables |
| Durable wire `1` | Ref/job/result/manifest/evidence/attempt/audit schemas; separate from metadata version |
| Artifact layout `1` | Private staging and immutable object directories |

`ap_schema_migrations` records each applied SQL checksum and timestamp; `PRAGMA user_version` records the current metadata version. Forward migration is transactional, including DDL, version changes and the ledger. Scripts execute as complete statements without `executescript`'s implicit transaction commit. Unknown databases, future versions, incompatible legacy definitions and checksum mismatches are rejected.

All existing scientific tables and records remain unchanged. New generic tables use the `ap_` prefix; legacy path-based states are preserved as history and are not silently imported as new states. New metadata schema versions enter the stack hash. Migration SQL is shipped inside the Python package.

```text
python -m autopost.cli db init --path runs/new.sqlite
python -m autopost.cli db migrate --path runs/existing.sqlite
python -m autopost.cli db status --path runs/existing.sqlite
```

`db init` continues to refuse an existing populated database. `db migrate` is idempotent. Constructing `SQLiteMetadataStore` also ensures the supported schema is present.

## Fixture scenarios

```text
python -m autopost.cli fixture run --scenario success
python -m autopost.cli fixture run --scenario duplicate_success
python -m autopost.cli fixture run --scenario conflict
python -m autopost.cli fixture run --scenario stale_parent
python -m autopost.cli fixture run --scenario incomplete_artifact
python -m autopost.cli fixture run --scenario executor_failure
```

Each scenario uses `runs/sprint2-fixture/<scenario>/` by default and writes a JSON report there. Use `--root` for an isolated location. Expected rejection/failure scenarios return a successful demonstration command, with their actual outcomes visible in the report. `duplicate_success` and `conflict` first accept a baseline and then deliver a second attempt. Repeating `success` preserves the same accepted state and budget charge.

`FixtureExecutor` also exposes `crash_before_finalize` and `crash_after_finalize` interruptions for tests. The acceptance suite turns these into real abrupt child-process exits and independently exits during the metadata transaction. Recovery reopens both stores, retries the logical job and proves that its effects commit only once. Concurrent delivery tests exercise duplicate acceptance and competing pointer updates.

Every fixture payload/report is marked `fixture_only`. No fixture completion, metric or checkpoint supports a scientific model-improvement claim. Real executors/providers, data-pipeline migrations, training, controller policy and operational reporting remain later-sprint work.
