# Sprint 2 audit

Audit date: 2026-11-06. Scope: roadmap version 1.1, Sprint 2; architecture Revision 3 §§8.1–8.3. Dependency: Sprint 1's local exit gate is recorded in `SPRINT_1_AUDIT.md` and the committed contract/config implementation.

| Requirement | Implementation and evidence |
|---|---|
| SQLite metadata backend behind its protocol | `store/metadata.py`; typed run, state, job, attempt, pointer, record, budget, result and audit operations |
| Versioned forward migrations preserve history | `store/db.py`, packaged `migrations/002_durable_stores.sql`; legacy snapshots, checksum/future-schema rejection, failed-DDL rollback tests |
| Staged immutable artifacts and content manifests | `store/artifacts.py`, `ArtifactManifest.create/content_digest`; bytes, descriptor, size, provenance, path and concurrent-finalization tests |
| Durable job/attempt journal | Reserve, status, result and retry/reopen tests; stable logical IDs and immutable terminal delivery |
| Atomic state/result/budget acceptance | One acceptance transaction; exactly-once retry, duplicate, conflict, summed-budget and immutable-record tests |
| Stale/conflicting results quarantined and audited | Durable incoming result/effects plus audit events; competing-parent and modified-result/effect tests |
| Crash before publication has no semantic effect | Abrupt child-process exit after staging, followed by store reopen and recovery |
| Crash after publication safely recovers | Abrupt exit after artifact finalization and before result journaling/metadata acceptance |
| Crash during metadata acceptance rolls back all effects | Abrupt exit after all transactional writes but before commit; separate injected database-failure rollback test |
| Raw completion manifests independent of scores | Evaluation fixture changes aggregates while raw artifact ref, manifest and bytes remain unchanged |
| Deterministic fixture executor | Six supported CLI scenarios, cancellation/retry and crash scenarios; no model/GPU dependencies |
| Generic persistence contains no gate policy | Named compare-and-set pointers and immutable generic records; no promotion/restore/pruning decisions |
| Versions and operational entrypoints | Wire 1 / metadata 2 / artifact layout 1; metadata version in stack hash and CPU manifest; `db migrate/status`, `fixture run`, Make targets |

## Local validation

- Final CPU fast suite: **149 passed, 1 slow test deselected** (including all new failure, migration, fixture and concurrent-delivery tests).
- Targeted Windows path/publication/crash checks: **23 passed** after fixing private extended path handling.
- Package wheel build: passed without network or dependency installation; wheel inspection confirmed both SQL resources and the backend/executor modules are included.
- Ruff: all checks passed on changed implementation and acceptance-test code.
- Quick synthetic self-check: passed, including 4,000 randomized scorer cases, gate branches, 200-run bootstrap simulations and 50 trials per joint-power effect.
- `pip check`: no broken requirements. `git diff --check`: passed.
- Public fixture CLI: all six scenarios pass their expected outcomes; repeating success returns `duplicate` without another budget charge.

The long statistical self-check was previously validated in Sprint 1; Sprint 2 does not modify scorer, statistical or gate algorithms. Its final regression run uses the same quick synthetic check required by CPU CI. GPU execution and scientific calibration remain outside this sprint.

Sprint 2's local exit gate passes. All validation used synthetic CPU fixtures; no model was downloaded, no GPU computation ran, and no scientific threshold was selected or changed.
