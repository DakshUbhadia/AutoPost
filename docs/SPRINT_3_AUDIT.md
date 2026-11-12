# Sprint 3 audit

Scope: roadmap version 1.1, Sprint 3; architecture Revision 3 §12. Sprint 2's committed acceptance gate is recorded in `SPRINT_2_AUDIT.md`.

| Requirement | Implementation / evidence |
|---|---|
| Versioned Task/data migration | Shared `contracts/tasks.py`; metadata migration 3; source revision/item identity and constrained derived measurability; legacy scientific tables retained |
| Pinned source adapters and raw evidence | Canonical, GSM8K and MATH local JSONL adapters; pinned source specs; original solutions/records and embedding caches retained separately for audit |
| Normalization, typing, rejection logs | Exact Fraction labels, Unicode/whitespace handling, symbolic training eligibility, explicit invalid/MC/yes-no/conflict/duplicate/decontamination reasons |
| MinHash and embedding-assisted dedup | Deterministic signatures, shingle confirmation, finite cosine cache and transitive source-lineage clustering; production pins and complete vectors required |
| Decontamination and cluster-level splits | Benchmark/transfer exclusions, family and item provenance, hidden parents held out, whole-cluster allocation, catalog reconciliation across builds |
| Immutable dataset manifests | Per-split content hashes, independent trainer/audit artifacts, atomic complete-build registration and idempotent publication |
| Development fixture partitions | Train/dev/pilot/screen/Gate-A/Gate-B/reservoir/transfer guard fixtures with stable IDs; symbolic data remains outside measurement |
| Disposable hidden framework | Three declarative programs; execution labels, bounded AST grammar, invariant campaigns, deterministic instance samples and program partitioning |
| Human-review workflow | Pending hidden pool, immutable digest-bound reviews, property-report reconstruction and matching-sample checks; no automatic human approval |
| Trainer reachability | On-policy/offline gateways, immutable membership, protected ID/cluster/problem/lineage checks and job reservation/result-acceptance enforcement |
| No final research assets | Final hidden/sealed partitions rejected by config, manifest, catalog membership and generation-job checks |

## Validation

- Final CPU fast suite: **203 passed, 1 slow test deselected** in 138.72 seconds. This includes the prior store/crash/migration suite and new source-adapter, canonicalization, clustering, cross-build contamination, trainer reachability, property-test and review-workflow tests.
- Quick synthetic self-check: **passed**, including 4,000 randomized scorer cases, all gate branches, bootstrap simulations and joint-power examples. Scorer/statistical/gate algorithms are unchanged; the long simulation was previously validated in Sprint 1.
- Ruff: **all checks passed** for new data modules and changed store, contract, CLI/config, fixture and acceptance-test code.
- Wheel build: passed without network or new dependency installation; archive inspection confirmed migration 3 and the data/task/catalog modules are packaged.
- `pip check`: no broken requirements. `git diff --check`: passed.
- Live CLI build: passed. Development data is generated under `runs/sprint3-development-data/`; its machine-readable build/contamination/property summary is retained in `docs/sprint3_data_validation.json`.
- Catalog and artifact checks: no forbidden training/measurement overlap; training artifacts exclude raw audit sources and hidden previews; identical rebuilds retain identical IDs and manifests.

The live disposable pool contains **24 `hidden_dev` instances from 3 programs**, each passing a **256-instance** invariant campaign. Its actual human-review state remains **pending**. All generated examples are engineering fixtures, and the cache is explicitly a synthetic fixture cache rather than BGE model output. No model/GPU execution or research calibration is claimed.

Sprint 3's local implementation exit gate passes. The human-review workflow is implemented and tested with explicitly synthetic review records in temporary test catalogs; no human approval was invented for the actual development pool. Final research hidden/sealed construction remains deferred by the roadmap.
