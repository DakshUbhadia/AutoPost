# Development data pipeline

Sprint 3 implements architecture §12 and the roadmap's CPU data-foundation gate. It produces engineering datasets and a disposable, inspectable `hidden_dev` pool. It creates no final research partitions or scientific improvement evidence.

## Build and inspect

```text
python -m autopost.cli data build-dev --root runs/development-data --seed 0
python scripts/build_hidden_perturbations.py --output runs/hidden_dev/previews.json
```

The dataset command prints a build ID and per-split `DatasetRef`s. Under `<root>/<build_id>/` it writes `index.json`, `build-report.json`, `contamination-report.json` and `hidden-review-packets.json`. Canonical immutable copies and raw source evidence are retained through `LocalArtifactStore`; SQLite records every split atomically. Repeating identical inputs, versions and seed produces identical IDs and bytes. Input ordering and filesystem locations do not participate in identity.

The checked-in fixture has 16 training clusters, 2 dev clusters, 3 pilot clusters, 2 each for screen, Gate-A, Gate-A reservoir, Gate-B, Gate-B reservoir and transfer guard, plus 24 hidden instances from 3 programs. Counts are engineering settings, not calibrated research set sizes. Every fixture source and its hand-specified embedding cache is explicitly marked synthetic. No model is downloaded or run.

## Pinned source adapters

`data build-dev` also accepts local snapshots:

```text
python -m autopost.cli data build-dev --root runs/local-data --config configs/data/development.yaml --source train.jsonl --source transfer.jsonl --source benchmark.jsonl --embeddings embeddings.json --programs hidden-programs.json
```

Each JSONL file begins with one source header:

```json
{"source":{"schema_version":1,"source_id":"example-train","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","role":"train","family_id":"example-family","license_id":"declared-license","adapter":"canonical","fixture_only":false}}
```

For the canonical adapter, following lines contain `source_item_id`, `problem`, `answer`, `skill`, optional `lineage_keys`, `multiple_choice`, `label_origin` and `source_payload`. `label_origin` must be `curated_reference`. The GSM8K adapter reads `question` and the reference after `####`; the MATH adapter reads `problem` plus an explicit `answer` or the final balanced `\boxed{...}` from `solution`. Both preserve the original record/solution in the audit payload. They do not run supplied code, invent labels or download data. Actual snapshots require exact 40-hex revisions; source IDs must be unique within a build.

Training, transfer and public-benchmark reference snapshots are required. Supply every applicable external holdout/benchmark snapshot, not merely an arbitrary benchmark file. Family and item-lineage declarations must faithfully identify upstream sources and copies. Invalid protected benchmark references fail the whole build, rather than silently weakening decontamination.

## Normalization, deduplication and splitting

1. Normalize Unicode and whitespace, preserve opaque mathematical text, canonicalize exact numeric labels and assign integer/rational/terminating-decimal or symbolic/interval/tuple types. Numeric references respect the independent exact scorer's supported size. Invalid labels, yes/no and multiple-choice items have explicit rejection reasons. Low-entropy integers are marked; training-slice quotas/exposure caps belong to Sprint 4.
2. Compute stable prompt IDs from normalized content, pinned origin, skill and normalization version. Track each item's source revision, source item ID and ancestry.
3. Cluster with deterministic 5-gram MinHash, verified shingle Jaccard, cached-vector cosine similarity and transitive source lineage. Exact matches and ancestry links are always considered. Contradictory reference labels quarantine the whole cluster; this is source-label consistency, not model-consensus exclusion.
4. Exclude training clusters touching benchmark references or transfer items. Exclude derived training sources belonging to the transfer source family. Duplicate/excluded/unused items retain reasons. Hidden-program parents are forced into the disposable pilot holdout.
5. Allocate whole clusters by a seeded hash order. Measurement partitions accept exact numeric items only; symbolic answers remain eligible for train/dev. Requested cluster counts must be available. Final checks prevent shared training/measurement problems or ancestry, including generated hidden instances.

The CPU reference implementation compares all pairs. It does not claim a large-corpus indexing or recall benchmark. A MinHash estimate is confirmed with exact shingle similarity; cosine and lineage comparisons remain separate. `configs/data/development.yaml` records the fixture build settings. Other size choices must be explicit, before claim-bearing experiments.

The embedding cache JSON contains `model_id`, exact `revision`, `fixture_only` and `vectors`, mapping `problem_key(normalized_text)` to finite, nonzero numeric vectors of one dimension. Missing vectors fail closed. The repository fixture cache is not BGE output and cannot process real snapshots. Real cached vectors must be computed for the pinned model outside this command (CPU is sufficient); their complete bytes and identity are retained for audit. No implicit model/library default or download supplies missing vectors.

## Immutable datasets and trainer isolation

Each split has a content-derived `DatasetManifest`/`DatasetRef`. Its input artifact contains only that split's tasks and manifest. Raw source snapshots, embedding caches, contamination reports and hidden previews live in a separate operator audit artifact. Neither its ref nor held-out records are passed in a trainer's dataset artifact.

`DatasetRepository` registers a complete build in one transaction. It reconciles exact, MinHash, embedding and source-lineage cluster aliases against earlier cataloged measurements and benchmark snapshots. New training slices overlapping protected history are rejected; protected aliases are retained as append-only provenance. A catalog-check report records the prior datasets, pinned embedding identity, linking edges and protected clusters. Cross-build comparison requires compatible cached vectors (or explicit new vectors for all relevant historical texts); it never silently skips missing embeddings.

`TrainingDataAccess.on_policy` returns canonical train/dev tasks. Its offline gateway additionally requires verified correct completions and provenance. Both check canonical membership and globally protected prompt IDs, cluster IDs, normalized problems and lineage. Relabeling or replacing IDs cannot turn a measurement row into an authorized training row. Only the narrow `TrainingCatalog` interface reaches these gateways.

Job reservation also validates dataset identity, split, provenance and finalized artifact content. Eligibility is checked again before accepting a result, closing the gap if measurement protection was added while a job ran. Unregistered inputs are allowed solely for explicitly marked synthetic fixture runs; such a run still cannot train on any registered measurement split. Synthetic datasets cannot enter unmarked production runs.

## Hidden programs and human review

`HiddenProgram` declares a versioned template, integer parameter ranges, one exact arithmetic answer expression, mandatory invariants and held-out parent lineage. Programs use a bounded AST arithmetic/comparison interpreter. Supplied Python functions, imports, calls, attributes, indexing, comprehensions and lambdas are rejected. No `eval`/`exec` is used. Templates accept plain parameter placeholders and cannot interpolate the answer.

Generation samples deterministically, derives labels by exact execution and rejects/regenerates invalid instances with counted reasons. The default campaign checks 256 accepted instances per program and selects a deterministic sample of 8 full generated questions/answers for human review. The catalog reconstructs property reports and generated labels before registration. Instances share their program's cluster ID. Program partition mechanics are hash-based and tested independently of research partition creation.

Generated `hidden_dev` instances are quarantined even after property checks pass. They remain inspectable through `DatasetRepository.inspect` and the review packets, but evaluation generation is blocked until a matching approval exists for every program. Review the actual generated instances, not only the template, then explicitly attest:

```text
python -m autopost.cli data review-hidden --root runs/development-data --packet runs/development-data/<build_id>/hidden-review-packets.json --program hidden-price --reviewer <reviewer-id> --decision approved --notes "Reviewed the generated-instance sample and constraints."
```

Use `--decision rejected` when appropriate. Reviews bind the exact program digest, reproducible property-report digest and exact sampled-instance IDs. Changing template/ranges/logic/report invalidates the old approval. Review records are immutable; corrections require a revised and retested program. The implementation does not invent human approval. Test approvals are synthetic and confined to temporary test catalogs.

Only `hidden_dev` is materialized. Build settings, manifests, database membership and generation-job checks reject `hidden_audit`, `hidden_sealed` and sealed transfer assets during this development stage. Enabling the later scientific protocol requires its explicit future preregistration/read-budget work.

## Schema migration and compatibility

Metadata migration 3 adds a versioned `ap_dataset_tasks` table with `source_ver`, source item identity, source role, task schema version and a constrained derived `measurable` value, plus dataset, provenance and review catalogs. The old Revision 2 `tasks` table remains unchanged as audit history and cannot feed new production training jobs. This resolves the old Python/SQL contract mismatch without relabeling or rewriting historical scientific records.

`autopost.interfaces.Task` reexports the shared frozen task contract and preserves positional construction for existing callers. Task/dataset/source/build/hidden/review wire schemas are version 1; metadata is version 3. These versions and build settings are recorded in stack/environment identities. Existing fixture history is never automatically reclassified into fixture provenance.
