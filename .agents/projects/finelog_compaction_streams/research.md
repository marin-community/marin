# Background Research Brief

- Effort: low, focused on the current object-native branch, production catalog geometry, and the migration/partitioning PR history.
- Stop rule: stop when code, tests, and the live catalog agree on the cause and no new source changes the proposed ownership boundary.
- Date: 2026-09-03.

## Question

Why do migrated object-native tables retain large nonterminal levels, and how should migration and scheduled maintenance prevent or repair that state without rereading the legacy archive?

## Current Marin Context

The `levanter.metrics` table is in `MIGRATION_PHASE_RETIRED` at table-spec version 1. `RETIRED` is migration-quiescent: the new version is active, the rollback observation window has closed, and migration no longer owns maintenance, while ordinary maintenance continues. Retirement removes old-version catalog rows and clears the active objects' `migration_backfill` markers and source identities.

A read-only production catalog inspection found 150 L2 objects containing 943,134,742 rows and 14,287,345,828 compressed bytes. The current exact-adjacency planner sees 142 runs. The largest is nine objects and 212,650,253 bytes, so no run reaches the 32-object or 268,435,456-byte trigger. Of the 149 neighboring pairs ordered by `min_seq`, 129 overlap and only eight are exactly adjacent. An overlap-connected sweep produces one component of 141 objects and 14,074,695,575 bytes plus a nine-object, 212,650,253-byte component.

The same production audit found no comparable nonterminal overlap backlog in other retired tables. `levanter.metrics` also has 81 L1 objects and about 2.8 GB; `iris.task_status` has two overlapping L0 neighbors totaling less than 200 KB. The active Levanter v1 `TableSpec` retains the canonical `(run_id, name, step, timestamp_ms)` sort but has no partition declaration. The legacy-only exact `run_id` partition policy is not consulted by object flush or compaction.

## Internal Prior Work

[Issue #8737](https://github.com/marin-community/marin/issues/8737) requires object-native storage, maintained object layouts, and reusable online table migration. [PR #8707](https://github.com/marin-community/marin/pull/8707) combined a `run_id`-first sort with exact `run_id` partition metadata after a seven-day selector scanned 1.068 billion rows across 17,411 segments and exceeded the query deadline. Object-native v1 kept the sort and omitted the partition. Artifact review concluded that the sort supplied most of the observed gain and partitioning did not justify another full-table rewrite without separate evidence.

The current migration selects all pre-fence sources newest-first and batches them by source count and compressed bytes. It does not preserve level, partition, or sequence-run boundaries. Each rewrite is advertised at the deepest input level. An unpartitioned batch can therefore combine sparse or disjoint sequence sets into an object whose footer interval overlaps other batches. See [`spec_migration.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/table/spec_migration.rs#L300-L365) and [`spec_migration.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/table/spec_migration.rs#L443-L463).

The compaction planner groups by level and exact physical partition. Partitioned streams may be sparse. Unpartitioned streams split unless `previous.max_seq + 1 == next.min_seq`, and thresholds apply to each resulting run. See [`planner.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/compaction/planner.rs#L38-L83). Object replacement itself is path-based and atomic. The executor already sorts and merges overlapping inputs without dropping or duplicating rows, and the merge tests cover overlapping ranges. Exact adjacency is a planner policy rather than a storage or commit requirement.

The migration provenance bit cannot identify already-retired repair candidates. [`retire_observed_migration`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/catalog/table_specs.rs#L633-L706) clears it before setting `RETIRED`. Repair must use the active table layout and current segment set.

## Negative / Failed Leads

- Aggregate level size is not the promotion trigger. The planner applies the byte and count limits to each inferred run.
- Lowering the target would metadata-promote isolated single objects without reducing query fanout. L3 is terminal, so this would preserve the fragmentation.
- Overlapping footer intervals do not prove duplicate rows. They describe sparse sequence sets coalesced by migration. Compaction is a stable union and must preserve any duplicates that genuinely exist.
- Three Echo searches for migration and compaction prior work failed because the service returned a non-JSON response. GitHub issues, PRs, branch history, code, and live read-only observations supplied the prior-work record.

## Evidence Map

### Claim: `RETIRED` is the settled migration state

- Support: `advance_phase` publishes an owed state and returns `false` in `RETIRED`; retirement closes the observation window and clears migration-only provenance.
- Contradictions: none found.
- Directness to Marin: exact production code path and live table status.
- Confidence: high.
- Action: allow ordinary maintenance to repair a retired table without reopening the migration state machine. Here, quiescent means migration performs no work; ordinary maintenance continues.

### Claim: exact adjacency is too strict for object-native compaction

- Support: the object catalog names the full live set; replacement removes exact paths under a fenced transaction; the executor supports overlap. Historical reserved sequence bands, partitions, and dropped writes create legitimate sparse ranges.
- Contradictions: legacy local compaction comments assume a contiguous prefix for deque splicing and filesystem recovery.
- Directness to Marin: exact planner, executor, controller, and catalog implementations.
- Confidence: high for object-backed tables; the legacy path should retain its existing policy.
- Action: make unpartitioned-run policy explicit. Treat one object-backed, unpartitioned level as a sparse stream, while retaining exact adjacency for the legacy filesystem path. Sparse membership survives bounded jobs even when a consumed interval was the bridge between other ranges.

### Claim: Levanter does not need a v2 rewrite in this rollout

- Support: active production v1 retains the canonical run-first sort; every live segment is unpartitioned; sparse-stream planning can clean its stranded L1/L2 objects in place. Partitioning would rewrite roughly 36 GiB without demonstrated incremental benefit.
- Contradictions: the static legacy policy and PR #8707 encode exact `run_id` partitioning, but they do not isolate how much improvement came from partition pruning versus the sort order.
- Directness to Marin: production RPC/catalog plus merged policy code.
- Confidence: high.
- Action: keep active Levanter v1 sorted and unpartitioned, remove automatic v2 registration, and defer object-native partitioning to a separately measured proposal.

## Recommended Next Experiments

### 1. Reproduce production planner geometry from metadata

- Minimum experiment: copy the production segment descriptors into a planner fixture and compare strict-adjacency with sparse-stream planning.
- Expected signal: strict mode returns no L2 job; sparse-stream mode selects a deterministic bounded prefix and repeated cycles drain the eligible L2 level.
- Cost or risk: metadata only; no object reads.

### 2. Verify sparse repair with real Parquet inputs

- Minimum experiment: copy a small set of overlapping Levanter objects, compact them through the object driver, and compare the complete decoded rows plus `seq`.
- Expected signal: inputs are replaced atomically, every row is preserved, and the next cycle makes progress.
- Cost or risk: bounded local object reads; do not infer uniqueness from footer overlap.

### 3. Rehearse all-table cleanup from smallest to largest

- Minimum experiment: rank copied production stores by live bytes and recent query/write frequency, run repeated maintenance on the smallest and coldest tables first, then advance through active cohorts with Levanter last.
- Expected signal: every table preserves complete rows and sequence values, each nonterminal sparse stream converges below the configured byte and count limits, query candidate bytes stay bounded, and Levanter remains sorted v1 without archive reads.
- Cost or risk: bounded local copies only; no production mutation during the rehearsal.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Object-native issue | GitHub issue | [#8737](https://github.com/marin-community/marin/issues/8737) | Maintained object layout and online migration goals | High | Primary project issue |
| Levanter layout | PR | [#8707](https://github.com/marin-community/marin/pull/8707) | Run-first sort, exact partitioning, and query motivation | High | Does not isolate partition benefit from sort benefit |
| Migration batching | Marin code | [`spec_migration.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/table/spec_migration.rs#L300-L463) | Cause of mixed-level, overlapping outputs | High | Current deployed branch |
| Compaction planning | Marin code | [`planner.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/compaction/planner.rs#L38-L102) | Per-run threshold and adjacency rule | High | Current deployed branch |
| Migration retirement | Marin code | [`table_specs.rs`](https://github.com/marin-community/marin/blob/c93e91f01190f2c6fd01d0dac841fb977724aeda/lib/finelog/rust/src/store/catalog/table_specs.rs#L633-L706) | Quiescent state and provenance clearing | High | Current deployed branch |
| Production catalog | Operational observation | `finelog-marin`, read-only SQLite/RPC, 2026-09-03 | Backlog size, geometry, and active v1 layout | High | No production mutation |

## Handoff

- Issue prior-work block: #8737 established maintained object layout; #8707 established the Levanter run-first sort and partition experiment. Object-native migration emitted deep-level sparse batches that the exact-adjacency planner cannot consume; active v1 still retains the useful sort.
- Settled design choices: clean every object-backed table through sparse planning; test copied stores from smallest/coldest to largest/most active; keep Levanter sorted and unpartitioned at v1; publish future migration rewrites at L1; retain generic partition collision and fanout guards for future specs; block rollout above the stated query-candidate and latency regressions.
- Deferred follow-up: service-level compaction-debt metrics, partial-source migration checkpoints, object-native Levanter partitioning, and a universal acknowledged-write receipt ledger.
- Stop reason: code, tests, Git history, and live catalog geometry agree on the cause and repair boundary.
