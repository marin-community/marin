# Finelog Compaction Stream Contract

This specification defines the planner boundary, migration output contract, and staged all-table cleanup proposed in [design.md](design.md). It changes no RPC, catalog table, state-document schema, or active Levanter table specification. It changes partitioned migration staging names to include the full partition fingerprint.

## Terms

A live object is an immutable data object referenced by the current local catalog revision. Query visibility is a separate property selected by the active table-spec version and migration phase.

A migration has two target-version classes. Backfill objects replace pre-fence source rows and carry source identities until activation proves the rewrite complete. Ordinary objects contain writes assigned after the migration fence. The classes remain separate through `OBSERVING` so abort can remove backfill objects and retain post-fence writes.

A catalog partition identity is the complete canonical `SegmentPartition`: partition spec ID plus ordered field names and values. Storage-directory hashing does not affect equality or query pruning.

## Planner API

File: `lib/finelog/rust/src/store/compaction/planner.rs`

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnpartitionedRunPolicy {
    StrictAdjacency,
    SparseStream,
}

pub fn plan(
    config: &CompactionConfig,
    segments: &[SegmentRow],
    unpartitioned: UnpartitionedRunPolicy,
) -> Option<CompactionJob>;
```

The caller supplies segments from one namespace, table-spec version, and migration class. `plan` groups them by level and catalog partition identity.

`StrictAdjacency` joins unpartitioned neighbors only when `previous.max_seq != i64::MAX` and `previous.max_seq + 1 == next.min_seq`. The local compaction driver passes this policy.

`SparseStream` treats every unpartitioned input at one level as one stream and sorts it by `(min_seq, max_seq, path)`. Overlaps, adjacency, and true gaps do not change membership. The object compaction driver passes this policy because its catalog replaces exact immutable paths and does not splice a sequence-contiguous filesystem deque.

Partitioned inputs use one sparse stream per catalog partition identity under either policy.

Levels are considered from L0 upward. Within a level, unpartitioned input sorts before partitioned input, followed by canonical `SegmentPartition` ordering. This retains current deterministic selection; no fairness guarantee is added.

A stream is eligible when compressed bytes reach the configured level target or input count reaches `max_segments_per_level`. The job contains the shortest ordered prefix reaching either limit and never contains more than `max_segments_per_level` planned inputs. Removing a prefix cannot split the remaining sparse stream, so repeated successful jobs leave every nonterminal stream below both thresholds even when an interval that bridged other inputs moves first.

## Object Compaction Contract

Files:

- `lib/finelog/rust/src/store/compaction/object_driver.rs`
- `lib/finelog/rust/src/store/compaction/executor.rs`
- `lib/finelog/rust/src/store/table/controller.rs`

One object job contains only segments that share namespace, table-spec version, migration class, level, and catalog partition identity.

The executor performs a sorted union. It does not deduplicate equal `seq` values or equal logical rows. Every complete decoded row from each consumed input appears once in the output. Fresh footer statistics derive from the written rows.

The decoded-Arrow setting is a soft admission limit. The executor reads and measures one input at a time. It may consume a shorter prefix than the planner selected. It always admits the first readable input, even when that input alone exceeds the limit, so transient decoded memory may exceed the configured value by one input.

When a one-input job requires no repartition or encoding rewrite, the commit atomically removes and reinserts the same immutable object reference at the next level. No data bytes are downloaded, encoded, or uploaded.

The lease and commit behavior is:

- the lease pins writer epoch, table-spec version, and the observed migration phase plus transition versions;
- unrelated catalog revisions may rebase;
- an input no longer live in the catalog, a changed table-spec version, a migration lifecycle transition, or a stolen writer fence rejects the replacement;
- the catalog removes exact input paths and inserts all outputs in one transaction;
- uploaded outputs from a rejected commit remain unreferenced for object GC.

## Migration Output Contract

File: `lib/finelog/rust/src/store/table/spec_migration.rs`

Every backfill rewrite publishes outputs at L1. Ordinary object-writer flushes remain sorted, partition-stamped L0 objects. Migration-produced L1 objects also carry the requested index artifacts and use directory-based staging. The migration never derives an output level from its sources.

The rewrite uses the desired `SourceLayout`:

- rows sort by the declared columns followed by `seq`;
- `SourceLayout.partition == None` is the canonical unpartitioned form;
- a `PartitionSpec` with no fields is rejected during table-spec validation;
- a declared partition splits rows and stamps each output with its catalog partition identity;
- output Parquet uses the declared maximum row-group size;
- a target partition's staging path keeps the bounded placement directory and prefixes the filename with the 64-hex-character SHA-256 of the canonical complete `SegmentPartition`; equal `seq` minima in distinct partitions therefore have distinct collision-resistant names even when their placement bucket is equal.

For an unpartitioned target, source selection sorts by `(min_seq, max_seq, path)` and takes inputs only from one overlap-connected component per batch. It does not cross a true sequence gap. For a partitioned target, partition metadata defines sparse streams and selection may span global sequence gaps.

The source-count limit and compressed input target are hard selection limits except that one source is always admitted for forward progress. The compressed target is not a hard output-size limit: compression ratio can change, a single source can be oversized, and partitioning can emit one output per distinct partition. The decoded-Arrow rule is the soft admission contract defined above.

Migration computes distinct target partitions while building local staged outputs, then constructs their descriptors and the projected namespace catalog before upload. It rejects the batch with `StatsError::ResourceExhausted` if output partition count exceeds 4,096 or canonical serialized catalog size would exceed 33,554,432 bytes. No output is uploaded after either rejection. Crossing a limit blocks the migration for operator action; this rollout does not add partial-source checkpoints.

Lifecycle behavior is:

- `DUAL_WRITE`, `BACKFILL`, and `VERIFY`: backfill objects are excluded from compaction because their source identities prove coverage; ordinary post-fence objects may compact separately.
- `OBSERVING`: the target version is active and the rewrite is complete. Backfill and ordinary objects may compact as separate classes. Replacement preserves `migration_backfill`; source identities may be cleared because no further coverage scan occurs. Abort removes every backfill-class object and reassigns ordinary target-version objects to the source version. A compaction commit whose lease observed an earlier phase is rejected.
- `MIGRATION_PHASE_RETIRED`: the source version is removed, migration checkpoint fields are cleared from the active objects, and future compaction may mix the former classes. This phase is migration-quiescent—ordinary maintenance continues—and irreversible through `AbortTableMigration`.

## Levanter Contract

This rollout leaves the recovered production `levanter.metrics` table at version 1. Its `SourceLayout` remains unpartitioned and retains sort columns `(run_id, name, step, timestamp_ms)`, maximum row-group size 131,072, and target object bytes 268,435,456. Recovery does not register a desired v2 definition or otherwise mutate the table-spec lifecycle. This contract does not define how a new object-native Levanter table is first registered.

The object compaction driver applies `SparseStream` to Levanter v1 like every other object-backed table. It merges or promotes only active v1 objects selected from the catalog and never reads version-0 archive objects. Rewritten outputs preserve the active v1 sort and remain unpartitioned.

The existing static Levanter `run_id` partition policy remains available to the legacy local driver. It is not projected into the object-backed `TableSpec`. Adding object-native `run_id` partitioning requires a later table-spec proposal and migration with measured benefit beyond the existing run-first sort.

## Persisted Shapes

No catalog schema or protobuf change is added. The existing Levanter v1 `table_specs` row and unpartitioned segment records remain unchanged.

Migration and compaction use the existing immutable object layout and content-addressed source references. No remote object prefix changes.

## Errors

No new public error type is added.

Existing lease conflicts remain nonfatal maintenance outcomes. Object read, decode, sort, partition, upload, and catalog errors fail the cycle while preserving every live input.

`AbortTableMigration` remains valid through `OBSERVING`. It returns the existing schema conflict after `MIGRATION_PHASE_RETIRED`; recovery then requires registering a later table-spec version.

## Required Tests

- Planner tests distinguish strict adjacency from sparse-stream membership for adjacency, partial overlap, nested ranges, true gaps, and `i64::MAX`.
- Planner tests prove deterministic input and stream ordering, catalog partition isolation, and the 32-input maximum.
- A copied production metadata fixture reproduces the 150-object Levanter L2 shape and selects a bounded sparse-stream job.
- An adversarial broad bridge followed by disjoint sub-target intervals reaches the stated nonterminal bound—bytes below the level target and count below `max_segments_per_level`—under repeated jobs, including executor truncation to one input.
- Executor tests compare complete decoded logical rows plus `seq` before and after overlap repair. They do not use a lossy row digest.
- Repeated object cycles leave every nonterminal sparse stream below both configured thresholds; a whole level already below both remains in place.
- Memory tests cover a shortened prefix and the single-oversized-input exception.
- Lease tests cover an unrelated revision rebase, input no longer live, table-version change, and writer-fence change.
- Crash tests cover upload before commit and restart-driven retry.
- Migration tests assert L1 output across mixed source levels, no cross-gap unpartitioned batch, target partition stamping, and collision-free staging for two distinct partitions deliberately assigned to the same placement bucket with equal `seq` minima.
- Lifecycle tests cover frozen backfill before activation, separate-class compaction during `OBSERVING`, abort after that compaction, merged-class compaction after retirement, and rejection of commits racing activation (`VERIFY` to `OBSERVING`), abort, or retirement.
- High-cardinality tests reject 4,097 output partitions and a projected catalog above 32 MiB before upload, including the one-source case.
- A recovered Levanter v1 fixture proves recovery creates no desired v2 definition, canonical sort remains active, outputs remain unpartitioned, and compaction reads only active v1 objects.
- Scheduler tests prove a successful object job requests the 100 ms retry cadence and returns to 30-second checks when no run is eligible.
- An offline old/new planner comparison over every active production catalog reports newly eligible objects, bytes, jobs, and terminal-range amplification before deployment.
- The copied-store rehearsal uses `infra.canary.probes` as the smallest cold table and `iris.task_status` as the smallest known affected table. The harness copies their catalog and referenced live objects into a disposable local root, retains the baseline catalog, opens without a production object endpoint in `ServeMode::Shadow`, and explicitly invokes `Store::maintain_namespace` until every nonterminal level is below both limits. It compares row and sequence aggregates, per-column aggregates, level geometry, and cold recovery.

## Rollout Gates

Deployment waits until the production `log` v0-to-v1 migration reports `MIGRATION_PHASE_RETIRED` and normal write, disk, and query health.

The production planner dry run must explain every newly eligible stream before the new planner is enabled. It simulates repeated jobs and the resulting terminal bounds. Replaying the production query corpus against the simulated catalog must keep p95 candidate bytes at or below 1.25 times baseline and every query at or below 2 times baseline. The current read-only audit found material overlap only in Levanter L1/L2 and two sub-200 KB `iris.task_status` L0 neighbors.

Copied-store testing starts with `infra.canary.probes`, then runs `iris.task_status`, whose production snapshot contains overlapping L0 intervals, sparse boundaries, and a 32-object L2. The disposable root contains no production storage endpoint, and the retained baseline supplies the before side of the comparison. The rehearsal must preserve row and sequence aggregates plus per-column count/length or numeric sums, converge each nonterminal stream below both configured thresholds, and recover the same result from an empty cache.

After the copied-store rehearsal passes, operators make one global go/no-go decision. The production binary enables sparse planning for every object-backed table, and every eligible table may start immediately. There is no namespace-specific storage contract, permanent allowlist, per-table execution order, or kill switch. A binary rollback prevents new jobs but does not reverse already committed level changes. Operators rollback on any row mismatch, disk pressure, memory regression, query deadline failure, p95 latency above 1.25 times baseline, or candidate bytes above the dry-run bound. No Levanter table-spec migration or `OBSERVING` phase is part of this rollout.

## Out of Scope

- Rewriting or deleting version-0 legacy archives.
- Deduplicating rows or sequence values.
- Recompacting terminal L3 objects solely to narrow their sequence bounds.
- Adding object-native Levanter `run_id` partitioning or registering Levanter v2.
- Defining first registration for new object-native Levanter tables.
- Adding idempotent batch identities or a durable acknowledged-write receipt ledger. Post-ack client receipts and canaries are at-least-once operational evidence, not exact completeness proof.
- Adding a per-table sparse-compaction allowlist or kill switch.
- Changing maintenance cadence, global concurrency, merge thread priority, object GC, or publication throttling.
- Removing the legacy static Levanter partition policy while legacy-local tables may still use it.
- Adding a public maintenance RPC or operator-triggered compaction command.
- Adding persisted sequence-topology or migration-provenance fields; sparse unpartitioned stream membership is derived from level and partition identity.
- Guaranteeing fair service among continuously eligible partitions; the planner retains deterministic canonical ordering.
