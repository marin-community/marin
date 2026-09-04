# Finelog Compaction Streams After Table Migration

Finelog will treat each unpartitioned object level as one sparse logical stream, so overlaps and legitimate sequence gaps cannot strand aggregate debt at a nonterminal level. Future migrations will publish rewritten data at L1 under the target layout and will not manufacture intervals across disconnected unpartitioned ranges. A server-owned Levanter v2 physical definition will restore the exact `run_id` partitioning lost during the object-native v1 migration.

The repair reads objects referenced by the active table state. It does not list, import, rewrite, or delete version-0 legacy archives.

## Challenges

The current planner treats unpartitioned files as one run only when `previous.max_seq + 1 == next.min_seq`. Production `levanter.metrics` has 150 L2 objects and 13.3 GiB, but exact adjacency divides them into 142 runs. The largest run has nine objects and 203 MiB, so no run reaches the 256 MiB or 32-object trigger. An interval sweep instead finds one overlap-connected component containing 141 objects and 13.1 GiB.

Strict adjacency cannot be a universal data invariant without renumbering valid rows. Historical telemetry uses reserved negative sequence bands, partitioned files are sparse by construction, and dropped writes can create real gaps. Overlap also does not prove duplication: two files can contain disjoint sparse row sets whose footer intervals overlap. The migration fix prevents avoidable cross-gap batches and deep-level publication; sparse-stream maintenance handles geometry that remains valid but nonadjacent.

`MIGRATION_PHASE_RETIRED` is migration-quiescent: migration performs no more work, while ordinary maintenance continues. Retirement removes the source version and clears migration provenance from active objects. Existing tables must therefore be repaired from their current layout and segment geometry.

The Levanter table also lost a required physical layout. PR #8707 partitioned it by exact `run_id`; object-native v1 has no partition declaration and every live object reports `partition_json = NULL`. Restoring that layout requires a v2 migration over about 36 GiB of active object-native data.

See [research.md](research.md) for code references, production measurements, and prior work.

## Costs / Risks

- Sparse-stream leveling can produce broad L3 sequence bounds. L3 is the configured maximum level and is not selected for further leveling. Before enablement, a simulated post-compaction catalog must keep p95 candidate bytes for the production query corpus within 1.25 times baseline and every query within 2 times baseline.
- Levanter v2 rewrites about 36 GiB once. L1 outputs at or above the 256 MiB target can advance by metadata-only promotion; smaller same-partition outputs require later merges. Version-0 archives remain untouched.
- A partition-changing batch may produce many objects because one output is written per distinct partition. Source count and input bytes do not hard-bound partition cardinality. A batch is rejected before upload if it would emit more than 4,096 partitions or make the serialized namespace catalog exceed 32 MiB. Exceeding either limit defers Levanter v2 until partial-source checkpoints have a separate design.
- The decoded-Arrow limit is a soft admission bound. The executor learns an input's decoded size only after reading it and permits one oversized input for forward progress, so peak transient memory can exceed the configured limit by that input.
- The current `log` namespace v0-to-v1 migration and Levanter v2 both consume the single migration slot. Starting v2 before `log` reaches `MIGRATION_PHASE_RETIRED` would extend both operations.

## Design

### Level object-native sparse streams

The object driver selects one namespace, table-spec version, and migration class before planning. The planner then selects a level and partition stream. Within a level, the complete canonical `SegmentPartition`, including spec ID, field names, and values, identifies a partitioned stream. Directory hashing does not affect stream equality.

Partitioned streams remain sparse and use their existing whole-stream planning. All unpartitioned objects at one level form one sparse stream, ordered by `(min_seq, max_seq, path)`. Sequence ranges guide deterministic input order; they are not membership or correctness boundaries. This remains correct because the object catalog replaces exact immutable paths and the executor performs a sorted union.

An eligible stream contributes the shortest prefix reaching either configured limit: the level byte target or `max_segments_per_level` (currently 256 MiB and 32 inputs for Levanter L1/L2). A planned job never exceeds the count limit. If decoded-memory admission consumes only one input, metadata promotion still drains that level; the remaining objects retain sparse-stream membership without relying on the interval that moved.

The executor performs a sorted union and preserves every input row. The catalog atomically replaces exact paths under a maintenance lease. A single-input level promotion atomically removes and reinserts the same source object reference with the next level; it does not rewrite bytes. These contracts make overlap safe without changing query-visible row semantics.

Legacy local compaction keeps exact adjacency. This proposal does not change its filesystem recovery or local mutation behavior.

The production Levanter L2 backlog is therefore eligible as one 150-object sparse stream. This demonstrates the general repair, but the rollout registers Levanter v2 before the first maintenance tick, so production Levanter is repartitioned from live v1 objects instead of first consolidating the faulty v1 layout.

### Publish migration rewrites at L1

Migration will publish every rewritten object at L1. Ordinary object-writer flushes remain sorted, partition-stamped L0 objects; L1 adds the index artifacts produced by migration and a directory-based staging layout. Publishing rewrites there prevents arbitrary source levels from placing new objects directly into L2 or terminal L3.

For an unpartitioned target, source batching follows overlap-connected components and never combines two components separated by a true sequence gap. For a partitioned target, the target partition metadata defines sparse stream membership and the executor may batch across global sequence gaps. Batch selection is deterministic and bounded by source count plus compressed input target. The executor applies the existing soft decoded-memory limit. Each partitioned staging filename includes the full SHA-256 fingerprint of the canonical `SegmentPartition`, so two partitions assigned to the same placement bucket cannot collide at an equal minimum sequence.

Before activation, backfill objects carry source identities that prove which source rows were rewritten; ordinary post-fence writes form a separate class. Compaction leaves the backfill class frozen through `DUAL_WRITE`, `BACKFILL`, and `VERIFY`. During `OBSERVING`, the new version is active and complete, so each class may compact separately while retaining its `migration_backfill` bit. An abort during `OBSERVING` removes the backfill class and reassigns post-fence writes to the prior version. At `MIGRATION_PHASE_RETIRED`, retirement clears the class marker and ordinary maintenance may mix all active-version objects. A compaction lease also pins the observed migration phase and transition versions; commit rejects an in-flight job if abort, activation, or retirement changed that lifecycle state.

### Restore Levanter with a server-owned physical policy

The server policy registry owns the physical overlay for `levanter.metrics`: sort order `(run_id, name, step, timestamp_ms)`, 256 MiB target objects, and identity partitioning from `run_id` to `run_id` with partition spec ID 1. The logical schema comes from the stored active definition for an existing table or the registration request for a new table. This restores the PR #8707 layout without asking each Levanter process to reconstruct a physical policy.

For an existing unpartitioned v1 table, bootstrap copies its logical schema and operating/artifact policies, applies the managed physical overlay, and registers v2. For an absent table, first registration stores v1 with the managed layout. A later logical-schema change still requires the normal next-version registration; the server overlay supplies physical fields and rejects a client-supplied conflicting layout.

Boot order is: open or adopt the local catalog, recover and claim remote HEADs, register managed definitions only on successfully claimed writable tables, update runtime policy, start maintenance, then serve traffic. Shadow, fenced, and unready tables do not mutate their catalogs during bootstrap. The desired v2 is therefore durable before the first scheduler tick can select Levanter v1 compaction.

The v2 source universe is the active v1 object set. Version-0 local files and their archived copies represent the retired prior storage generation and are not scanned.

### Schedule and roll out through existing maintenance

An object-backed cycle performs one job and reports pending on progress. The scheduler retries that table after 100 ms; it returns to 30-second checks when no run is eligible. Existing process-wide limits, low-priority compaction thread, writer fencing, and GC remain unchanged. Stream choice keeps the existing priority: lower levels first, then canonical partition order. Continuous input above compaction capacity can starve a later partition; debt metrics or introspection should make that overload visible.

Before deployment, run both planners over a copy of every active production catalog, simulate repeated jobs, and report newly eligible objects, bytes, terminal bounds, and candidate bytes for the production query corpus. The rollout waits for the `log` v0-to-v1 migration to reach `MIGRATION_PHASE_RETIRED`, then deploys the new binary. Startup registers Levanter v2 in the boot sequence above.

Set the v2 observation window long enough to run acceptance checks. During `OBSERVING`, require v1 pre-fence source rows to equal v2 backfill rows. Compare complete decoded samples against retained v1 object references, bounded by the migration fence. Capture acknowledged post-fence writes independently at the writer boundary and compare their count and complete rows with the ordinary v2 class; canary rows supplement this check but do not replace it. Require every v2 object to carry partition spec ID 1 and run representative queries shaped as `WHERE run_id = ? AND name IN (...)`. Abort before the deadline on a row mismatch, missing acknowledged write, missing partition, failed pruning, write regression, disk pressure, any query deadline failure, or p95 latency above 1.25 times the pinned v1 baseline. After `MIGRATION_PHASE_RETIRED`, rollback requires a new table-spec migration; subsequent checks are monitoring.

## Testing

Planner tests cover strict local adjacency, object sparse-stream behavior across partial overlap, nested intervals, `i64::MAX`, and true gaps, deterministic ordering, canonical partition isolation, and the hard input-count limit. A copied production-catalog fixture reproduces the 150-object Levanter L2 shape. An adversarial bridge-interval fixture runs repeated memory-truncated cycles and proves the remaining nonterminal stream falls below both configured thresholds because membership does not depend on the bridge interval.

Small copied Parquet files verify complete decoded rows before and after overlap repair, partial-prefix progress under the Arrow admission rule, metadata-only single-input promotion, an unrelated flush rebasing under the lease, stale-input and fence rejection, and recovery after upload but before commit. Repeated cycles must reach the expected steady state.

Migration scenarios mix source levels, negative sequence bands, overlaps, gaps, and partitions across several batches. They assert L1 output, no cross-gap unpartitioned batch, collision-free staging for two partitions with the same placement bucket and minimum sequence, checkpoint protection before activation, separate-class compaction during `OBSERVING`, successful abort after that compaction, and ordinary compaction after retirement. Race tests reject a compaction commit after concurrent activation, abort, or retirement. Fanout tests enforce the 4,096-output and 32 MiB catalog limits before upload.

The marin-dev Levanter rehearsal compares pre-fence v1 rows with v2 backfill rows for fixed run IDs, verifies exact partition pruning, and records query latency. The production dry run and `OBSERVING` checks are required rollout artifacts.

## Deferred Follow-up

This PR uses the offline catalog report for rollout evidence. Service-level compaction-debt metrics and partial-source migration checkpoints remain follow-up work. The fixed fanout, catalog-size, and query-regression gates block deployment instead of changing architecture during rollout.

## Open Questions

- Should the first rollout enable sparse-stream planning for every object-backed table after the catalog dry run, or gate it to the known affected namespaces for one release? The implementation keeps the policy explicit so either rollout choice uses the same storage contract.
- Which durable writer-boundary record should be the authority for acknowledged post-fence writes during Levanter v2 observation? The acceptance rule requires an independent record; the exact operational source should be selected before deployment.
