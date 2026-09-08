# Finelog Compaction Streams After Table Migration

Finelog will treat each unpartitioned object level as one sparse logical stream, so overlaps and legitimate sequence gaps cannot strand aggregate debt at a nonterminal level. Future migrations will publish rewritten data at L1 under the target layout and will not manufacture intervals across disconnected unpartitioned ranges. The rollout applies sparse-stream cleanup to every object-backed table after a copied-store rehearsal on the smallest cold table and the smallest known affected table.

Levanter remains sorted by `(run_id, name, step, timestamp_ms)` and unpartitioned. This rollout does not register a v2 definition or rewrite its roughly 36 GiB active object set merely to add `run_id` partition metadata. The repair reads objects referenced by the active table state and does not list, import, rewrite, or delete version-0 legacy archives.

## Challenges

The current planner treats unpartitioned files as one run only when `previous.max_seq + 1 == next.min_seq`. Production `levanter.metrics` has 150 L2 objects and 13.3 GiB, but exact adjacency divides them into 142 runs. The largest run has nine objects and 203 MiB, so no run reaches the 256 MiB or 32-object trigger. An interval sweep instead finds one overlap-connected component containing 141 objects and 13.1 GiB.

Strict adjacency cannot be a universal data invariant without renumbering valid rows. Historical telemetry uses reserved negative sequence bands, partitioned files are sparse by construction, and dropped writes can create real gaps. Overlap also does not prove duplication: two files can contain disjoint sparse row sets whose footer intervals overlap. The migration fix prevents avoidable cross-gap batches and deep-level publication; sparse-stream maintenance handles geometry that remains valid but nonadjacent.

`MIGRATION_PHASE_RETIRED` is migration-quiescent: migration performs no more work, while ordinary maintenance continues. Retirement removes the source version and clears migration provenance from active objects. Existing tables must therefore be repaired from their current layout and segment geometry.

PR #8707 introduced both the Levanter `run_id`-first sort and exact `run_id` partitioning. The object-native v1 table retained the useful sort but not partition metadata. We do not have evidence that partition pruning materially improved the relevant queries beyond the sort order, so partitioning is not worth a full-table rewrite in this rollout.

See [research.md](research.md) for code references, production measurements, and prior work.

## Costs / Risks

- Sparse-stream leveling can produce broad L3 sequence bounds. L3 is the configured maximum level and is not selected for further leveling. Before enablement, a simulated post-compaction catalog must keep p95 candidate bytes for the production query corpus within 1.25 times baseline and every query within 2 times baseline.
- Enabling cleanup for every object-backed table creates maintenance work beyond the known Levanter backlog. The copied-store rehearsal therefore starts with the smallest cold table, then exercises real overlap and sparsity in the smallest known affected table. Production enablement is global: the current binary has no per-table kill switch, and rolling back the binary cannot undo objects already promoted to terminal L3.
- The decoded-Arrow limit is a soft admission bound. The executor learns an input's decoded size only after reading it and permits one oversized input for forward progress, so peak transient memory can exceed the configured limit by that input.
- Sparse cleanup should not compete with the current `log` v0-to-v1 migration for disk and maintenance capacity. Production deployment waits for `log` to reach `MIGRATION_PHASE_RETIRED`.

## Design

### Level object-native sparse streams

The object driver selects one namespace, table-spec version, and migration class before planning. The planner then selects a level and partition stream. Within a level, the complete canonical `SegmentPartition`, including spec ID, field names, and values, identifies a partitioned stream. Directory hashing does not affect stream equality.

Partitioned streams remain sparse and use their existing whole-stream planning. All unpartitioned objects at one level form one sparse stream, ordered by `(min_seq, max_seq, path)`. Sequence ranges guide deterministic input order; they are not membership or correctness boundaries. This remains correct because the object catalog replaces exact immutable paths and the executor performs a sorted union.

An eligible stream contributes the shortest prefix reaching either configured limit: the level byte target or `max_segments_per_level` (currently 256 MiB and 32 inputs for Levanter L1/L2). A planned job never exceeds the count limit. If decoded-memory admission consumes only one input, metadata promotion still drains that level; the remaining objects retain sparse-stream membership without relying on the interval that moved.

The executor performs a sorted union and preserves every input row. The catalog atomically replaces exact paths under a maintenance lease. A single-input level promotion atomically removes and reinserts the same source object reference with the next level; it does not rewrite bytes. These contracts make overlap safe without changing query-visible row semantics.

Legacy local compaction keeps exact adjacency. This proposal does not change its filesystem recovery or local mutation behavior. The production Levanter L2 backlog becomes one eligible sparse stream and drains through ordinary maintenance without changing table-spec version or physical partitioning.

### Publish migration rewrites at L1

Migration will publish every rewritten object at L1. Ordinary object-writer flushes remain sorted, partition-stamped L0 objects; L1 adds the index artifacts produced by migration and a directory-based staging layout. Publishing rewrites there prevents arbitrary source levels from placing new objects directly into L2 or terminal L3.

For an unpartitioned target, source batching follows overlap-connected components and never combines two components separated by a true sequence gap. For a partitioned target, the target partition metadata defines sparse stream membership and the executor may batch across global sequence gaps. Batch selection is deterministic and bounded by source count plus compressed input target. The executor applies the existing soft decoded-memory limit. Each partitioned staging filename includes the full SHA-256 fingerprint of the canonical `SegmentPartition`, so two partitions assigned to the same placement bucket cannot collide at an equal minimum sequence.

Before activation, backfill objects carry source identities that prove which source rows were rewritten; ordinary post-fence writes form a separate class. Compaction leaves the backfill class frozen through `DUAL_WRITE`, `BACKFILL`, and `VERIFY`. During `OBSERVING`, the new version is active and complete, so each class may compact separately while retaining its `migration_backfill` bit. An abort during `OBSERVING` removes the backfill class and reassigns post-fence writes to the prior version. At `MIGRATION_PHASE_RETIRED`, retirement clears the class marker and ordinary maintenance may mix all active-version objects. A compaction lease also pins the observed migration phase and transition versions; commit rejects an in-flight job if abort, activation, or retirement changed that lifecycle state.

### Keep Levanter sorted and unpartitioned

The active Levanter v1 definition already carries the canonical sort order `(run_id, name, step, timestamp_ms)`, the 256 MiB object target, and the established row-group size. Its lack of partition metadata is intentional for this rollout. Startup does not register a managed v2, mutate its table-spec lifecycle, or schedule a migration rewrite.

This contract is limited to the recovered production v1 definition. The legacy local driver may continue using its static `run_id` partition policy while legacy-local tables exist; object compaction follows the active `TableSpec` and therefore leaves production v1 unpartitioned. A future proposal for newly created object-native Levanter tables must identify the authoritative synchronous registration path. A future partitioning proposal must also demonstrate an incremental benefit over sorting and justify its rewrite cost separately.

### Schedule and roll out through existing maintenance

An object-backed cycle performs one job and reports pending on progress. The scheduler retries that table after 100 ms; it returns to 30-second checks when no run is eligible. Existing process-wide limits, low-priority compaction thread, writer fencing, and GC remain unchanged. Stream choice keeps the existing priority: lower levels first, then canonical partition order. Continuous input above compaction capacity can starve a later partition; debt metrics or introspection should make that overload visible.

Before deployment, run both planners over every active production catalog. Then copy the smallest cold table and the smallest table with observed overlap into a disposable local object root. The harness retains the baseline catalog, has no endpoint for production storage, opens in shadow mode so no scheduler runs, and explicitly invokes the same `maintain_namespace` entry point production maintenance uses. It drives the affected table until every nonterminal level is below the configured limits and compares row counts, sequence aggregates, per-column aggregates, resulting geometry, and cold-restart recovery with the baseline.

After the rehearsal passes, make one global production go/no-go decision and deploy a binary that enables sparse-stream planning for every object-backed table. This is deliberate cleanup, not a namespace allowlist: the storage contract is uniform, and unit plus scenario tests cover the larger shapes. Production waits for the `log` migration to report `MIGRATION_PHASE_RETIRED`. Once deployed, all eligible tables may begin immediately; monitoring detects regressions but is not an execution-order control. A binary rollback can stop new jobs, but already committed level changes remain valid and are not reversed.

### Future migration write authority

No table-spec migration is introduced by this rollout, so an independent post-fence write ledger is not a deployment gate. A later physical migration has five plausible authorities:

1. The target catalog and its persisted high-water mark prove that Finelog made rows durable before acknowledging them, but they are not independent of the target being verified and do not preserve RPC batch identity after compaction.
2. A client or forwarder receipt recorded after `WriteRows` succeeds is independent but only at-least-once evidence: the client can crash after the acknowledgement and before recording it, while a lost response can cause a retry that duplicates rows because `WriteRows` has no idempotency token. Levanter would also need client-side capture because its metrics do not pass through the regional log forwarders.
3. A durable producer-side pre-send journal with a stable batch identity could close the acknowledgement gap, but exact reconciliation also requires `WriteRows` to accept and deduplicate that identity. This is a protocol change.
4. A server-side receipt journal atomically committed with the append could cover every writer uniformly, but it adds another durable schema and recovery contract to the ingest path solely for verification.
5. Canary writes are cheap and easy to inspect but sample behavior rather than proving completeness.

For a future partition migration, use retained source references for pre-fence parity. For exact post-fence proof, either reconcile against an independent durable producer source or add option 3; use option 4 only when universal proof is required and producer-side identity is unavailable. Post-ack receipts and canaries remain useful operational signals, not completeness proofs.

## Testing

Planner tests cover strict local adjacency, object sparse-stream behavior across partial overlap, nested intervals, `i64::MAX`, and true gaps, deterministic ordering, canonical partition isolation, and the hard input-count limit. A copied production-catalog fixture reproduces the 150-object Levanter L2 shape. An adversarial bridge-interval fixture runs repeated memory-truncated cycles and proves the remaining nonterminal stream falls below both configured thresholds because membership does not depend on the bridge interval.

Small copied Parquet files verify complete decoded rows before and after overlap repair, partial-prefix progress under the Arrow admission rule, metadata-only single-input promotion, an unrelated flush rebasing under the lease, stale-input and fence rejection, and recovery after upload but before commit. Repeated cycles must reach the expected steady state.

Migration scenarios mix source levels, negative sequence bands, overlaps, gaps, and partitions across several batches. They assert L1 output, no cross-gap unpartitioned batch, collision-free staging for two partitions with the same placement bucket and minimum sequence, checkpoint protection before activation, separate-class compaction during `OBSERVING`, successful abort after that compaction, and ordinary compaction after retirement. Race tests reject a compaction commit after concurrent activation, abort, or retirement. Fanout tests enforce the 4,096-output and 32 MiB catalog limits before upload.

The copied-store rollout test uses `infra.canary.probes` as the smallest cold table and `iris.task_status` as the smallest table with observed overlap. The latter contains real overlapping L0 intervals, sparse boundaries, and a 32-object L2. The shadow store has no production storage endpoint or maintenance scheduler; the harness explicitly drives the real maintenance entry point. The larger Levanter geometry remains covered by the copied production-catalog planner fixture and adversarial executor scenarios.

### Rehearsal result

The September 4 rehearsal copied 25.5 MiB: 78 canary rows and 3,108,480 `iris.task_status` rows. Ordinary maintenance reduced `iris.task_status` from 39 L0, 46 L1, 32 L2, and one L3 object to 7 L0, 15 L1, one L2, and two L3 objects. The L2-to-L3 job completed in 1.45 seconds with the optimized binary. Row count, minimum and maximum sequence, sequence sum, and count/length or numeric sums for every logical column matched before maintenance, after maintenance, and after recovery into an empty cache. The remaining overlap is between the two terminal L3 objects; no nonterminal overlap remains.

## Deferred Follow-up

This PR uses the offline catalog report for rollout evidence. Service-level compaction-debt metrics, partial-source migration checkpoints, Levanter `run_id` partitioning, and a general acknowledged-write receipt ledger remain follow-up work. The fixed fanout, catalog-size, and query-regression gates block deployment instead of changing architecture during rollout.

## Open Questions

- After global production enablement, when should cleanup be declared complete? The proposal is to wait until every nonterminal stream is below both thresholds, then observe two additional maintenance intervals with representative query replay and no regression gate firing.
