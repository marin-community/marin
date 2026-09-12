# Claude Fable design review

Claude Fable reviewed the research, design, and interface spec against marin#9122, MarinSkyRL#540, release `.9`, the pinned MarinSkyRL runtime, and Harbor. It approved the tagged source, metadata selection, and lazy extraction direction, but rejected the original spec as implementation-ready.

## Blocking findings

1. Release `.9` is hash-shuffled across 66 roughly 58 MB row groups. A representative cohort touches most groups, and PyArrow decodes a full `task_binary` column chunk for one referenced row. A process-local remote cache would therefore read about 3.8 GB per rollout coordinator. The implementation caches the one immutable Parquet object per node, groups each rollout batch's reads by row group, and shares the extracted-task cache across coordinator processes. Tar extraction remains rollout-batch lazy. A separate row-group payload cache is deferred until profiling shows repeated local decompression is material.
2. The release has 39 environment identities against a shared 40-snapshot Daytona quota. The proposed one-slot LRU policy was unsafe around non-Harbor snapshots and concurrent jobs. Snapshot cleanup and reservation have been removed from the direct-ingestion slice and require a separate quota-aware design.
3. A hand-maintained runtime-interface string could silently reuse stale snapshots, and the original provenance omitted `manifest.json`'s `verify_tool_ref`. The revision carries the verifier ref. Runtime-interface salting is deferred with the snapshot lifecycle work and, when implemented, will be derived from locked Harbor, harbor-config, and Daytona versions plus a materializer schema version.

## Contract clarifications adopted

- Exact source, tag, and mode semantics; detection of individually unknown requested values.
- Stable limit byte encoding, nested-prefix behavior, and locality-oriented iteration order.
- Launch-host selection and Ray-side count/digest agreement before rollouts.
- A mode selector so deterministic runs can exclude judge tasks.
- Required-column type checks that allow future extra columns.
- Stable UID and opaque prompt semantics.
- Rejection of solutions and unsafe archives, plus comparison of the published `dockerfile_id` when a selected row is loaded. Recomputing the publisher fingerprint requires a shared normalization API and is deferred.
- Fatal handling of packed task corruption to preserve complete GRPO groups.

## Explicit deferrals

The direct-ingestion change does not update Harbor. Harbor receives a concrete directory only after MarinSkyRL materializes an assigned rollout batch. Admission-time unpacking would require a generic async Harbor task-loader hook and is only justified if batch-level extraction measures as materially early. Snapshot lifecycle, runtime-interface salting, capabilities, judge credentials, publisher layout changes, and advanced selectors remain follow-up work.

The complete reviewer output is preserved in Loom session `0dey0v76` as a typed `result`.
