# Read TaskTrove Clean directly from MarinSkyRL

TaskTrove Clean is ready for row-level consumption, but MarinSkyRL still requires a second artifact containing one directory and many objects per selected task. The 160-task smoke staged 2,685 objects for 18 minutes before Ray started. A SkyRL experiment should name the clean release and a source/tag/mode selection. The runtime may cache the single packed Parquet object on each allocated node, but it must not extract task directories until a rollout batch needs them. [The research brief](research.md) records the release and runtime evidence; [the Fable review](review.md) records the design corrections.

## Constraints found in review

Release `.9` is one 3.84 GB Parquet file with 66 hash-shuffled row groups. A representative cohort touches most row groups, and PyArrow decodes a whole `task_binary` column chunk for any referenced row in that group. Remote row-at-a-time access therefore approaches a full 3.84 GB read and multiplies the transfer across rollout coordinator processes. The primary win is eliminating thousands of object operations, not avoiding the packed bytes.

The release has 39 environment identities while the shared Daytona organization has a 40-snapshot quota. Changing snapshot cleanup during the ingestion work could evict environments used by concurrent runs or silently trigger Harbor's per-trial build fallback. Snapshot admission and lifecycle need a separate, quota-aware change.

Harbor already accepts and validates a concrete task directory and hashes the full environment tree. It does not need to know about Parquet, TaskTrove selectors, or Marin artifact identities. Batch-time extraction occurs immediately before `TrialConfig` construction, when every extracted task is about to run. A generic Harbor admission-time task-loader hook would only move extraction across its concurrency semaphore, so it is deferred unless measurements show batch materialization is too early.

## Selection contract

Add `TaskTroveSelection` with exact source names, exact tags, exact modes, tag match mode, an optional task limit, and a seed. Values within each field are OR'd except tags under `ALL`, which require the requested tag set to be a subset of the row's Arrow `list<string>`. Sources, tags, and modes combine with AND. At least one source, tag, or mode is required. Every named source and tag must occur in the release, so a typo cannot silently narrow a cohort.

Selection membership is independent of Parquet order. A limit keeps the lexicographically smallest SHA-256 digests of `seed`, `source`, and `path`, encoded with NUL separators. Since `(source, path)` is unique in the release, the same seed gives a nested cohort: limit 100 is a subset of limit 200. The selected references are iterated in `(row_group, row)` order for locality. Family and language filters and disjoint train/validation splits remain future extensions.

## Launch protocol and preflight

Add `TaskTroveDataSource` beside `ArtifactDataSource` in Marin. It resolves the release artifact plus `tasks/part-00000.parquet` into a `tasktrove_parquet` locator containing the exact object URI, node-local packed cache root, artifact identity, `manifest.json`'s `verify_tool_ref`, and selection. The existing directory source becomes an explicitly tagged `directory` locator. Both participate in the SkyRL step fingerprint and dependency graph.

MarinSkyRL parses the tagged union and runs the selector on the launch host before allocating the GPU gang. This metadata-only pass validates required columns and types, checks every requested value, calculates the selected reference count, distinct `dockerfile_id` count, and a digest of the ordered references, and writes those facts into `resolved-skyrl.json`. The launch host and allocated nodes need read access to the exact in-region S3 object; credential failure is a preflight error.

The task runtime copies only that exact Parquet object into an identity-keyed node-local directory. This is an automatic cache internal to the run, not an exploded staging artifact. One Iris task performs the copy per node before Ray starts, so all rollout coordinators on that node share it. The copy is size-checked and atomically published. For release `.9`, the worst-case node-local transfer is the full 3.84 GB object. No tar archive is opened at this point.

The Ray-side dataset repeats the metadata scan against the local file and asserts that the count and reference digest equal the launch preflight. A disagreement means the immutable source contract was violated and aborts before rollouts.

## Dataset and lazy task materialization

`TerminalBenchTaskDataset` accepts directory and packed source values. A packed item uses a stable display URI as `prompt`, a stable `dataset_identity/source/path` UID, and a serializable row reference in `env_extras`. All consumers of `prompt` continue to treat it as opaque until the Harbor runner, which replaces it with a local path before calling `build_trial_config`.

The packed materializer groups a rollout batch by `(dataset_path, row_group)`. A node-local lock prevents concurrent coordinators from publishing the same extracted task twice. A coordinator reads each required `task_binary` row group once per batch and never requests `solution_binary`. Because the whole Parquet file is already node-local, a separate row-group payload cache and prefetch layer is deferred until profiling shows repeated decompression is material.

For each assigned reference, the materializer validates source, path, mode, and catalog `dockerfile_id`, then extracts the tar archive into an identity-keyed directory. It accepts regular files only, rejects absolute paths, parent traversal, links, and any `solution/` entry, and publishes with an atomic rename. It requires `instruction.md`, `task.toml`, and `environment/Dockerfile`. MarinSkyRL treats the published `dockerfile_id` as catalog provenance because the publisher's normalization algorithm is not a shared runtime API. Repeated samples and later epochs reuse the same local directory. A catalog, row, or archive mismatch is fatal task-data corruption; the run stops without manufacturing a GRPO group with missing members.

Provenance retains dataset URI and identity, selection, reference count and digest, verifier ref, source, path, mode, `dockerfile_id`, and the MarinSkyRL runtime commit. Environment digest remains Harbor's own full-environment digest. A runtime-interface salt derived from locked Harbor/Daytona versions and a materializer schema is useful for cross-version snapshot safety, but it belongs in the quota-aware snapshot follow-up and is not required to make packed ingestion correct.

## Implementation slices

The first slice changes MarinSkyRL's tagged request parsing, launch preflight, exact-file node cache, packed dataset, batch-grouped lazy extraction, and CPU tests. The second Marin change adds the public source/selector API, advances the MarinSkyRL pin, and replaces `rl_smoke.py`'s exploded `sample_step` with a direct source selection that excludes judge mode.

Snapshot admission, protected LRU cleanup, cap-full telemetry, runtime-interface salting, capabilities, and judge credentials remain tracked under MarinSkyRL#540 and marin#9122. They should not be bundled into the direct-ingestion PR: the shared quota has concurrency semantics that need an explicit reservation design.

## Verification

CPU tests use a synthetic two-row-group file and cover exact source matching, tag ALL/ANY, mode filtering, source/tag/mode AND, stable nested limits, unknown sources, launch/Ray digest disagreement, and solution rejection. Dataset construction returns packed references instead of task paths. The materializer extracts only selected `task_binary` rows and publishes the task directory atomically.

The in-region integration gate selects a deterministic non-judge source from release `.9`, records preflight time, the single-object node copy time and bytes, batch materialization time, time to first trial, and object count. Success means no exploded object-store artifact, no archive extraction before a rollout batch, and completion of at least one Harbor trial. The 18-minute/2,685-object smoke is the baseline.
