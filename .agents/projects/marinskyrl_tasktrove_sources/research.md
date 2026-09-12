# Background Research Brief

- Effort / stop rule / date: medium; stopped after the issue, release PR, prior selector PR, pinned MarinSkyRL and Harbor code, and three Echo queries converged on the same boundary; 2026-09-11.

## Question

How should a MarinSkyRL `terminal_bench` run select TaskTrove Clean rows by source or tags, consume their packed Harbor tasks without an exploded staging artifact, and reuse the corresponding Daytona environments across runs?

## Current Marin Context

[marin#9122](https://github.com/marin-community/marin/issues/9122) defines the cross-repository task-environment contract. [MarinSkyRL#540](https://github.com/marin-community/MarinSkyRL/issues/540) owns packed Parquet ingestion, snapshot reuse, capability checks, and judge plumbing. Dataset-side golden and judge validation stays in [marin#9083](https://github.com/marin-community/marin/issues/9083).

TaskTrove Clean release `2026.09.10.9` contains 1,449,686 tasks in one 3,837,032,177-byte Parquet file with 66 row groups. Its selector columns include `source`, `family`, `mode`, `dockerfile_id`, `language`, and `tags`; each `task_binary` contains the Harbor task, including `environment/Dockerfile` and the pinned verifier install. The release deliberately avoids source-partitioned copies so consumers can select cohorts from one immutable object ([release contract](https://github.com/marin-community/marin/blob/0c8d8e231194981f8aafe6b0ba3d8894891abff1/experiments/post_training/tasktrove/README.md#L54-L87)).

Marin's SkyRL adapter currently has one data locator shape: an artifact URI, a node-local directory, and a relative path. `SkyRLSpec` accepts only `ArtifactDataSource`, so a caller cannot express a Parquet selector ([adapter](https://github.com/marin-community/marin/blob/0c8d8e231194981f8aafe6b0ba3d8894891abff1/lib/marin/src/marin/rl/skyrl.py#L104-L182)). The TaskTrove smoke therefore reads rows, expands each archive, uploads the resulting directory tree, and passes that second artifact to SkyRL ([smoke export](https://github.com/marin-community/marin/blob/0c8d8e231194981f8aafe6b0ba3d8894891abff1/experiments/post_training/tasktrove/rl_smoke.py#L65-L144)).

## Internal Prior Work

[MarinSkyRL PR #477](https://github.com/marin-community/MarinSkyRL/pull/477) added `org/repo[@revision]::subdir` selectors for older Hugging Face TaskTrove layouts. It resolves a Hub revision and extracts every packed task in the selected directory before training. This is useful precedent for immutable source provenance, but it selects storage subdirectories instead of row metadata and preserves the exploded-directory boundary.

The pinned MarinSkyRL protocol serializes every train and validation input as the same `DataLocator` and converts it to a local path before launch ([protocol](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/cloud/iris/protocol.py#L40-L86), [launcher adaptation](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/cloud/iris/iris_backend.py#L270-L308)). `TerminalBenchTaskDataset` then lists task directories and puts the local path in `prompt` and `env_extras` ([dataset](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/skyrl-train/skyrl_train/trajectory_runners/harbor/dataset.py#L6-L100)). The Harbor runner finally passes that path to `build_trial_config` ([runner](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/skyrl-train/skyrl_train/trajectory_runners/harbor/runner.py#L965-L999)). A packed reference can travel through `env_extras`, then become a local task directory at this last boundary.

The clean-release smoke measured the current cost. A 1,273-task export produced 8,265 objects and held two H100 nodes for 58 minutes before Ray started. A 160-task export still produced 2,685 objects and took 18 minutes to stage. The third attempt then completed two training steps, proving the downstream Harbor and SkyRL path once the tasks were local ([PR #9061](https://github.com/marin-community/marin/pull/9061)). Fable's design review found that `.9` is hash-shuffled across 66 roughly 58 MB row groups. A representative cohort touches most groups, so PyArrow binary-column access approaches a full 3.84 GB read even when only a small row subset is selected.

Harbor already gives equivalent environment trees stable auto-snapshot names. It hashes every file under `environment/` ([environment hash](https://github.com/marin-community/harbor/blob/952a005353273590631b8f970f4664a121710b32/src/harbor/utils/container_cache.py#L35-L64)) and names the snapshot from that digest plus the Daytona target ([snapshot name](https://github.com/marin-community/harbor/blob/952a005353273590631b8f970f4664a121710b32/src/harbor/environments/daytona/snapshots.py#L175-L196)). MarinSkyRL undermines reuse by deleting every Harbor snapshot idle for more than two hours before each launch ([purge policy](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/cloud/iris/iris_backend.py#L147-L155), [purge implementation](https://github.com/marin-community/MarinSkyRL/blob/d7e7351f97a2f03ced985de91bd779c4f99e3ba9/cloud/iris/iris_backend.py#L823-L867)). Harbor's environment digest also does not include the MarinSkyRL/Harbor sandbox build interface, even though Harbor appends agent tooling during image construction.

## Negative / Failed Leads

- Echo searches for the clean-release SkyRL smoke and issue #9122 returned no relevant indexed prior-work artifact. One result described a different curriculum smoke; it did not change this design. The GitHub issues, PRs, and pinned code are the primary sources.
- Reusing the existing Hugging Face selector syntax would still download and unpack a storage subtree. TaskTrove Clean's cohorts are Parquet rows, not directories.
- Passing the Parquet file through the current `ArtifactDataSource` is insufficient. MarinSkyRL materializes the whole artifact tree on every node, and `TerminalBenchTaskDataset` rejects files because it only accepts directories containing `instruction.md`. A packed locator can instead cache only the exact Parquet object and retain its row predicate.
- Treating `dockerfile_id` alone as the snapshot key is unsafe. Harbor deliberately includes other `environment/` files, and its Daytona target is also part of the current key.

## Evidence Map

### Claim: Packed selection belongs in the launch protocol and terminal-bench dataset

- Support: The release exposes typed selector columns and one packed payload per row; the current protocol only represents local directory paths.
- Contradictions: The earlier Hub selector preserves immutable provenance, but its selection unit is a repository subdirectory.
- Directness to Marin: exact current release, adapter, and pinned runtime.
- Confidence: high.
- Action: add a distinct TaskTrove Parquet locator and carry a structured selection through Marin, the MarinSkyRL request, Hydra, and `env_extras`.

### Claim: Selection can defer task extraction, but `.9` cannot avoid reading most packed bytes

- Support: Parquet readers can read `source`, `tags`, `path`, and identity columns without reading `task_binary`; the release has only 66 row groups.
- Contradictions: Fetching one binary row decodes its containing Parquet column chunk. Because `.9` is hash-shuffled, a realistic cohort touches most of the 3.84 GB file.
- Directness to Marin: exact release layout and row-group count.
- Confidence: high for correctness, medium for throughput until measured.
- Action: cache the exact immutable Parquet object once per allocated node, share selected row-group payload caches across rollout coordinator processes, and extract only the unique tasks assigned to a rollout batch. The payload prefetch never opens the task archives.

### Claim: Snapshot reuse needs a lifecycle fix, but it should not ship in the ingestion slice

- Support: Harbor already hashes environment content and serializes creation by snapshot name.
- Contradictions: the existing hash omits the sandbox build-interface version, the two-hour purge removes otherwise reusable snapshots, and 39 release environments nearly exhaust the shared 40-slot quota.
- Directness to Marin: pinned Harbor and MarinSkyRL code.
- Confidence: high.
- Action: design snapshot admission and protected cleanup separately. Derive any future runtime-interface salt from locked dependency versions and a materializer schema. Do not teach Harbor about TaskTrove or Parquet.

## Recommended Next Experiments

### 1. Packed selector and lazy materialization CPU test

- Minimum experiment / baseline: create a synthetic Parquet file with multiple sources, overlapping tags, two row groups, and distinct archives. Assert `sources` use OR semantics, requested tags use ALL semantics, both clauses combine with AND, a deterministic limit is stable, dataset construction creates no task directories, and one assigned batch extracts only its referenced rows.
- Expected signal / falsifier: selected IDs and extracted paths match exactly; any unselected extracted archive or second read of the same row group within a batch falsifies the design.
- Cost or risk / sources: CPU-only; covers the primary contract without S3 or Daytona.

### 2. Exact-file node-cache benchmark

- Minimum experiment / baseline: select the smoke's 160-task cohort directly from release `.9`, record launch-host metadata time, the one-object node copy, first-batch row-read and extraction time, and Ray-ready time. Compare against 2,685 objects and 18 minutes.
- Expected signal / falsifier: no exported task-directory artifact, one packed copy per node, and no task extraction before the assigned rollout batch. Re-reading the remote object per coordinator would falsify the cache strategy.
- Cost or risk / sources: one in-region read of the release plus a CPU launcher; do not copy the corpus across regions.

### 3. Snapshot reuse smoke

- Minimum experiment / baseline: run two deterministic tasks sharing one environment identity twice. Confirm the second run resolves the same Daytona snapshot, then exercise TTL/LRU cleanup with a fake client.
- Expected signal / falsifier: no snapshot build on the second run; a changed runtime-interface marker produces a different identity.
- Cost or risk / sources: one small Daytona smoke after CPU tests; the fake-client suite covers destructive cleanup policy first.

## Hypothesis Queue Update

1. Implement the packed source/selection contract and CPU tests.
2. Measure real `.9` range-read behavior before tuning row-group caches or changing the published Parquet layout.
3. Design runtime-salted environment identity and quota-safe snapshot admission as a separate change.
4. Keep capability declarations and judge credential plumbing in MarinSkyRL#540 as follow-up work because release `.9` has no capability columns and no configured judge endpoint.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Task environment contract | GitHub issue | [marin#9122](https://github.com/marin-community/marin/issues/9122) | cross-repository completion criteria | high | current parent issue |
| Packed ingestion issue | GitHub issue | [MarinSkyRL#540](https://github.com/marin-community/MarinSkyRL/issues/540) | runtime scope and measured staging cost | high | no comments or overlapping PR |
| Clean release | PR | [marin#9061](https://github.com/marin-community/marin/pull/9061) | release shape, counts, smoke results | high | merged at `0c8d8e2311` |
| Prior selector | PR | [MarinSkyRL#477](https://github.com/marin-community/MarinSkyRL/pull/477) | immutable Hub selector precedent | high | directory selector, not row selector |
| Marin adapter | Marin code | [skyrl.py](https://github.com/marin-community/marin/blob/0c8d8e231194981f8aafe6b0ba3d8894891abff1/lib/marin/src/marin/rl/skyrl.py#L104-L182) | current data-source contract | high | exact current main revision |
| MarinSkyRL runtime | external code | [protocol and loader](https://github.com/marin-community/MarinSkyRL/tree/d7e7351f97a2f03ced985de91bd779c4f99e3ba9) | current launch and directory-only data path | high | exact Marin pin |
| Harbor snapshot cache | external code | [snapshot implementation](https://github.com/marin-community/harbor/tree/952a005353273590631b8f970f4664a121710b32) | current environment hash and reuse | high | exact MarinSkyRL lock pin |
| Echo searches | internal search | executions 5505, 5508, 5509 | checked for prior decisions | low | no relevant result; feedback recorded |

## Handoff

- Issue `Prior work` block: TaskTrove Clean already publishes one immutable Parquet object with `source`, `tags`, `dockerfile_id`, and packed Harbor task columns. MarinSkyRL PR #477 provides immutable selector provenance but still explodes a selected Hub directory. Harbor already caches Daytona environments by environment-tree digest; MarinSkyRL's two-hour purge prevents durable reuse and the digest lacks a runtime-interface version. Implement a distinct packed-task locator, metadata-only selection, batch-grouped lazy extraction, runtime-salted environment identities, and quota-aware cleanup.
- Review outcome: the first selector also exposes `mode`; `family` and `language` wait. Exact-file node caching replaces repeated remote row-group reads. Snapshot lifecycle and any Harbor task-loader extension are separate follow-ups. The packed path must be measured against the 18-minute smoke baseline before it is declared complete.
