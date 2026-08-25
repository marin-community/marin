# Background Research Brief

- Effort: medium
- Stop rule: stop after the live measurements, Marin code, available cluster storage classes, and primary storage documentation converge on a ranked experiment.
- Date: 2026-07-23

## Question

Why does a CoreWeave finelog mirror need 28-95 seconds to bind its listener after a restart, and which storage or finelog changes can reduce that recovery time?

## Current Marin Context

RNO2A stores the finelog cache on a 250 GiB `shared-vast` PVC. The volume held 2.5 GiB across 315 segment files and was 1% full during the incident. The PV is NFSv3 with `nconnect=32`, multiple VAST remote ports, `lookupcache=pos`, `noatime`, and `noacl`. All three Marin CoreWeave clusters expose only the `shared-vast` StorageClass; no persistent SSD or block class is selectable today.

The live resource patch restarted RNO2A at 02:20:49 UTC. The server logged `finelog-server listening` at 02:21:17.415 UTC, about 28.4 seconds later. Warm read-only probes took 0.244 seconds to `stat` every local file and 1.30 seconds to read 8 KiB from every file. Earlier liveness-driven restarts took 85-95 seconds before bind.

## Internal Prior Work

Before PR #7540, `lib/finelog/rust/src/store/namespace.rs` adopted local segments during `Namespace::open`, then called `catalog.upsert_segment` once for every adopted segment. `lib/finelog/rust/src/store/catalog.rs` executed each upsert as a standalone SQLite statement. On an NFS-backed SQLite database, routine startup could therefore issue hundreds of implicit commits and synchronous filesystem operations before the listener bound.

PR #7540 changes that path to one transaction per namespace, eliminates a second full Parquet-footer scan that only recomputed `next_seq`, selects SQLite's rollback-journal `PERSIST` mode with `synchronous=FULL`, and emits structured phase timings. WAL remains disabled because SQLite requires WAL clients to share a memory index and explicitly does not support WAL over network filesystems.

Iris already keeps its CoreWeave controller SQLite state on node-local NVMe through `storage.local_state_dir`; `lib/iris/docs/coreweave.md` documents that `/mnt/local` is the bare-metal node's NVMe RAID. Finelog differs because its PVC provides node-independent persistence for active segments that have not yet reached object storage.

## External Prior Art

CoreWeave documents `emptyDir` and `hostPath` as node-local NVMe options. `emptyDir` is erased with the pod; `hostPath` survives pod replacement on the same node but couples data to that node. CoreWeave's distributed file storage documentation identifies `shared-vast` as the general PVC path. Dedicated VAST storage can provide block access and custom QoS, but it is a separately provisioned product rather than a StorageClass currently present in these clusters.

VAST QoS can prioritize metadata operations and define minimum, maximum, or burst throughput. The live PV has no attached QoS policy. Increasing a PVC's requested capacity has no established performance effect without a capacity-based QoS policy.

CoreWeave recommends larger object-store requests and notes that LOTA caches objects above 4 MiB. Remote recovery that touches many small segment objects will not benefit fully from LOTA. SQLite's documentation warns that WAL is unsuitable on general network filesystems; batching rollback-journal transactions is the safer first change.

## Negative / Failed Leads

- The PVC is 99% free, so capacity exhaustion is excluded.
- The container recorded zero cgroup OOM kills and peaked below its 32 GiB limit. More memory does not address pre-bind filesystem work.
- Warm metadata and small-read probes are much shorter than startup. Raw footer reads alone do not explain the 28.4-second interval.
- The current NFS mount already uses connection spreading and metadata-friendly options. Mount tuning has little obvious headroom and cannot remove SQLite commit round trips.
- Switching SQLite to WAL is not a valid general NFS optimization.

## Evidence Map

### Claim: standalone startup catalog upserts are the leading bottleneck

- Support:
  - Marin code: one `catalog.upsert_segment` call per adopted segment in `Namespace::open`.
  - Live RNO2A: about 315 local segment files and 28.4 seconds before bind.
  - Warm storage probes: 0.244 seconds for metadata and 1.30 seconds for 8 KiB per file.
  - SQLite: a standalone write statement opens an implicit transaction; default synchronous rollback-journal behavior can require filesystem synchronization.
- Contradictions:
  - No phase timers or syscall trace yet prove how much time is spent in catalog commits.
- Directness to Marin: exact live dataset, deployment, PVC, and startup code path.
- Confidence: high inference, not yet experimentally isolated.
- Action: deploy PR #7540 and compare its new phase timings with the 28.4-second baseline.

### Claim: a larger PVC will not make this deployment faster

- Support:
  - The PVC is 1% full.
  - The PV has no QoS policy; VAST documents capacity-based performance only when that QoS mode is configured.
- Contradictions:
  - CoreWeave could apply undocumented tenant-level policies outside the PV manifest.
- Directness to Marin: exact live PV and StorageClass.
- Confidence: high.
- Action: request view-level latency and `qos_wait` data from CoreWeave before changing storage size.

## Recommended Next Experiments

### 1. Measure the batched startup path

- Minimum experiment: run five controlled restarts of PR #7540 and record catalog open, local adoption, footer reconciliation, catalog refresh, namespace rehydration, and total store timings.
- Baseline/control: the observed 28.4-second restart and five restarts from an identical copied store on `shared-vast` using the prior build.
- Expected signal: at least 50% lower container-start-to-listener-bind time and one catalog commit per startup or namespace instead of one per segment.
- Falsifier: startup improves less than 20% and the footer/adoption phase remains dominant.
- Cost/risk: low implementation cost; preserve FULL durability and crash-safe transaction semantics.
- Sources: Marin `namespace.rs` and `catalog.rs`; SQLite transaction and pragma documentation.

### 2. Compare shared VAST with node-local NVMe

- Minimum experiment: mount an `emptyDir` on the same CPU node, copy the same store, and run repeated startup probes without production traffic.
- Baseline/control: identical image, CPU limit, store contents, and node; only filesystem differs.
- Expected signal: NVMe isolates the filesystem contribution to startup.
- Falsifier: NVMe startup is within 20% of VAST after catalog batching.
- Cost/risk: diagnostic only. Do not cut production to `emptyDir`; it loses data on pod or node replacement.
- Sources: CoreWeave local storage documentation; Marin CoreWeave runbook.

### 3. Measure and prioritize the VAST view

- Minimum experiment: ask CoreWeave for metadata latency, IOPS, and `qos_wait` for VAST view `110833` / quota `110935` during restart windows; request a trial prioritized view if contention appears.
- Baseline/control: restart latency and VAST counters before and during the trial.
- Expected signal: lower tail startup time when `qos_wait` or metadata latency was previously elevated.
- Falsifier: `qos_wait` remains near zero and storage latency does not correlate with slow boots.
- Cost/risk: provider coordination and possible storage cost.
- Sources: VAST QoS and connection-balancing documentation.

### 4. Increase remote segment size

- Minimum experiment: inventory remote object sizes and count footer/object requests during reconcile; compact objects toward at least 15 MiB where retention/query behavior permits.
- Baseline/control: reconcile request count, bytes, and elapsed time before and after compaction tuning.
- Expected signal: fewer requests and higher LOTA hit usefulness for objects above 4 MiB.
- Falsifier: remote reconcile is already background-only and contributes no readiness delay, or objects already exceed the thresholds.
- Cost/risk: compaction CPU and altered query granularity.
- Sources: CoreWeave object-storage best practices.

## Hypothesis Queue Update

- Add: implicit SQLite transaction count dominates routine NFS startup.
- Add: intermittent VAST metadata contention explains the 28-95 second spread after transaction count is controlled.
- Add: remote small-object layout limits LOTA during cold recovery.
- Falsify / stop: increasing PVC capacity as a performance fix.
- Promote: deploy and measure the batched catalog refresh, single footer pass, and phase timing in PR #7540.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Finelog namespace startup | Marin code | `lib/finelog/rust/src/store/namespace.rs` | one catalog upsert per adopted segment | high | exact current code |
| Finelog catalog | Marin code | `lib/finelog/rust/src/store/catalog.rs` | standalone SQLite writes | high | exact current code |
| CoreWeave local storage | official docs | https://docs.coreweave.com/products/storage/local-storage | NVMe `emptyDir` and `hostPath` lifecycle | high | current product docs |
| CoreWeave storage overview | official docs | https://docs.coreweave.com/products/storage | available storage families | high | current product docs |
| Distributed file storage | official docs | https://docs.coreweave.com/products/storage/distributed-file-storage/create-volumes | `shared-vast` PVC behavior | high | current product docs |
| Object storage performance | official docs | https://docs.coreweave.com/products/storage/object-storage/improving-performance/best-practices | small-object and LOTA thresholds | high | current product docs |
| Dedicated VAST release | official docs | https://docs.coreweave.com/changelog/release-notes/dedicated-vast-storage | block access and custom QoS option | medium | separate product availability requires provider confirmation |
| VAST QoS overview | vendor docs | https://kb.vastdata.com/documentation/docs/qos-overview-3 | metadata QoS and capacity-based policies | high | live PV has no policy in its manifest |
| VAST client balancing | vendor docs | https://kb.vastdata.com/docs/client-to-protocol-server-cnode-balancing | connection/VIP tuning | high | live mount already uses multiple remote ports |
| SQLite over networks | official docs | https://www.sqlite.org/useovernet.html | WAL/network-filesystem caveat | high | rules out a tempting shortcut |
| SQLite pragmas | official docs | https://sqlite.org/pragma.html | journal and synchronous defaults | high | behavior should be confirmed against runtime connection settings |
| SQLite transactions | official docs | https://www.sqlite.org/lang_transaction.html | implicit transaction per standalone statement | high | explains commit count |

## Handoff

- Suggested issue title: `[finelog] Benchmark startup catalog refresh on shared VAST`
- Open questions: measured phase timing after PR #7540; VAST `qos_wait`; remote object-size distribution; acceptable durability model for a local-NVMe active cache.
- Stop reason: internal code, live measurements, installed StorageClasses, and primary external sources converge on a falsifiable first experiment.
