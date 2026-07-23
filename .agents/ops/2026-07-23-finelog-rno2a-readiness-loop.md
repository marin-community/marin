---
date: 2026-07-23
system: finelog
severity: degraded
resolution: mitigated
pr: https://github.com/marin-community/marin/pull/7540
issue: none
---

## TL;DR

- RNO2A recorded six liveness restarts and 79 readiness failures in roughly three hours during concurrent ingest and large compactions.
- Exit 137 came from kubelet liveness termination, not an OOM: cgroup counters showed zero OOM kills and memory peaked at 12.4 GB under a 32 GiB limit.
- The 250 GiB PVC was 1% full. More disk capacity would not address the failure.
- A live patch raised reserved compute, added a five-minute startup probe, and moved all probes to HTTP `/health`; the replacement pod remained Ready with zero restarts.
- Global Kubernetes defaults, startup phase logging and batching, a dedicated Grafana dashboard, the readiness runbook, and a startup/PVC research brief were proposed in PR #7540.

## Original problem report

The RNO2A finelog mirror intermittently reported `HTTP /health probe is not Ready`. The request was to determine whether the server was overloaded, apply a targeted live fix, raise global Kubernetes provisioning where justified, investigate CoreWeave PVC and startup performance, add a Grafana debugging view, and open a pull request.

## Investigation path

1. The live Deployment and pod were inspected with the explicit RNO2A kubeconfig and context. The old pod had restarted nine times and retained TCP liveness and readiness probes with a 500m CPU request, 2 CPU limit, 4 GiB memory request, and 32 GiB memory limit.
2. Pod events showed six liveness kills and 79 readiness failures over roughly three hours. Previous logs showed concurrent `WriteRows` calls taking 8–24 seconds while multi-million-row compactions ran.
3. Exit 137 was compared with kubelet events and `/sys/fs/cgroup/memory.events`. The kubelet explicitly killed the container after failed liveness probes; `oom` and `oom_kill` were both zero.
4. Current and peak memory, filesystem use, and node capacity were measured. Memory peaked at 12.4 GB, the PVC used 2.5 of 250 GiB, and the 192-CPU control node had substantial unreserved compute and memory.
5. Earlier restart logs showed 85–95 seconds before the listener bound, while liveness began after 15 seconds. The probe policy amplified a transient stall into repeated cold starts.
6. The Deployment was patched in place to request 2 CPUs and 16 GiB, allow 8 CPUs and 32 GiB, add a five-minute HTTP startup probe, and use HTTP `/health` for liveness and readiness. The rollout completed without changing the PVC or Secret.
7. The replacement pod started at 02:20:49 UTC and logged its listener at 02:21:17.415 UTC, about 28.4 seconds later. It remained Ready with zero restarts through the final check.
8. The mounted VAST/NFS volume and finelog startup code were inspected. Warm metadata and small-read probes were much shorter than startup, while `Namespace::open` performed a standalone SQLite catalog upsert for every adopted segment. Batching those writes became the highest-priority follow-up experiment.

## User course corrections

- The user asked for a targeted server patch before the configuration change, so the safe probe and resource patch was applied and verified live first.
- The user broadened the provisioning change from RNO2A to global Kubernetes finelog defaults, so no cluster-specific override was added.
- The user requested an investigation rather than an immediate storage migration, so PVC options and falsifiable experiments were recorded without changing durability or storage products.
- The user added a dedicated Grafana dashboard requirement, so pod, resource, probe, PVC, fleet-health, and warning-event diagnostics were included in the same pull request.

## Root cause

Concurrent ingest and large compactions intermittently delayed the health endpoint long enough to fail readiness and liveness. The liveness policy then killed the process during both overload and slow store reopening, creating a restart loop. Memory exhaustion, PVC capacity, and node capacity were excluded.

The remaining 28–95 second startup time was not fully isolated. The leading hypothesis was hundreds of implicit SQLite catalog transactions over the shared VAST/NFS volume, based on the startup code path and live file count. PR #7540 added phase timers, batched those writes, and removed a duplicate footer scan so the next deployment could confirm or falsify it.

## Fix

The live RNO2A Deployment was patched to request 2 CPUs and 16 GiB of memory, allow 8 CPUs and 32 GiB, and defer liveness and readiness until a five-minute HTTP startup probe succeeded. PR #7540 applied those values as the global Kubernetes finelog defaults, kept per-cluster overrides available, batched startup catalog writes, removed a duplicate footer scan, selected network-safe persistent rollback journaling, added structured startup timings and the dedicated Grafana dashboard, and documented the diagnostic procedure.

## How OPS.md could have shortened this

The previous runbook did not explain that exit 137 could be a liveness kill, nor did it list cgroup OOM counters, peak memory, PVC use, and previous logs as the first decision points. The added readiness section now provides those commands and directs operators to compare ingest/compaction stalls with probe events before increasing memory or storage.

Startup phase timing had been absent from the server. PR #7540 added timers around catalog open, local segment adoption, footer reconciliation, catalog refresh, engine rehydration, and remote reconcile so subsequent incidents could separate storage latency from SQLite transaction overhead.

## Artifacts

- Pull request: https://github.com/marin-community/marin/pull/7540
- Session: https://loom.rjp.io/s/cy3crgzz
- Startup and PVC research: `.agents/projects/finelog-pvc-startup/research.md`
- Updated runbook: `lib/finelog/OPS.md`
- Grafana dashboard: `infra/grafana/dashboards/finelog.json`
- Final live pod: `finelog-cw-rno2a-5d99f7b464-x956q`, Ready with zero restarts
