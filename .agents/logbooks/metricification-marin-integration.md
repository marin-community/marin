---
topic: metricification-marin-integration
description: Marin M1-M5 producer, query, dashboard, and alert integration
author: Marin metricification coordinator
---

# Metricification Marin Integration: Task Logbook

## Scope

- Goal: Land the single remaining Marin integration/dashboard review unit.
- Primary metrics: Post-exit identity joins, progress/work rates, phase
  fractions, bounded outliers, failure timelines, and telemetry completeness.
- Constraints: Simple v1 direct Finelog transport; one draft Marin PR; no vLLM
  modification; approved vLLM and Rigging feeders stay in this PR; probes are
  explicit opt-in and observe-only.
- Coordinating issue/PR: #7804, #7681, and #7795; no branch PR yet.

## Current TL;DR

- `origin/main` contains the simple telemetry foundation from #7839 plus early
  Levanter, Zephyr, centralized vLLM, and Grafana producers.
- The branch now supplies canonical Iris identity and centralized vLLM exporter
  health. Broader producers still emit legacy attributes, lack Marin RL
  operational telemetry, and expose only partial Levanter/Zephyr phase and
  outlier evidence.
- The benchmark commit recommends logical namespaces, lazy binding, and a run
  catalog, while deferring physical partitioning and generic rollups. The v1
  reset contract keeps this PR on `telemetry_v1`.
- The approved vLLM feeder consists of three ordered commits ending at
  `45d9b2bd3b67`. The approved Rigging feeder consists of three ordered commits
  ending at `805dd5e47483`.
- User scope correction limits this PR to those feeders, the benchmark, minimum
  canonical identity/exporter health, and the vLLM end-to-end query. M2–M4 are
  follow-up workitems.

## Baseline

- Date: 2026-08-01
- Code refs: `900e245ee` (`origin/main`), `729cc2d6a` (branch benchmark)
- Baseline numbers: Benchmark fixtures cover up to 1,000,000 rows and 111 files;
  the incident anchor is 988,423,984 rows. Runtime defaults are 10,000 records,
  16 MiB, and a 5-second maximum shutdown budget.

## Entry Log

### 2026-08-01 00:52 UTC - Contract and code audit

- Hypothesis: Existing foundation code provides reusable producer and query
  seams, so M1-M4 can be additive without changing transport semantics.
- Commit Hash: `729cc2d6a334d7e1a848797a921d8c6b00734e3a`
- Command: `weaver artifact show metricification-{plan,design,foundation-contract,integration-workplan}`; repository `rg`; `gh issue view 7681/7804/7795`; Echo federated searches for canonical identity, RL, Levanter, and Zephyr telemetry.
- Config: Current checkout, no feeder commits, no live cluster access.
- Result: Confirmed the direct exporter/Finelog endpoint, Levanter telemetry
  tracker, Zephyr coordinator snapshots, centralized Marin vLLM adapter, and
  initial Grafana stall rules. Echo returned no additional indexed result for
  the searched metricification terms. #7681 identifies the unmerged host-side
  rank reporter as prior work, but its broadcast implementation does not
  gather all ranks and should not be copied as-is.
- Interpretation: Keep the foundation transport fixed. Centralize resource and
  signal vocabulary, then instrument existing timing/counter boundaries. Use a
  real all-gather for rank summaries and retain only bounded top-k evidence.
- Next action: Implement M1 typed identity/common families and behavior tests.

### 2026-08-01 01:18 UTC - Review slice integrated

- Hypothesis: Canonical Iris identity and exporter self-health are sufficient
  to make the two approved feeders operable without broad producer fan-out.
- Commit Hash: `74e02e306` with M1 integration at `ecba5432d`, vLLM feeder at
  `939239589`, and benchmark at `729cc2d6a`.
- Command: Ordered cherry-picks for both feeder series; focused Rigging, Iris,
  Marin vLLM, and Grafana pytest commands; changed-file lint/type gate.
- Config: Rigging probes remain explicit opt-in with DCGM/Ray precedence and no
  automated recovery. The vLLM endpoint requires `job_id`, `root_run_uid`, or
  `execution_uid` plus raw lower/upper time bounds.
- Result: Rigging 519 passed; vLLM/Grafana 64 passed; Iris identity 6 passed;
  Marin vLLM forwarding 2 passed. Iris stamps root/execution/job/task/attempt/
  worker/process identity and `serving_job_id`; the vLLM polling cadence emits
  exporter queue, loss, attempt/failure/retry/rejection, oldest-record age, and
  last-success freshness snapshots.
- Overhead: Unconfigured exporter-health calls measured 2.11 µs/call (100,000
  calls, best of five). A configured vLLM poll emits at most nine fixed-name
  health rows every 15 seconds (0.6 rows/s); probe defaults are explicit
  ten-minute samples with five-second NVIDIA and eight-second NCCL deadlines.
- Interpretation: A post-exit operator can select one served job/root/execution
  and answer whether token throughput, request queue/KV saturation, latency,
  outcomes, or the worst producer replica explains a slow or silent serve.
- Next action: Run the pre-PR review gate, open the single draft PR, and track
  the deferred M2 Levanter, M3 RL, M4 Zephyr, and remaining M5 work separately.
