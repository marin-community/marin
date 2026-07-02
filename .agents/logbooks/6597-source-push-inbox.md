---
topic: 6597-source-push-inbox
issue: TBD
description: Source-push inbox MGPU MoE permute_up plus W13 research follow-up to #6597.
author: dlwh
---

# 6597 Source-Push Inbox: Research Logbook

## Current TL;DR
- `stable`: Package-private source-push inbox path has a stable target-shape result at `8.353ms`, `218.19 TFLOP/s/rank`, `85.23 GB/s/rank`, with no drops on the 48-repeat consolidation run.
- The result beats the current serial hybrid estimate from ring prologue plus Pallas W13: `9.692ms` by `1.339ms`, about `13.8%` lower elapsed time.
- The current best branch for production-facing consolidation is `codex/6597-source-push-inbox-production`, PR #6841.
- This research branch tracks follow-up experiments and negative results so production work can stay clean.

## Scope
- Goal: Track source-push inbox MGPU MoE forward research that follows #6597, especially production relevance, decomposition, and next-step overlap/kernel design work.
- Primary metrics: target-shape elapsed time for `permute_up + W13`, W13 TFLOP/s/rank over full path time, send GB/s/rank, drops, metadata mismatches, and max absolute diff for correctness smokes.
- Constraints: Use H100 cluster `cw-us-east-02a`; do not restart or bounce Iris; babysit H100 jobs; preserve production PR cleanliness.
- Coordinating issue/PR: Parent issue #6597, production PR #6841, coordinating experiment issue TBD.
- Experiment ID prefix: `MOE-SP`
- Shared tags: `MOE-SP`, `issue-6597`, `source-push-inbox`, `mgpu-moe`

## Current Baseline
- Date: 2026-07-02
- Code refs:
  - Production branch: `codex/6597-source-push-inbox-production`
  - Research branch: `research/6597-source-push-inbox`
  - PR: https://github.com/marin-community/marin/pull/6841
- Target-shape baseline:
  - Job: `/dlwh/repro-source-push-consolidation-216-20260702-1555`
  - Commit: `34b86bf7b` for main consolidation run; integration-smoke fix landed at `24a200d03`
  - Winning profile: `hopper_source_push_inbox_rough_balanced_216`
  - Median over 48 repeats: `0.008353267879491406s` (`8.353ms`).
  - W13 TFLOP/s/rank: `218.1942535108326`
  - Send GB/s/rank: `85.23213027766897`
  - Dropped entries/rows: `0/0`
- Comparison baseline supplied by user:
  - Ring gather/dispatch prologue: `4.553ms`
  - Isolated Pallas W13: `5.139ms`
  - Serial hybrid estimate: `9.692ms`
  - Useful-row throughput over serial hybrid: `177 TFLOP/s/rank`
  - Capacity-padded throughput over serial hybrid: `222 TFLOP/s/rank`

## Hypothesis Queue

### Active
- `MOE-SP-001`: Source-push inbox remains the most production-relevant forward path if the production metadata path can feed it without changing the winning profile. Evidence: compact-routing smoke in `MOE-SP-003`. Next test: wire a production caller or a closer production-layout smoke at target scale.
- `MOE-SP-002`: The remaining tax is not comm-only; it is send/slot floor plus receiver/WGMMA scheduling. Evidence: `send_only=5.593ms`, `store_zero=5.650ms`, `full_wgmma=8.313ms`. Next test: isolate receiver wait/tail counters before changing kernel structure.
- `MOE-SP-003`: `send_pipeline_depth=2` may be a small win, but the observed delta is too small to rebaseline from one consolidation matrix. Evidence: `8.332ms` versus `8.353ms`. Next test: isolated 96-repeat A/B if this knob becomes decision-relevant.

### Blocked
- `MOE-SP-004`: True producer/consumer overlap may require a supported async remote-GMEM refill path or a structural split. Blocker: current Pallas/Mosaic support constraints and previous ring path showing blocking staging behavior. Resume when a viable async refill mechanism or warp-specialized design is proposed.

### Falsified / Dead End
- `MOE-SP-005`: Increasing `n_groups_per_job` broadly should improve throughput. Why stopped: `n_groups_per_job=5` regressed to `14.426ms`.
- `MOE-SP-006`: Reducing worker allocation is harmless. Why stopped: `num_send_sms=1` regressed to `10.601ms`; `num_sms=16` regressed to `13.912ms`.

### Promoted
- `MOE-SP-007`: Stable source-push inbox profile should be preserved as the named profile and production branch baseline. Decision: PR #6841, profile `hopper_source_push_inbox_rough_balanced_216`.

## Decision Log
- 2026-07-02: Preserve a single public stable source-push profile and move failed/experimental modes out of ordinary sweep choices. Evidence: PR #6841.
- 2026-07-02: Treat source-push inbox as faster than the serial ring+Pallas hybrid estimate on comparable target shape. Evidence: `8.353ms` median versus `9.692ms` serial estimate.
- 2026-07-02: Keep `send_pipeline_depth=1` as the named stable profile despite `depth=2` being a tiny win in one matrix, because it needs isolated repeat validation before changing the proven profile.

## Negative Results Index
- Ring destination-pull path is correct but slower and not a performance path without true async refill or producer/consumer split. Prior work is recorded in #6597 and the production branch history.
- `n_groups_per_job=5`, `num_send_sms=1`, and `num_sms=16` are clear regressions in the consolidation matrix.
- `direct_self_compute=true` was a small regression in this profile: `8.466ms` versus `8.353ms`.

## Entry Log

### 2026-07-02 09:08 - MOE-SP-001 consolidation issue setup
- Hypothesis: The stable source-push inbox result is strong enough to deserve its own experiment issue and research branch linked from #6597 and PR #6841.
- Commit Hash: `24a200d03` on `codex/6597-source-push-inbox-production`; research branch starts from this commit.
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name repro-source-push-consolidation-216-20260702-1555 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 3600s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_inbox_consolidation.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --suite all \
    --jsonl scratch/source_push_consolidation_216_20260702_1555.jsonl
  ```
- Config:
  - Target shape: `ep_size=8`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`
  - Stable profile: `hopper_source_push_inbox_rough_balanced_216`
  - Main knobs: `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=queue`, `n_groups_per_job=2`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=1`
- Result:
  - Winning: `8.353ms`, `218.19 TFLOP/s/rank`, `85.23 GB/s/rank`, drops `0/0`
  - `metadata_mode=remote_slot`: `8.499ms`, `214.46 TFLOP/s/rank`
  - `output_mode=debug`: `8.352ms`, `218.23 TFLOP/s/rank`
  - `hidden_output_mode=full`: `8.359ms`, `218.06 TFLOP/s/rank`
  - `n_groups_per_job=1`: `8.502ms`, `214.39 TFLOP/s/rank`
  - `n_groups_per_job=5`: `14.426ms`, `126.34 TFLOP/s/rank`
  - `send_pipeline_depth=2`: `8.332ms`, `218.75 TFLOP/s/rank`
  - `num_send_sms=1`: `10.601ms`, `171.92 TFLOP/s/rank`
  - `num_sms=16`: `13.912ms`, `131.02 TFLOP/s/rank`
  - `direct_self_compute=true`: `8.466ms`, `215.28 TFLOP/s/rank`
- Interpretation: The stable path is faster than the serial ring+Pallas hybrid estimate and should be tracked as the current proven source-push baseline. The decomposition shows a `5.6ms` send/store-zero floor and about `2.7ms` additional full-WGMMA cost.
- Next action: Open the coordinating experiment issue and link it to this logbook, parent issue #6597, PR #6841, and the research branch.

### 2026-07-02 09:08 - MOE-SP-002 compact-routing integration smoke
- Hypothesis: The source-push inbox kernel can be fed from production-like compact routing metadata and source-pack-like payload layout without changing the winning profile.
- Commit Hash: `24a200d03`
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name repro-source-push-integration-smoke-20260702-1603 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 1200s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_inbox_consolidation.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --suite integration \
    --jsonl scratch/source_push_integration_smoke_20260702_1603.jsonl
  ```
- Config:
  - Small legal WGMMA smoke shape: `tokens_per_rank=256`, `hidden_dim=128`, `intermediate_dim=128`, `experts_per_rank=4`
  - Input mode: compact routing metadata and source-pack-like packed payload
- Result:
  - Job: `/dlwh/repro-source-push-integration-smoke-20260702-1603`
  - Median time: `1.623ms` on small smoke shape
  - Metadata mismatches: `0`
  - Dropped entries/rows: `0/0`
  - Max abs diff: `0.007767200469970703`
- Interpretation: This is an integration correctness smoke, not a target-shape performance number. It shows the source-push inbox path can consume production-like compact routing inputs.
- Next action: Use this as the bridge from package-private harness to production caller integration work.
