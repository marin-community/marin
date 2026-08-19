---
topic: moe-tasktrove-v326
issue: https://github.com/marin-community/marin/issues/8449
description: Qwen3.6 and Gemma 4 TaskTrove v3.26 evaluation campaign
author: benjaminfeuer
---

# MoE TaskTrove v3.26: Task Logbook

## Current TL;DR

The 564-run campaign is in integration. Harbor PR #83 supplies the Pi hosted-vLLM endpoint path. No evaluation job has been submitted. Four one-task smokes must validate both models with Terminus-2 and Pi before the eight-job queue opens.

## Scope

- Goal: run 141 TaskTrove v3.26 datasets with two models and two Harbor harnesses, materializing 300 tasks per cell.
- Primary metrics: pass@1 over scoreable trials, scoreable trial count, infrastructure exception count, decode throughput, and agent-timeout incidence.
- Constraints: 32,768-token context; 16,384-token output allowance; prior Qwen3-Coder rollout policy except measured MoE serving/concurrency changes; at most eight active Iris jobs; at least 271 scoreable trials per completed cell.
- DRI: Benjamin Feuer.
- Coordinating issue: https://github.com/marin-community/marin/issues/8449
- Campaign tracker: `/Users/benjaminfeuer/Documents/experiments/active/moe-data-quality/TRACKER.md`
- W&B: not used for Harbor evaluation jobs.
- Checkpoints: not applicable.

## Baseline

- Date: 2026-08-19
- Marin code: `be0f70b169` (`agent/moe-tasktrove-v326`, based on `origin/main`)
- OpenThoughts-Agent code: `c5cccf33d79511172ac910f36385007a123d5aa1` (`penfever/working`)
- Harbor code: `41f4320c0471ea3362a6d3160df8b6c75f0126f7` (`main`, includes PR #83)
- TaskTrove: `open-thoughts/TaskTrove@6ac7c547ee2a8108836887e6530eb7dddf02dd9a` (latest v3.26 revision at integration time)
- Qwen weights observed at HF revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Gemma weights observed at HF revision: `24548b62aa021d562695c04aaf7758a1ea47990b`
- Prior campaign: `/Users/benjaminfeuer/Documents/experiments/active/qwen3-coder-data-quality`
- Historical Qwen3.6 serving signal: decode occupied 99.5–99.8% of request time and AgentTimeoutError affected 40–76% of trials on a B200 profile; see https://echo.oa.dev/wiki/50.

## Decision Log

- `MDQ-001`: use the latest v3.26 TaskTrove commit, including the duplicate `verifier.env` repairs recorded at https://echo.oa.dev/wiki/182.
- `MDQ-002`: keep the prior 16,384-token output allowance. A smaller allowance confounded the historical Qwen3.6 comparison.
- `MDQ-003`: run both harnesses through CoreWeave controller ingress so Pi reaches the co-located vLLM endpoint.
- `MDQ-004`: require decode-rate and timeout evidence from smokes before fixing per-harness concurrency.
- `MDQ-005`: keep no more than eight campaign jobs in Iris states PENDING, BUILDING, or RUNNING.

## Entry Log

### 2026-08-19 13:49 UTC - Bootstrap campaign record

- Hypothesis: Qwen3.6-35B-A3B and Gemma-4-26B-A4B can complete the prior TaskTrove rollout contract on one H100x8 Iris worker per cell; TP/DP and Harbor concurrency may need model-specific adjustment.
- Commit Hash: pending initial logbook commit.
- Command: no launch command executed.
- Config: 141 datasets × 2 models × 2 harnesses; 300 tasks; context 32,768; output 16,384; eight active jobs.
- Result: Harbor PR #83 merged and local Harbor advanced to `41f4320c0471`. TaskTrove advanced from the tracker's older v3.26 pin `a9c9bd35cb4f` to `6ac7c547ee2a`; the tracker and manifest must be regenerated before launch.
- Interpretation: the prior Harbor blocker is resolved. Model registration, durable campaign artifacts, and four smokes remain.
- Next action: register Gemma serving, create the harness configs and v3.26 manifest, then dry-run and submit one-task smokes.
