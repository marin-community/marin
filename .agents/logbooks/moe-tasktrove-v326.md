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

### 2026-08-19 14:08 UTC - Pin launch configuration and build manifest

- Hypothesis: Qwen TP2×DP4 and Gemma TP1×DP8 will use all eight H100s while preserving the prior 32k/16k rollout contract; four one-task smokes will determine whether either topology or 32-way Harbor concurrency needs adjustment.
- Commit Hash: OpenThoughts-Agent `6b57e1a31c` on pushed branch `penfever/working`; Harbor `41f4320c0471`.
- Command: dry-ran `artifacts/launch_cell.py <qwen36|gemma4> <terminus-2|pi> 1 --smoke --dry-run` for all four combinations. Planned mirror command: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m scripts.iris.launch_mirror hf-to-gcs --cluster marin --tpu v6e-4 --priority interactive --no-wait --output-mode gcs --gcs-output-dir gs://marin-us-central2/tmp/ttl=14d/ot-agent/model-mirrors/benjaminfeuer --secrets-env <filtered-secrets> --repo Qwen/Qwen3.6-35B-A3B --repo google/gemma-4-26B-A4B --gcs-prefix gs://marin-models-us/ot-agent/models --job_name mdq-tasktrove-v326-model-mirrors`.
- Config: Terminus-2 and Pi share 32 concurrent trials, one attempt, 7,200-second agent timeout, 14,400-second verifier timeout, Daytona 2 CPU/4 GiB/4 GiB, and 32,768 input/16,384 output limits. Pi is pinned to 0.73.1 with model-specific `qwen-chat-template` or `chat-template` thinking format. Qwen uses TP2×DP4; Gemma uses TP1×DP8. Both retain the prior Qwen3-Coder server sampling defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05).
- Result: all four dry runs resolved controller ingress and exact pinned TaskTrove selectors. The manifest contains 141 datasets, 564 grid cells, and 169,200 attempts; seven datasets require deterministic round-robin expansion. Both models are absent from `gs://marin-models-us/ot-agent/models`.
- Interpretation: no further Harbor changes are required. Model mirroring is the only launch prerequisite remaining before the four smoke jobs.
- Next action: mirror both model repositories once, verify cache hits, then submit and monitor all four smoke jobs.

### 2026-08-19 14:07 UTC - Advance OTA Harbor runtime pin

- Hypothesis: the merged Harbor source alone is insufficient unless OTA's frozen worker environment and baked images resolve the same commit.
- Commit Hash: OpenThoughts-Agent `30aec355` on pushed branch `penfever/working`; Harbor `41f4320c0471`.
- Command: `uv lock --upgrade-package harbor`; `uv run pytest -q tests/unit/agents/installed/test_pi.py` in the Harbor checkout.
- Config: `uv.lock` and every `docker/Dockerfile.*` `HARBOR_COMMIT` pin now resolve `41f4320c0471ea3362a6d3160df8b6c75f0126f7`.
- Result: Harbor's Pi unit suite passed 22/22. The mirror worker had shown OTA's previous lock resolving pre-PR commit `772e20f7`; the new pushed OTA revision removes that drift before any eval launch.
- Interpretation: smoke workers built from `30aec355` will contain the hosted-vLLM Pi endpoint implementation. The already-running mirror job is unaffected because it does not execute Harbor.
- Next action: wait for `/benjaminfeuer/mdq-tasktrove-v326-model-mirrors`, confirm both regional cache hits, then launch the four smokes from OTA `30aec355`.
