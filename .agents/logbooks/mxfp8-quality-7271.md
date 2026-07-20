---
topic: mxfp8-quality-7271
issue: https://github.com/marin-community/marin/issues/7271
description: Fixed-token BF16 versus hybrid MXFP8 quality gate on GB200
author: Matt Wittmann
---

# MXFP8 Quality Gate: Task Logbook

## Current TL;DR

- The first paired smoke was stopped before training: simulated epoching reduced rare finite components to zero sequences at the 20-step budget.
- The regression fix disables simulated epoching only for shortened shape smokes; the full fixed-token run retains the intended simulated schedule. Seventeen focused tests pass.
- The primary gate is a matched-token d2560/L26/E128/top-4 run: 31,474 steps, batch 512, sequence length 4096, and 66,005,762,048 tokens per arm.
- A 20-step, full-topology smoke is next. No quality or performance result is claimed yet.

## Scope

- Goal: determine whether the production hybrid MXFP8 recipe from #7282 preserves training quality while reducing time to a matched loss.
- Primary metrics: paired train-loss delta, final/cooldown-tail loss delta, Paloma macro/domain loss, uncheatable-eval loss, tokens/s, MFU, and time to matched loss.
- Constraints: same seed, data order, schedule, topology, and optimizer; only the FP8 model config may differ. Use region-local `s3://marin-us-east-02a/marin` data and checkpoints.
- Coordinating issue: https://github.com/marin-community/marin/issues/7271

## Baseline

- Date: 2026-07-19
- Code refs: #7282 implementation at `c3cb334f8`; quality launcher at `485e23f7f`.
- Baseline numbers: the corrected #7282 64-GPU production-shape smoke measured 299,894 tok/s BF16 versus 392,287 tok/s MXFP8/XLA, or 1.308x. That was a 50-step throughput result, not a long-run quality result.

## Entry Log

### 2026-07-19 17:23 - MXFP8Q-001 preflight ready

- Hypothesis: the exact d2560 quality cell compiles, trains, evaluates, checkpoints, and resumes on 8xGB200x4 per arm with finite paired losses and a stable MXFP8 throughput advantage.
- Commit Hash: `485e23f7f4d89d1d2f218c777766497d6a8d7ae4`
- Command: submit two CPU coordinators through Iris, each running `python -m experiments.grug.moe.launch_mxfp8_quality` with `MXFP8_QUALITY_PAIR_ID=MXFP8Q-001-smoke`, `MXFP8_QUALITY_STEPS=20`, and arm `bf16` or `mxfp8`. Pass `WANDB_API_KEY` from agenix and `MARIN_PREFIX=s3://marin-us-east-02a/marin` to each coordinator.
- Config: d2560/L26/E128/top-4, f1280, shared-f2560, seq4096, batch512, seed0; 8 replicas x 4 GB200, mesh replica2/data2/expert8; full MuonH/Newton-Schulz; ring MoE, scanned layers, recompute-all; MXFP8 uses fused grouped experts, per-tensor FP8 dense GEMMs, BF16 EP wire, and the XLA producer. Paloma and uncheatable evaluation run at the forced final callback. Checkpoints are hourly plus forced final on S3.
- Result: pending submission.
- Interpretation: the launcher passed 16 focused tests and a final read-only review with no findings. Four CPU/`recipe="auto"` variant-contract failures are inherited unchanged from #7282.
- Next action: submit and babysit both 20-step arms; require finite loss, final evaluation, S3 checkpoint, and exact-ID resume before the full 31,474-step pair.

### 2026-07-19 18:02 - MXFP8Q-001 smoke stopped before training

- Hypothesis: shortening `experiment_budget` to 20 steps while retaining a 10.37T-token `target_budget` truncates at least one rare finite mixture component to zero sequences.
- Commit Hash: fix pending commit; failed launch used `c0be963da` with launcher snapshot `485e23f7f`.
- Command: paired Iris coordinators `/mwittmann/mxfp8q-001-smoke-bf16-coord` and `/mwittmann/mxfp8q-001-smoke-mxfp8-coord`; each launched an 8xGB200x4 child gang.
- Config: both arms used the same 20-step shape-smoke data config with simulated epoching enabled.
- Result: both arms failed before step 0 and before W&B initialization with `MixtureDataset in RESTART_STRATEGY encountered an empty finite dataset (async_len() returned 0)`. The coordinators and child gangs were stopped while retrying.
- Interpretation: `LmDataConfig` slices every component to `int(true_length * experiment_budget / target_budget)`. At 20 steps the ratio is about `4.04e-6`, so sufficiently rare components become empty; restart sampling rejects empty finite datasets. This is shared smoke infrastructure, not a BF16/MXFP8 quality signal.
- Next action: keep simulated epoching for the full 31,474-step comparison, disable it for shortened shape smokes as in the existing GB200 scale-smoke pattern, add regression coverage for both modes, and relaunch the exact pair.
