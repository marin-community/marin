---
topic: Cleaned Datakit SFT production run
issue: https://github.com/marin-community/marin/issues/8954
description: Production SFT of the 67B/2B-active long-context checkpoint on cleaned chat data
author: William Held
---

# Cleaned Datakit SFT: Task Logbook

## Scope

- Goal: Train for 2,000 steps on 80% cleaned Datakit SFT and 20% pretraining replay.
- Primary metrics: training loss, throughput, numerical stability, and checkpoint completion.
- Constraints: 262,144-token context; non-preemptible v4-2048 in us-central2-b; whole-conversation greedy packing.
- Coordinating issue: https://github.com/marin-community/marin/issues/8954

## Baseline

- Date: 2026-09-07
- Code refs: `20c022c51a6a6817130f6c422d5da645b366347f`
- Baseline numbers: initialize from step 157,000; source checkpoint size 1,079,160,353,673 bytes; 2,000 SFT steps.

## Entry Log

### 2026-09-07 14:10 - Launch preflight

- Hypothesis: The cleaned chat mixture can continue the long-context model without malformed turn, reasoning, or tool-call structure entering training.
- Commit Hash: pending retention and logbook commit
- Command: `uv run python experiments/june_tpu_67b_a2b/moe/sft_datakit_chat_mix.py`
- Config: v4-2048; batch 256; sequence length 262,144; 80% SFT; 20% replay; 2,000 steps; MuonH at 5e-5; 3% warmup; cosine decay; weights-only initialization from step 157,000.
- Result: Preflight confirmed the source checkpoint metadata records step 157,000 and the durable output root is unused. The source checkpoint is 1.005 TiB. Periodic resume checkpoints use the 14-day regional temporary bucket with one retained; only the forced final checkpoint is durable.
- Interpretation: Source lineage, output identity, data region, and checkpoint retention are suitable for submission.
- Next action: Commit and push the launch record, submit once, then verify scheduling, W&B identity, initialization, and early training health.

### 2026-09-07 14:22 - Coordinator stopped before TPU dispatch

- Hypothesis: The pushed production recipe will dispatch the requested training child after pruning completed data dependencies.
- Commit Hash: `77308d1a9fc61afd226f5cf98a6dabc14f1be39f`
- Job: `/held/sft-datakit-20260907-prod`
- Result: The coordinator pruned all completed tokenized chat dependencies and resolved the intended 2,000-step training configuration and step-157000 checkpoint. Dispatch then failed because `WANDB_API_KEY` was absent from the coordinator environment. No TPU child was created and training did not start.
- Interpretation: The data graph and training configuration reached dispatch intact. The failure is isolated to launch-time secret forwarding and did not consume TPU capacity or write model state.
- Next action: Supply `WANDB_API_KEY` through Iris's standard `-e` environment forwarding, use a fresh coordinator job name, and repeat the initial health gate.

### 2026-09-07 16:37 - Prepare local recipe for 2,000 steps

- Hypothesis: The cleaned Datakit mixture can train the long-context checkpoint with intact conversation, reasoning, and tool boundaries.
- Source code: `bc6b4710e5a739ba2b4839913368310ed60efe8c`, based on local revision `30462366bea3cb7d861fdc577dd3573023464a83`. The user selected this local recipe, then requested 2,000 steps and only the final checkpoint permanent. This entry restores the earlier logbook from remote revision `7e5d2f5bd4c6ac04c1203946c969530030f6ccb1` and preserves its history.
- DRI: William Held. Monitoring owner: this session, with 15-minute health checks after startup.
- Planned coordinator: `/held/sft-datakit-20260907-prod-2000`. Cluster: `marin`. Coordinator resources: 1 CPU, 2 GB RAM, 5 GB disk, production priority, us-central2, no timeout.
- Training: non-preemptible v4-2048 in us-central2-b, 256 workers, context parallelism 4, expert parallelism 1, batch 256, sequence length 262,144, 80% SFT and 20% replay. MuonH at 5e-5, 3% warmup, cosine decay, no gradient clipping. Final step: 2,000.
- Durable output: `gs://marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2_7599f720`. This is the user-selected recipe's resolved output; no objects existed at preflight.
- Checkpoints: final permanent checkpoint at `<output>/checkpoints/step-2000/`; temporary saves every 30 minutes, retaining one, under `gs://marin-us-central2/tmp/ttl=14d/checkpoints-temp/marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2_7599f720/checkpoints`. No permanent interval saves. Recovery checkpoint size is estimated at 1.005 TiB from the previously measured optimizer-inclusive source; steady-state temporary retention is about 1.005 TiB, with old and new checkpoints overlapping during saves.
- Initialization: weights only from `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4-102e3c/checkpoints/step-157000/`. Preflight read `metadata.json` and confirmed numeric step 157,000 and `is_temporary=false`.
- Data: normalized/rendered/tokenized SFT artifacts under the regional Datakit paths; replay from `gs://marin-us-central2/datakit/store/june-67b-a2b-length64k/2026.08.24`. The recipe checks storage region and validates all packed SFT components before training.
- W&B: enabled, project `marin_moe_sft`, run ID and name `moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2`, resume `allow`. Forward the supplied key through the coordinator environment; never include it in the run record.
- Runtime: Iris bundles the local workspace. Record the actual source SHA and bundle after submission. No Ray spill or Ray session upload is configured by this Grug recipe; profiler uses its existing defaults and operational logs go through Iris.
- Validation: repository lint passed on the recipe. Capturing the resolved launcher configuration without dispatch confirmed 2,000 steps, no permanent intervals, a 30-minute temporary interval, and one retained temporary checkpoint.
- Next action: publish this source and record on a separate run branch, submit once with the command below, then verify dispatch, W&B identity, checkpoint initialization, and training progress.

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name sft-datakit-20260907-prod-2000 --priority production \
  --cpu 1 --memory 2GB --disk 5GB --region us-central2 \
  -e UV_LOCK_TIMEOUT 900 -e MARIN_PREFIX gs://marin-us-central2 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- uv run python experiments/june_tpu_67b_a2b/moe/sft_datakit_chat_mix.py
```

### 2026-09-07 16:55 - Resume LCR optimizer with WSD

This entry supersedes the unsubmitted 16:37 plan. The user requested carrying the long-context optimizer state, retaining absolute step 157,000, and replacing cosine decay with warmup-stable-decay. The WSD split below is the stated working assumption; no alternative split was supplied.

- Source code: `3bab24750e1261a0958f5391b30bc12db6770e26`. The submitted workspace will add only this logbook and its input manifest. The source is being published under `held/sft-datakit-8954-wsd-2000` to preserve the existing remote branch.
- DRI and monitoring contact: William Held. This active assistant session owns launch verification and subsequent 15-minute checks. Success at startup requires the intended TPU child, matching W&B identity, restored step 157,000, advancing steps, finite losses, and LR values following the new schedule. Numerical failures, missing state, or an unexpected lineage stop automatic recovery and require the DRI's decision.
- Planned job: `/held/sft-datakit-20260907-prod-2000` on cluster `marin`, using the command in the preceding entry. The prior coordinator is terminal failed; the new job is not yet submitted. The supplied W&B key was authenticated successfully and will be forwarded with `-e WANDB_API_KEY`.
- Training contract: 2,000 additional updates, from step 157,000 through final state 159,000; non-preemptible v4-2048 in us-central2-b; 256 workers; batch 256; sequence length 262,144; context parallelism 4; expert parallelism 1; 80% SFT and 20% replay. The model, optimizer buffers, and counters load from the step-157000 checkpoint listed above.
- LR schedule for both MuonH and Adam: linear warmup from 0 to 5e-5 over steps 157,000–157,060; stable at 5e-5 through step 158,800; linear decay to 5e-6 at step 159,000. The new schedule evaluates the preserved optimizer counter minus 157,000. Muon momentum and Adam moment/bias-correction state remain intact.
- W&B: entity `held`, project `marin_moe_sft`, ID/name `moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2`, resume `allow`. No existing run with this ID was returned by the preflight query.
- Durable output: `gs://marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2_c8284c64`. Its only planned permanent checkpoint is `checkpoints/step-159000/`.
- Recovery storage: `gs://marin-us-central2/tmp/ttl=14d/checkpoints-temp/marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_v2_c8284c64/checkpoints`; save every 30 minutes and retain one complete temporary checkpoint. Both the durable and temporary roots were absent at preflight. These preserve the user-selected recipe's storage policy.
- Size estimate: all 72 expected model/optimizer arrays match the source checkpoint's shape and dtype. Logical array payload is 539,331,833,888 bytes (about 502 GiB), including optimizer state. Actual stored bytes may differ; use the earlier measured 1.005 TiB source checkpoint as a conservative storage allowance. Old and new temporary checkpoints overlap while a replacement is written.
- Data identity: [resolved input manifest](sft-datakit-2026-09-07-inputs.json) lists all 159 tokenized SFT artifacts and the output/checkpoint roots. Replay remains the 2026.08.24 us-central2 store listed above. Packed-source validation is a required runtime dependency and has not yet been executed by this launch.
- Runtime: Iris uploads a workspace bundle; record its ID and actual checkout SHA after submission. Profiling is disabled. This Grug recipe does not use Ray, so Ray spill and Ray session uploads do not apply. Logs are available through Iris.
- Completed checks: source metadata and scalar optimizer counters all report step 157,000; all 72 array shapes/dtypes match; resolved launcher uses full-state initialization, final step 159,000, and the specified checkpoint policy. Repository-wide lint and type checking passed. The checkpoint, Grug contract, and recipe tests passed (38 passed, 1 skipped), including numerical WSD updates and recovery with preserved optimizer buffers.
- Test limitation: the affected-test runner selected 69 paths but could not install `torchcodec==0.10.0` on Linux aarch64. The focused tests ran successfully in the existing environment. The combined run exposed duplicate optimizer registration names; the June TPU names are now distinct.
- Next action: publish the record, submit once, then verify the startup contract. Do not treat successful submission as successful training.

### 2026-09-07 16:58 - Coordinator submitted; preprocessing active

- Submitted at 16:55:39 PDT: `/held/sft-datakit-20260907-prod-2000`, using the recorded command with the W&B key forwarded. Source SHA: `360142508301b1e54e147762f3e8692ce781b1d3`; clean working tree; workspace bundle 10.7 MB, ID `a253e1113ae2a6db3ffc19832e09405a06c9eb1f76a8cffb8bcebe8e386db6fb`.
- Result at 16:58: coordinator running, zero failures and preemptions. Twelve Zephyr transform/normalization coordinators were running and worker jobs were being scheduled. Raw inputs were reused, but this revision's processed-chat artifacts were not all cached. No TPU training child had been submitted.
- Interpretation: launch environment setup passed; the data graph must finish preprocessing and packing validation before TPU dispatch. W&B training state and model initialization remain unverified.
- Dashboard: https://iris.oa.dev/#/job/%2Fheld%2Fsft-datakit-20260907-prod-2000
- Next action: monitor preprocessing, then verify TPU dispatch and the WSD/full-state startup contract. Monitoring state is in the session's ignored `scratch/20260907_sft_datakit_monitoring_state.json`.
