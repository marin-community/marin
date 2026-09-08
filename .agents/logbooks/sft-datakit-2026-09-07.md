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


### 2026-09-07 - Restore the production base and relaunch with optimizer state

- Source: `c32cba41c21ab6c9edbeb892f660d041708aa689`, directly descended from the production revision pinned in issue #8954, `77308d1a9fc61afd226f5cf98a6dabc14f1be39f`. The launch-record commit adds documentation only; its SHA and submitted bundle will be recorded after submission.
- Prior attempt: `/held/sft-datakit-20260907-prod-2000` used stale data-processing code from `360142508301b1e54e147762f3e8692ce781b1d3`. It rebuilt data and encountered malformed reasoning rows. It was cancelled before TPU dispatch. Existing data and source checkpoints were preserved.
- Preflight: all 158 resolved SFT cache steps report SUCCESS, and all 158 training shard ledgers are finished. Exact paths are in `sft-datakit-2026-09-07-inputs.json`. The full library tree and dependency files match the production base. Lint and type checks passed; 38 tests passed and one skipped, with one checkpoint-write timeout passing on isolated rerun.
- Training: non-preemptible v4-2048 in us-central2-b; batch 256; context 262,144; context parallelism 4; expert parallelism 1; 80% SFT and 20% replay. Replay remains at `gs://marin-us-central2/datakit/store/june-67b-a2b-length64k/2026.08.24`.
- Initialization: preserve the model, optimizer moments, momentum, and counters from step 157,000 at `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4-102e3c/checkpoints/step-157000/`. Metadata confirms step 157,000. Train 2,000 updates through absolute step 159,000 with 60 warmup, 1,740 stable, and 200 linear-decay steps evaluated at the preserved optimizer counter minus 157,000; peak LR 5e-5 and floor 5e-6 for both Muon and Adam schedules.
- Output: `gs://marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07_c0b2c01b`. This root is unused. Permanent checkpoints are under its `checkpoints/` directory; save only the final checkpoint permanently.
- Recovery: one temporary checkpoint every 30 minutes under `gs://marin-us-central2/tmp/ttl=14d/checkpoints-temp/marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07_c0b2c01b/checkpoints`, with a 14-day lifecycle. Source physical size is 1,079,160,353,673 bytes; assuming comparable serialization and state size, budget approximately 1.005 TiB each for the retained recovery checkpoint and final checkpoint, with transient overlap while committing a replacement.
- Tracker: entity `held`, project `marin_moe_sft`, ID/name `moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07`, resume `allow`. No existing run has that ID. The API key is forwarded explicitly and omitted from this record.
- Coordinator: `/held/sft-datakit-20260907-prod-wsd`; cluster `marin`; 1 CPU, 2 GB memory, 5 GB disk in us-central2; production priority; no timeout. DRI: William Held. This assistant session owns startup checks and subsequent 15-minute monitoring. Iris/Finelog retain runtime logs; Zephyr scratch, if needed, uses the regional lifecycle-managed temporary prefix. Ray spill is not applicable to this JAX/Fray run.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name sft-datakit-20260907-prod-wsd --priority production --cpu 1 --memory 2GB --disk 5GB --region us-central2 -e UV_LOCK_TIMEOUT 900 -e MARIN_PREFIX gs://marin-us-central2 -e WANDB_API_KEY "$WANDB_API_KEY" -- uv run python experiments/june_tpu_67b_a2b/moe/sft_datakit_chat_mix.py`
- Next action: submit once, confirm the cached data dependencies are skipped, and verify TPU dispatch and full-state initialization.


### 2026-09-07 17:28 PDT - Cached inputs reused and TPU training dispatched

- Submitted `/held/sft-datakit-20260907-prod-wsd` at 17:26:18 PDT from clean, pushed source `d033b52537755baf458dfbfafaae6e84f7aa70c0`; bundle `a843178ff0e84c71a03d0adf41fccc74fd298431fee39b0d861a6af5d9db2559` (11.1 MB). The coordinator received the W&B credential.
- The coordinator reused completed dependencies and dispatched `grug-train-moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07` at 17:27:23 PDT. The job tree contains only the coordinator and training child, with no preprocessing jobs. Training is pending matching TPU workers; full-state initialization and training metrics are not yet observed.
- [Iris job](https://iris.oa.dev/#/job/%2Fheld%2Fsft-datakit-20260907-prod-wsd) · [Stale-checkout incident record](https://marina.oa.dev/echo/wiki/359).
- Next action: monitor allocation, then checkpoint restoration and the first training updates.


### 2026-09-07 19:49 PDT - Use continuous-token replay

- The user requested the default `TokenSeqDataset` path for replay and retained greedy whole-conversation SFT packing. Remove replay's `pack=True` override; the SFT packer and all 158 SFT cache identities remain unchanged. A regression using two documents verifies that replay chunks continue across document boundaries. All four recipe tests pass.
- The previous coordinator `/held/sft-datakit-20260907-prod-wsd` was cancelled at 19:48 PDT before any observed training updates. Sampled workers were building a greedy index for a 587,952,610-document replay component. [Startup investigation](https://marina.oa.dev/echo/wiki/360).
- New coordinator: `/held/sft-datakit-20260907-prod-wsd-continuous`. Source is the commit containing this entry; the actual SHA and bundle are recorded after submission. This remains descended from the complete production base `77308d1a9f`.
- New output: `gs://marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07_replay_continuous_0ff1e29f`; verified unused. Temporary checkpoints: `gs://marin-us-central2/tmp/ttl=14d/checkpoints-temp/marin-us-central2/grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07_replay_continuous_0ff1e29f/checkpoints`. Save a recovery checkpoint every 30 minutes, retaining only the latest, and only the final step-159000 checkpoint permanently; source-size estimate remains 1.005 TiB per checkpoint.
- W&B entity/project: `held/marin_moe_sft`; ID/name: `moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07_replay_continuous`; resume `allow`. The distinct ID records the corrected replay behavior.
- Preserve the prior launch's step-157000 full-state source checkpoint, 2,000 updates, 60/1,740/200 WSD schedule, batch 256, context 262,144, 80% SFT/20% replay, non-preemptible v4-2048 in us-central2-b, and 256-worker topology. Dataset roots are unchanged. DRI: William Held. Startup monitoring remains owned by this assistant session; the background Iris status monitor alone cannot detect stalled training progress.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name sft-datakit-20260907-prod-wsd-continuous --priority production --cpu 1 --memory 2GB --disk 5GB --region us-central2 -e UV_LOCK_TIMEOUT 900 -e MARIN_PREFIX gs://marin-us-central2 -e WANDB_API_KEY "$WANDB_API_KEY" -- uv run python experiments/june_tpu_67b_a2b/moe/sft_datakit_chat_mix.py`
- Next action: submit once, verify cached dependencies are skipped, and inspect startup progress beyond replay dataset construction.


### 2026-09-07 19:53 PDT - Continuous-token replay submitted

- Submitted `/held/sft-datakit-20260907-prod-wsd-continuous` at 19:51:22 PDT from clean, pushed source `a77786f182d4aa58cee051236624138ebdff59ec`, bundle `67a428ea21b7bba780c471d16edc77c12da17b9182ef652fe79ea470fc8d6f6c` (11.1 MB). The coordinator received the W&B credential.
- The coordinator dispatched the training child at 19:52:25 PDT. At 19:52:57 PDT both coordinator and training child were RUNNING. There are no preprocessing children. Dataset startup and the first training updates remain to be verified.
- [Iris job](https://iris.oa.dev/#/job/%2Fheld%2Fsft-datakit-20260907-prod-wsd-continuous). Next action: verify progress beyond dataset construction and checkpoint restoration.
