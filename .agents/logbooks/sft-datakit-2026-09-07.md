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
