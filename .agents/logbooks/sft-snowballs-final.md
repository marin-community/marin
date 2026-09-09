---
topic: sft-snowballs-final
issue: https://github.com/marin-community/marin/issues/8977
description: Five-way Snowball LCE SFT and evaluation campaign derived from issue 8225.
author: benfeuer
---

# Snowball LCE Final: Task Logbook

## Scope

- Goal: publish five Snowball LCE bases, run the full Chat/Thinking/OpenCode/Nemotron-Terminal SFT graph for each on RNO2A, and evaluate every checkpoint under the fixed policy.
- Primary metrics: 17-task non-agentic results for all 25 checkpoints and matching-harness agentic reward for ten final checkpoints.
- Constraints: RNO2A interactive priority; preserve per-base architecture; no unapproved cross-region transfer over 10 GB; reproducible pinned inputs and durable artifacts.
- Coordinating issues: [#8977](https://github.com/marin-community/marin/issues/8977), [#8225](https://github.com/marin-community/marin/issues/8225), [#7958](https://github.com/marin-community/marin/issues/7958).

## Current TL;DR

- Fixed policy written. Five native sources identified. Two base HF uploads owned by `/held` are running but not yet validated. RNO2A launch is gated on a Snowball HF-loader smoke test and accessible pinned SFT data.

## Entry Log

### 2026-09-08 20:31 EDT - Policy reconstruction

- Hypothesis: first-class Snowball HF initialization can avoid copying five roughly 1 TB native GCS checkpoints to the RNO2A region while preserving each model's `qk_mult`.
- Commit Hash: `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`
- Command: read-only inspection of issues #8977/#8225/#7958, PR #8172 and current training/evaluation launchers.
- Config: five step-157000 262K-context bases; Base -> Chat -> Thinking -> two independent 1,888-step agentic branches; evaluation policy recorded in the external experiment directory.
- Result: `POLICY.md` and `STATE.md` created; two in-progress `/held` uploads observed; no training submitted.
- Interpretation: validate the HF path before GPU fan-out; never race another owner's upload or perform an unapproved native cross-region mirror.
- Next action: monitor uploads, validate the Snowball HF SFT path, and resolve the remaining three base exports.

### 2026-09-08 20:50 EDT - Packed-mask blocker and RNO2A smoke launch

- Hypothesis: the first-class Snowball adapter can train the published HF checkpoint at the historical 64-H100 topology after preserving packed segment IDs and expressing an intra-node `expert=8`, cross-node `data=8` mesh.
- Commit Hash: uncommitted launch bundle based on `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`; WIP commit pending checks.
- Command: `iris --config lib/iris/config/marin.yaml job run --target-cluster cw-rno2a --job-name snowball-final-qk157-smoke-coord ... -- python -m experiments.sft.configs.snowball_lce_final --base qk157 --stage smoke --version 2026.09.08.1 --run`
- Config: `qk157@2b1f526273b8968b307a0098c08fb4321bb91e35`, one update, seq 32768, global batch 64, 8 nodes x 8 H100, ring expert parallelism 8, AdamH/Adam at 5e-5, unique JAX port 19301, RNO2A interactive.
- Result: job `/benfeuer/snowball-final-qk157-smoke-coord` submitted. Local packed-boundary regression reproduced: changing segment 0 changed all 128 checked hidden values in segment 1 (maximum absolute change 0.01901957). The fix makes segment 1 bit-identical. Mesh and launcher unit tests pass.
- Interpretation: the previous adapter could report a healthy run while violating packed-example isolation. Full training remains gated on the smoke reaching a finite update and producing a reloadable checkpoint.
- Next action: monitor the exact job handle; validate output and record its immutable code commit.

### 2026-09-08 20:52 EDT - Smoke attempt 1 failed before model load

- Hypothesis: Levanter tokenizer staging accepts the same `repo@revision` syntax as its HF checkpoint converter.
- Commit Hash: uncommitted launch bundle based on `1234cbe4bfbcdc24c1bcfb20741f557c1eec567e`.
- Command: `/benfeuer/snowball-final-qk157-smoke-coord`.
- Config: qk157 smoke policy above; tokenizer path was `open-athena/snowball-67b-a2b-base-262k-qk157@2b1f...`.
- Result: falsified before model loading. `snapshot_download` rejected the `@revision` suffix as an invalid repo ID on all eight ranks. No update ran and no checkpoint was written.
- Interpretation: model checkpoint loading supports `RepoRef` strings; tokenizer staging does not. For the next smoke, use the validated repository's current tokenizer path while retaining the immutable model revision. Add revision-aware tokenizer staging before full training.
- Next action: relaunch with a new artifact version and job handle; retain this failed attempt in the record.
