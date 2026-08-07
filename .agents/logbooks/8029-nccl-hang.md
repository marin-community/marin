---
topic: nccl-hang
issue: https://github.com/marin-community/marin/issues/8029
description: Reproduce and isolate the GB200 NCCL wedge, then validate the 300B FSDP model across rack counts.
author: power
---

# NCCL Hang: Hero Run Logbook

## Run Contract

- DRI: User in the Weaver session; Codex owns launch monitoring.
- Goal: Validate the NCCL 2.30.7 fix at eight racks, then run the 300B Grug FSDP model for 1000 steps at 2, 4, 8, and 12 racks under crash supervision.
- Stop and escalation criteria: Stop the matrix on an 8-rack minimal-repro wedge. For the 300B matrix, classify any XLA deadman abort or illegal instruction, preserve provenance, and resume the environment-variable arms at the failing scale.
- Issue: https://github.com/marin-community/marin/issues/8029; progress is kept in its single Weaver-backed comment.
- W&B: `marin-community/marin_moe`, group `moe-hero-fsdp`; each model run uses its run ID as the W&B ID and display name with resume policy `allow`.
- Output roots: `s3://marin-us-east-02a/marin/grug/<run-id>/2026.08.07`; the dry plan resolves `grug/<run-id>/2026.08.07`.
- Initialization: None.
- Final step: 1000 for every run.
- Checkpoint policy: Metrics-only diagnostic runs with `--no-save-checkpoints`; no multi-terabyte final or rollback checkpoint is written.
- Detection: One `GPUHangSupervisor` per GPU task, XLA per-execution termination after 60 seconds, no in-pod restart, and the existing 15-minute progress watchdog fallback.
- Babysitter cadence: Two minutes through admission and first progress, then at most 15 minutes until terminal state.

## Launched Instances

No instances launched from the clean run commit yet.

## Event Log

### 2026-08-07 02:11 UTC - Two-rack NCCL 2.30.7 gate completed

- Status: The minimal reproducer completed 1000 steps with `NCCL_NVLS_ENABLE=0` and another 1000 with `NCCL_NVLS_ENABLE=1` on one two-rack allocation.
- Evidence: `/power/wedge-sup-nccl2307-nvls-2rack-coord` reported NCCL 2.30.7 on all 32 tasks; both arms reached step 999 without a watchdog abort.
- Decision: Prepare a clean, supervised eight-rack minimal gate before launching the 300B matrix.
- Next: Commit and push the run implementation and contract.
