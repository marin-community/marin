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

### 2026-08-07 02:27 UTC - wedge-sup-nccl2307-8rack-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --timeout 43200 --max-retries 0 --job-name wedge-sup-nccl2307-8rack-20260807-coord -- python -m experiments.grug.recovery.launch_wedge_supervised --run-id wedge-sup-nccl2307-8rack-20260807 --dp-racks 8 --num-steps 1000 --version 2026.08.07 --run`.
- Job: `/power/wedge-sup-nccl2307-8rack-20260807-coord`; child `/power/wedge-sup-nccl2307-8rack-20260807-coord/grug-train-wedge-sup-nccl2307-8rack-20260807`.
- Git SHA: `40a504c48f70aabc12084ecc703b53afe1846a5f`.
- Dirty tree: No; the commit was pushed to `origin/weaver/infra-debug-nccl-hang` before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 128 workers, four GB200 GPUs each, eight NVL72 racks on `cw-us-east-08a`.
- W&B: None; this is a synthetic minimal reproducer.
- Output root: `s3://marin-us-east-02a/marin/grug/wedge-sup-nccl2307-8rack-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: None.
- Babysitter: Codex, two-minute cadence through first progress and terminal state.

## Event Log

### 2026-08-07 02:11 UTC - Two-rack NCCL 2.30.7 gate completed

- Status: The minimal reproducer completed 1000 steps with `NCCL_NVLS_ENABLE=0` and another 1000 with `NCCL_NVLS_ENABLE=1` on one two-rack allocation.
- Evidence: `/power/wedge-sup-nccl2307-nvls-2rack-coord` reported NCCL 2.30.7 on all 32 tasks; both arms reached step 999 without a watchdog abort.
- Decision: Prepare a clean, supervised eight-rack minimal gate before launching the 300B matrix.
- Next: Commit and push the run implementation and contract.

### 2026-08-07 02:28 UTC - Eight-rack gate queued

- Status: The coordinator is running and all 128 GPU tasks are in the gang scheduler's build gate with zero failures or preemptions.
- Evidence: Iris resolved the expected output root and submitted the child from the clean source commit.
- Decision: Keep the single production-priority request and wait for full-gang admission.
- Next: Verify NCCL provenance and step progress immediately after admission.
