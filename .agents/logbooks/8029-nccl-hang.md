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

### 2026-08-07 02:34 UTC - moe-hero-fsdp-nccl2307-2rack-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-1000-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_supervised --run-id moe-hero-fsdp-nccl2307-2rack-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-1000-20260807-coord`; expected child `grug-train-moe-hero-fsdp-nccl2307-2rack-1000-20260807`.
- Git SHA: `1ba73a2951247e7bdd2e87974ef36625e2c6a478`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 workers, four GB200 GPUs each, two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Babysitter: Codex, two-minute cadence through first progress, then at most 15 minutes.

### 2026-08-07 02:50 UTC - moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807-coord -e WANDB_API_KEY <set> -e NCCL_RUNTIME_CONNECT 0 -- python -m experiments.grug.moe_hero_fsdp.launch_supervised --run-id moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807-coord`; expected child `grug-train-moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807`.
- Git SHA: `b3e8e4beda543c4413297635dd10e2e96b4d8957`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 workers, four GB200 GPUs each, two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-rtconnect0-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Environment ablation: `NCCL_RUNTIME_CONNECT=0`.
- Babysitter: Codex, two-minute cadence through first progress, then at most 15 minutes.

### 2026-08-07 02:57 UTC - moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807-coord -e WANDB_API_KEY <set> -e CUDA_MODULE_LOADING EAGER -- python -m experiments.grug.moe_hero_fsdp.launch_supervised --run-id moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807-coord`; expected child `grug-train-moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807`.
- Git SHA: `b18bdc949e77a683b91982a5c21f3462762d960a`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 workers, four GB200 GPUs each, two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-moduleeager-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Environment ablation: `CUDA_MODULE_LOADING=EAGER`.
- Babysitter: Codex, two-minute cadence through first progress, then at most 15 minutes.

### 2026-08-07 03:07 UTC - moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_failsafe_control --run-id moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807-coord`; expected child `grug-train-moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807`.
- Git SHA: `1f8d8377f499884e2bea9b65489aff221cd03494`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 workers, four GB200 GPUs each, two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-failsafeonly-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Diagnostic control: Same XLA failsafe flags as the supervisor, direct trainer process, zero task retries.
- Babysitter: Codex, two-minute cadence through first progress, then at most 15 minutes.

### 2026-08-07 03:31 UTC - moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch --run-id moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807-coord`; expected child `grug-train-moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807`.
- Git SHA: `1ac2c4331e55439932da32a614c04f912ed31df4`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 workers, four GB200 GPUs each, two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Diagnostic control: Ordinary launcher with no supervisor, recovery XLA flags, environment ablation, or profiler/debugger attachment.
- Babysitter: Codex, two-minute cadence through first progress, then at most 15 minutes.

## Event Log

### 2026-08-07 03:47 UTC - Clean one-process-per-node control stopped

- Result: The ordinary 2-rack trainer used 32 JAX processes for 128 GPUs and advanced through step 7 without `bad_alloc`, `length_error`, or a collective stall.
- Decision: Stopped `/power/moe-hero-fsdp-nccl2307-2rack-clean-1000-20260807-coord`; it did not test the requested one-process-per-GPU topology.
- Next: Set `processes_per_task=4` for each four-GPU Iris task and start the 2-rack supervised gate with XLA's 60-second execution deadman.

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

### 2026-08-07 02:31 UTC - Eight-rack minimal gate passed

- Status: All 128 tasks completed successfully in 2 minutes 32 seconds with no deadman abort.
- Evidence: Task 0 reported 512 devices, mesh `(8, 64, 1, 1)`, driver 595.71.05, NCCL 2.30.7+cuda13.3, and `no wedge reproduced for ablation baseline through 1000 steps`.
- Decision: Close the minimal-repro ablation ledger and start the 300B FSDP matrix sequentially at two racks.
- Next: Run 1000 supervised steps at two racks; promote to four racks only after a clean terminal result.

### 2026-08-07 02:34 UTC - Two-rack 300B gate submitted

- Status: The production-priority coordinator is pending with zero failures or preemptions.
- Evidence: Iris accepted the 9.8 MB bundle from the clean source commit.
- Decision: Keep the single two-rack request; do not submit the four-rack gate until this run is terminal.
- Next: Verify the resolved plan, W&B identity, NCCL provenance, and first training step.

### 2026-08-07 02:45 UTC - Two-rack 300B gate wedged before step zero

- Status: The real 300B FSDP model wedged on its first `jit_train_step` execution with NCCL 2.30.7; no training step completed.
- Evidence: Worker 1 reported a 128-device clique initialization stuck at 02:43:57, XLA's hang watchdog reported `jit_train_step` unfinished after one minute at 02:44:47, and the child aborted at 02:44:57. The supervisor classified `crash returncode=-6 last_step=None`, exhausted its zero-restart budget, and Iris coscheduled the other 31 workers down.
- Decision: Stop rack-count promotion and retry the environment-variable matrix at the failing two-rack scale, ordered by low expected steady-state cost and relevance to communicator initialization.
- Next: Start with `NCCL_RUNTIME_CONNECT=0`, then promote a clean arm through 1000 steps before resuming the 4/8/12-rack matrix; continue through the catalog if it wedges.

### 2026-08-07 02:56 UTC - Runtime-connect arm failed before step zero

- Status: `NCCL_RUNTIME_CONNECT=0` did not resolve the 300B run and changed the terminal failure from the XLA watchdog to `std::bad_alloc`.
- Evidence: The 128-device clique initially warned at 10 seconds, completed after 14.5 seconds, and the first `jit_train_step` then aborted on `std::bad_alloc` at 02:55:24. The supervisor classified `crash returncode=-6 last_step=None`; 31 sibling tasks were coscheduled down.
- Decision: Reject this arm because it does not reach a training step and continue in low-impact order.
- Next: Run `CUDA_MODULE_LOADING=EAGER` at the same two-rack shape.

### 2026-08-07 03:02 UTC - Eager-module arm failed before step zero

- Status: `CUDA_MODULE_LOADING=EAGER` also completed clique initialization and then aborted on `std::bad_alloc` before a training step.
- Evidence: The clique completed after 10.95 seconds at 03:01:45; `std::bad_alloc` followed nine seconds later, and the supervisor classified `crash returncode=-6 last_step=None`.
- Decision: Pause environment arms because `std::bad_alloc` is new relative to the historical unsupervised 300B runs. Split the supervisor parent from its XLA failsafe flags before interpreting further ablations.
- Next: Run the identical two-rack model directly with the same XLA flags and zero task retries, capture cgroup/GPU memory, and attach the Iris native profiler around first execution.

### 2026-08-07 03:20 UTC - Direct failsafe control reproduced a C++ exception

- Status: The first `jit_train_step` failed without a supervisor parent; rank 2 terminated on `std::length_error: basic_string::_M_create` before step zero.
- Evidence: Immediately before execution, task 0 used 186.5 GB of an 850 GiB cgroup limit with no `memory.events`, about 143.7 GiB of each 186 GiB GPU, and 172 GiB RSS. A five-second native sample placed the main thread in JAX/XLA compilation, including SPMD partitioning and HLO passes, rather than NCCL execution.
- Decision: Exonerate the supervisor parent process. Treat the shared XLA failsafe flags, especially progress tracking, as the leading explanation for the new `bad_alloc`/`length_error` family.
- Next: Repeat the same direct control with LLDB breakpoints on `abort` across ranks to capture the throwing native stack, then set `xla_gpu_execution_progress_tracking=0` while preserving the execution and NCCL-init timeouts.
