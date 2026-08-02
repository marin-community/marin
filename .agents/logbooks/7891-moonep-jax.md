---
topic: moonep-jax
issue: https://github.com/marin-community/marin/issues/7891
description: MoonEP and global histogram quantile balancing on one A08 NVL72 rack
author: rav
---

# MoonEP JAX: Task Logbook

## Current TL;DR

The work starts from PR #7890 at `e38ae4f8`. The first gate requires correct gradients and zero drops for 25 EP64 steps. The final gate requires at least 21.7% median MFU.

## Scope

- Goal: Implement global histogram QB and MoonEP expert routing in Levanter.
- Primary metrics: Assignment drops, output and gradient parity, median MFU, and tokens per second.
- Constraints: Use one A08 NVL72 rack. Do not change cluster configuration. Keep one active rack request.
- Coordinating issue: https://github.com/marin-community/marin/issues/7891
- Baseline PR: https://github.com/marin-community/marin/pull/7890

## Baseline

- Date: 2026-08-02.
- Code reference: `e38ae4f8b2477d420575b7335676328b5dd88172`.
- MHEP-004: 25 steps, 24.1231% median MFU, and 9.6786% final drops.
- MHEP-008: 200 steps, 23.6969% median MFU, and 7.4113% final drops.
- Hardware: 16 A08 nodes with four GB200 GPUs on one NVL72 rack.
- Model: Batch 1024, sequence 4096, 256 experts, top-8, and EP64.

## References

- MoonEP: https://github.com/moonshotAI/moonep at `0f385f038fc33bec22e3bcf5a07a8a22693e754c`.
- Kimi K3 report: https://arxiv.org/abs/2607.24653.
- Local MoonEP clone: `.agents/tmp/moonep`.
- Local report: `.agents/tmp/moonep/moonep-paper.pdf`.
- Report SHA-256: `936a7a3b655947b014ba96b8a790c3cdb6ea8b37eea514e4b7b52655e20af0f8`.

## Hypothesis Queue

### Active

- `MNEP-H1`: A global 1,000-bin histogram estimates the pooled QB target within one bin width.
- `MNEP-H2`: The MoonEP allocation gives each EP64 rank exactly `S*K` assignments with no drops.
- `MNEP-H3`: Sparse remote expert copies and grouped GEMM reach at least 21.7% median MFU after profile-based changes.

### Blocked

- None.

### Falsified / Dead End

- Receiver-ECHO from #7279 reached 18.2099% median MFU and did not remove all drops.
- Three-choice spill from #7279 reached 24.0829% median MFU and left 5.8806% final drops.

### Promoted

- None.

## Experiment Matrix

| ID | Transport | QB | Gate |
| --- | --- | --- | --- |
| MNEP-001 | Fixed all-to-all | Global histogram | QB-only effect |
| MNEP-002 | MoonEP JAX | Local exact | Zero-drop MoonEP effect |
| MNEP-003 | MoonEP JAX | Global histogram | Combined correctness and MFU |
| MNEP-004 | MoonEP JAX with QuACK | Global histogram | Rack retry after XLA OOM |

## Entry Log

### 2026-08-02 09:08 UTC - Research start

- Hypothesis: A portable JAX path can prove MoonEP semantics before native CUDA fabric work.
- Commit Hash: `e38ae4f8b2477d420575b7335676328b5dd88172`.
- Command: Source preparation and issue creation only.
- Config: PR #7890 EP64 hero configuration.
- Result: Created issue #7891. The MoonEP clone and report are present and verified.
- Interpretation: No open MoonEP issue or competing branch was found.
- Next action: Add the global histogram QB reference and behavior tests.

### 2026-08-02 09:19 UTC - Global histogram QB reference

- Hypothesis: A 1,000-bin pooled histogram removes the error from averaging per-rank quantiles.
- Commit Hash: `b3a4e0b65`.
- Command: `uv run pytest tests/test_moe_hero_ep.py tests/test_grug_variant_contracts.py -q`.
- Config: Global histogram range `[min(bias)-1, max(bias)+1]` with one integer reduction per layer.
- Result: Pooled quantile, local-average counterexample, and two-device reduction checks pass. Pyrefly reports zero errors.
- Interpretation: The implementation matches the report algorithm within one bin width and preserves the local estimator as a control.
- Next action: Implement the MoonEP allocation planner and compare it with the independent reference.

### 2026-08-02 10:06 UTC - Four-GPU MoonEP correctness gate

- Hypothesis: The portable MoonEP schedule preserves dense MoE outputs and gradients under full owner skew.
- Commit Hash: `e4a085e56`.
- Command: Four-GPU GB200 pytest for dense output and gradient parity.
- Config: Eight experts, top-2, four EP ranks, full routing to one owner, and four-token compute padding.
- Result: The output and gradients for tokens, gate/up weights, and down weights pass. The test completed in 97.00 seconds.
- Interpretation: Token dispatch, sparse expert copies, return order, and automatic differentiation are correct. The Pallas grouped GEMM corrupted isolated rows, so the portable path selects the XLA grouped GEMM.
- Next action: Run MNEP-003 on one NVL72 and profile the combined path.

### 2026-08-02 10:10 UTC - MNEP-003 rack launch contract

- Goal: Verify zero transport errors, finite training, and initial MFU for combined MoonEP and global histogram QB.
- Run ID: `mnep-003-combined-25-20260802-1010`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mnep-003-combined-25-20260802-1010-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-003-combined-25-20260802-1010 --num-steps 25 --moe-implementation moonep_jax --moonep-token-padding 128 --qb-method global_histogram --qb-histogram-bins 1000 --version 2026.08.02 --run`.
- Config: EP64, batch 1024, sequence 4096, 256 experts, top-8, and 16 workers with four GB200 GPUs each.
- Output: `s3://marin-us-east-02a/marin/grug/mnep-003-combined-25-20260802-1010/2026.08.02`.
- W&B identity: Project `rav_moe`, group `moe-hero-ep`, and run name `mnep-003-combined-25-20260802-1010` in offline capture mode.
- Stop criteria: Stop on task retry, OOM, non-finite loss, transport errors, or incomplete step 25.
- Next action: Commit and tag this contract, submit once, and monitor to a terminal state.

### 2026-08-02 10:17 UTC - MNEP-003 failed before step zero

- Job: `/rav/mnep-003-combined-25-20260802-1010-coord`.
- Result: All 16 workers initialized, but the XLA grouped GEMM requested a 180.62 GiB temporary and OOMed before step zero.
- Evidence: Iris recorded one task failure and zero preemptions. The parent job was stopped before its automatic retry could repeat the OOM.
- Interpretation: XLA grouped GEMM is a correctness reference only. It is not a viable rack compute path.
- Next action: Select the existing QuACK SM100 grouped GEMM and rerun the four-GPU output and gradient gate.

### 2026-08-02 10:42 UTC - QuACK correctness gate and MNEP-004 contract

- Code change: Add an explicit XLA or QuACK compute backend. QuACK requires SiLU and 128-aligned hidden and intermediate dimensions.
- Padding change: Every local and copied expert has one non-empty padding bucket. This removes duplicate group boundaries in QuACK weight gradients and adds eight rows per GPU for the hero shape.
- Four-GPU result: Bfloat16 outputs and gradients pass against the dense oracle under full owner skew. The QuACK test completed in 124.32 seconds.
- Run ID: `mnep-004-quack-combined-25-20260802-1042`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mnep-004-quack-combined-25-20260802-1042-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-004-quack-combined-25-20260802-1042 --num-steps 25 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --version 2026.08.02 --run`.
- Config: EP64, batch 1024, sequence 4096, 256 experts, top-8, and 16 workers with four GB200 GPUs each.
- Stop criteria: Stop on task retry, OOM, non-finite loss, transport errors, or incomplete step 25.
- Next action: Commit and tag the QuACK snapshot, submit MNEP-004 once, and monitor it to a terminal state.

### 2026-08-02 10:56 UTC - MNEP-004 full-program memory failure

- Result: Task 0 failed twice with exit 139 while XLA compiled the first training step. The other 15 tasks stopped through atomic rescheduling.
- Evidence: The first attempt reported 146.43 GiB before rematerialization and 141.52 GiB after rematerialization. The compiler target was 133.22 GiB.
- Isolation: Full-shape QuACK forward and gradient calls passed on one GB200 with 525,312 rows, hidden size 5,120, intermediate size 1,280, and eight groups.
- Interpretation: The QuACK kernel supports the hero shape. The complete training graph exceeds the default compiler memory target and enters a failing rematerialization path.
- Action: MNEP-004 was stopped before a third rack attempt.

### 2026-08-02 10:56 UTC - MNEP-005 memory-fraction smoke contract

- Goal: Compile and execute three combined steps with a 92% JAX memory fraction.
- Run ID: `mnep-005-quack-mem92-smoke-3-20260802-1056`.
- Config: MNEP-004 settings plus `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` and no task retries.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Submit one rack request. Promote the memory setting to the 25-step run only if this smoke test passes.

### 2026-08-02 11:07 UTC - MNEP-005 collective-pool OOM

- Result: The program compiled and reached executable launch. All tasks then failed while XLA requested a separate 15.00 GiB collective pool.
- Evidence: A 92% JAX preallocation used about 175.4 GiB on each GPU before execution. The remaining device memory could not hold the collective pool.
- Interpretation: The compiler needs more than the default 75% program pool, but collectives need at least 15 GiB outside that pool.
- Action: Stop the automatic child retry. Test an 84% fraction, which gives about 159 GiB to the program and about 30 GiB outside it.

### 2026-08-02 11:07 UTC - MNEP-006 memory-window smoke contract

- Goal: Compile and execute three combined steps with an 84% JAX memory fraction.
- Run ID: `mnep-006-quack-mem84-smoke-3-20260802-1107`.
- Config: MNEP-005 settings except `XLA_PYTHON_CLIENT_MEM_FRACTION=0.84`.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Promote 84% to a 25-step run only if MNEP-006 passes.

### 2026-08-02 11:14 UTC - MNEP-006 compiler failure

- Result: The 84% run failed with exit 139 during XLA scheduling, before executable launch.
- Interpretation: The allocator split is sufficient for the program and collective pools, but the scheduler still uses its independent memory limit.
- Source check: XLA computes the scheduler limit from 80% of device memory, input and output size, and `xla_gpu_memory_limit_slop_factor`. The default factor is 95.
- Action: Keep the 84% allocator split and set the scheduler slop factor to 106. This raises the observed 133.22 GiB target above the 146.43 GiB original schedule without reducing collective memory.

### 2026-08-02 11:14 UTC - MNEP-007 scheduler-limit smoke contract

- Goal: Compile and execute three combined steps with an 84% JAX memory fraction and a 106% XLA scheduler slop factor.
- Run ID: `mnep-007-quack-mem84-slop106-smoke-3-20260802-1114`.
- Config: MNEP-006 settings plus `XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=106`.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Promote both memory settings to a 25-step run only if MNEP-007 passes.

### 2026-08-02 11:28 UTC - MNEP-007 NCCL device-communicator failure

- Result: The full train step compiled without the prior scheduler warning.
- Failure: All workers then exited 139 in `ncclDevCommCreate` before step zero.
- Source check: JAX 0.11.0 pins XLA `131bf41acb4650e4391a640c3f1859c1c86ad74b`.
- Source check: This XLA revision requests one device communicator for its NCCL-backed ragged all-to-all barrier.
- Interpretation: The 84% allocator split and 106% scheduler factor solved the two HBM limits.
- Interpretation: NCCL 2.28.9 cannot create the new device communicator on this NVL72.
- Action: Stop the automatic retry and disable only the NCCL-backed ragged barrier.
- Fallback: XLA then uses its standard NCCL ragged all-to-all across the multi-host EP64 group.

### 2026-08-02 11:28 UTC - MNEP-008 host-NCCL smoke contract

- Goal: Compile and execute three combined steps without the NCCL device communicator.
- Run ID: `mnep-008-quack-host-nccl-smoke-3-20260802-1128`.
- Config: EP64, batch 1024, sequence 4096, 256 experts, top-8, and global histogram QB with 1,000 bins.
- Runtime: Use an 84% main pool, a 106% scheduler factor, and the standard NCCL ragged path.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Promote these settings to a 25-step run only if MNEP-008 passes.

### 2026-08-02 11:49 UTC - MNEP-008 correct but below the MFU gate

- Result: The 16-worker child job completed all three steps without a transport error.
- Correctness: Final loss was 9.4065, and all three steps reported zero dropped assignments.
- Performance: Two MFU samples had a 6.7023% median with a 38.617-second final step.
- Interpretation: Standard NCCL ragged transport proves rack correctness, but it misses the 21.7% MFU gate.
- Action: Keep this run as the correctness fallback and restore the one-shot path with a compatible NCCL release.

### 2026-08-02 11:49 UTC - NCCL 2.29.7 restores the one-shot gate

- Reproducer: JAX 0.11.0 and NCCL 2.28.9 crashed the four-GPU value and gradient test in `ncclDevCommCreate`.
- Treatment: The same environment with NCCL 2.29.7 passed the test in 143.21 seconds.
- Source check: NCCL 2.29 adds versioned device API structures and cross-version checks.
- Decision: Pin the Marin and Levanter CUDA 13 extras to NCCL 2.29.7.
- Decision: Remove the host-NCCL fallback flag and retain the 84% pool with the 106% scheduler factor.

### 2026-08-02 11:49 UTC - MNEP-009 NCCL 2.29.7 smoke contract

- Goal: Execute three combined steps through XLA's one-shot ragged path on one NVL72.
- Run ID: `mnep-009-quack-nccl2297-smoke-3-20260802-1149`.
- Config: EP64, batch 1024, sequence 4096, 256 experts, top-8, and global histogram QB with 1,000 bins.
- Runtime: Use NCCL 2.29.7, an 84% main pool, and a 106% scheduler factor.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Launch the 25-step MFU gate only if MNEP-009 passes.

### 2026-08-02 12:14 UTC - MNEP-009 one-shot kernel failure

- Result: The EP64 training step compiled, but many ranks reported a CUDA illegal address at the same timestamp before step zero.
- Evidence: XLA failed while it recorded the completion event for an asynchronous collective. The workers then exited with signal 5.
- Interpretation: NCCL 2.29.7 fixes device-communicator creation, but XLA 0.11.0's older multi-node one-shot kernel is not correct at EP64.
- Source check: OpenXLA PR 46116 adds a later NCCL LSA and GIN device kernel for multi-node ragged all-to-all.
- Four-GPU gate: JAX `0.11.1.dev20260731` with the new device-kernel flag passed MoonEP output and gradient parity in 119.88 seconds.
- Decision: Pin the exact July 31 GPU nightly and enable its device kernel only for MoonEP.

### 2026-08-02 12:14 UTC - MNEP-010 device-kernel smoke contract

- Goal: Execute three combined steps through XLA's NCCL LSA and GIN device kernel on one NVL72.
- Run ID: `mnep-010-device-kernel-smoke-3-20260802-1214`.
- Config: EP64, batch 1024, sequence 4096, 256 experts, top-8, and global histogram QB with 1,000 bins.
- Runtime: Use JAX `0.11.1.dev20260731`, NCCL 2.29.7, an 84% main pool, a 106% scheduler factor, and the device-kernel flag.
- Stop criteria: Stop on any task failure, non-finite loss, transport error, or incomplete step 3.
- Next action: Launch the 25-step correctness and MFU gate only if MNEP-010 passes.

### 2026-08-02 12:35 UTC - MNEP-010 lacks the multi-node LSA fix

- Result: The EP64 program compiled, but rank 10 failed before step zero with a CUDA illegal address.
- Evidence: All four local GPU threads failed when XLA recorded the completion event for an asynchronous collective.
- Source: JAX `0.11.1.dev20260731` pins XLA `2d26accb73ec90841df9f6156d877b19782300dd` from July 30.
- Source: XLA `0d993720fac7a1b5f7522b1c525a9354ebb04f0b` reads the NCCL LSA domain size from `nLsaTeams` on multi-node NVLink.
- Gap: The LSA fix is 67 XLA commits after the MNEP-010 wheel source.
- Decision: Build JAX at `f9f6bbaced02ef315d20b34facec09e79f356503`, which pins fixed XLA `5d53e1e40cd08655e8fe52f104f35b57ce35a626`.
- Next action: Do the four-GPU parity gate, then do one EP64 three-step gate with the fixed XLA wheel.

### 2026-08-02 12:53 UTC - Direct weight sends pass the four-GPU parity gate

- Change: Use multiple ragged updates per peer to send each expert from its owner buffer.
- Change: Use a custom gradient path to sum duplicate expert gradients at the owner.
- Memory: This removes the 66-row forward sender buffer for each weight projection at the hero shape.
- Gate: The four-GPU full-owner-skew test passed output and input, `w13`, and `w2` gradient comparisons.
- Result: `1 passed, 146 warnings in 169.59s` on four GB200 GPUs.
- Next action: Repeat this gate with the fixed XLA wheels before the EP64 rack run.

### 2026-08-02 13:05 UTC - Fixed XLA build passes local runtime gates

- Build: JAX `f9f6bbaced02ef315d20b34facec09e79f356503` with XLA `5d53e1e40cd08655e8fe52f104f35b57ce35a626`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-20260802`.
- Hash: JAX `40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d`.
- Hash: JAX PJRT `fd2724cd9f128ea1a0d1f74029ce6fcdaf7915db1a351b088316cc821ac2408d`.
- Hash: JAX plugin `d04ee6bdc956979fa0c43ed95bfdba7bc4f665ceceb34531ef792cff742ddf95`.
- Hash: jaxlib `03e838842547a66af13bc93a533ce1943dc0f2eb83026a94994eca7f47c072b4`.
- Gate: The fixed-XLA four-GPU output and gradient test passed in 130.54 seconds.
- Setup: `--moonep-jax-wheel-build lsa-20260802` keeps standard GPU setup, checks all hashes, and restores CUDA 13 libraries.
- Setup gate: A clean generated task environment installed the fixed build and found four GB200 GPUs.
- Next action: Submit MNEP-011 for three combined EP64 steps on one NVL72 rack.

### 2026-08-02 13:14 UTC - MNEP-011 reaches the experimental peer-memory kernel

- Run ID: `mnep-011-fixed-xla-smoke-3-20260802-1307`.
- Snapshot: `mnep-011-fixed-xla-smoke-3-20260802-1307` at `849e4f4c36de7d8e0ed10c948b522612e146227a`.
- Setup: All 16 workers installed the four checked wheels and reported NCCL 2.29.7.
- Progress: The full training program compiled and reached the fused loss before executable launch.
- Failure: Task 10 exited 133 before step zero with `CUDA_ERROR_CONTAINED` from invalid peer GPU memory access.
- Placement: The workers used the `DH1-393-US-EAST-08A` NVLink domain on one physical rack.
- Action: Stop the parent as soon as Iris assigned attempt one.
- Interpretation: The XLA LSA-size fix is present, but its experimental direct peer-memory kernel is not correct for this EP64 program.
- Next action: Profile the correct standard NCCL path, then replace its largest transport cost with a bounded static collective.

### 2026-08-02 13:39 UTC - MNEP-012 proves the correct rack path

- Run ID: `mnep-012-host-nccl-profile-3-20260802-1325`.
- Snapshot: `mnep-012-host-nccl-profile-3-20260802-1325` at `c57925bb3bd1cc95aaa9ba4fa6f45993a789a171`.
- Result: All 16 workers completed three combined steps on attempt zero.
- Correctness: Loss was 9.40749740600586, with zero dropped assignments and no transport error.
- Performance: Median MFU was 6.6395957732540705% across two measured steps. The final duration was 38.981918029952794 seconds.
- Profile gap: Grug skips callback hooks for completed steps zero and one. The requested step-one window did not start.
- Decision: Require profile windows to start at step three or later, then repeat five steps with a step-three profile window.

### 2026-08-02 14:21 UTC - One-process MoonEP passes the four-GPU parity gate

- Layout: The Iris supervisor ran four JAX processes with one GB200 device for each process.
- Runtime: The fixed XLA build used the NCCL LSA and GIN device kernel.
- Skew: Every token selected experts zero and one, which were both owned by process zero.
- Result: All four processes matched the dense output and the input, `w13`, and `w2` gradients.
- Capacity: The run reported zero dropped assignments.
- Snapshot: `mnep-014-one-proc-device-smoke-3-20260802-1402` at `eac3ee03760b3fa056a5eb0e2660444d9872f5df`.
- Rack gate: MNEP-014 is waiting for one complete NVL72 with this process layout.

### 2026-08-02 14:29 UTC - Report audit fixes the QB score domain

- Report rule: QB selects Top-(k+1) from `sigmoid(router_logits) + bias`.
- Report bound: The required bias is in `[min(bias) - 1, max(bias) + 1]` because the scores are in `(0, 1)`.
- Gap: The first histogram implementation used raw router logits for selection, cutoffs, and margins.
- Fix: Use sigmoid scores for these three operations. Keep the bias out of the mixture weights and router gradients.
- Gate: A crafted large-logit case now selects the sigmoid-score route and keeps every required bias in the report's range.
- Action: Replace the queued MNEP-014 rack run with a corrected snapshot after the code gate passes.

### 2026-08-02 14:43 UTC - MNEP-016 proves the low-host admission request

- Run ID: `mnep-016-sigmoid-qb-low-host-smoke-3-20260802-1443`.
- Command: The existing GPU pod ran `.agents/tmp/launch_mnep016.py` and submitted the child job directly.
- Config: Each of 16 workers requested four GB200 GPUs, 16 CPUs, 88 GiB RAM, and 256 GiB disk.
- Admission: Kueue admitted all 16 workers on `DH1-393-US-EAST-08A` without a resource wait.
- Failure: Every worker stopped during setup with `ModuleNotFoundError: No module named 'experiments.grug.moe_hero_ep'`.
- Cause: A direct child submission reused the GPU pod source bundle. The bundle did not contain the MoonEP experiment package.
- Action: Stop the child job before GPU setup and submit a top-level coordinator from the current worktree.

### 2026-08-02 14:55 UTC - MNEP-017 bundled rack smoke contract

- Goal: Run three combined MoonEP and global QB steps with one JAX process for each GPU.
- Snapshot: `mnep-017-sigmoid-qb-low-host-bundled-smoke-3-20260802-1451` at `18d3b9832`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-017-sigmoid-qb-low-host-bundled-smoke-3-20260802-1451-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-017-sigmoid-qb-low-host-bundled-smoke-3-20260802-1451 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-20260802 --processes-per-task 4 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.
- Config: EP64, 16 workers, four processes for each worker, fixed XLA, sigmoid-score QB, 1,000 bins, and the device kernel.
- Source: The top-level coordinator bundled the current worktree before federation to `cw-us-east-08a`.
- State: The coordinator is active. The 16 GPU tasks wait for MNEP-013 to release the NVL72.
- Stop criteria: Stop on the first retry, setup error, transport error, non-finite loss, or dropped assignment.

### 2026-08-02 15:05 UTC - MNEP-013 profile serialization exceeds the barrier timeout

- Run ID: `mnep-013-host-nccl-profile-5-20260802-1341`.
- Snapshot: `mnep-013-host-nccl-profile-5-20260802-1341` at `a0180e182`.
- Config: The run used five steps, a two-step profile from step three, the host-NCCL path, and one process for each task.
- Model result: Fifteen ranks completed all five model steps. The last reported loss was 8.16.
- Profile result: Rank zero entered `jax.profiler.stop_trace` and used 42 GiB of host RAM.
- Failure: The remaining ranks entered `barrier_sync()` and exceeded its 200-second timeout before rank zero serialized the trace.
- Evidence: Iris reported exit 133 for task zero and one failure across 16 tasks.
- Artifact: No profile file was written before the failure.
- Interpretation: The model step is correct. The profile window and HLO data are too large for the profile barrier.
- Action: Use one profile step without HLO data before a change to the common barrier timeout.

### 2026-08-02 15:05 UTC - MNEP-017 rules out process layout as the EP64 fix

- Run ID: `mnep-017-sigmoid-qb-low-host-bundled-smoke-3-20260802-1451`.
- Snapshot: `mnep-017-sigmoid-qb-low-host-bundled-smoke-3-20260802-1451` at `18d3b9832`.
- Config: The run used 16 workers, four JAX processes for each worker, one GPU for each process, 16 CPUs, and 88 GiB RAM.
- Admission: Kueue placed the run on one NVL72 without a resource wait.
- Setup: All processes installed the fixed JAX build and NCCL 2.29.7. The full EP64 program compiled.
- Failure: All four local ranks on task 10 reported `CUDA_ERROR_ILLEGAL_ADDRESS` before step zero.
- Stack: The failure occurred in `AsyncExecution::ExecutionGuard`, `AsyncStartThunk`, and `WhileThunk` during completion-event recording.
- Evidence: Iris reported exit 250 for task 10 and one failure across 16 tasks.
- Interpretation: One process for each GPU does not make the multi-node ragged device kernel correct.
- Decision: Stop process-layout tests. Replace the device kernel with a bounded standard collective, or optimize the correct host-NCCL path.

### 2026-08-02 15:30 UTC - XLA two-slice decomposition passes the local parity gate

- Hypothesis: XLA can replace the unsafe cross-slice ragged kernel and keep its fast device kernel inside each slice.
- Snapshot: `f1353baa0`.
- Runtime: The fixed XLA build used a two-GPU slice override on four GB200 GPUs.
- HLO: The optimized program used all-gather groups `{0,2}` and `{1,3}`, plus a standard metadata all-to-all across the same groups.
- HLO: The remaining ragged collective groups were `{0,1}` and `{2,3}`.
- Result: The full-owner-skew test passed output and input, `w13`, and `w2` gradient parity in 123.91 seconds.
- Source: [OpenXLA PR 46570](https://github.com/openxla/xla/pull/46570) describes this decomposition as the efficient multi-host fallback when the copy kernel is not available.
- Decision: Use two 32-GPU slices for the next EP64 rack test.

### 2026-08-02 15:30 UTC - MNEP-018 two-slice rack smoke contract

- Goal: Compile and execute three combined MoonEP and global QB steps through the two-slice XLA decomposition.
- Run ID: `mnep-018-two-slice-device-smoke-3-20260802-1530`.
- Snapshot: `f1353baa0`.
- Config: EP64, 16 workers, one process for each worker, fixed XLA, sigmoid-score QB, 1,000 bins, 16 CPUs, and 88 GiB RAM.
- Transport: Standard collectives connect two 32-GPU slices. The NCCL LSA and GIN kernel operates only inside each slice.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-018-two-slice-device-smoke-3-20260802-1530-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-018-two-slice-device-smoke-3-20260802-1530 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-20260802 --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.
- Stop criteria: Stop on the first retry, setup error, transport error, non-finite loss, or dropped assignment.

### 2026-08-02 15:50 UTC - MNEP-018 needs more cross-slice collective headroom

- Result: Both task-zero attempts compiled the complete train step, then failed before step zero in the cross-slice NCCL all-to-all.
- Error: Every local GPU reported `NCCL WARN Cuda failure 2 'out of memory'`. JAX reported the failure in `jit_train_step`.
- Isolation: Other tasks only lost the task-zero coordination service. No direct peer-memory or ragged device-kernel fault appeared.
- Interpretation: The two-slice rewrite reached its standard cross-slice transport. The 84% JAX pool leaves enough space for the host-ragged collective arena, but not this additional all-to-all.
- Action: Stop the retry and test an 80% pool. This pool is about 152.5 GiB and remains above the measured 146.43 GiB program schedule.

### 2026-08-02 15:52 UTC - MNEP-019 memory-window smoke contract

- Goal: Execute three combined MoonEP and global QB steps through the two-slice transport with enough cross-slice collective headroom.
- Run ID: `mnep-019-two-slice-mem80-smoke-3-20260802-1552`.
- Config: Match MNEP-018, but reduce the JAX main pool from 84% to 80%.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, compiler OOM, collective OOM, transport error, non-finite loss, or dropped assignment.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-019-two-slice-mem80-smoke-3-20260802-1552-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-019-two-slice-mem80-smoke-3-20260802-1552 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-20260802 --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.

### 2026-08-02 16:05 UTC - MNEP-019 isolates collective overlap memory

- Result: The 80% pool compiled the complete train step, but task zero again failed before step zero in the decomposed NCCL all-to-all.
- Device state: The main CUDA async pool held about 148 GiB. The separate collective space had about 37 GiB available.
- Shape: Each rank has 524,288 token assignments. One bfloat16 hidden buffer is exactly 5 GiB, and the two-slice dispatch all-gather produces 10 GiB.
- Source: XLA documents that `xla_gpu_experimental_parallel_collective_overlap_limit` controls the number of in-flight collectives. The current value is four.
- Interpretation: Four concurrent 10 GiB decomposed buffers exceed the 20% collective space. Lowering the main pool further would put the measured program schedule at risk.
- Decision: Keep the 80% main pool and limit MoonEP to three in-flight collectives. Keep the existing four-collective limit for the fixed all-to-all baseline.

### 2026-08-02 16:05 UTC - MNEP-020 bounded-overlap smoke contract

- Goal: Execute three combined MoonEP and global QB steps with the two-slice transport and at most three in-flight collectives.
- Run ID: `mnep-020-two-slice-overlap3-smoke-3-20260802-1605`.
- Config: Match MNEP-019 with an 80% main pool, but reduce the MoonEP collective overlap limit from four to three.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, memory error, transport error, non-finite loss, or dropped assignment.

### 2026-08-02 16:20 UTC - MNEP-020 proves the two-slice rack path

- Snapshot: `95cc400df`.
- Result: All 16 workers completed three combined MoonEP and global QB steps on attempt zero.
- Runtime: The child finished in 10 minutes and 10 seconds, including setup and compilation. No worker retried or reported a transport error.
- Memory: An 80% main pool and at most three in-flight collectives removed the cross-slice all-to-all OOM.
- Gate: Iris proves complete rack execution. Final loss, drop count, and MFU extraction waits for the delayed Finelog tail.
- Next action: Capture one no-HLO profile step, then use the trace and steady-state metrics for the first optimization.

### 2026-08-02 16:21 UTC - MNEP-021 compact profile contract

- Goal: Capture one representative XPlane step for the correct two-slice MoonEP and global QB path.
- Run ID: `mnep-021-two-slice-profile-5-20260802-1621`.
- Config: Match MNEP-020 for five steps. Start the profile at step three, capture one step, and omit HLO protobuf data.
- Gate: Require attempt-zero completion, a written XPlane artifact, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, profile barrier timeout, memory error, transport error, non-finite loss, or dropped assignment.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-021-two-slice-profile-5-20260802-1621-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-021-two-slice-profile-5-20260802-1621 --num-steps 5 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-20260802 --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --profile-start-step 3 --profile-num-steps 1 --version 2026.08.02 --run`.

### 2026-08-02 16:50 UTC - MNEP-020 exact correctness and throughput result

- Result: All 16 workers completed three combined steps on attempt zero.
- Correctness: Final loss was 9.981470108032227, with zero dropped assignments.
- Performance: Median MFU was 3.1512622180758267% across two measured steps.
- Performance: Throughput was 51,066.91275525412 tokens per second. Final step time was 82.13349454081617 seconds.
- Gap: The fallback is 6.9 times below the required 21.7% median MFU.

### 2026-08-02 16:50 UTC - MNEP-021 profile isolates transport cost

- Result: All 16 workers completed five combined steps on attempt zero.
- Correctness: Final loss was 8.176630973815918, with zero dropped assignments.
- Performance: The final unprofiled step reached 2.9958644336442806% MFU and 48,548.65671981346 tokens per second.
- Artifact: `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mnep-021-two-slice-profile-5-20260802-1621/plugins/profile/steps-3-to-4`.
- Profile: The local ragged kernel used 10.645 seconds across 80 calls on the traced process.
- Profile: The transport barrier used 3.294 seconds across 400 calls.
- Profile: Cross-slice all-gather used 3.548 seconds across 548 calls.
- Interpretation: Transport and its barriers dominate the correct fallback path.
- Decision: Test XLA's direct EP64 device path with NCCL 2.30.7 before source changes.
- Local gate: NCCL 2.30.7 passed four-GPU output and gradient parity in 108.50 seconds.

### 2026-08-02 16:57 UTC - MNEP-022 direct-device smoke contract

- Goal: Execute three combined MoonEP and global QB steps through the direct EP64 device kernel.
- Run ID: `mnep-022-direct-nccl2307-smoke-3-20260802-1657`.
- Runtime: Use fixed XLA `5d53e1e`, NCCL 2.30.7, an 84% main pool, and four in-flight collectives.
- Config: Use 16 workers, four GB200 GPUs per worker, and one JAX process per worker.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, memory error, transport error, non-finite loss, or dropped assignment.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-022-direct-nccl2307-smoke-3-20260802-1657-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-022-direct-nccl2307-smoke-3-20260802-1657 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-20260802 --moonep-transport direct_device --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.

### 2026-08-02 17:07 UTC - MNEP-022 rules out the NCCL runtime-only fix

- Snapshot: `mnep-022-direct-nccl2307-smoke-3-20260802-1657` at `3e38281e4`.
- Setup: All 16 workers used fixed XLA `5d53e1e` and NCCL 2.30.7 on one NVL72.
- Failure: Task 10 and all four local GPU threads failed before step zero.
- Error: CUDA reported an illegal address while XLA recorded an asynchronous completion event.
- Stack: The failure passed through `AsyncExecution`, `AsyncStartThunk`, and `WhileThunk`.
- Result: A runtime-only update to NCCL 2.30.7 did not make XLA's direct EP64 device kernel correct for the 5 GiB hero buffers.
- Build check: The fixed JAX/XLA wheels used NCCL 2.29.7 headers. The local four-GPU gate did not use multi-node GIN.
- Source check: NCCL 2.30.7 changes GIN structures and resource-sharing fields, so the runtime-only test did not test the new device API.
- Action: Stop the parent when Iris started attempt one.
- Decision: Rebuild the fixed JAX/XLA wheels against NCCL 2.30.7, then repeat local and rack correctness gates before a source patch.

### 2026-08-02 17:18 UTC - Fixed XLA rebuild uses NCCL 2.30.7 headers

- Source: JAX `f9f6bbace` and XLA `5d53e1e` with CUDA 13.0, cuDNN 9.12, NCCL 2.30.7, and `sm_100`.
- Build: The PJRT target recompiled both ragged all-to-all CUDA sources and completed successfully.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-20260802`.
- PJRT SHA-256: `a1bb00b9ed594e7d1b85251bce63660bb85c5f7a661d618af677cee481a4572a`.
- Host wheels: The JAX, jaxlib, and CUDA plugin hashes match the prior source-identical build.
- Next gate: Install this complete wheel set and repeat full-owner-skew parity on four local GPUs.
