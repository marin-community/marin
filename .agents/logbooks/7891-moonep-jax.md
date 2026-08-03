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

### 2026-08-02 17:25 UTC - NCCL 2.30.7 header build passes local parity

- Setup: The clean checkout at `ae283d1e8` used the rebuilt PJRT wheel and the NCCL 2.30.7 runtime.
- Device gate: JAX found all four GB200 GPUs in the development pod.
- Correctness: The full-owner-skew test passed output and input, `w13`, and `w2` gradient parity.
- Result: `1 passed, 3 warnings in 109.07s`.
- Scope: This gate uses one local LSA domain. The rack gate must test multi-node GIN.

### 2026-08-02 17:26 UTC - MNEP-023 rebuilt-header smoke contract

- Goal: Execute three combined MoonEP and global QB steps through the direct EP64 device kernel.
- Run ID: `mnep-023-direct-nccl2307-headers-smoke-3-20260802-1726`.
- Snapshot: `mnep-023-direct-nccl2307-headers-smoke-3-20260802-1726`.
- Runtime: Use fixed XLA `5d53e1e` built against NCCL 2.30.7, with the matching runtime, an 84% main pool, and four in-flight collectives.
- Config: Use 16 workers, four GB200 GPUs per worker, one JAX process per worker, sigmoid-score global histogram QB, and QuACK grouped GEMM.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, memory error, transport error, non-finite loss, or dropped assignment.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-023-direct-nccl2307-headers-smoke-3-20260802-1726-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-023-direct-nccl2307-headers-smoke-3-20260802-1726 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-nccl-2307-20260802 --moonep-transport direct_device --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.

### 2026-08-02 17:31 UTC - MNEP-023 still fails the concurrent direct path

- Placement: All 16 workers used NVLink domain `DH1-129-US-EAST-08A` on one NVL72.
- Failure: Task 10 attempt zero exited with `SIGTRAP` before step zero. The run wrote no training metrics.
- Retry: Iris started attempt one for the worker group. Stop the parent under the test contract.
- Result: Building the fixed device kernel against NCCL 2.30.7 did not remove the EP64 failure.
- Source: Each direct multi-node kernel uses GIN signal index zero and CTA-indexed world barriers from one cached device communicator.
- Hypothesis: Four concurrent collective launches reuse the same synchronization slots and can corrupt device progress.
- Decision: Set the direct transport overlap limit to one, then repeat the same rack gate.

### 2026-08-02 17:35 UTC - MNEP-024 serialized direct smoke contract

- Goal: Execute three combined MoonEP and global QB steps through one direct EP64 device kernel at a time.
- Run ID: `mnep-024-direct-serialized-smoke-3-20260802-1735`.
- Snapshot: `mnep-024-direct-serialized-smoke-3-20260802-1735`.
- Treatment: Match MNEP-023, but reduce the direct collective overlap limit from four to one.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.
- Stop criteria: Stop on the first retry, memory error, transport error, non-finite loss, or dropped assignment.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 --job-name mnep-024-direct-serialized-smoke-3-20260802-1735-coord -e WANDB_MODE offline -e WANDB_PROJECT rav_moe -- python -m experiments.grug.moe_hero_ep.launch --run-id mnep-024-direct-serialized-smoke-3-20260802-1735 --num-steps 3 --moe-implementation moonep_jax --moonep-token-padding 128 --moonep-grouped-gemm quack --qb-method global_histogram --qb-histogram-bins 1000 --moonep-jax-wheel-build lsa-nccl-2307-20260802 --moonep-transport direct_device --processes-per-task 1 --worker-cpu 16 --worker-ram-gb 88 --version 2026.08.02 --run`.

### 2026-08-02 17:44 UTC - MNEP-024 rules out collective concurrency

- Placement: All 16 workers used NVLink domain `DH1-124-US-EAST-08A` on one NVL72.
- Setup: Every worker verified the rebuilt fixed wheels and the NCCL 2.30.7 runtime.
- Failure: Tasks 8 through 15 reported a CUDA illegal address before step zero.
- Error: All four local GPU threads failed while XLA recorded an asynchronous completion event.
- Stack: The failure passed through `AsyncExecution`, `AsyncStartThunk`, and `WhileThunk`.
- Result: A direct overlap limit of one did not change the failure, so concurrent kernel slot reuse is not the cause.
- Action: Stop the parent when Iris started attempt one.
- Decision: Restore overlap four and isolate the 5 GiB ragged activation transport without the model.

### 2026-08-02 17:52 UTC - Focused EP64 transport probe

- Shape: Each rank sends a 5 GiB bfloat16 buffer with 524,288 rows and 5,120 elements per row.
- Pattern: Every rank sends equal 8,192-row segments to all 64 ranks.
- Check: Each receiver samples the first, middle, and last column and verifies the source rank for every row.
- Runtime: Use the rebuilt fixed wheels, NCCL 2.30.7, and XLA's direct device kernel.
- Purpose: Remove model, optimizer, grouped GEMM, and QB work from the failing path.

### 2026-08-02 17:54 UTC - The 5 GiB probe passes one local LSA domain

- Setup: One process used four local GB200 GPUs, the rebuilt PJRT wheel, and NCCL 2.30.7.
- Result: The probe passed in 3.891037 seconds with checksum `2359296` and zero sampled mismatches.
- Interpretation: Large input and output offsets are correct inside one local LSA domain.
- Next gate: Run the same shape at EP64 to exercise the multi-node path without training work.

### 2026-08-02 17:54 UTC - MNEP-025 focused rack probe contract

- Goal: Execute and verify one balanced 5 GiB ragged all-to-all on one NVL72.
- Run ID: `mnep-025-ragged-5g-probe-20260802-1754`.
- Snapshot: `mnep-025-ragged-5g-probe-20260802-1754`.
- Shape: Use 524,288 rows per rank, 5,120 bfloat16 elements per row, and 64 equal peer segments.
- Gate: Require attempt-zero completion, checksum `49545216`, zero sampled mismatches, and no transport error.
- Stop criteria: Stop on any retry, memory error, transport error, or value mismatch.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 3600 --max-retries 0 --job-name mnep-025-ragged-5g-probe-20260802-1754-coord -- python -m experiments.grug.moe_hero_ep.ragged_device_probe --run-id mnep-025-ragged-5g-probe-20260802-1754 --rows-per-rank 524288 --row-elements 5120 --moonep-jax-wheel-build lsa-nccl-2307-20260802`.

### 2026-08-02 18:01 UTC - MNEP-025 is inconclusive

- Result: Task 0 received SIGKILL after 28.17 seconds. The other 15 tasks stopped through coscheduling.
- Difference: This run did not report the CUDA illegal-address error from the training probes.
- Interpretation: The 5 GiB shape can exceed a process or pod resource limit before it isolates the direct transport fault.
- Next gate: Reduce each rank to 640 MiB while keeping every local and remote peer active.

### 2026-08-02 18:01 UTC - MNEP-026 small balanced rack probe contract

- Goal: Execute one balanced 640 MiB ragged all-to-all on one NVL72.
- Run ID: `mnep-026-ragged-640m-probe-20260802-1801`.
- Shape: Use 65,536 rows per rank, 5,120 bfloat16 elements per row, and 64 equal peer segments.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.
- Stop criteria: Stop on any memory error, transport error, or value mismatch.

### 2026-08-02 18:04 UTC - MNEP-026 confirms a host resource fault

- Result: Task 0 received SIGKILL during NCCL communicator setup after 26.7 seconds.
- Evidence: A peer then reported that task 0 closed the bootstrap socket during `ncclCommInitRankConfig`.
- Interpretation: The probe did not reach compilation or the direct transport kernel.
- Cause: The probe requested 32 GiB of host memory. The training workers request 256 GiB.
- Fix: Match the probe CPU, host memory, and disk requests to the training workers.
- Next gate: Repeat the 640 MiB probe with the corrected worker resources.

### 2026-08-02 18:06 UTC - MNEP-027 corrected rack probe contract

- Goal: Execute one balanced 640 MiB ragged all-to-all on one NVL72 with the corrected worker resources.
- Run ID: `mnep-027-ragged-640m-resource-fixed-20260802-1806`.
- Snapshot: `mnep-027-ragged-640m-resource-fixed-20260802-1806` at `3e128c559`.
- Shape: Use 65,536 rows per rank, 5,120 bfloat16 elements per row, and 64 equal peer segments.
- Resources: Request 32 CPUs, 256 GiB of host memory, and four GB200 GPUs for each worker.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.

### 2026-08-02 18:11 UTC - MNEP-027 isolates the remote-domain failure

- Topology: All 16 workers ran in NVLink domain `DH1-125-US-EAST-08A`.
- Result: A lower-domain rank reported 98,304 sampled mismatches before task 10 failed with a CUDA illegal address.
- Interpretation: The mismatch count equals all three sampled columns for the 32,768 rows from the remote NVLink domain.
- Interpretation: Local-domain copies are correct. Cross-domain GIN writes do not produce valid output.
- Source comparison: NVIDIA's hybrid all-to-all example reserves only hybrid barriers and GIN signals.
- Source difference: XLA also reserves separate LSA and rail-GIN barriers that its hybrid kernel does not use.
- Next gate: Build XLA with the device communicator requirements matched to NVIDIA's hybrid example.

### 2026-08-02 18:17 UTC - Hybrid-resource PJRT build

- Patch: Reserve hybrid barriers and GIN signals, but do not reserve unused separate LSA and rail-GIN barriers.
- Source: JAX `f9f6bbace`, XLA `5d53e1e`, NCCL 2.30.7, CUDA 13.0, cuDNN 9.12, and `sm_100`.
- PJRT SHA-256: `ad8ee4dff204460f10bff5eb468957b332131203b628bf02ad2bcc0fdff73d0f`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-resources-20260802`.
- Local gate: The four-GPU 640 MiB probe passed in 3.456048 seconds with checksum `294912` and zero mismatches.
- Next gate: Run the same balanced probe at EP64 on one NVL72.

### 2026-08-02 18:20 UTC - MNEP-028 hybrid-resource rack probe contract

- Goal: Test the corrected hybrid device communicator with a balanced cross-domain transfer.
- Run ID: `mnep-028-ragged-hybrid-resources-20260802-1820`.
- Snapshot: `mnep-028-ragged-hybrid-resources-20260802-1820` at `1c9607519`.
- Shape: Use 65,536 rows per rank, 5,120 bfloat16 elements per row, and 64 equal peer segments.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.

### 2026-08-02 18:23 UTC - MNEP-028 falsifies the resource-layout hypothesis

- Result: All eight workers for ranks 32 through 63 reported CUDA illegal addresses after communicator setup.
- Result: The lower-domain workers stayed active until Iris stopped them through coscheduling.
- Interpretation: Removing the two unused device resources did not change the cross-domain failure.
- Next gate: Reduce each remote GIN put from 10 MiB to 1 MiB without changing the peer set.

### 2026-08-02 18:23 UTC - MNEP-029 transfer-size gate

- Goal: Test whether a 1 MiB GIN put crosses the NVLink-domain boundary correctly.
- Run ID: `mnep-029-ragged-64m-probe-20260802-1823`.
- Snapshot: `mnep-028-ragged-hybrid-resources-20260802-1820` at `1c9607519`.
- Shape: Use 65,536 rows per rank, 512 bfloat16 elements per row, and 64 equal peer segments.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.

### 2026-08-02 18:25 UTC - MNEP-029 fails with 1 MiB puts

- Result: The eight workers for ranks 32 through 63 failed again. The lower-domain workers stayed active.
- Interpretation: A 1 MiB per-peer transfer does not remove the directional cross-domain failure.
- Next gate: Test 4 KiB per peer, which matches the scale of NVIDIA's educational hybrid example.

### 2026-08-02 18:25 UTC - MNEP-030 minimum transfer gate

- Goal: Test whether the cross-domain GIN path can move any payload for this XLA allocation.
- Run ID: `mnep-030-ragged-256k-probe-20260802-1825`.
- Shape: Use 65,536 rows per rank, two bfloat16 elements per row, and 64 equal peer segments.
- Per-peer payload: 4 KiB.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.

### 2026-08-02 18:29 UTC - MNEP-030 rules out payload size

- Result: The 4 KiB per-peer case failed with the same upper-domain CUDA illegal addresses.
- Interpretation: The fault does not depend on the payload size from 4 KiB through 10 MiB.
- Next gate: Replace XLA's deprecated strong legacy completion signal with the explicit weak signal from NVIDIA's example.

### 2026-08-02 18:29 UTC - Weak-signal PJRT build

- Patch: Use `ncclGin_WeakSignalInc` for each remote put.
- PJRT SHA-256: `c71148f3901030525093480bbdf6582d255d7b34af5564a636ac409b24de1ffa`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-weak-20260802`.
- Local gate: The four-GPU 640 MiB probe passed in 3.867382 seconds with checksum `294912` and zero mismatches.
- Next gate: Repeat the 4 KiB and 10 MiB per-peer cases at EP64.

### 2026-08-02 18:32 UTC - MNEP-031 weak-signal rack probe contract

- Goal: Test explicit weak GIN signals with the minimum 4 KiB per-peer payload.
- Run ID: `mnep-031-ragged-hybrid-weak-20260802-1832`.
- Snapshot: `mnep-031-ragged-hybrid-weak-20260802-1832` at `81a394ee4`.
- Gate: Require attempt-zero completion, checksum `6193152`, zero sampled mismatches, and no transport error.

### 2026-08-02 18:39 UTC - MNEP-031 rules out weak signals

- Result: The workers for ranks 32 through 63 failed with the same CUDA illegal address.
- Result: Explicit weak GIN completion signals did not change the failure.
- Next gate: Confirm the physical topology and inspect NCCL's LSA team selection.

### 2026-08-02 18:44 UTC - Topology correction

- Correction: All 64 GPUs are in one physical NVL72 and one MNNVL fabric.
- NCCL reports one 64-rank clique, one 64-rank NVLink domain, and four local ranks on each worker.
- NCCL still creates a four-rank LSA team because `p2pCrossClique` is false and `computeLsaSize` uses the worker-size greatest common divisor.
- XLA therefore uses LSA for three peers and GIN for the other 60 peers. This is a logical transport split, not a physical rack split.
- The earlier `remote-domain`, `lower-domain`, and `upper-domain` terms in this logbook refer to this logical rank split.

### 2026-08-02 19:05 UTC - Full-MNNVL LSA runtime

- Source: Clone XLA at `5d53e1e40cd` into `.agents/tmp/xla-5d53` for direct source inspection.
- Finding: XLA follows the LSA team size that NCCL exports. It cannot use direct LSA pointers for peers outside that team.
- Patch: Extend NCCL's LSA team to all ranks when the communicator covers one complete MNNVL domain.
- Patch file: `experiments/grug/moe_hero_ep/nccl_patches/0001-use-full-mnnvl-domain-for-lsa.patch`.
- Build: NCCL 2.30.7 at `73cf112`, CUDA 13.0.48, CCCL 2.8.5, GCC 14, and `sm_100`.
- Runtime SHA-256: `e38471a61852b2ec56265a1d39b866a33d65b340498380c1ba2101c77e729b38`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/nccl-2.30.7-full-mnnvl-lsa-20260802/libnccl.so.2.30.7`.
- Local gate: The four-GPU probe passed in 3.337760 seconds with checksum `294912` and zero mismatches.
- Caveat: NCCL disables MNNVL for the one-worker probe, so only the rack gate can test the new team selection.
- Next gate: Run the minimum balanced probe at EP64 and require the `LSA extended to full MNNVL domain` log record.

### 2026-08-02 19:12 UTC - MNEP-032 setup-order failure

- Run ID: `mnep-032-ragged-full-mnnvl-lsa-20260802-1906`.
- Result: Task 10 failed with the known CUDA illegal address before the output check.
- Cause: The final CUDA setup step reinstalled `nvidia-nccl-cu13` after the fixed runtime was copied into the virtual environment.
- Interpretation: MNEP-032 used stock NCCL and did not test the full-MNNVL LSA patch.
- Fix: Run CUDA library restoration before the fixed JAX and NCCL artifact installation.
- Next gate: Repeat the same EP64 probe and verify the patched NCCL version and LSA log record.

### 2026-08-02 19:27 UTC - MNEP-033 rejects the full-MNNVL LSA extension

- Run ID: `mnep-033-ragged-full-mnnvl-lsa-20260802-1914`.
- Snapshot: `mnep-033-ragged-full-mnnvl-lsa-20260802-1914` at `758d562cc`.
- Result: The corrected setup installed the patched NCCL runtime after the CUDA setup.
- Result: Task 10 failed with the same CUDA illegal address. Iris stopped the other 15 tasks through coscheduling.
- Interpretation: Extending the LSA team to all 64 ranks is not valid for this communicator.
- Decision: Keep NCCL's four-rank LSA team and repair XLA's hybrid GIN path.
- Next gate: Test GIN barriers without memory fences, as used by NVIDIA's NCCL 2.30.7 hybrid example.

### 2026-08-02 19:35 UTC - XLA source audit changes the next hypothesis

- Source: Fetch OpenXLA main at `92f13a5889` and compare it with pinned XLA `5d53e1e40cd`.
- Result: No later XLA commit changes the device ragged all-to-all kernel or thunk.
- Result: NCCL 2.30.7 defines `ncclGinFenceLevel::Relaxed` as an alias for `None`. A fence-name build has no semantic change.
- Finding: XLA allows 64 CTAs for this kernel. NVIDIA's NCCL 2.30.7 GIN all-to-all example uses 16 CTAs.
- Finding: An OpenXLA PR 41903 review also flags 64 CTAs as unusually large compared with NCCL EP.
- Patch: Cap the device kernel and its reserved world barriers at 16 CTAs.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0003-cap-device-kernel-at-16-ctas.patch`.
- PJRT SHA-256: `6a87208443b820f2e37c6e4517d22d7b9d1f143b224b1c6d91550d9cae604b2e`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-cta16-20260802`.
- Local gate: The four-GPU 640 MiB probe passed in 3.553119 seconds with checksum `294912` and zero mismatches.
- Next gate: Repeat the 4 KiB EP64 probe on one NVL72.

### 2026-08-02 19:43 UTC - MNEP-034 rules out CTA pressure

- Run ID: `mnep-034-ragged-hybrid-cta16-20260802-1940`.
- Snapshot: `mnep-034-ragged-hybrid-cta16-20260802-1940` at `3aa760408`.
- Result: Task 10 failed with the same CUDA illegal address. Iris stopped the other 15 tasks through coscheduling.
- Interpretation: Reducing the launch and barrier grid from 64 to 16 CTAs does not fix the cross-node path.
- Next gate: Replace XLA's legacy hybrid barrier with the NCCL 2.30 world-GIN barrier API.

### 2026-08-02 19:46 UTC - NCCL 2.30 world-GIN PJRT build

- Finding: XLA PR 41903 still uses `barrierCount` and `ncclBarrierSession`, which NVIDIA documents as the pre-2.30 compatibility path.
- Patch: Use `worldGinBarrierCount` and `ncclGinBarrierSession` for NCCL 2.30 or newer.
- Patch: Keep a separate LSA barrier to order local P2P copies before each world-GIN barrier.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0004-use-nccl-230-world-gin-barriers.patch`.
- PJRT SHA-256: `c2521e40e7cd87f445b42445ba5221771b89c754caf5fa81c92a7b47add6cb31`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-world-gin-20260802`.
- Local gate: The four-GPU 640 MiB probe passed in 3.785112 seconds with checksum `294912` and zero mismatches.
- Next gate: Repeat the 4 KiB EP64 probe on one NVL72.

### 2026-08-02 19:53 UTC - MNEP-035 rules out the legacy barrier API

- Run ID: `mnep-035-ragged-world-gin-20260802-1948`.
- Snapshot: `mnep-035-ragged-world-gin-20260802-1948` at `85b74dcce`.
- Result: Task 10 failed with the same CUDA illegal address on ranks 40 through 43.
- Result: NCCL formed one 64-rank communicator on one NVL72 and completed initialization.
- Interpretation: The NCCL 2.30 world-GIN barrier does not fix the device transport fault.
- Next gate: Disable the data transfer but keep both barriers. A normal value mismatch will show that the barriers complete.

### 2026-08-02 20:06 UTC - Barrier-only XLA diagnostic build

- Patch: Keep the entry and exit barriers, but do not run LSA copies, GIN puts, signal waits, or GIN flushes.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0005-disable-transfer-for-barrier-isolation.patch`.
- PJRT SHA-256: `d3e391455196736d8793e4a983c63ed1644fe90e8ce87e9f56635fa43c83196c`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-barrier-only-20260802`.
- Expected rack result: The kernel completes and the probe reports sampled value mismatches.

### 2026-08-02 20:15 UTC - MNEP-036 isolates the fault to barrier setup

- Run ID: `mnep-036-ragged-barrier-only-20260802-2008`.
- Snapshot: `mnep-036-ragged-barrier-only-20260802-2008` at `22add9a56`.
- Result: Task 10 failed with the same CUDA illegal address.
- Result: The diagnostic kernel did not run copies, puts, signal reads, signal waits, or GIN flushes.
- Interpretation: The fault is in the world-GIN object or barrier path.
- Next gate: Keep only the LSA barriers and remove the GIN object and world barrier.

### 2026-08-02 20:17 UTC - LSA-barrier-only XLA diagnostic build

- Patch: On a split LSA team, run only the entry and exit LSA barriers.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0006-use-only-lsa-barriers-for-isolation.patch`.
- PJRT SHA-256: `2a9411320bbc9fc36ce21c60eb9a2825b3c54b2a6afcbb75cbfa0fb9ed3a1023`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-lsa-barrier-only-20260802`.
- Expected rack result: The kernel completes and the probe reports sampled value mismatches.

### 2026-08-02 20:23 UTC - MNEP-037 reproduces with only the LSA barrier

- Run ID: `mnep-037-ragged-lsa-barrier-only-20260802-2018`.
- Snapshot: `mnep-037-ragged-lsa-barrier-only-20260802-2018` at `8add8fae7`.
- Result: Task 10 failed with the same CUDA illegal address.
- Interpretation: A split communicator fails before any data transfer or world-GIN barrier.
- Node check: Kubernetes reports node `s9jtxs64` as ready, with no GPU, NVLink, memory, or PCI fault condition.
- Next gate: Return from the device kernel before it reads the NCCL device communicator.

### 2026-08-02 20:25 UTC - Empty XLA device-kernel diagnostic build

- Patch: Return at the start of the ragged all-to-all device kernel.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0007-return-from-device-kernel-for-isolation.patch`.
- PJRT SHA-256: `458f3277349276d5a13a3c652d625339f34c8602ed5a230f9bea861e1005fa44`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-kernel-20260802`.
- Expected rack result: The kernel completes and the probe reports sampled value mismatches.

### 2026-08-02 20:30 UTC - MNEP-038 moves the fault before device code

- Run ID: `mnep-038-ragged-noop-kernel-20260802-2026`.
- Snapshot: `mnep-038-ragged-noop-kernel-20260802-2026` at `383312217`.
- Result: Task 10 failed with the same CUDA illegal address.
- Result: The device kernel returned before it read the NCCL device communicator or any buffer.
- Interpretation: XLA's host-side device communicator or symmetric-window setup causes the fault.
- Next gate: Request only LSA device resources and keep the device kernel empty.

### 2026-08-02 20:33 UTC - LSA-only host-resource XLA diagnostic build

- Patch: Request only LSA device resources during clique preparation and execution.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0008-request-only-lsa-device-resources.patch`.
- PJRT SHA-256: `c2b5461dfbfd53dfcebf76af16eb55736c02aa934020e81edc2477d59851f973`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-lsa-host-20260802`.
- Expected rack result: The kernel completes and the probe reports sampled value mismatches.

### 2026-08-02 20:38 UTC - MNEP-039 rules out GIN resource setup

- Run ID: `mnep-039-ragged-noop-lsa-host-20260802-2034`.
- Snapshot: `mnep-039-ragged-noop-lsa-host-20260802-2034` at `285c4838f`.
- Result: Task 10 failed with the same CUDA illegal address.
- Result: XLA requested only LSA device resources and launched an empty kernel.
- Interpretation: GIN resource creation is not the source of the fault.
- Next gate: Return before `RunCollective` gets device state or launches a kernel.

### 2026-08-02 20:39 UTC - Empty RunCollective XLA diagnostic build

- Patch: Return from `RunCollective` before buffer lookup, device communicator lookup, or kernel launch.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0009-return-before-device-collective-execution.patch`.
- PJRT SHA-256: `ba8fb4ba686e18ec710f6495e7f4e8d407cadf5076e8786a06314d30443e6eb4`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-run-collective-20260802`.
- Expected rack result: The collective returns and the probe reports sampled value mismatches.

### 2026-08-02 20:44 UTC - MNEP-040 clears prepare-time setup

- Run ID: `mnep-040-ragged-noop-run-collective-20260802-2041`.
- Snapshot: `mnep-040-ragged-noop-run-collective-20260802-2041` at `cf43590dc`.
- Result: The collective completed without a CUDA fault.
- Result: Task 0 reported the expected 189 sampled value mismatches.
- Interpretation: Clique preparation, symmetric allocation, and communicator initialization are safe.
- Next gate: Get the LSA device communicator, then return before kernel launch.

### 2026-08-02 20:46 UTC - Return-after-device-communicator XLA build

- Patch: Complete buffer lookup and LSA device-communicator lookup, then return before kernel launch.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0010-return-after-device-communicator-lookup.patch`.
- PJRT SHA-256: `4b1ddc5011aa44126ff711c4efe2fe889ef43d380e31609e560437cc6cbee0cd`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-device-comm-20260802`.
- Expected rack result: The collective completes and the probe reports sampled value mismatches.

### 2026-08-02 21:54 UTC - MNEP-041 isolates a smaller host path

- Run ID: `mnep-041-ragged-noop-after-device-comm-20260802-2048`.
- Snapshot: `mnep-041-ragged-noop-after-device-comm-20260802-2048` at `8c7f23d9f`.
- Result: Task 10 failed with the same CUDA illegal address after 27 seconds.
- Result: The device kernel did not start.
- Interpretation: The fault occurs before the return after `GetDeviceComm`.
- Next gate: Return before `GetDeviceComm`, but after all prior host checks.

### 2026-08-02 21:55 UTC - Return-before-device-communicator XLA build

- Patch: Complete buffer and symmetric-memory lookup, then return before `GetDeviceComm`.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0011-return-before-device-communicator-lookup.patch`.
- PJRT SHA-256: `b3c2d632c70628d026af8bc87f09d48c405b03a57dcba69880327b66ecd748d2`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-before-device-comm-20260802`.
- Expected rack result: The collective completes and the probe reports sampled value mismatches.

### 2026-08-02 22:04 UTC - MNEP-042 moves the boundary before `GetDeviceComm`

- Run ID: `mnep-042-ragged-noop-before-device-comm-20260802-2200`.
- Snapshot: `mnep-042-ragged-noop-before-device-comm-20260802-2200` at `cc813e39a`.
- Result: Task 10 failed with the same CUDA illegal address after 21 seconds.
- Result: The code returned before `GetDeviceComm` and before the device kernel.
- Interpretation: `GetDeviceComm` is not the source of the fault.
- Echo milestone: `#1864`.
- Next gate: Convert the device buffers, then return before the next host lookup.

### 2026-08-02 22:05 UTC - Return-after-buffer-conversion XLA build

- Patch: Convert the five device buffers, then return before all other host checks.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0012-return-after-device-buffer-conversion.patch`.
- PJRT SHA-256: `3475c89c11683f1106290ab487f55087bd924b6de9c787153f83f9403ce9b2bf`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-buffer-conversion-20260802`.
- Expected rack result: The collective completes and the probe reports sampled value mismatches.

### 2026-08-02 22:11 UTC - MNEP-043 clears device-buffer conversion

- Run ID: `mnep-043-ragged-noop-after-buffer-conversion-20260802-2207`.
- Snapshot: `mnep-043-ragged-noop-after-buffer-conversion-20260802-2207` at `8fcd363b6`.
- Result: The collective completed without a CUDA fault.
- Result: Task 1 reported the expected 189 sampled value mismatches.
- Interpretation: `ConvertToDeviceBuffers` is safe.
- Echo milestone: `#1865`.
- Next gate: Include peer-access state and symmetric-memory lookup, then return before `Comm::NumRanks`.

### 2026-08-02 22:12 UTC - Return-after-symmetric-lookup XLA build

- Patch: Resolve both symmetric-memory windows, then return before `Comm::NumRanks`.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0013-return-after-symmetric-memory-lookup.patch`.
- PJRT SHA-256: `a5c642dd2476faf51287ad7cd5d4734b27d9ab5cc4fd1cb2353d575de8e536a8`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-symmetric-lookup-20260802`.
- Expected rack result: The collective completes and the probe reports sampled value mismatches.

### 2026-08-02 22:18 UTC - MNEP-044 finds a conditional fallback

- Run ID: `mnep-044-ragged-noop-after-symmetric-lookup-20260802-2214`.
- Snapshot: `mnep-044-ragged-noop-after-symmetric-lookup-20260802-2214` at `f42c97c20`.
- Result: Task 10 failed with the same CUDA illegal address after 27 seconds.
- Finding: The diagnostic return was inside the device-path predicate.
- Interpretation: A false predicate lets a rank start the older fallback path.
- Next gate: Replace the fallback with an error that reports all device-path predicates.

### 2026-08-02 22:19 UTC - Device-predicate report XLA build

- Patch: Report the four device-path predicate values instead of starting the fallback.
- Patch file: `experiments/grug/moe_hero_ep/xla_patches/0014-report-device-path-predicate.patch`.
- PJRT SHA-256: `3a4cf444687179fea6545a25cec73ec930dd875ec1bd033e8495262c02430042`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-report-device-predicate-20260802`.
- Expected rack result: The probe reports the false predicate without a CUDA fault.

### 2026-08-02 22:28 UTC - MNEP-045 finds the missing symmetric-buffer flag

- Run ID: `mnep-045-ragged-report-device-predicate-20260802-2221`.
- Snapshot: `mnep-045-ragged-report-device-predicate-20260802-2221` at `e31b9d9c0`.
- Result: XLA reported `uses_device_kernel=0`, `supports_device_comm=1`, `has_collective_memory=1`, and `has_lsa_size=1`.
- Finding: `UsesDeviceKernel()` requires both the device-kernel option and the NCCL symmetric-buffer option.
- Finding: Marin set the device-kernel option but did not set the symmetric-buffer option.
- Fix: Add `--xla_gpu_experimental_enable_nccl_symmetric_buffers=true` to the MoonEP XLA defaults.
- Local gate: All 37 MoonEP tests pass.
- Next gate: Use the current diagnostic PJRT at EP64. The expected value mismatch proves that XLA selected the device path and completed symmetric-memory lookup.

### 2026-08-02 22:32 UTC - MNEP-046 proves direct-path selection

- Run ID: `mnep-046-ragged-enable-symmetric-20260802-2229`.
- Snapshot: `mnep-046-enable-symmetric-buffers-20260802-2228` at `5ff74aba4`.
- Result: NCCL created a 188 GB symmetric virtual-address window for the 64-rank communicator.
- Result: XLA entered the direct device path, completed both symmetric-memory lookups, and returned before the diagnostic kernel.
- Gate: The probe reported the expected 189 sampled value mismatches and no CUDA fault.
- Interpretation: The earlier fallback fault did not test XLA's new device kernel. The missing symmetric-buffer flag caused the fallback.
- Echo milestone: `#1873`.
- Next gate: Run the clean XLA `5d53e1e` and NCCL 2.30.7 wheel with the corrected flags.

### 2026-08-02 22:33 UTC - MNEP-047 clean direct-device probe contract

- Goal: Verify a balanced ragged all-to-all through clean XLA on all 64 GPUs of one NVL72.
- Run ID: `mnep-047-ragged-clean-direct-20260802-2233`.
- Runtime: JAX `f9f6bbace`, XLA `5d53e1e`, NCCL 2.30.7, and no XLA source patch.
- Shape: 64 rows for each rank, 32 bfloat16 elements for each row, and one row for each peer.
- Gate: Require attempt-zero completion, checksum `6048`, zero sampled mismatches, and no transport error.

### 2026-08-02 23:03 UTC - MNEP-047 proves clean EP64 transport correctness

- Result: The clean XLA and NCCL 2.30.7 child completed on attempt zero in 1 minute and 50 seconds.
- Correctness: Checksum `6048`, zero sampled mismatches, and no transport error.
- Kernel time: The compiled 64-rank probe completed in 8.607078 seconds.
- Interpretation: The runtime flag fix is sufficient for the balanced direct-device transport probe. No XLA source patch is required for this gate.
- Next gate: Run three full MoonEP and global QB training steps through the same clean direct path.

### 2026-08-02 23:03 UTC - MNEP-048 full-program direct smoke contract

- Goal: Execute three full MoonEP and global QB training steps through the clean EP64 direct path.
- Run ID: `mnep-048-clean-direct-smoke-3-20260802-2303`.
- Runtime: JAX `f9f6bbace`, XLA `5d53e1e`, NCCL 2.30.7, an 84% main pool, and four in-flight collectives.
- Config: Global histogram QB with 1,000 bins, 128-token MoonEP padding, and QuACK grouped GEMM.
- Gate: Require attempt-zero completion, finite loss, zero dropped assignments, and no transport error.

### 2026-08-02 23:29 UTC - MNEP-048 fails deterministically at step two

- Result: The child made three complete failed attempts because the common dispatch helper defaulted to three failure retries.
- Failure: Every rank reported a `NaN` loss at step two on all three attempts.
- Correction: The coordination-service connection errors followed rank exit and were not the root failure.
- Action: Stop the fourth attempt and set `max_retries_failure=0` in the MoonEP training dispatch.
- Interpretation: The balanced direct-transport probe is not a sufficient full-program correctness gate.
- Next gate: Set the collective overlap limit to one. This tests whether concurrent direct kernels corrupt shared device-communicator state.

### 2026-08-02 23:51 UTC - MNEP-049 rules out the overlap limit

- Treatment: Reduce the direct collective overlap limit from four to one.
- Result: The run still reported a NaN loss at step two.
- Interpretation: The overlap limit does not order all remote writes before their consumers.
- Next gate: Add compiled finite checks at the main training boundaries.

### 2026-08-02 23:59 UTC - MNEP-050 passes with full boundary checks

- Run ID: `mnep-050-finite-diagnostics-3-20260802-2359`.
- Result: All three steps completed with finite inputs, loss, gradients, updates, outputs, and QB values.
- Result: The final loss was `9.42`, and the run dropped no expert assignments.
- Interpretation: A compiled consumer changes the failing schedule.
- Next gate: Test a host wait without the compiled checks.

### 2026-08-03 00:07 UTC - MNEP-051 rules out a host wait

- Treatment: Wait for the complete training state after each step, with no compiled finite checks.
- Result: The run still reported a NaN loss at step two.
- Interpretation: The missing order is inside the compiled step. Host completion between steps is not sufficient.
- Next gate: Keep only the full-gradient finite consumer.

### 2026-08-03 00:15 UTC - MNEP-052 passes with a full-gradient consumer

- Run ID: `mnep-052-gradient-completion-3-20260803-0015`.
- Result: All three steps completed on attempt zero with finite gradients and a final loss of `9.7`.
- Result: Steady step times were 49 and 54 seconds.
- Interpretation: The full-gradient consumer can change the direct-kernel schedule, but it is not yet a data dependency of the optimizer.
- Next gate: Test a smaller expert-gradient consumer.

### 2026-08-03 00:23 UTC - MNEP-053 small expert consumer passes once

- Run ID: `mnep-053-expert-gradient-completion-3-20260803-0023`.
- Result: All three steps completed on attempt zero with finite sampled expert gradients and a final loss of `9.42`.
- Result: Steady step times were 48 to 49 seconds.
- Interpretation: This single pass did not prove that a narrow consumer orders all direct writes.
- Next gate: Capture a five-step profile with the narrow consumer.

### 2026-08-03 00:33 UTC - MNEP-054 rejects the narrow consumer

- Run ID: `mnep-054-direct-guard-profile-5-20260803-0033`.
- Result: Step one was finite. At step two, one worker reported a non-finite sampled expert gradient and a NaN loss.
- Interpretation: The three sampled expert tensors do not order all gradient writes.
- Next gate: Repeat the profile with the full-gradient consumer.

### 2026-08-03 00:45 UTC - MNEP-055 rejects an independent full scan

- Run ID: `mnep-055-full-gradient-profile-5-20260803-0045`.
- Result: All 16 workers reported finite gradients at step one and non-finite gradients at step two. The loss was NaN.
- Interpretation: An independent full-gradient output does not make the optimizer wait for that reduction.
- Decision: Make every gradient passed to the optimizer depend on the full finite reduction.

### 2026-08-03 00:57 UTC - MNEP-056 data-dependent guard contract

- Run ID: `mnep-056-optimizer-gradient-gate-profile-5-20260803-0057`.
- Snapshot: `mnep-056-optimizer-gradient-gate-20260803-0057` at `eb2f5ad38`.
- Treatment: Gate every optimizer gradient on the full-gradient finite result before the update.
- Gate: Require five finite steps on attempt zero, no dropped assignments, and a step-three XPlane profile.

### 2026-08-03 01:11 UTC - MNEP-056 passes four steps before profile timeout

- Correctness: All 16 workers reported finite gradients through step four. This passes the old step-two failure point.
- Profile: Process zero captured all four local GPUs and retained about two million events. CUPTI then dropped more events.
- Failure: Profile flush took more than the 200-second distributed barrier limit. A non-tracing worker timed out before process zero reached the barrier.
- Interpretation: The terminal failure is in profile export. The run did not report a training NaN.
- Decision: Capture one representative GPU from process zero, then repeat the five-step gate.

### 2026-08-03 01:30 UTC - MNEP-057 device filter does not bound collection

- Run ID: `mnep-057-one-gpu-profile-5-20260803-0115`.
- Correctness: All 16 workers again reported finite gradients through step four.
- Profile: The tracer still reached two million events. CUPTI counts process events before it filters the selected GPU.
- Failure: The non-tracing workers timed out at the profile barrier while process zero flushed the capture.
- Decision: Limit activity and callback events to 250,000 and aggregate repeated kernels.

### 2026-08-03 01:31 UTC - MNEP-058 bounded profile contract

- Run ID: `mnep-058-bounded-profile-5-20260803-0131`.
- Snapshot: `mnep-058-bounded-profile-20260803-0131` at `1963ee40d`.
- Treatment: Keep the one-GPU filter, cap activity and callback events at 250,000, and aggregate repeated kernels.
- Gate: Require five finite steps, successful profile upload, and an operation-level profile summary.

### 2026-08-03 01:47 UTC - MNEP-058 keeps training finite but exceeds the profile barrier

- Correctness: All workers kept finite gradients and loss through step four for the third consecutive guarded rack run.
- Failure: Fifteen workers reached the export barrier. Process zero needed more than 200 seconds to flush 250,000 retained events.
- Interpretation: The terminal error is profile finalization. The training error did not return.
- Decision: Increase the profile barrier limit to 600 seconds and reduce each event cap to 100,000.

### 2026-08-03 01:49 UTC - MNEP-059 long profile barrier contract

- Run ID: `mnep-059-long-profile-barrier-5-20260803-0151`.
- Snapshot: `mnep-059-long-profile-barrier-20260803-0149` at `d975e2b7c`.
- Treatment: Use a 600-second profile barrier, retain at most 100,000 activity and callback events, aggregate repeated kernels, and keep the one-GPU filter.
- Gate: Require five finite steps, successful profile upload, and an operation-level profile summary.
- Submission note: The first coordinator stopped before GPU allocation because its version label was invalid. The corrected coordinator uses `mnep-059-dev`.

### 2026-08-03 02:05 UTC - MNEP-059 passes the rack correctness and profile gate

- Result: All 16 workers completed all five steps on attempt zero.
- Correctness: Every worker reported finite complete gradients at every step. The final logged loss was `8.18`, and the run dropped no expert assignments.
- Profile: Process zero wrote a 370 MB XPlane and a 65 MB trace, then uploaded the session to `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mnep-059-long-profile-barrier-5-20260803-0151`.
- Profile export: Finalization took about six minutes. The 600-second barrier let all workers complete.
- Decision: Use this run as the overlap-one correctness baseline. Test four in-flight direct collectives without profiler overhead.

### 2026-08-03 02:06 UTC - MNEP-060 overlap-four throughput contract

- Run ID: `mnep-060-direct-overlap-4-5-20260803-0206`.
- Treatment: Raise `--xla_gpu_experimental_parallel_collective_overlap_limit` from one to four through an explicit XLA flag.
- Gate: Require five finite steps on attempt zero and compare the steady step time with MNEP-059's 40 to 44 seconds.

### 2026-08-03 02:21 UTC - MNEP-060 rejects overlap four

- Result: All 16 workers completed five finite steps on attempt zero, with no dropped expert assignments.
- Timing: The four steady steps took 41, 43, 45, and 45 seconds. Their median was 44 seconds.
- Comparison: MNEP-059 took 40, 41, and 44 seconds outside its profile window. Its median was 41 seconds.
- Decision: Keep the direct collective overlap limit at one. A limit of four did not improve throughput.
- Estimate: The analytic ideal step time is 2.59 seconds. The current 41-second baseline is about 6.3% MFU, and the target 21.7% MFU needs a step time of at most 11.9 seconds.

### 2026-08-03 02:24 UTC - MNEP-059 profile summary lacks GPU events

- Result: In-cluster analysis read the complete 370 MB XPlane, with 20,461,033 complete events and no truncation signal.
- Limitation: The aggregated trace contains host events but no useful GPU operations or collectives. It cannot identify GPU hot spots.
- Decision: Disable host tracing and GPU aggregation for the next capture. Keep the one-GPU filter and the 100,000 activity and callback caps.
- Next gate: Remove one dispatch collective by packing each expert ID with its token payload, then repeat the rack correctness and timing test.

### 2026-08-03 02:28 UTC - MNEP-061 packed dispatch contract

- Run ID: `mnep-061-packed-dispatch-5-20260803-0228`.
- Snapshot: `mnep-061-packed-dispatch-20260803-0228` at `eca29a26f`.
- Treatment: Bit-pack each int32 expert ID into the token dtype and send both in one ragged all-to-all. This removes one dispatch collective from every MoE layer.
- Local gate: Exact ID recovery passed for BF16, FP16, and FP32 payloads. The MoonEP planner and abstract distributed lowering tests passed.
- Rack gate: Require five finite steps on attempt zero, no dropped assignments, and a lower steady step time than the 41-second MNEP-059 baseline.

### 2026-08-03 02:37 UTC - MNEP-061 passes four-GPU numerical parity

- Environment: The supplied four-GB200 pod, clean XLA `5d53e1e40cd`, and NCCL 2.30.7.
- Sync: The packed transport source in `/tmp/rav-codex-mnep061` had the same SHA-256 as the pushed source.
- Result: The existing dense-reference test passed for the output and the input, W13, and W2 gradients.
- Runtime: The test completed in 110.25 seconds, including compilation.
- Next gate: Complete the five-step EP64 rack run and compare its steady step time with MNEP-059.

### 2026-08-03 02:40 UTC - MNEP-061 rejects packed dispatch rows

- Result: All 16 workers completed five finite steps on attempt zero. The final loss was `8.63`, and the run dropped no expert assignments.
- Timing: The four steady steps took 52, 65, 62, and 64 seconds. Their median was 63 seconds.
- Comparison: MNEP-059's median was 41 seconds. Approximate MFU fell from 6.3% to 4.1%.
- Cause: The packed metadata changed the BF16 token row from 5,120 to 5,122 elements. This breaks the aligned transport row used by the direct device kernel.
- Decision: Reject and revert packed rows. Keep the 5,120-element token payload and reconstruct expert IDs from the global plan instead.

### 2026-08-03 02:43 UTC - MNEP-062 GPU profile contract

- Run ID: `mnep-062-gpu-profile-5-20260803-0243`.
- Snapshot: `mnep-062-gpu-profile-20260803-0243` at `759ec0c0e`.
- Treatment: Restore the aligned baseline transport. Capture one GPU step with host tracing and GPU event aggregation disabled.
- Limits: Keep one GPU, 100,000 activity events, 100,000 callback events, and the 600-second profile barrier.
- Gate: Require five finite steps, successful profile upload, and a kernel and collective time summary from the GPU timeline.

### 2026-08-03 02:50 UTC - MNEP-063 metadata-free dispatch contract

- Planned run ID: `mnep-063-rebuilt-metadata-5-20260803-0250`.
- Snapshot: `mnep-063-rebuilt-metadata-20260803-0250` at `09417ad7d`.
- Treatment: Sort each sender's assignments by destination and expert. Rebuild the received expert IDs from source and destination interval intersections in the shared allocation plan.
- Expected effect: Remove one int32 ragged all-to-all from every MoE layer while keeping the aligned 5,120-element BF16 token row.
- Local gate: The planner and abstract distributed lowering tests passed. The dense-reference output and input and weight gradient test passed on four GB200 GPUs in 117.99 seconds, including compilation.
- Rack gate: Require five finite steps on attempt zero, no dropped assignments, and a steady step time below the 41-second aligned baseline.

### 2026-08-03 02:55 UTC - MNEP-062 profile run is invalid

- Result: All 64 workers completed the first finite step. The run then stopped in the next direct collective while the four-GPU parity test used another tray in the same NVL72 rack.
- Evidence: After the parity test ended, all workers remained active but the sampled rack GPUs had 0% use and about 171 GB allocated memory. No worker failure or retry occurred.
- Decision: Stop MNEP-062. Do not run the development pod and an EP64 rack job at the same time. Run MNEP-063 next, then profile that exact implementation with exclusive rack use.

### 2026-08-03 03:04 UTC - MNEP-063 passes correctness but not the throughput gate

- Result: All 16 workers completed five finite steps on attempt zero. The final sampled losses were `8.78` and `8.17`, and the run dropped no expert assignments.
- Timing: The four steady steps took 40, 44, 45, and 42 seconds. Their median was 43 seconds, which is about 6.0% MFU.
- Comparison: MNEP-059 had a 41-second median and about 6.3% MFU. Removing the expert-ID collective did not give a measured improvement.
- Decision: Keep the metadata-free design because it reduces work without changing the aligned token row. Use a GPU timeline to find the much larger cost.

### 2026-08-03 03:05 UTC - MNEP-064 metadata-free GPU profile contract

- Run ID: `mnep-064-metadata-gpu-profile-5-20260803-0305`.
- Treatment: Capture completed step 3 from the exact MNEP-063 implementation. Disable host tracing and GPU event aggregation, and trace one GPU with 100,000 activity and callback events.
- Isolation: Keep the supplied development tray idle for the complete EP64 run.
- Gate: Require five finite steps, successful profile upload, and a kernel and collective time summary from the GPU timeline.

### 2026-08-03 03:24 UTC - MNEP-064 finds serialized planner work

- Result: All 16 workers completed five finite steps on attempt zero. Process zero uploaded the 3.4 MB XPlane and 0.6 MB trace to `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mnep-064-metadata-gpu-profile-5-20260803-0305`.
- Capture limit: The trace retained 29,680 complete events from the first part of the profiled step. Later activity buffers were dropped after the 100,000-event limit, so the trace does not give a full-step percentage.
- Finding: The retained section is dominated by repeated allocation-loop kernels. The same planner kernel names occur about 2,234 times before the capture limit. `_balance_owner_groups` runs its 66-step expert allocation loop once for each of 64 independent owner ranks.
- Decision: Batch the 64 independent owner allocations and keep the same 66-step greedy sequence for each owner.

### 2026-08-03 03:27 UTC - MNEP-065 batched planner contract

- Treatment: Replace the serialized per-owner allocation loop with one batched loop over all owner ranks. The allocation result stays equal to the MoonEP reference.
- Microbenchmark: On one GB200 GPU at the EP64 planner shape, median planner time fell from 53.53 ms to 1.61 ms, a 33.3x reduction.
- Correctness: All five CPU planner reference tests passed. The four-GPU dense-reference output and input and weight gradient test passed in 115.30 seconds, including compilation.
- Static checks: The required lint, formatting, license, syntax, and Pyrefly checks passed.
- Rack gate: Require five finite steps on attempt zero, no dropped assignments, and a steady step time below the 43-second MNEP-063 result.

### 2026-08-03 03:38 UTC - MNEP-065 raises MFU to about 9.6%

- Result: All 16 workers completed five finite steps on attempt zero. The final loss was `8.18`, and the run dropped no expert assignments.
- Timing: The four steady steps took 26, 27, 27, and 27 seconds. Their median was 27 seconds, which is about 9.6% MFU.
- Comparison: MNEP-063 had a 43-second median and about 6.0% MFU. Batched planning reduced median step time by 37% and raised MFU by about 60%.
- Memory: Sampled worker allocation fell from about 171 GB to about 160 GB.
- Decision: Keep batched planning. Replace the large token ragged collectives with a bounded standard all-to-all fast path and retain ragged transport as the exact fallback.

### 2026-08-03 03:47 UTC - MNEP-066 bounded token all-to-all contract

- Treatment: Pad each sender-to-rank token block to 9/8 of the mean block size and use one standard all-to-all for dispatch and combine. Compact the received blocks back to the exact MoonEP receiver order.
- Fallback: If any sender-to-rank block exceeds the fixed bound, all ranks use the exact ragged collective. The branch condition comes from the replicated send matrix.
- Correctness: The four-GPU dense-reference test passed for both the balanced fixed fast path and the full-skew ragged fallback. Both outputs and the input and weight gradients passed in 199.53 seconds, including two compilations.
- Local checks: Five planner tests passed, both GPU cases skipped on CPU as required, and the required lint, formatting, license, syntax, and Pyrefly checks passed.
- Rack gate: Require five finite steps on attempt zero, no dropped assignments, and a steady step time below the 27-second MNEP-065 result.

### 2026-08-03 04:00 UTC - MNEP-066 does not improve rack time

- Result: The coordinator and all 16 workers succeeded on attempt zero. The five training steps had finite loss and dropped no expert assignments.
- Timing: The observed steady steps took 28, 26, 28, and about 26 seconds. The median stayed near 27 seconds, or about 9.6% MFU.
- Decision: Reject the bounded token all-to-all as a performance change. Keep the batched planner result from MNEP-065 and profile the remaining device time before the next optimization.

### 2026-08-03 04:16 UTC - MNEP-067 post-planner profile contract

- Baseline: Restore the MNEP-065 exact ragged token transport and its 27-second median step time.
- Capture: Profile step three on process zero after the batched planner change. Increase the GPU activity and callback limits to one million events.
- Trace data: Enable the host trace and HLO metadata so that the full device timeline has named regions and compiler operation names.
- Gate: Require five finite steps, no dropped assignments, a successful profile upload, and a full-step kernel and collective summary.

### 2026-08-03 04:25 UTC - MNEP-067 finds weight transport as the main limit

- Result: All 16 workers completed five finite steps on attempt zero. The loss stayed finite, and the run dropped no expert assignments.
- Timing: The steady step time stayed near 27 seconds, or about 9.6% MFU.
- Profile: The complete 28.35-second trace contains 1,265,679 complete events with no suspected truncation. Communication used 56.1% of device time.
- Finding: Ragged all-to-all used 15.18 seconds across 576 calls. The six large W13 and W2 transfer families used 14.12 seconds; the six token transfer families used about 1.06 seconds.
- Decision: Add a global no-drop gate for the fixed-capacity all-to-all path. Use exact MoonEP when any sender-to-expert cell exceeds the fixed capacity.

### 2026-08-03 04:42 UTC - MNEP-068 passes both four-GPU branches

- Treatment: Use fixed all-to-all with capacity factor 1.1 only when the maximum sender-to-expert count fits the fixed cell. The branch predicate is equal on all expert ranks.
- Fallback: Use exact MoonEP when any sender-to-expert cell exceeds the fixed capacity.
- Correctness: The balanced fast-path case and the full-skew fallback case both matched the dense output and the input, W13, and W2 gradients on four GB200 GPUs.
- Runtime: Both cases passed in 54.36 seconds, including compilation.
- Rack gate: Require five finite steps on attempt zero, no dropped assignments, and at least 21.7% median MFU.

### 2026-08-03 04:49 UTC - MNEP-068 rejects the dynamic collective branch

- Result: Step 1 completed with finite loss and finite gradients. Step 2 produced a NaN loss and a non-finite gradient diagnostic. The job stopped without a retry.
- Timing: The step interval was 28 seconds. This matches exact MoonEP, so the 1.1 fixed-capacity gate did not pass for the rack routes.
- Cause: The stable exact MoonEP path became unstable only after it was placed in a runtime conditional with a different collective sequence. The four-GPU test did not find this multi-host fault.
- Decision: Remove the runtime conditional. Keep exact MoonEP as a static mode, and measure a separate static fixed-capacity mode before selecting a no-drop capacity for the QB run.

### 2026-08-03 04:57 UTC - MNEP-069 static capacity measurement contract

- Change: Exact MoonEP and QB-fixed all-to-all are separate compile-time modes. No runtime branch contains different collective sequences.
- Correctness: Both static modes matched the dense output and the input, W13, and W2 gradients on four GB200 GPUs. The test took 140.28 seconds, including compilation.
- Treatment: Run global bucketed QB with the static fixed schedule and capacity factor 1.1.
- Gate: Require five finite steps on attempt zero. Measure median MFU and the exact reported assignment overflow before selecting the final capacity.

### 2026-08-03 05:06 UTC - MNEP-069 rejects capacity factor 1.1

- Result: All 16 workers completed five finite steps on attempt zero. The loss stayed finite.
- Timing: Median MFU was `25.02%`, and the final step took `10.25` seconds.
- Invalid result: The run dropped `1,182,560,647` expert assignments, or `73.42%` of all assignments.
- Finding: One-step-late global QB did not keep sender-to-expert cells inside the 1.1 fixed capacity during this short optimizer schedule.
- Decision: Do not count the `25.02%` result. Keep exact MoonEP and remove the repeated world barriers from the XLA GIN transport.

### 2026-08-03 05:34 UTC - MNEP-070 sparse GIN transport contract

- Run ID: `mnep-070-sparse-gin-probe-20260803-0534`.
- Snapshot: `c57e4cc15`.
- Treatment: Use one GIN ready signal and one strong completion signal for each nonempty remote update. Use LSA barriers for local peers.
- Probe: Run one balanced direct-device ragged all-to-all across 64 GB200 GPUs with 64 rows per rank and 32 elements per row.
- Gate: Require checksum `6048`, zero sampled mismatches, attempt zero, and completion on all 16 workers.

### 2026-08-03 05:45 UTC - MNEP-070b lower-resource transport contract

- MNEP-070 did not receive a rack. Kueue excluded 126 of 205 nodes on CPU capacity and 78 nodes on GPU capacity.
- Action: Stop MNEP-070 before GPU admission. Reduce each probe worker from 32 CPUs, 256 GiB RAM, and 256 GiB disk to 8 CPUs, 64 GiB RAM, and 64 GiB disk.
- Run ID: `mnep-070b-sparse-gin-probe-20260803-0545` at `e8ec1b033`.
- Gate: Keep the same 64-GPU checksum, mismatch, and attempt-zero requirements.

### 2026-08-03 05:47 UTC - MNEP-070b passes the rack transport gate

- Result: All 16 workers succeeded on attempt zero. Each worker completed in 34 to 40 seconds, including environment setup and JAX startup.
- Correctness: The probe raises an error for any sampled mismatch. All workers exited with code zero, so all sampled values matched.
- Scheduling: The 8-CPU and 64-GiB worker request received a complete NVLink domain in less than one minute.
- Decision: Use the sparse GIN wheel for a five-step exact MoonEP run with global histogram QB.

### 2026-08-03 05:50 UTC - MNEP-071 exact sparse GIN contract

- Run ID: `mnep-071-exact-sparse-gin-5-20260803-0550` at `dca3852a2`.
- Treatment: Use exact MoonEP, global histogram QB with 1,000 bins, direct device transport, and the sparse GIN XLA wheel.
- Evidence: Send tracker data to `marin-community/rav_moe`. Keep finite gradient diagnostics enabled.
- Gate: Require five finite steps on attempt zero, no dropped expert assignments, and at least `21.7%` median MFU.

### 2026-08-03 05:52 UTC - MNEP-071b lower-CPU training contract

- MNEP-071 did not receive GPUs. Kueue excluded 130 of 205 nodes on the 32-CPU request, one node on RAM, and 74 nodes on GPU capacity.
- Action: Stop MNEP-071 before GPU admission. Keep 256 GiB RAM and reduce the worker request to eight CPUs.
- Run ID: `mnep-071b-exact-sparse-gin-5-20260803-0552` at `dca3852a2`.
- Gate: Keep the MNEP-071 correctness and MFU requirements.

### 2026-08-03 05:55 UTC - MNEP-071c lower-memory training contract

- MNEP-071b removed the CPU constraint, but Kueue excluded 125 nodes on the 256-GiB RAM request.
- Action: Stop MNEP-071b before GPU admission. Keep eight CPUs and reduce host RAM to 64 GiB.
- Run ID: `mnep-071c-exact-sparse-gin-5-20260803-0555` at `dca3852a2`.
- Gate: Keep the MNEP-071 correctness and MFU requirements. Treat any host out-of-memory error as a failed resource estimate.

### 2026-08-03 06:01 UTC - MNEP-071c rejects 64 GiB host RAM

- Result: All 16 workers started. Process zero received SIGKILL with exit 137 during XLA compilation, before step one.
- Scope: The failure occurred before the sparse GIN transport ran. It does not give a transport result.
- Decision: Retry with eight CPUs and 128 GiB host RAM. Keep the model, wheel, QB, and correctness gate unchanged.
- Retry: `mnep-071d-exact-sparse-gin-5-20260803-0601` at `dca3852a2`.

### 2026-08-03 06:09 UTC - MNEP-071d passes correctness but not MFU

- Result: All 16 workers completed five finite steps on attempt zero. The run dropped no expert assignments, and the final loss was `8.1779`.
- Throughput: Median MFU was `9.698%`, mean MFU was `9.663%`, and the final sampled step took `26.688` seconds. Throughput was `157,160.6` tokens per second.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-071d-exact-sparse-gin-5-20260803-0601).
- Comparison: The exact MNEP-067 baseline was about `9.6%` MFU. Sparse GIN did not remove the measured weight-transport cost.
- Decision: Keep the transport correctness result. Capture a full GPU profile before the next XLA change.

### 2026-08-03 06:09 UTC - MNEP-072 sparse GIN profile contract

- Run ID: `mnep-072-sparse-gin-profile-5-20260803-0609` at `24944a477`.
- Treatment: Repeat MNEP-071d and profile one complete step from step three on process zero.
- Trace: Enable host labels, HLO metadata, and one million GPU activity and callback events.
- Gate: Require five finite steps, no dropped assignments, successful profile upload, and a full ragged all-to-all time comparison with MNEP-067.

### 2026-08-03 06:28 UTC - MNEP-072 sparse GIN profile result

- Result: All 16 workers completed five finite exact steps with no dropped assignments.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-072-sparse-gin-profile-5-20260803-0609).
- Profile: [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmnep-072-sparse-gin-profile-5-20260803-0609).
- Measurement: Ragged all-to-all used `14.9604 s` across 576 calls and `54.6%` of the device time.
- Comparison: MNEP-067 used `15.1776 s` for the same 576 calls. The sparse handshake saved only `0.2172 s`.
- Decision: Keep the sparse handshake for correctness. Restore parallel GIN writes before another rack test.

### 2026-08-03 06:43 UTC - MNEP-073 multi-context GIN probe contract

- Run ID: `mnep-073-multicontext-gin-probe-20260803-0643` at `e0214a306`.
- Treatment: Divide each remote row across all GIN contexts. Use a world barrier and one weak completion signal for each chunk.
- Shape: Use four updates for each peer, 256 rows for each rank, and 5,120 BF16 elements for each row.
- Local gate: The XLA build passed. The four-GPU exact MoonEP output and input, W13, and W2 gradient test passed.
- Rack gate: Require attempt-zero completion, checksum `24192`, zero sampled mismatches, and success on all 16 workers.

### 2026-08-03 06:48 UTC - MNEP-073 passes the rack transport gate

- Result: All 16 workers succeeded on attempt zero in `43.31 s`, including setup and JAX startup.
- Correctness: The checker raises for any sampled mismatch. All workers returned code zero, so the checksum is `24192` with zero sampled mismatches.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-073-multicontext-gin-probe-20260803-0643-coord).
- Decision: Use the same wheel for a five-step exact MoonEP MFU gate.

### 2026-08-03 06:48 UTC - MNEP-074 multi-context MFU contract

- Run ID: `mnep-074-multicontext-gin-5-20260803-0648`.
- Treatment: Run exact MoonEP and global histogram QB on EP64. Divide W13, W2, and token transfers across all GIN contexts.
- Resources: Use 16 workers with four GB200 GPUs, eight CPUs, and 128 GiB host RAM for each worker.
- Gate: Require five finite steps, no dropped assignments, attempt zero, and at least `21.7%` median MFU.

### 2026-08-03 06:59 UTC - MNEP-074 passes correctness but not MFU

- Result: All 16 workers completed five steps on attempt zero. All gradient checks were finite, and the final loss was `8.1781`.
- Routing: The run dropped zero expert assignments.
- Throughput: Median MFU was `10.004%`, mean MFU was `9.976%`, and the final sampled step took `25.752 s`.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-074-multicontext-gin-5-20260803-0648).
- Comparison: Median MFU increased from the MNEP-071d result of `9.698%`, but the result did not meet the `21.7%` gate.
- Decision: Keep the multi-context correctness result. Profile one complete step before the next transport change.

### 2026-08-03 07:01 UTC - MNEP-075 multi-context profile contract

- Run ID: `mnep-075-multicontext-gin-profile-5-20260803-0701`.
- Treatment: Repeat MNEP-074 and profile one complete step from step three on process zero.
- Trace: Enable host labels, HLO metadata, and one million GPU activity and callback events.
- Gate: Require five finite steps, zero dropped assignments, a profile upload, and a full transport-time comparison with MNEP-072.

### 2026-08-03 07:34 UTC - MNEP-075 identifies exposed token transport

- Result: All 16 workers completed five finite steps on attempt zero with no dropped assignments.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-075-multicontext-gin-profile-5-20260803-0701).
- Profile: [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmnep-075-multicontext-gin-profile-5-20260803-0701).
- Measurement: Ragged all-to-all used `15.0939 s` across 576 calls. Only `2.3611 s`, or `15.64%`, overlapped compute. The exposed part was `12.7328 s`.
- Breakdown: The six large token dispatch and combine families used `14.1204 s`. The weight families used less than `1.0 s`.
- Comparison: MNEP-072 used `14.9604 s`. More GIN contexts did not increase effective bandwidth.
- Decision: Keep the correct direct transport. Split token dispatch, grouped GEMM, and combine into pipeline buckets before more GIN work.

### 2026-08-03 07:49 UTC - MNEP-076 two-bucket rack contract

- Treatment: Split every source-to-destination token message into two contiguous buckets. Dispatch, compute, and return each bucket independently so XLA can overlap the next transfer with the current grouped GEMM.
- Local gate: The one-bucket and two-bucket paths match an independent dense output and input, W13, and W2 gradients on four GB200 GPUs. Both report zero routing errors.
- Checks: The MoonEP unit set, the 41 hero-EP tests, formatting, lint, types, syntax, and file checks pass.
- Rack gate: Require five finite EP64 steps on attempt zero, zero dropped assignments, and a median MFU above MNEP-074's `10.004%`.
- Next action: Profile the treatment if it improves MFU. Increase the bucket count only when the profile shows useful communication and compute overlap.

### 2026-08-03 08:12 UTC - MNEP-076 rejects interleaved token buckets

- Result: All 16 workers completed five finite steps on attempt zero. The run dropped no expert assignments, and the final loss was `8.6264`.
- Throughput: Median MFU was `9.439%`, median measured step time was `27.314 s`, and median throughput was `153,559` tokens per second.
- Comparison: MNEP-074 reached `10.004%` median MFU. Splitting each transfer into two smaller transfers added work without enough overlap.
- Cause: The graph put `combine 0` before `dispatch 1` in collective order. XLA preserves this order, so it could not start the next dispatch during the first bucket's expert GEMM.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-076-bucket2-5-20260803-0753).

### 2026-08-03 08:18 UTC - MNEP-077 reordered pipeline contract

- Treatment: Put all token dispatch collectives before the first token combine collective. This permits XLA to run `dispatch N+1` during bucket `N` compute and to run `combine N` during bucket `N+1` compute.
- Correctness: The two-bucket path matches the dense output and the input, W13, and W2 gradients on four GB200 GPUs. It reports zero routing errors.
- Local checks: The MoonEP test set reports 17 passed and 9 skipped. Formatting, lint, types, syntax, and file checks pass.
- Rack gate: Require five finite EP64 steps on attempt zero, zero dropped assignments, and median MFU above MNEP-074's `10.004%`.
- Next action: Profile the reordered pipeline if it improves MFU. If communication remains exposed, replace token transport with a persistent symmetric-memory kernel that uses a fixed small set of communication SMs.

### 2026-08-03 08:37 UTC - MNEP-077 rejects graph-only reordering

- Result: All 16 workers completed five finite steps on attempt zero. The run dropped no expert assignments.
- Throughput: P50 MFU was `9.262%`, median measured step time was `27.210 s`, and median throughput was `154,144` tokens per second.
- Comparison: MNEP-074 reached `10.004%` p50 MFU, and the interleaved MNEP-076 run reached `9.439%`. Moving the collective order did not create a useful compute window.
- Decision: Reject graph-only reordering as a throughput result. Keep the ordering needed for later buckets, but remove the serial receive-order scatter and return gather before adding more buckets.
- Evidence: [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-077-reordered-pipeline-5-20260803-0823).

### 2026-08-03 08:45 UTC - MNEP-078 zero-copy expert layout contract

- Treatment: Use four source-to-destination expert segments for each peer. Dispatch writes each segment directly into its final padded expert group, and combine reads the expert outputs directly from the same group layout.
- Removed work: Delete the full receive-order-to-expert scatter and the expert-to-receive-order gather from every token dispatch and combine pair.
- CPU gate: A two-rank, two-bucket round-trip test preserves every exact message slice through expert order and back to sender order.
- GPU gate: Both one-bucket and two-bucket paths match the dense output and the input, W13, and W2 gradients on four GB200 GPUs. Both report zero routing errors.
- Rack gate: Require five finite EP64 steps on attempt zero, zero dropped assignments, and median MFU above MNEP-074's `10.004%`.
- Next action: Profile the treatment if it improves MFU. If transfer remains exposed, use the direct layout in a persistent communication kernel with a small fixed SM set and double buffers.

### 2026-08-03 09:00 UTC - MNEP-078 rejects zero-copy layout alone

- Result: All 16 workers completed five finite steps on attempt zero. The run dropped no expert assignments.
- Throughput: P50 MFU was `7.793%`, median measured step time was `33.213 s`, and median throughput was `126,284` tokens per second.
- Comparison: MNEP-074 reached `10.004%` p50 MFU. Direct placement removed two full-buffer layout kernels, but its four ragged slices per peer made the current XLA collective path slower.
- Decision: Keep the direct expert layout as the target for a fused transport path, but reject the current multi-slice XLA ragged collective as a throughput result.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-078-zero-copy-layout-5-20260803-0850-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-078-zero-copy-layout-5-20260803-0850).

### 2026-08-03 09:00 UTC - MNEP-079 zero-copy profile contract

- Treatment: Record one measured step from the MNEP-078 graph without another code change.
- Question: Measure the new ragged collective time, removed layout time, compute overlap, and cost of four slices per peer.
- Decision gate: Use the measured exposed transfer as the baseline for a persistent transport kernel with fixed communication SMs and double buffers.

### 2026-08-03 09:15 UTC - MNEP-079 confirms exposed transport

- Result: All 16 workers completed five finite steps on attempt zero and uploaded the requested one-step profile.
- Ragged transport: 864 device-kernel calls used `26.424 s`. Only `4.091 s` overlapped compute, for a `15.5%` overlap fraction and `22.333 s` of exposed ragged transport.
- Kernel shape: The deployed device kernel used a 64-CTA grid with 512 threads per CTA. The trace reports 25% theoretical occupancy for each launch.
- Cause: Two token buckets add dispatch and combine calls in the forward pass, the rematerialized forward pass, and the collective transpose. The current graph still starts most transfers outside the expert GEMM windows.
- Decision: Add an explicit value-preserving dependency from dispatch bucket N to dispatch bucket N+1. This makes the next transfer and the current bucket GEMM ready at the same point. Build a 16-CTA transport variant in parallel for the next kernel gate.
- Evidence: [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmnep-079-zero-copy-profile-5-20260803-0900) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-079-zero-copy-profile-5-20260803-0900).

### 2026-08-03 09:20 UTC - MNEP-080 forced pipeline contract

- Treatment: Pass the next dispatch metadata through an XLA optimization barrier with one scalar from the prior dispatch result. The metadata value does not change and the dependency does not copy the token buffer.
- Expected schedule: Dispatch 0 completes first. Dispatch 1 then runs with bucket 0 GEMM. Combine 0 can then run with bucket 1 GEMM.
- Correctness: Local lowering, 23 MoonEP tests, 41 hero tests, and the exact two-bucket four-GB200 output and gradient gate pass.
- Rack gate: Require five finite EP64 steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-078's `7.793%`.
- Next action: Profile the treatment if it improves throughput, then apply the measured best fixed communication-CTA count.

### 2026-08-03 09:33 UTC - MNEP-080 rejects overlap without a remote-write fence

- Result: All 16 workers completed step 1 with finite gradients. All workers then reported non-finite gradients and a NaN loss at step 2. The later coordination errors were shutdown results.
- Performance signal: The interval between the two finite-diagnostic boundaries fell from MNEP-078's `33.213 s` median step to about `24 s`, but the invalid result is not an MFU measurement.
- Scope: The same overlap graph passes output and input, W13, and W2 gradient checks on four GB200 GPUs. The failure is in the cross-rack GIN path.
- Cause hypothesis: The multi-context kernel ends with a relaxed world barrier. NCCL 2.30 states that an all-context barrier with a `Put` fence is required before prior remote puts are visible in local memory.
- Decision: Keep eager dispatch as the default. Make compute overlap explicit, cap transport at 16 CTAs, and add an all-GIN-context `Put` fence before the next rack gate.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-080-forced-pipeline-5-20260803-0927-coord).

### 2026-08-03 09:42 UTC - fenced 16-CTA overlap contract

- Kernel: Apply sparse GIN, multi-context GIN, the 16-CTA cap, and the all-context remote-put fence to XLA `5d53e1e40c` with NCCL `2.30.7`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cta16-fence-20260803`.
- PJRT SHA-256: `4d3f0da2320322ebafb01770bee19440ec2479f04da726761a67332eb9013f68`.
- Local gate: The exact two-bucket compute-overlap output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Five finite steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-078's `7.793%`.

### 2026-08-03 09:58 UTC - MNEP-081 proves fenced overlap correctness

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments. The fence fixed the step-2 failure from MNEP-080.
- Performance: Steady durations were `62.791 s`, `59.593 s`, and `62.236 s`. The p50 MFU was `4.159%` at `67,394` tokens/s.
- Comparison: MNEP-078 reached `7.793%` p50 MFU. The all-context `Put` fence is correct but adds more cost than the overlap removes.
- Decision: Keep the fence build as correctness evidence, not as the throughput path. Test the same 16-CTA kernel without the added fence to isolate CTA count from fence cost.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-081-fenced-overlap-5-20260803-0948-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-081-fenced-overlap-5-20260803-0948).

### 2026-08-03 09:59 UTC - CTA16 and rematerialization treatments

- CTA16 artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cta16-20260803`, PJRT SHA-256 `f6f661a75a992a137e641a183deee79c9c79978d8fff062395980f0b3faf87ca`.
- CTA16 gate: Run the exact MNEP-081 graph without patch 0017. Require five finite steps before using its throughput result.
- Rematerialization: Add an explicit `save_moe` launcher mode. The MNEP-079 profile contains rematerialized token transfers; saving the tagged MoE tensors can remove those duplicate collectives if the program fits in HBM.
- Order: Isolate CTA16 correctness and speed first. Then combine the best safe transport with `save_moe` and measure HBM, MFU, and tokens/s.

### 2026-08-03 10:09 UTC - MNEP-082 rejects CTA16 transport

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments.
- Performance: The p50 duration was `62.450 s`. The p50 MFU was `4.144%` at `67,162` tokens/s.
- Comparison: MNEP-081 reached `4.159%` p50 MFU with the broad fence. Thus, the 16-CTA cap caused most of the slowdown.
- Decision: Reject CTA16 as a performance path. Test CTA32 to increase transport throughput and keep GPU resources free for expert compute.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-082-cta16-overlap-5-20260803-0959-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-082-cta16-overlap-5-20260803-0959).

### 2026-08-03 10:15 UTC - CTA32 overlap contract

- Kernel: Apply sparse GIN, multi-context GIN, and a 32-CTA transport cap to XLA `5d53e1e40c` with NCCL `2.30.7`.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cta32-20260803`.
- PJRT SHA-256: `a9d724a350612b982757ac38dbac07f5a9305b55d62e96f9196eedcf4cb9f1b4`.
- Local gate: The exact two-bucket overlap output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Require five finite steps on attempt zero, zero dropped assignments, and p50 MFU above CTA16.

### 2026-08-03 10:28 UTC - MNEP-083 rejects CTA32 transport

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments.
- Performance: The p50 duration was `43.602 s`. The p50 MFU was `5.936%` at `96,194` tokens/s.
- Comparison: CTA32 improved on CTA16, but MNEP-074 reached `10.004%` p50 MFU without bucket overlap.
- Decision: Reject the CTA cap as the primary overlap control. Restore CTA64 and fence only each active GIN context.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-083-cta32-overlap-5-20260803-1018-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-083-cta32-overlap-5-20260803-1018).

### 2026-08-03 10:28 UTC - CTA64 active-context fence contract

- Kernel: Keep the 64-CTA grid and add a `Put` fence only to each GIN context that transfers a remote chunk.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cta64-active-fence-20260803`.
- PJRT SHA-256: `f0666ab5da1982fecd925b38120abdb70af99ad5db54f449226c34598b7c5ef7`.
- Local gate: The exact two-bucket overlap output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Require five finite steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-074.

### 2026-08-03 10:40 UTC - MNEP-084 rejects the active-context fence

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments.
- Performance: The p50 duration was `33.333 s`. The p50 MFU was `7.765%` at `125,830` tokens/s.
- Comparison: The active-context fence is safe and faster than CTA32, but MNEP-074 reached `10.004%` p50 MFU.
- Decision: Keep the fence as a correctness option. Restore one token message per peer before the next overlap gate.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-084-cta64-active-fence-5-20260803-1030-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-084-cta64-active-fence-5-20260803-1030).

### 2026-08-03 10:44 UTC - Receive-order bucket pipeline contract

- Treatment: Dispatch one contiguous message to each peer, then move received rows into the expert-grouped compute buffer.
- Purpose: Reduce each token collective from multiple expert segments per peer to one segment per peer.
- Overlap: Dispatch bucket 1 and the layout plus expert compute for bucket 0 become ready at the same boundary.
- Local gate: The exact two-bucket output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Require five finite steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-074.

### 2026-08-03 10:49 UTC - Strong GIN completion contract

- Kernel: Replace each weak GIN completion signal with a strong signal for the same remote chunk.
- Purpose: Make the bundled remote put complete before the receiver observes the signal, without an added world barrier.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cta64-strong-signal-20260803`.
- PJRT SHA-256: `4ff4a481124ed4348f764ec5c75b6bce178f1b4975958d47586a61af8a75f4c8`.
- Local gate: The receive-order two-bucket output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Require five finite steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-074.

### 2026-08-03 10:58 UTC - MNEP-085 rejects receive-order overlap

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments.
- Performance: The p50 duration was `27.641 s`. The p50 MFU was `9.364%` at `151,744` tokens/s.
- Comparison: One token message per peer is faster than the expert-segment transport, but MNEP-074 reached `10.004%` p50 MFU.
- Decision: Keep the lower-fanout receive-order layout. Change the transport kernel so that local-only CTAs leave after their local barrier while the smaller active GIN set waits for remote completion.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-085-receive-strong-5-20260803-1050-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-085-receive-strong-5-20260803-1050).

### 2026-08-03 11:00 UTC - Local-only CTA release contract

- Kernel: CTAs that copy only local LSA rows use an LSA barrier and then leave. CTAs that transfer remote rows keep the world barrier and wait for strong GIN completion.
- Purpose: Keep the remote transfer on the small active GIN set and release the other SMs for expert compute.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-release-local-ctas-20260803`.
- PJRT SHA-256: `6df224e50a9a37965f671cc6a7f36a545baff0998777a9d1021fbb1477ef1707`.
- Local gate: The receive-order two-bucket output and input, W13, and W2 gradient checks pass on four GB200 GPUs.
- Rack gate: Require five finite steps on attempt zero, zero dropped assignments, and p50 MFU above MNEP-074.

### 2026-08-03 11:12 UTC - MNEP-086 rejects local-only CTA release alone

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments.
- Performance: The p50 duration was `27.620 s`. The p50 MFU was `9.371%` at `151,855` tokens/s.
- Comparison: MNEP-085 reached `9.364%` p50 MFU. Releasing the inactive CTAs did not change end-to-end speed.
- Decision: Keep the kernel change as a valid overlap resource control, but do not claim a speed gain. Profile this exact graph to test if XLA still serializes the ready GEMM with the ragged collective.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-086-release-local-ctas-5-20260803-1105-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-086-release-local-ctas-5-20260803-1105).

### 2026-08-03 11:14 UTC - MNEP-087 CTA-release profile contract

- Treatment: Repeat MNEP-086 and record one complete step from step three on process zero.
- Question: Measure ragged collective and grouped GEMM concurrency after inactive CTAs leave. Check if the ready GEMM starts during the remote wait.
- Gate: Require five finite steps, zero dropped assignments, a profile upload, and an event-level comparison with MNEP-075.

### 2026-08-03 11:22 UTC - MNEP-087 confirms serialized expert GEMM

- Result: All 16 workers completed five finite steps on attempt zero with zero dropped assignments and uploaded one full-step profile.
- Performance: The p50 duration was `28.081 s`. The p50 MFU was `9.217%` at `149,365` tokens/s.
- Transport: 864 ragged all-to-all kernels used `15.584 s`. GPU compute overlapped `2.999 s`, or `19.24%`, of that transport.
- Critical result: The 606 QuACK grouped GEMM kernels used `2.014 s`, but exactly `0.000 s` overlapped a ragged all-to-all kernel.
- Cause: XLA assigns the ragged operation to a communication stream, but the latency-hiding schedule places each ready grouped GEMM after the collective completes. Releasing 60 local-only CTAs cannot help until the schedule starts the independent compute inside the collective window.
- Decision: Add a narrow XLA latency-hiding rule that starts a ready ragged all-to-all before an independent compute candidate while keeping the one-collective overlap limit.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-087-release-local-profile-5-20260803-1113-coord), [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-087-release-local-profile-5-20260803-1113), and [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmnep-087-release-local-profile-5-20260803-1113).

### 2026-08-03 12:38 UTC - CUTLASS scheduling-group overlap proof

- XLA: Keep scheduling annotations on `CuteDSLRT_NvJaxCutlassCall` and `CuteDSLRT_NvJaxCutlassCallNoCudaGraph` in the GPU latency-hiding scheduler.
- JAX: Attach the scheduling group to the QuACK FFI custom call. Pair dispatch bucket 1 with both expert GEMMs for bucket 0.
- HLO result: The final GPU schedule contains `ragged-all-to-all-start.4`, the gated QuACK GEMM, the down QuACK GEMM, and `ragged-all-to-all-done.4` on four adjacent lines in group `789000`.
- Correctness: The two-bucket forward probe returned the expected value with zero routing errors. The independent dense-reference test passed output and input, W13, and W2 gradient checks on four GB200 GPUs.
- Artifact: `s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-multicontext-cutlass-overlap-20260803`.
- PJRT SHA-256: `1aeb6b867f97af3deeadb5fe187c21327dca856a313264237f7e27548738815b`.
- Next: Pair combine bucket 0 with expert compute bucket 1, then run the one-NVL72 correctness, throughput, and profile gates.

### 2026-08-03 12:57 UTC - Both transport stages overlap expert compute

- Dispatch: The final GPU schedule starts dispatch bucket 1, runs both expert GEMMs for bucket 0, and then completes the dispatch in scheduling group `789000`.
- Combine: The final GPU schedule starts combine bucket 0, runs both expert GEMMs for bucket 1, and then completes the combine in scheduling group `790000`.
- Fusion barrier: A small Triton row gather keeps XLA horizontal fusion from adding a false dependency between the two buckets. Its custom VJP uses a Triton row scatter because each valid receive position is unique.
- Correctness: The two-bucket forward probe returned `1680.0` with zero routing errors. The four-GB200 dense-reference test passed the output and input, W13, and W2 gradient checks in `187.31 s`.
- Gate: Push this stage, then run five finite steps on one NVL72. Compare p50 MFU and tokens/s with MNEP-074 and record a full-step profile if the throughput gate passes.

### 2026-08-03 13:08 UTC - MNEP-088 exposes a rematerialized-gradient group collision

- Result: The 64-GPU job formed one `nvlink.domain`, installed the fixed runtime, and failed during full training-graph compilation before step 0.
- Error: XLA rejected scheduling groups `789000` and `790000` because a later `scheduled_cutlass_call` had an operand with the same annotation and unannotated operations between them.
- Cause: The custom VJP reused the forward transport group ID for activation-gradient QuACK calls. Rematerialization put the recomputed forward calls and gradient calls in one computation, which joined separate windows into a group with gaps.
- Fix: Keep scheduling IDs on the two forward expert GEMMs only. Leave activation-gradient GEMMs unannotated until they have a separate backward transport schedule.
- Test: Rematerialize the MoE call in the four-GB200 dense-reference gradient gate before MNEP-089.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-088-cutlass-overlap-5-20260803-1303-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-088-cutlass-overlap-5-20260803-1303).

### 2026-08-03 13:14 UTC - Rematerialized gradient gate passes

- Change: Activation-gradient QuACK calls no longer reuse the forward transport group ID.
- Regression gate: The four-GB200 dense-reference test now applies `jax.checkpoint` to the MoonEP call before differentiation.
- Result: The rematerialized graph compiled without a scheduling-group error and matched output and input, W13, and W2 gradients in `237.29 s`.
- Decision: Commit the correction, release the debug tray, and submit MNEP-089 as one controlled retry of the five-step rack gate.

### 2026-08-03 13:24 UTC - MNEP-089 exposes the collective transpose collision

- Result: The CUTLASS collision from MNEP-088 is gone. Full training-graph compilation then failed before step 0 on annotated `ragged-all-to-all-start` operations.
- Error: Groups `789000` and `790000` had a prior operation with the same annotation in their operand trees and unannotated operations between the two calls.
- Cause: JAX copies the metadata context from a forward ragged collective into its automatic transpose. The rematerialized reverse-scan body then joins the recomputed forward collective and its transpose into one group with gaps.
- Fix: Wrap annotated zero-output ragged collectives in a custom VJP. Keep the annotation on the forward call and implement the standard ragged transpose without metadata.
- Regression gate: Run two rematerialized MoonEP layers in `jax.lax.scan` on four GB200 GPUs, then compare the final output and all gradients with a dense two-layer scan.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-089-cutlass-overlap-5-20260803-1315-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-089-cutlass-overlap-5-20260803-1315).

### 2026-08-03 13:31 UTC - Reverse-scan collective gate passes

- Custom VJP: Annotated zero-output ragged collectives keep the group ID in the forward pass. Their transpose exchanges input and output offsets and runs the reverse ragged collective without metadata.
- Regression shape: Two distinct expert-weight sets run through two rematerialized MoonEP layers in `jax.lax.scan`, with a residual connection between layers.
- Scale: The test uses expert weight scale `0.05`. The rack model uses about `0.007`; the test remains more numerically demanding without adding two layer gradients into one shared weight tensor.
- Result: The forward and reverse scans compiled without an annotation collision. Output and input, W13, and W2 gradient parity passed the existing tolerances on four GB200 GPUs in `85.49 s`.
- Decision: Commit and submit the next five-step rack gate.

### 2026-08-03 13:43 UTC - MNEP-090 isolates block-remat duplication

- Result: The CUTLASS-gradient and collective-transpose collisions are gone. Full training-graph compilation still failed before step 0 on a forward `scheduled_cutlass_call` in groups `789000` or `790000`.
- Cause: Block rematerialization makes a second forward MoE region after JAX builds the scan transpose. The cloned region keeps the original numeric scheduling group IDs, so XLA sees two separate transport-compute windows as one group with a gap.
- Decision: Test `remat_mode=save_moe` first. This saves the tagged MoE values, removes the cloned dispatch window, and avoids extra EP communication in backward. Keep an XLA group-split fix as a fallback if the rack cannot hold the saved values.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-090-cutlass-overlap-5-20260803-1335-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-090-cutlass-overlap-5-20260803-1335).

### 2026-08-03 13:43 UTC - MNEP-091 save-MoE rack gate

- Treatment: Use the two-bucket annotated CUTLASS overlap graph with `remat_mode=save_moe` on one NVL72.
- Gate: Require all 16 workers to compile and complete five finite steps on attempt zero with zero dropped assignments.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-091-save-moe-overlap-5-20260803-1343-coord).

### 2026-08-03 13:49 UTC - MNEP-091 cannot save the MoE state

- Result: Full training-graph compilation requested `557.88 GiB` on each GPU and failed before step 0.
- Decision: Keep block rematerialization. Reduce the custom expert VJP residual to inputs and weights, then recompute only the gated activation in its backward rule.
- Evidence: [Iris job](https://iris.oa.dev/#/job/%2Frav%2Fmnep-091-save-moe-overlap-5-20260803-1343-coord) and [W&B run](https://wandb.ai/marin-community/rav_moe/runs/mnep-091-save-moe-overlap-5-20260803-1343).

### 2026-08-03 14:36 UTC - Backward transport-compute overlap gate passes

- Backward schedule: Transposed dispatch and combine collectives use group IDs `791000` and `792000`. Each group overlaps the independent bucket's QuACK `dh` GEMM.
- Rematerialization: The expert VJP saves only inputs and weights. It recomputes the gated activation once and does not recompute the down projection.
- Fusion barrier: A Triton row scatter and its gather transpose keep XLA from merging both bucket-layout gradients into one multi-output fusion. The merged fusion created a false dependency from the communication start to the other bucket's GEMM.
- Correctness: The standard ragged transport passed the dense-reference output and input, W13, and W2 gradient gate with two rematerialized scan layers and the latency-hiding scheduler enabled on four GB200 GPUs in `89.17 s`.
- Direct-device note: The one-process, four-GPU symmetric-memory test exits with signal 11 in both one-bucket and two-bucket modes on this tray. The process-per-GPU rack path remains the direct-device correctness gate.
- Next: Submit five finite steps on one NVL72 with block rematerialization, then profile exact ragged and QuACK overlap and compare MFU with MNEP-074.
