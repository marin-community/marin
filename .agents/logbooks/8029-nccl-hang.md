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

### 2026-08-07 03:55 UTC - moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807

- Command: `uv run --frozen iris --cluster=marin job run --target-cluster cw-us-east-08a --priority production --no-wait --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 43200 --max-retries 0 --job-name moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_supervised --run-id moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807-coord`; child `/power/moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807-coord/grug-train-moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807`.
- Git SHA: `fe7ed155797efcebb3ff11b0a678da8b11aac0be`.
- Dirty tree: No; the commit was pushed before submission.
- Source bundle: Iris workspace bundle, 9.8 MB; no content ID was reported.
- Hardware: 32 Iris tasks, four GB200 GPUs and four JAX processes per task, 128 processes/GPUs across two NVL72 racks on `cw-us-east-08a`.
- W&B: ID and display name `moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807`, project `marin_moe`, group `moe-hero-fsdp`, resume `allow`.
- Output root: `s3://marin-us-east-02a/marin/grug/moe-hero-fsdp-nccl2307-2rack-1ppg-supervised-1000-20260807/2026.08.07`.
- Initialization: None.
- Final step: 1000.
- Checkpoint policy: Metrics only; no checkpoints.
- Diagnostics: One supervisor per GPU process with XLA's 60-second execution deadman, progress tracking, and NCCL-init timeout; no environment ablation or debugger.
- DRI: User; monitor `scratch/20260807-0355_monitoring_state.json` polls every 60 seconds and exits on terminal state, explicit hang/failure signal, 30-minute startup stall, or 15-minute post-progress stall.

### 2026-08-07 04:13 UTC - mhf-2rack-1ppg-clean-20260807

- Command: `... --job-name mhf-2rack-1ppg-clean-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch --run-id mhf-2rack-1ppg-clean-20260807 --dp-racks 2 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/mhf-2rack-1ppg-clean-20260807-coord`.
- Git SHA: `f1ee764317`.
- Hardware: 32 Iris tasks, four GB200 GPUs and four JAX processes per task, 128 processes across two NVL72 racks on `cw-us-east-08a`.
- Diagnostic control: One process per GPU, no supervisor, no recovery XLA flags, no environment ablation.
- Purpose: Separate the process topology from the recovery instrumentation as the cause of the pre-step-0 `std::bad_alloc`.

### 2026-08-07 04:16 UTC - mhf-2rack-1ppg-sup-notrack-20260807

- Command: as above with `-e XLA_FLAGS "--xla_gpu_execution_progress_tracking=0"` and `launch_supervised`.
- Job: `/power/mhf-2rack-1ppg-sup-notrack-20260807-coord`.
- Environment ablation: `XLA_FLAGS=--xla_gpu_execution_progress_tracking=0`.
- Purpose: Test whether the deadman's thunk-reporting path, not the trainer, threw the allocation errors.

### 2026-08-07 04:24 UTC - mhf-2rack-1ppg-sup-dm600-20260807

- Command: as above with `-e XLA_FLAGS "--xla_gpu_execution_progress_tracking=0 --xla_gpu_execution_terminate_timeout=600s"`.
- Job: `/power/mhf-2rack-1ppg-sup-dm600-20260807-coord`.
- Purpose: Confirm the supervised path clears step 0 once the deadman exceeds a cold first execution.

### 2026-08-07 04:40 UTC - mhf-8rack-1ppg-sup-base-20260807

- Command: `... --job-name mhf-8rack-1ppg-sup-base-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_supervised --run-id mhf-8rack-1ppg-sup-base-20260807 --dp-racks 8 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/mhf-8rack-1ppg-sup-base-20260807-coord`.
- Git SHA: `bab625f751`.
- Hardware: 128 Iris tasks, 512 GPUs across eight NVL72 racks on `cw-us-east-08a`.
- Detection: In-code defaults — 600s XLA execution deadman, thunk reporting off, no restart budget.
- Purpose: Reproduce the #7344 wedge at the scale where it is documented (steps 17-200).
- Outcome: No wedge. Stopped by operator at step 301 after 1:49:07, 18.9 s/it, loss 2.68 from 8.82 at step 9. No OOM, no fatal, no RAS capture triggered. Watch stats were enabled at interval 20 and cleared every watch step.

### 2026-08-07 06:43 UTC - mhf-8rack-1ppg-stock-20260807

- Command: `... --job-name mhf-8rack-1ppg-stock-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_stock_control --run-id mhf-8rack-1ppg-stock-20260807 --dp-racks 8 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/mhf-8rack-1ppg-stock-20260807-coord`.
- Git SHA: `99333900c9`; clean tree, pushed before submission.
- Hardware: 128 Iris tasks, 512 GPUs across eight NVL72 racks on `cw-us-east-08a`.
- Detection: None in-process. No supervisor parent, no XLA execution deadman, no thunk reporting, no retries. `--xla_gpu_nccl_termination_timeout_seconds=600` remains because `marin.training.training` sets it independently of the recovery framework.
- Purpose: Restore the historical stock configuration to test whether the recovery instrumentation, not chance, explains the clean supervised run.
- Caveat: Differs from `mhf-8rack-1ppg-sup-base-20260807` in two variables, not one — instrumentation and watch state. Watch is off here per operator direction. Watch-on is the follow-up arm if this runs clean.
- Outcome: No wedge. Stopped by operator at step 205 after 1:11, 18.8 s/it, loss 2.87. Cleared the whole documented 17-200 span.

### 2026-08-07 08:07 UTC - mhf-12rack-1ppg-stock-20260807

- Command: `... --job-name mhf-12rack-1ppg-stock-20260807-coord -e WANDB_API_KEY <set> -- python -m experiments.grug.moe_hero_fsdp.launch_stock_control --run-id mhf-12rack-1ppg-stock-20260807 --dp-racks 12 --num-steps 1000 --no-save-checkpoints --version 2026.08.07 --run`.
- Job: `/power/mhf-12rack-1ppg-stock-20260807-coord`.
- Git SHA: `e9c27d3bc6`; clean tree, pushed before submission.
- Hardware: 192 Iris tasks, 768 GPUs across twelve NVL72 racks on `cw-us-east-08a`.
- Detection: None in-process; monitor plus a 240s stall tripwire drive capture externally.
- Purpose: Escalate scale after two clean 8-rack arms.
- Caveat: #8029 records no 12-rack stock wedge. The only 12-rack data point is `CUDA_LAUNCH_BLOCKING` clean through 2,580 steps, so this is an untested scale rather than a known-higher-hazard one.

## Event Log

### 2026-08-07 08:06 UTC - Reproduction failed twice at 8 racks; instrumentation is not the suppressor

- Supervised (deadman + supervisor parent): clean to step 305, 18.9 s/it.
- Stock control (no instrumentation, provenance verified from `/proc` on every trainer process): clean to step 205, 18.8 s/it.
- Five recorded 8-rack FSDP wedges all sit at or below step 152. Two independent clean runs totalling about 506 steps is evidence the hazard rate has changed, not that two draws missed.
- The instrumentation hypothesis from the previous update is dead: removing it did not restore the wedge.
- Throughput is identical with and without the instrumentation (18.8 vs 18.9 s/it), so the deadman is free if it ever does prove protective.
- Ruled out: driver/VBIOS drift; the FSDP hero running the EP64 mesh (`expert_axis_size=1`, `replica_axis_size=8`, so the 17-200 reference applies rather than EP64's ~1,290); `5a7aa95fc9` reduced RAS collection, which predates reproductions still recorded on 08-06.
- Not yet explained: what changed between the historical wedges and now. Node-set specificity is untested — RAS at `--detail stall` carries no host identity, so it would need per-task capture and a wedging run to compare against.

### 2026-08-07 06:50 UTC - Stock control flag provenance verified

- Read from every python process in task 0 via `/proc/<pid>/environ`, excluding the probe's own pid.
- Topology is `iris.hooks.multigpu_main --nproc 4` (pid 1) over four `_callable_runner.py` trainers (pids 815, 817, 819, 821), one JAX process per GPU. No `levanter.recovery.child` grandchild, so no supervisor layer.
- Every trainer carries `XLA_FLAGS=--xla_gpu_enable_command_buffer= --xla_gpu_nccl_termination_timeout_seconds=600`, with no `xla_gpu_execution_terminate_timeout` and no `xla_gpu_execution_progress_tracking`.
- The contrast against `mhf-8rack-1ppg-sup-base-20260807` is therefore the per-execution deadman, thunk reporting, and the supervisor parent, holding command buffers and the clique timeout equal.

### 2026-08-07 06:40 UTC - Driver and VBIOS unchanged since the historical wedges

- `nvidia-smi` on the live eight-rack job reports driver `595.71.05` and VBIOS `97.00.B9.00.99` on every GB200 sampled.
- The driver matches the fleet-wide constant recorded in #8029, so a fleet update does not explain non-reproduction. VBIOS is recorded here for the first time.

### 2026-08-07 06:20 UTC - Two monitor hang patterns matched strings the deployed binary cannot print

- `avoid infinite hangs` and `hang_watchdog` appear nowhere in the deployed jaxlib 0.11.0. The monitor was greping for text that cannot occur.
- Detection was never at risk: the progress-stall check and the 240s tripwire catch a wedge structurally, and a deadman kill takes the job terminal. Classification was — a real wedge would have been recorded as `failed` rather than `hang_signal`.
- Replaced with a match on the absl FATAL severity prefix (`F20260807 ...`), which `LOG(FATAL)` always emits regardless of wording.
- Open: whether arming the per-execution deadman inserts a host-device sync. The abort message is not a literal in the binary, so it could not be traced that way. If it does, every supervised arm is confounded against the unsupervised historical control.

### 2026-08-07 05:15 UTC - The recurring two-rack OOM is the watch-stats program

- Two attempts of one config died on the byte-identical 117.02 GiB allocation inside `jit_train_step` at step 80 and step 20, both exact multiples of the 20-step watch interval.
- Pool fragmentation decides which watch step it lands on, which is why the step differed. Eight racks cleared every watch step, so it is fragmentation-dependent rather than a hard sizing failure.
- Watch stats are off for hero runs from `eb6a320a18`. This was a competing risk censoring wedge trials at 2 racks.

### 2026-08-07 04:49 UTC - Eight-rack baseline flag provenance verified

- Read from the trainer grandchildren (`python -m levanter.recovery.child`, pids 1353-1356) via `/proc/<pid>/environ`, not the supervisor parents, which carry no deadman flags by design.
- `XLA_FLAGS = --xla_gpu_enable_command_buffer= --xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_execution_terminate_timeout=600s --xla_gpu_execution_progress_tracking=0`.
- `NCCL_DEBUG=WARN`, `JAX_ENABLE_PGLE=1`, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, NCCL 2.30.7+cuda13.3, 128 tasks x 4 processes = 512 ranks.
- The NCCL clique timeout is 600s, not the recovery default of 120s: `marin.training.training` sets it first and `_merge_xla_flag` preserves the existing value. Clique acquisition therefore cannot trip the abort ahead of the execution deadman.

### 2026-08-07 04:38 UTC - Recovery instrumentation, not the model, caused every pre-step-0 failure

- Finding: `--xla_gpu_execution_progress_tracking=8` is "number of thunks to report in progress tracking on execution timeout". It runs only when the deadman fires, and on a module this size it throws `std::bad_alloc` / `std::length_error`, replacing the abort diagnostic with an allocation error.
- Finding: The 60s per-execution deadman cannot survive step 0 at rack scale. The first `jit_train_step` execution measured 261s at 2 racks (loop entry 04:32:16, step 1 at 04:36:38); it carries `ncclCommSplit` across every rank, lazy CUDA module loading, and a PGLE profile pass.
- Evidence: A thread dump of the clean run's rank 0 four minutes into the first `train_step` showed `active+gil` in `jax/_src/interpreters/partial_eval.py`, i.e. jaxpr construction, not a blocked collective.
- Consequence: The 02:46, 02:56 (`NCCL_RUNTIME_CONNECT=0`), 03:03 (`CUDA_MODULE_LOADING=EAGER`), 03:23 (failsafe-only) and 03:55 (one process per GPU) arms all aborted on their own deadman before step 1 and carry no ablation verdict. `CUDA_MODULE_LOADING=EAGER` in particular is untested, and lazy module loading resolves inside exactly the first execution those runs never survived.
- Fix: `bab625f751` sets the hero deadman to 600s and thunk reporting to 0, with a regression test pinning the deadman above the measured first execution.
- Result: One process per GPU at 300B is healthy — 18.5 s/step, loss 6.59 at step 25 on the clean 2-rack control.


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
