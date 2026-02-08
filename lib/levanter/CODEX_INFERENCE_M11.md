# CODEX Inference M11: Interleaved SimPO Train -> Host-DP Inference -> Train

Status: IN PROGRESS - BLOCKED (2026-02-08 runtime blocker on `v5p-32`)

## Goal

Implement and validate an in-process workflow in `src/levanter/main/train_simpo.py`:
1. Train for 2 steps.
2. Pause training and run multi-host host-data-parallel inference for `128` prompts with `max_new_tokens=2048`.
3. Resume the same training run for 5 additional steps (total steps `= 7`).

## What Has Been Implemented

- Exact-step inference scheduling (`eval_at_steps`) with `eval_every` fallback.
- Inference mode switch:
  - `global_mesh`
  - `host_data_parallel`
- Deterministic host prompt sharding for host-DP callback execution.
- Host payload gather-to-leader for global metrics/samples in host-DP mode.
- Per-host JSONL output writing by eval step (`host_data_parallel_output_dir`).
- Callback exception handling updated to propagate failures (no swallow).
- M11 configs created:
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_train2_infer128_train5.yaml`
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a.yaml`
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a_train2.yaml`
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_phase_a_global.yaml`

## TPU Runtime Progress (What Was Done)

### Repeated phase-A experiments (same failure family)

- Ran multiple variants (`run20` through `run32`) across:
  - host-DP and global inference modes,
  - cleanup mode variants,
  - ragged/non-ragged variants,
  - different decode microbatch knobs.
- Representative logs:
  - `/tmp/levanter_run_m11_v5p32_run23_phaseA_hostdp.log`
  - `/tmp/levanter_run_m11_v5p32_run24_phaseA_cleanup_end.log`
  - `/tmp/levanter_run_m11_v5p32_run25_phaseA_global.log`
  - `/tmp/levanter_run_m11_v5p32_run26_phaseA_ragged.log`

Common outcome:
- Inference decode runs and reaches end-of-decode state.
- Resume into post-inference training fails with the same TPU runtime crash signature.

### Focused runs from latest pass

- `run31` log: `/tmp/levanter_run_m11_v5p32_run31_phaseA_train3_tpr256_r64_mqt512.log`
  - Reached `DecodeStats[after_decode]: active=32 pages_in_use=1056 free=3168`.
  - Immediately after inference, training resume failed with:
    - `jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly`
    - `TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id`

- `run32` log: `/tmp/levanter_run_m11_v5p32_run32_phaseA_syncbar_train3_tpr256_r64_mqt512.log`
  - Same phase-A setup after adding extra device-sync barriers (see patch below).
  - Same outcome and same crash signature as `run31`.

- `run33` log: `/tmp/levanter_run_m11_v5p32_run33_phaseA_train2_syncbar.log`
  - Started a train2-only termination test to see if stopping immediately after inference avoids resume crash.
  - Observed active decode progress (iter0 + steady decode iterations around ~292-294 tok/s per host).
  - No successful completion marker captured before interruption, so this run is inconclusive.

## Patch Added During Runtime Debugging

File: `src/levanter/main/train_simpo.py`

- Imported `sync_global_devices` and added explicit device-level barriers:
  - Before inference callback body:
    - `sync_global_devices(f"levanter_inference_pre_device_{step_number}")`
  - In callback `finally` before the post callback barrier:
    - `sync_global_devices(f"levanter_inference_post_device_{step_number}")`

Result:
- Did not resolve the post-inference resume crash (`run32` still failed identically to `run31`).

## Current Blocker (Where We Are Stuck)

M11 is blocked by a TPU runtime failure when transitioning from completed inference back into training within the same process:

- `jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly`
- `TensorCoreSequencer ... scheckne`
- launch-group mismatch:
  - `unexpected peer shows up in the launch group with a different launch id`

This failure has reproduced across multiple config variants and persisted after adding explicit pre/post device sync barriers.

## Current Completion State

- Code-level M11 functionality is implemented.
- Runtime objective is not yet met because train->infer works, but infer->train resume is not stable on the tested `v5p-32` path.
- M11 is not complete yet; M12 remains blocked on M11 runtime unblock.
