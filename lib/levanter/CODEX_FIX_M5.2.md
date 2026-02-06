# M5.2 Debug Log (Codex): Per-Round Reset Failure on Multi-Host TPU

## Objective

Reproduce and debug the per-round reset crash using a real workload config:

- `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml`

The target failure window is the transition from round 0 to round 1, where
`engine.generate()` starts and `InferenceEngine.reset()` runs.

## Ground Rules

1. Use foreground launch so all logs are streamed immediately.
2. Route every run to a dedicated `/tmp/` log file.
3. Keep environment changes explicit per run.
4. Append each attempt here with:
   - exact command
   - log path
   - key boundary lines
   - result

## Canonical Command Template

```bash
python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- \
  -e TPU_STDERR_LOG_LEVEL 0 \
  -e TPU_MIN_LOG_LEVEL 0 \
  -e JAX_ASYNC_ERROR_CHECKING 1 \
  -e JAX_TRACEBACK_FILTERING off \
  -e JAX_TRANSFER_GUARD disallow \
  uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml \
  2>&1 | tee /tmp/<run_name>.log
```

## Run Log

### 2026-02-05 - Setup

- Removed stale notes file: `CLAUDE_FIX_M5.2.md`
- Created this file to track all M5.2 attempts from scratch.

### 2026-02-05 - Baseline Recheck (no extra TPU debug env)

- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_user_verify_5prompts_round2.log`
- Log:
  - `/tmp/levanter_run_m52_user_verify_5prompts_round2.log`
- Key evidence:
  - Round 0 completes (`Round 0: generate done`).
  - Round 1 starts (`Round 1: generate start`).
  - Crash occurs immediately after `Generate reset: start` on both hosts.
  - Worker failure line: `Command execution on worker {0,1} failed with exit status 1`.
- Result:
  - **FAILED** at round-boundary reset.

### 2026-02-05 - Debug Env Attempt A (launch syntax mistake)

- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -e JAX_TRANSFER_GUARD disallow -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2.log`
- Key evidence:
  - Container fails immediately with `tini` error:
    - `[FATAL tini (7)] exec -e failed: No such file or directory`
- Root cause:
  - `-e` flags were placed after the first `--`, so they were treated as container command tokens, not launch env args.
- Result:
  - **INVALID RUN** (wrapper invocation bug, no workload signal).

### 2026-02-05 - Debug Env Attempt B (correct launch syntax + transfer guard)

- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -e JAX_TRANSFER_GUARD disallow -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2_v2.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2_v2.log`
- Key evidence:
  - TPU runtime and low-level driver logs are emitted (debug env applied correctly).
  - Python stacktrace fails during initialization with:
    - `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Disallowed host-to-device transfer ...`
- Root cause:
  - `JAX_TRANSFER_GUARD=disallow` blocks normal host-to-device transfers during startup (`trainer.initialize` seed broadcast path), so this does not test M5.2 behavior.
- Result:
  - **FAILED EARLY** (instrumentation too strict for normal startup).

### 2026-02-05 - Debug Env Attempt C (physical reset, no transfer guard)

- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2_v3.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2_v3.log`
- Key evidence:
  - Round 0 fully completes:
    - `Round 0: generate done`
    - `Round 0: samples table done`
  - Immediately after round 0 completion:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id than the current group leader`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 begins but reset never completes:
    - `Round 1: generate start`
    - `Generate reset: start`
    - then slice health failure / worker exit.
- Result:
  - **FAILED** at the same round-boundary window, but now with concrete TPU runtime signature.

### 2026-02-05 - Debug Env Attempt D (logical reset isolated config, invalid path)

- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path /tmp/sample_llama8b_multihost_real_5prompts_2048_reset_logical_round2_cleanup_none_noop_isolated.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2_logical_isolated.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2_logical_isolated.log`
- Key evidence:
  - Both workers error before workload start:
    - `FileNotFoundError: ... /tmp/sample_llama8b_multihost_real_5prompts_2048_reset_logical_round2_cleanup_none_noop_isolated.yaml`
- Root cause:
  - Config file was created on local machine `/tmp`, but container executes in `/opt/marin`; that path is not mounted there.
- Result:
  - **INVALID RUN** (pathing issue).

### 2026-02-05 - Debug Env Attempt E (logical reset isolated config, corrected path)

- Isolated config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_logical_round2_cleanup_none_noop_isolated.yaml`
  - Diff from baseline: only `reset_mode: logical` (all sizing unchanged: `max_pages=240`, `max_seqs=5`, `max_prefill_size=512`).
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_logical_round2_cleanup_none_noop_isolated.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2_logical_isolated.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2_logical_isolated.log`
- Key evidence:
  - Round 0 completes.
  - Immediately after round 0:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - same `unexpected peer ... different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts with `Generate reset: start` and then fails slice-wide (`Slice health check failed`).
- Result:
  - **FAILED** with the same signature as physical reset.

### 2026-02-05 - Debug Env Attempt F (physical reset + skip round barrier + no samples table)

- Isolated config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_skip_barrier_nosamples.yaml`
  - Added:
    - `skip_round_barrier: true`
    - `skip_samples_table: true`
  - Kept:
    - `reset_mode: physical`
    - same sizing (`max_pages=240`, `max_seqs=5`, `max_prefill_size=512`)
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_skip_barrier_nosamples.yaml 2>&1 | tee /tmp/levanter_run_m52_dbg_env_round2_physical_skipbarrier_nosamples.log`
- Log:
  - `/tmp/levanter_run_m52_dbg_env_round2_physical_skipbarrier_nosamples.log`
- Key evidence:
  - Round 0 completes.
  - Same TPU runtime error appears immediately after round 0:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts and reset starts, then slice fails:
    - `Round 1: generate start`
    - `Generate reset: start`
    - `Slice health check failed ...`
- Result:
  - **FAILED** with the same signature even when round barrier and sample table logging are disabled.

### 2026-02-05 - Debug Env Attempt G (H1: force post-generate sync on all hosts)

- Hypothesis:
  - Round-boundary failure is caused by hosts not being synchronized immediately after `engine.generate()`.
- Code/config changes:
  - Added debug knobs in `sample_lm_multihost.py`:
    - `force_post_generate_block_until_ready`
    - `force_post_generate_barrier`
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h1_postgen_sync.yaml`
    - Enabled:
      - `force_post_generate_block_until_ready: true`
      - `force_post_generate_barrier: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h1_postgen_sync.yaml 2>&1 | tee /tmp/levanter_run_m52_h1_postgen_sync.log`
- Log:
  - `/tmp/levanter_run_m52_h1_postgen_sync.log`
- Key evidence:
  - Both hosts execute the new sync steps:
    - `Round 0: post-generate block_until_ready start/done`
    - `Round 0: post-generate barrier start/done`
  - Failure still occurs immediately after round 0 completion:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
    - round 1 reaches `Generate reset: start` then slice failure.
- Result:
  - **FAILED** with unchanged signature.
- Takeaway:
  - Simple post-`generate()` synchronization on all hosts is **not sufficient** to prevent the launch-group mismatch.

### 2026-02-05 - Debug Env Attempt H (H2: early barrier before leader postprocess)

- Hypothesis:
  - If all hosts synchronize before leader-only tracker/logging work, the round boundary should stop diverging.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h2_early_barrier.yaml`
  - Key flags:
    - `barrier_before_leader_postprocess: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h2_early_barrier.yaml 2>&1 | tee /tmp/levanter_run_m52_h2_early_barrier.log`
- Log:
  - `/tmp/levanter_run_m52_h2_early_barrier.log`
- Key evidence:
  - Barrier executes on both hosts:
    - `Round 0: pre-postprocess barrier start/done`
  - Leader enters postprocess path:
    - `Round 0: metrics log start`
    - `Round 0: samples table start`
  - Then same TPU signature:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Host 1 reaches round 1 generate while host 0 is still in leader postprocess:
    - `Round 1: generate start` (host 1)
    - later `Round 0: samples table done` (host 0)
- Result:
  - **FAILED** with unchanged signature.
- Takeaway:
  - Early barrier alone does not prevent launch-group mismatch once leader-only postprocess begins.

### 2026-02-05 - Debug Env Attempt I (H3: skip leader postprocess entirely)

- Hypothesis:
  - Leader-only tracker/logging path is the trigger; removing it should stabilize round transition.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h3_skip_leader_postprocess.yaml`
  - Key flags:
    - `skip_leader_postprocess: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h3_skip_leader_postprocess.yaml 2>&1 | tee /tmp/levanter_run_m52_h3_skip_leader_postprocess.log`
- Log:
  - `/tmp/levanter_run_m52_h3_skip_leader_postprocess.log`
- Key evidence:
  - `Round 0: leader postprocess skipped by config`
  - Both hosts start round 1 cleanly (`Round 1: generate start` on both hosts).
  - Round 1 also completes (`Round 1: generate done`).
  - Launcher terminates cleanly: `Job finished with no error.`
- Result:
  - **PASSED**.
- Takeaway:
  - Removing leader postprocess avoids the failure window.

### 2026-02-05 - Debug Env Attempt J (H4: skip samples table only)

- Hypothesis:
  - `wandb.Table`/samples logging is the specific trigger.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h4_skip_samples_only.yaml`
  - Key flags:
    - `skip_samples_table: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h4_skip_samples_only.yaml 2>&1 | tee /tmp/levanter_run_m52_h4_skip_samples_only.log`
- Log:
  - `/tmp/levanter_run_m52_h4_skip_samples_only.log`
- Key evidence:
  - Leader still does metrics path:
    - `Round 0: metrics log start`
    - `Round 0: metrics log done`
  - Same launch-id mismatch appears immediately after:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts but slice fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Disabling samples table alone is not sufficient.

### 2026-02-05 - Debug Env Attempt K (H5: skip metrics + skip samples)

- Hypothesis:
  - The combined leader tracker logging path (metrics and samples) is the trigger.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h5_skip_metrics_and_samples.yaml`
  - Key flags:
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h5_skip_metrics_and_samples.yaml 2>&1 | tee /tmp/levanter_run_m52_h5_skip_metrics_and_samples.log`
- Log:
  - `/tmp/levanter_run_m52_h5_skip_metrics_and_samples.log`
- Key evidence:
  - `Round 0: metrics log skipped by config`
  - Both hosts enter round 1 normally.
  - Round 1 completes.
  - Launcher ends cleanly: `Job finished with no error.`
- Result:
  - **PASSED**.
- Takeaway:
  - Removing both metrics and samples logging stabilizes round transition.

### 2026-02-05 - Debug Env Attempt L (H6: skip metrics only, keep samples)

- Hypothesis:
  - Metrics logging alone is the trigger; removing it should be enough.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h6_skip_metrics_only.yaml`
  - Key flags:
    - `skip_metrics_log: true`
    - `skip_samples_table: false` (default)
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h6_skip_metrics_only.yaml 2>&1 | tee /tmp/levanter_run_m52_h6_skip_metrics_only.log`
- Log:
  - `/tmp/levanter_run_m52_h6_skip_metrics_only.log`
- Key evidence:
  - `Round 0: metrics log skipped by config`
  - Leader still executes samples table:
    - `Round 0: samples table start`
    - `Round 0: samples table done`
  - Same launch-id mismatch appears right after:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Metrics logging is not the sole trigger; samples path can still trigger the failure.

### 2026-02-05 - Debug Env Attempt M (H7: force samples rows, bypass `wandb.Table`)

- Hypothesis:
  - `wandb.Table` construction/serialization is the trigger; forcing plain row logging should avoid the failure.
- Code/config changes:
  - Added config flag in `sample_lm_multihost.py`:
    - `force_samples_rows_log: bool = False`
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h7_force_samples_rows.yaml`
    - Enabled:
      - `force_samples_rows_log: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h7_force_samples_rows.yaml 2>&1 | tee /tmp/levanter_run_m52_h7_force_samples_rows.log`
- Log:
  - `/tmp/levanter_run_m52_h7_force_samples_rows.log`
- Key evidence:
  - Round 0 completes.
  - Forced path executes:
    - `Round 0: samples rows log start (forced)`
    - `Round 0: samples rows log done (forced)`
  - Immediately after, same TPU signature:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Failure is not specific to `wandb.Table`; plain samples rows logging still triggers the same boundary failure.

### 2026-02-05 - Debug Env Attempt N (H8: noop tracker backend)

- Hypothesis:
  - W&B backend/network side effects are the trigger; using `tracker.type: noop` should avoid failure.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h8_tracker_noop.yaml`
  - Key change:
    - `trainer.tracker.type: noop`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h8_tracker_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h8_tracker_noop.log`
- Log:
  - `/tmp/levanter_run_m52_h8_tracker_noop.log`
- Key evidence:
  - Round 0 completes and leader postprocess runs:
    - `Round 0: metrics log start/done`
    - `Round 0: samples table start/done`
  - Same launch-group failure appears right after:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice health fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Failure is not fixed by swapping tracker backend to `noop`; backend integration alone is not the root cause.

### 2026-02-05 - Debug Env Attempt O (H9: metrics-only + noop tracker)

- Hypothesis:
  - If tracker backend side effects were the issue, minimal metrics-only logging on `noop` should pass.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h9_metrics_only_noop.yaml`
  - Key flags:
    - `trainer.tracker.type: noop`
    - `skip_samples_table: true`
    - metrics logging enabled (default)
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h9_metrics_only_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h9_metrics_only_noop.log`
- Log:
  - `/tmp/levanter_run_m52_h9_metrics_only_noop.log`
- Key evidence:
  - Round 0 completes.
  - Leader performs only metrics log path:
    - `Round 0: metrics log start`
    - `Round 0: metrics log done`
  - Same TPU failure immediately follows:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Even minimal metrics logging with `noop` tracker is sufficient to reproduce the failure.

### 2026-02-05 - Debug Env Attempt P (H10: samples-only rows, metrics disabled)

- Hypothesis:
  - Samples logging alone (without metrics and without `wandb.Table`) might be safe.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h10_samples_only_rows.yaml`
  - Key flags:
    - `skip_metrics_log: true`
    - `force_samples_rows_log: true`
    - `skip_samples_table: false`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h10_samples_only_rows.yaml 2>&1 | tee /tmp/levanter_run_m52_h10_samples_only_rows.log`
- Log:
  - `/tmp/levanter_run_m52_h10_samples_only_rows.log`
- Key evidence:
  - Round 0 completes.
  - Metrics are skipped:
    - `Round 0: metrics log skipped by config`
  - Samples rows forced path runs:
    - `Round 0: samples rows log start (forced)`
    - `Round 0: samples rows log done (forced)`
  - Same TPU failure immediately follows:
    - `Core halted unexpectedly ... TensorCoreSequencer ... scheckne`
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice health fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Samples logging alone is sufficient to trigger the same failure signature.

### 2026-02-05 - Debug Env Attempt Q (H11: defer tracker logs until after rounds)

- Hypothesis:
  - In-loop tracker logging is the destabilizing operation; if we defer all tracker logs until after rounds complete, round-boundary failure should disappear.
- Code/config changes:
  - Added config flag in `sample_lm_multihost.py`:
    - `defer_tracker_logs_until_end: bool = False`
  - Behavior when enabled:
    - Round-loop metrics/samples payloads are buffered.
    - `levanter.tracker.log(...)` calls are flushed once after the rounds finish.
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h11_defer_tracker_logs.yaml`
    - Enabled:
      - `defer_tracker_logs_until_end: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h11_defer_tracker_logs.yaml 2>&1 | tee /tmp/levanter_run_m52_h11_defer_tracker_logs.log`
- Log:
  - `/tmp/levanter_run_m52_h11_defer_tracker_logs.log`
- Key evidence:
  - Round 0 completes and both hosts enter round 1:
    - `Round 0: generate done`
    - `Round 1: generate start` on both hosts
  - Round 1 completes:
    - `Round 1: generate done`
  - Deferred flush executes once:
    - `Deferred tracker log flush start (4 entries)`
    - `Deferred tracker log flush done`
  - Launcher ends cleanly:
    - `Job finished with no error.`
  - Note: TPU teardown still prints `Failed to unregister debug metadata: FAILED_PRECONDITION: Unexpected run_id ...`, but these were non-fatal in this run.
- Result:
  - **PASSED**.
- Takeaway:
  - Deferring tracker logs out of the round loop eliminates the round-boundary launch-id mismatch in this workload.

### 2026-02-06 - Debug Env Attempt R (H12: defer tracker logs + noop tracker)

- Hypothesis:
  - If H11 is truly backend-agnostic, combining deferred logging with `tracker.type: noop` should also pass.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h12_defer_tracker_noop.yaml`
  - Key flags:
    - `trainer.tracker.type: noop`
    - `defer_tracker_logs_until_end: true`
- Commands:
  - First attempt:
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h12_defer_tracker_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h12_defer_tracker_noop.log`
  - Rerun after cleanup:
    - `gcloud alpha compute tpus tpu-vm ssh simpo_worker_2 --quiet --worker=all --zone=us-central1-a --command='sudo docker rm -f levanter || true'`
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h12_defer_tracker_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h12_defer_tracker_noop_rerun.log`
- Logs:
  - First attempt (inconclusive infra interruption):
    - `/tmp/levanter_run_m52_h12_defer_tracker_noop.log`
  - Rerun (valid):
    - `/tmp/levanter_run_m52_h12_defer_tracker_noop_rerun.log`
- Key evidence (rerun):
  - Round 0 completes and both hosts enter round 1.
  - Round 1 completes (`Round 1: generate done` on both hosts).
  - Deferred flush executes:
    - `Deferred tracker log flush start (4 entries)`
    - `Deferred tracker log flush done`
  - Launcher ends cleanly:
    - `Job finished with no error.`
  - No `unexpected peer ... different launch id` / no round-boundary `FAILED_PRECONDITION` signature.
  - Non-fatal teardown warnings remain (`Failed to unregister debug metadata: Unexpected run_id`).
- Result:
  - **PASSED** (rerun).
- Takeaway:
  - Deferred logging fix is backend-agnostic; it works with `wandb` (H11) and `noop` (H12).

### 2026-02-06 - Debug Env Attempt S (H13: leader-only timing skew with logs disabled)

- Hypothesis:
  - If the failure is purely due to leader-vs-follower timing skew, introducing a large leader-only delay should reproduce failure even when tracker logging is disabled.
- Code/config changes:
  - Added config flag in `sample_lm_multihost.py`:
    - `leader_postprocess_sleep_sec: float = 0.0`
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h13_sleep_no_logs.yaml`
  - Key flags:
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `leader_postprocess_sleep_sec: 10.0`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h13_sleep_no_logs.yaml 2>&1 | tee /tmp/levanter_run_m52_h13_sleep_no_logs.log`
- Log:
  - `/tmp/levanter_run_m52_h13_sleep_no_logs.log`
- Key evidence:
  - Round 0 completes.
  - Leader executes explicit delay:
    - `Round 0: leader postprocess sleep start (10.000s)`
    - `Round 0: leader postprocess sleep done`
  - Both hosts still enter round 1 cleanly:
    - `Round 1: generate start` on both hosts
  - Round 1 completes and leader sleeps again for 10s.
  - Launcher ends cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no round-boundary `FAILED_PRECONDITION` signature.
- Result:
  - **PASSED**.
- Takeaway:
  - Timing skew alone (even a forced 10s leader-only delay) is not sufficient to trigger the launch-group mismatch when in-loop tracker logging is disabled.

### 2026-02-06 - Debug Env Attempt T (H14 run1: baseline config after promoting deferred logs default)

- Hypothesis:
  - If deferred logging is the real fix, the original failing baseline config should pass unchanged once deferred logging is default.
- Code change:
  - In `sample_lm_multihost.py`, set:
    - `defer_tracker_logs_until_end: bool = True` (default)
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h14_default_defer_baseline_run1.log`
- Log:
  - `/tmp/levanter_run_m52_h14_default_defer_baseline_run1.log`
- Key evidence:
  - Baseline round boundary succeeds:
    - `Round 0: generate done`
    - `Round 1: generate start` on both hosts
  - Full 2-round run completes:
    - `Round 1: generate done` on both hosts
  - Deferred flush executes:
    - `Deferred tracker log flush start (4 entries)`
    - `Deferred tracker log flush done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No round-boundary `unexpected peer ... different launch id` / no `FAILED_PRECONDITION` continuator-halted signature.
- Result:
  - **PASSED** (run 1).
- Takeaway:
  - The original baseline config now passes with no special debug config changes when deferred logging is default.

### 2026-02-06 - Debug Env Attempt U (H14 run2: baseline regression repeat with default deferred logs)

- Hypothesis:
  - H14 run1 was not a one-off; unchanged baseline should pass again with default deferred logging.
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h14_default_defer_baseline_run2.log`
- Log:
  - `/tmp/levanter_run_m52_h14_default_defer_baseline_run2.log`
- Key evidence:
  - Round boundary succeeds on both hosts:
    - `Round 0: generate done`
    - `Round 1: generate start`
  - Round 1 completes:
    - `Round 1: generate done`
  - Deferred flush executes:
    - `Deferred tracker log flush start (4 entries)`
    - `Deferred tracker log flush done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer ... different launch id` / no round-boundary continuator-halted signature.
- Result:
  - **PASSED** (run 2).
- Takeaway:
  - Baseline pass with default deferred logging is reproducible (2/2).

### 2026-02-06 - Debug Env Attempt V (H15: legacy-mode control with deferred logging disabled)

- Hypothesis:
  - Forcing legacy behavior (`defer_tracker_logs_until_end: false`) should reproduce the original failure.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml`
  - Key change:
    - `defer_tracker_logs_until_end: false`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml 2>&1 | tee /tmp/levanter_run_m52_h15_legacy_defer_false.log`
- Log:
  - `/tmp/levanter_run_m52_h15_legacy_defer_false.log`
- Key evidence:
  - Round 0 leader postprocess runs in-loop:
    - `Round 0: metrics log start/done`
    - `Round 0: samples table start/done`
  - Immediately after, failure signature returns:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 begins and reset starts on both hosts, then slice fails:
    - `Round 1: generate start`
    - `Generate reset: start`
    - `Slice health check failed ...`
- Result:
  - **FAILED** (expected legacy regression).
- Takeaway:
  - Disabling deferred logging restores the original failure.

### 2026-02-06 - Debug Env Attempt W (H16: minimal in-loop tracker probe, wandb backend)

- Hypothesis:
  - If call timing is the root trigger, a single minimal in-loop `tracker.log` scalar should still fail.
- Code/config changes:
  - Added config flag in `sample_lm_multihost.py`:
    - `emit_minimal_tracker_probe: bool = False`
  - Added probe path in leader postprocess:
    - logs `{"sample/minimal_probe": float(round_index)}` via `levanter.tracker.log(...)` when enabled.
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h16_minimal_probe_wandb.yaml`
  - Key flags:
    - `defer_tracker_logs_until_end: false`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: true`
    - `trainer.tracker.type: wandb`
- Commands:
  - First attempt:
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h16_minimal_probe_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_h16_minimal_probe_wandb.log`
  - Rerun (valid after cleanup):
    - `gcloud alpha compute tpus tpu-vm ssh simpo_worker_2 --quiet --worker=all --zone=us-central1-a --command='sudo docker rm -f levanter || true'`
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h16_minimal_probe_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_h16_minimal_probe_wandb_rerun.log`
- Logs:
  - First attempt (inconclusive transport interruption):
    - `/tmp/levanter_run_m52_h16_minimal_probe_wandb.log`
  - Rerun (valid):
    - `/tmp/levanter_run_m52_h16_minimal_probe_wandb_rerun.log`
- Key evidence (rerun):
  - Round 0 completes:
    - `Round 0: generate done`
    - `Round 0: metrics log skipped by config`
    - `Round 0: minimal tracker probe start/done`
  - Immediately after probe call:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then fails:
    - `Round 1: generate start`
    - `Slice health check failed ...`
- Result:
  - **FAILED** (rerun).
- Takeaway:
  - A single minimal scalar in-loop tracker call is sufficient to trigger the failure; complex payloads are not required.

### 2026-02-06 - Debug Env Attempt X (H17: minimal in-loop tracker probe, noop backend)

- Hypothesis:
  - If backend effects are not required, the same minimal in-loop probe should fail even with `tracker.type: noop`.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h17_minimal_probe_noop.yaml`
  - Key flags:
    - `defer_tracker_logs_until_end: false`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: true`
    - `trainer.tracker.type: noop`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h17_minimal_probe_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h17_minimal_probe_noop.log`
- Log:
  - `/tmp/levanter_run_m52_h17_minimal_probe_noop.log`
- Key evidence:
  - Round 0 completes and probe executes:
    - `Round 0: generate done`
    - `Round 0: metrics log skipped by config`
    - `Round 0: minimal tracker probe start/done`
  - Immediate boundary failure repeats:
    - `An unexpected peer shows up in the launch group with a different launch id ...`
    - repeated `FAILED_PRECONDITION: The program continuator has halted unexpectedly`
  - Round 1 starts then slice health fails.
- Result:
  - **FAILED**.
- Takeaway:
  - Failure is backend-agnostic and tied to in-loop tracker call timing/path, not W&B-specific side effects or payload complexity.

### 2026-02-06 - Debug Env Attempt Y (H18: teardown warning correlation check)

- Hypothesis:
  - `Failed to unregister debug metadata: FAILED_PRECONDITION: Unexpected run_id` warnings are non-fatal teardown noise, not the trigger for the round-boundary crash.
- Method:
  - Correlated warning presence across known PASS and FAIL logs using `rg` counts.
  - PASS set:
    - `/tmp/levanter_run_m52_h11_defer_tracker_logs.log`
    - `/tmp/levanter_run_m52_h12_defer_tracker_noop_rerun.log`
    - `/tmp/levanter_run_m52_h13_sleep_no_logs.log`
    - `/tmp/levanter_run_m52_h14_default_defer_baseline_run1.log`
    - `/tmp/levanter_run_m52_h14_default_defer_baseline_run2.log`
  - FAIL set:
    - `/tmp/levanter_run_m52_h15_legacy_defer_false.log`
    - `/tmp/levanter_run_m52_h16_minimal_probe_wandb_rerun.log`
    - `/tmp/levanter_run_m52_h17_minimal_probe_noop.log`
- Key evidence:
  - PASS logs:
    - all include `Job finished with no error`
    - all include non-zero `Unexpected run_id` warning counts (`8` to `40`)
  - FAIL logs:
    - all include `Slice health check failed`
    - all have `Unexpected run_id` warning count `0`
- Result:
  - **PASSED** (hypothesis supported).
- Takeaway:
  - `Unexpected run_id` warnings occur in successful teardown paths and are not the root trigger for the round-boundary launch-id mismatch crash.

### 2026-02-06 - Debug Env Attempt Z (H19: minimal probe relocated via deferred logging)

- Hypothesis:
  - If call-site timing/location is the trigger, the same minimal probe should be safe when deferred until after rounds complete.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h19_minimal_probe_deferred_wandb.yaml`
  - Key flags:
    - `defer_tracker_logs_until_end: true`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: true`
    - `trainer.tracker.type: wandb`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h19_minimal_probe_deferred_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_h19_minimal_probe_deferred_wandb.log`
- Log:
  - `/tmp/levanter_run_m52_h19_minimal_probe_deferred_wandb.log`
- Key evidence:
  - Round 0 completes and in-loop probe markers execute:
    - `Round 0: generate done`
    - `Round 0: minimal tracker probe start/done`
  - Round 1 starts cleanly on both hosts:
    - `Round 1: generate start`
  - Round 1 completes and deferred flush executes:
    - `Round 1: generate done`
    - `Deferred tracker log flush start (2 entries)`
    - `Deferred tracker log flush done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no round-boundary `FAILED_PRECONDITION` / no slice health failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Same probe payload is safe when deferred; call-site timing/location is the differentiator.

### 2026-02-06 - Debug Env Attempt AA (H20: deferred minimal probe with noop backend)

- Hypothesis:
  - If relocation (defer) is truly backend-agnostic, the deferred minimal probe should also pass with `tracker.type: noop`.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h20_minimal_probe_deferred_noop.yaml`
  - Key flags:
    - `defer_tracker_logs_until_end: true`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: true`
    - `trainer.tracker.type: noop`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h20_minimal_probe_deferred_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h20_minimal_probe_deferred_noop.log`
- Log:
  - `/tmp/levanter_run_m52_h20_minimal_probe_deferred_noop.log`
- Key evidence:
  - Round 0 completes and probe executes:
    - `Round 0: generate done`
    - `Round 0: minimal tracker probe start/done`
  - Round 1 starts and completes cleanly:
    - `Round 1: generate start`
    - `Round 1: generate done`
  - Deferred flush executes:
    - `Deferred tracker log flush start (2 entries)`
    - `Deferred tracker log flush done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no round-boundary `FAILED_PRECONDITION` / no slice health failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Deferred minimal-probe relocation is backend-agnostic (`wandb` and `noop` both pass).

### 2026-02-06 - Debug Env Attempt AB (H21: hardening guard rejects legacy metrics+samples mode)

- Hypothesis:
  - New startup guard should fail fast on the original unsafe multi-round legacy mode.
- Code change under test:
  - `src/levanter/main/sample_lm_multihost.py`
  - Added `_validate_tracker_logging_safety(...)` and invoked it during startup config validation.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml 2>&1 | tee /tmp/levanter_run_m52_h21_guard_legacy_reject.log`
- Log:
  - `/tmp/levanter_run_m52_h21_guard_legacy_reject.log`
- Key evidence:
  - Startup validation fails on both hosts:
    - `Config validation failed`
    - `ValueError: Unsafe tracker logging configuration for multi-host multi-round sampling ...`
  - No round execution markers are reached.
- Result:
  - **PASSED** (guard behavior as intended).
- Takeaway:
  - Unsafe legacy path is blocked early with explicit guidance.

### 2026-02-06 - Debug Env Attempt AC (H22: hardening guard rejects minimal-probe-only unsafe mode)

- Hypothesis:
  - Guard should also reject unsafe config where only minimal in-loop probe is enabled.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h16_minimal_probe_wandb.yaml`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h16_minimal_probe_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_h22_guard_probe_reject.log`
- Log:
  - `/tmp/levanter_run_m52_h22_guard_probe_reject.log`
- Key evidence:
  - Startup validation fails with the same explicit `ValueError` guard message.
  - No round markers reached.
- Result:
  - **PASSED** (guard behavior as intended).
- Takeaway:
  - Guard catches all known unsafe in-loop emission variants, not just metrics/samples combo.

### 2026-02-06 - Debug Env Attempt AD (H23: guard allows defer=false when in-loop emission is disabled)

- Hypothesis:
  - Guard is selective: it should allow `defer=false` when all in-loop tracker paths are disabled.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h23_defer_false_no_inloop_emission.yaml`
  - Key flags:
    - `defer_tracker_logs_until_end: false`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: false`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h23_defer_false_no_inloop_emission.yaml 2>&1 | tee /tmp/levanter_run_m52_h23_defer_false_no_inloop_emission.log`
- Log:
  - `/tmp/levanter_run_m52_h23_defer_false_no_inloop_emission.log`
- Key evidence:
  - Round boundary succeeds:
    - `Round 0: generate done`
    - `Round 1: generate start`
  - Round 1 completes:
    - `Round 1: generate done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator `FAILED_PRECONDITION` / no slice health failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Guard is not over-broad; it permits safe defer=false configurations.

### 2026-02-06 - Debug Env Attempt AE (H24: guard allows single-round legacy in-loop logging)

- Hypothesis:
  - Guard should only block unsafe multi-round mode; single-round legacy in-loop logging should be allowed.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round1_cleanup_none_noop_h24_single_round_legacy_inloop.yaml`
  - Key flags:
    - `n_rounds: 1`
    - `defer_tracker_logs_until_end: false`
    - metrics/samples logging enabled (default behavior)
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round1_cleanup_none_noop_h24_single_round_legacy_inloop.yaml 2>&1 | tee /tmp/levanter_run_m52_h24_single_round_legacy_inloop.log`
- Log:
  - `/tmp/levanter_run_m52_h24_single_round_legacy_inloop.log`
- Key evidence:
  - Single round completes with in-loop logs:
    - `Round 0: generate done`
    - `Round 0: metrics log start/done`
    - `Round 0: samples table start/done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No crash signature appears.
- Result:
  - **PASSED**.
- Takeaway:
  - Guard scope is correctly targeted to multi-round unsafe behavior rather than blocking all legacy usage.

### 2026-02-06 - Debug Env Attempt AF (H25: local regression tests for guard decision matrix)

- Hypothesis:
  - Guard logic should be stable and explicitly covered by fast local tests so regressions are caught before TPU runs.
- Code/config updates:
  - Added tests:
    - `tests/inference/test_sample_lm_multihost_config_guard.py`
  - Added explicit safety flag to baseline real config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml`
    - `defer_tracker_logs_until_end: true`
- Command:
  - `uv run pytest tests/inference/test_sample_lm_multihost_config_guard.py -q`
- Key evidence:
  - `7 passed, 1 warning in 0.02s`
  - Test matrix covers:
    - unsafe reject cases (legacy metrics+samples, probe-only)
    - safe allow cases (deferred, no-emission, single-round, single-host, skip-leader-postprocess)
- Result:
  - **PASSED**.
- Takeaway:
  - Guard behavior is now protected by cheap deterministic unit tests.

### 2026-02-06 - Debug Env Attempt AG (H26: post-guard integration with skip-leader bypass)

- Hypothesis:
  - Guard should permit the explicit `skip_leader_postprocess=true` bypass path, and that path should remain stable.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h26_skip_leader_defer_false.yaml`
  - Key flags:
    - `n_rounds: 2`
    - `defer_tracker_logs_until_end: false`
    - `skip_leader_postprocess: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h26_skip_leader_defer_false.yaml 2>&1 | tee /tmp/levanter_run_m52_h26_skip_leader_defer_false.log`
- Log:
  - `/tmp/levanter_run_m52_h26_skip_leader_defer_false.log`
- Key evidence:
  - Round 0 and round 1 execute:
    - `Round 0: generate done`
    - `Round 1: generate start`
    - `Round 1: generate done`
  - Leader postprocess is skipped by config:
    - `Round 0: leader postprocess skipped by config`
    - `Round 1: leader postprocess skipped by config`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no round-boundary `FAILED_PRECONDITION` / no slice health failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Guard preserves intentional bypass path and it remains stable in integration.

### 2026-02-06 - Debug Env Attempt AH (H27: docs/config hardening note)

- Hypothesis:
  - Safety guidance should live in user-facing docs and the baseline real config to reduce accidental reintroduction of unsafe settings.
- Changes:
  - Baseline config now explicitly sets deferred logging:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml`
    - Added `defer_tracker_logs_until_end: true` and comment.
  - Added docs note:
    - `docs/reference/Configuration.md`
    - Under tracker notes, documented that multi-host multi-round sampling should keep deferred tracker logging enabled unless in-loop tracker emission is disabled.
- Result:
  - **COMPLETED**.
- Takeaway:
  - Safety expectations are now present both in example config and reference docs.

### 2026-02-06 - Debug Env Attempt AI (H28: all-host in-loop probe, noop backend, skip leader postprocess)

- Hypothesis:
  - If in-loop tracker emission is symmetric across hosts, round-boundary failure may not occur even with `defer_tracker_logs_until_end=false`.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h28_allhost_probe_noop_skip_leader.yaml`
  - Key flags:
    - `n_rounds: 2`
    - `defer_tracker_logs_until_end: false`
    - `skip_leader_postprocess: true`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe_all_hosts: true`
    - tracker backend: `noop`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h28_allhost_probe_noop_skip_leader.yaml 2>&1 | tee /tmp/levanter_run_m52_h28_allhost_probe_noop_skip_leader.log`
- Log:
  - `/tmp/levanter_run_m52_h28_allhost_probe_noop_skip_leader.log`
- Key evidence:
  - Both hosts emit all-host probe in both rounds:
    - `Round 0: all-host minimal tracker probe start/done` on host 0 and host 1
    - `Round 1: all-host minimal tracker probe start/done` on host 0 and host 1
  - Reset path remains healthy across round boundary:
    - `Generate reset: start` / `Generate reset: done` on both hosts for both rounds
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
- Result:
  - **PASSED**.
- Takeaway:
  - In-loop tracker emission is not universally unsafe; symmetric all-host emission can be stable in this mode.

### 2026-02-06 - Debug Env Attempt AJ (H29: all-host in-loop probe, wandb backend, skip leader postprocess)

- Hypothesis:
  - H28 behavior should hold under `wandb`; backend should not be the discriminating factor.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h29_allhost_probe_wandb_skip_leader.yaml`
  - Key flags mirror H28, with tracker backend switched to `wandb`.
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h29_allhost_probe_wandb_skip_leader.yaml 2>&1 | tee /tmp/levanter_run_m52_h29_allhost_probe_wandb_skip_leader.log`
- Log:
  - `/tmp/levanter_run_m52_h29_allhost_probe_wandb_skip_leader.log`
- Key evidence:
  - Both hosts emit all-host probe in both rounds:
    - `Round 0: all-host minimal tracker probe start/done` on host 0 and host 1
    - `Round 1: all-host minimal tracker probe start/done` on host 0 and host 1
  - Wandb receives probe metrics from both hosts:
    - `sample/minimal_probe_all_hosts`
    - `sample/minimal_probe_all_hosts_process_index`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - `Failed to unregister debug metadata: ... Unexpected run_id` warnings appear at teardown, but run still succeeds.
  - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Symmetric all-host in-loop logging also passes with `wandb`; this further weakens any backend-specific failure theory.

### 2026-02-06 - Debug Env Attempt AK (H30: all-host in-loop probe, noop backend, leader postprocess enabled)

- Hypothesis:
  - If tracker emission is symmetric across hosts, the round-boundary failure should not require `skip_leader_postprocess=true`; it should still pass with leader postprocess enabled as long as leader-only tracker emission paths remain disabled.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h30_allhost_probe_noop_leader_enabled_guard_bypass.yaml`
  - Key flags:
    - `n_rounds: 2`
    - `defer_tracker_logs_until_end: false`
    - `skip_leader_postprocess: false`
    - `skip_metrics_log: true`
    - `skip_samples_table: true`
    - `emit_minimal_tracker_probe: false`
    - `emit_minimal_tracker_probe_all_hosts: true`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h30_allhost_probe_noop_leader_enabled_guard_bypass.yaml 2>&1 | tee /tmp/levanter_run_m52_h30_allhost_probe_noop_leader_enabled_guard_bypass.log`
- Log:
  - `/tmp/levanter_run_m52_h30_allhost_probe_noop_leader_enabled_guard_bypass.log`
- Key evidence:
  - Round boundary still healthy with leader postprocess enabled:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Generate reset: start/done` on both hosts in both rounds
  - All-host probe emits on both hosts in both rounds:
    - `Round 0: all-host minimal tracker probe start/done`
    - `Round 1: all-host minimal tracker probe start/done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
  - `Unexpected run_id` teardown warnings still appear in this passing run.
- Result:
  - **PASSED**.
- Takeaway:
  - The failure is not simply "in-loop tracker call exists"; symmetric all-host in-loop emission survives the round boundary even with leader postprocess enabled.

### 2026-02-06 - Debug Env Attempt AL (H31: guard precision refinement + no-bypass integration recheck)

- Hypothesis:
  - Guard should block only known-unsafe leader-side in-loop tracker emission, not symmetric all-host-only emission; after refinement, H30 topology should run cleanly without any bypass.
- Code/config changes:
  - Guard precision update:
    - `src/levanter/main/sample_lm_multihost.py`
    - Guard now keys on leader-side in-loop tracker paths (`metrics`, `samples`, `emit_minimal_tracker_probe`) instead of all in-loop emissions.
  - Guard test matrix update:
    - `tests/inference/test_sample_lm_multihost_config_guard.py`
    - Added/updated case so all-host-only probe mode is allowed.
  - Removed temporary bypass path (not retained in code).
  - Added config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h31_allhost_probe_noop_leader_enabled.yaml`
- Local validation:
  - `uv run pytest tests/inference/test_sample_lm_multihost_config_guard.py -q`
  - Result: `9 passed, 1 warning in 0.02s`
- TPU command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h31_allhost_probe_noop_leader_enabled.yaml 2>&1 | tee /tmp/levanter_run_m52_h31_allhost_probe_noop_leader_enabled.log`
- Log:
  - `/tmp/levanter_run_m52_h31_allhost_probe_noop_leader_enabled.log`
- Key evidence:
  - Boundary survives without bypass:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Round 1: generate done` on both hosts
  - All-host probe still emits symmetrically:
    - `Round 0/1: all-host minimal tracker probe start/done` on both hosts
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
- Result:
  - **PASSED**.
- Takeaway:
  - Guard can be narrowed to leader-side in-loop emission without losing protection, and the refined policy matches observed runtime behavior.

### 2026-02-06 - Debug Env Attempt AM (H32: backend parity recheck, wandb + leader-enabled all-host probe)

- Hypothesis:
  - After guard refinement, the leader-enabled all-host-only mode should remain stable on `wandb` (not just `noop`), confirming backend independence in this topology.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h32_allhost_probe_wandb_leader_enabled.yaml`
  - Key flags mirror H31, with tracker backend switched to `wandb`.
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h32_allhost_probe_wandb_leader_enabled.yaml 2>&1 | tee /tmp/levanter_run_m52_h32_allhost_probe_wandb_leader_enabled.log`
- Log:
  - `/tmp/levanter_run_m52_h32_allhost_probe_wandb_leader_enabled.log`
- Key evidence:
  - Round boundary remains healthy:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Generate reset: start/done` on both hosts in both rounds
  - Round 1 completes on both hosts:
    - `Round 1: generate done`
    - `Round 1: all-host minimal tracker probe start/done`
  - Wandb receives all-host probe metrics from both hosts:
    - `sample/minimal_probe_all_hosts`
    - `sample/minimal_probe_all_hosts_process_index`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
  - `Unexpected run_id` teardown warnings still appear in this passing run.
- Result:
  - **PASSED**.
- Takeaway:
  - Refined-guard behavior generalizes across tracker backends in leader-enabled all-host-only mode.

### 2026-02-06 - Debug Env Attempt AN (H33: post-refinement regression pair)

- Hypothesis:
  - After guard refinement, known-unsafe legacy mode should still be rejected early, while baseline deferred mode should still pass end-to-end.
- Subtest A (unsafe expected reject):
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml`
  - Command:
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop_h15_legacy_defer_false.yaml 2>&1 | tee /tmp/levanter_run_m52_h33_unsafe_guard_reject.log`
  - Log:
    - `/tmp/levanter_run_m52_h33_unsafe_guard_reject.log`
  - Key evidence:
    - Both hosts fail fast during validation:
      - `Config validation failed`
      - `ValueError: Unsafe tracker logging configuration ...`
    - Launcher exits with worker non-zero status as expected.
  - Result:
    - **EXPECTED REJECT / PASSED**.
- Subtest B (baseline expected pass):
  - Config:
    - `config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml`
  - Command:
    - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_5prompts_2048_reset_physical_round2_cleanup_none_noop.yaml 2>&1 | tee /tmp/levanter_run_m52_h33_baseline_safe_recheck.log`
  - Log:
    - `/tmp/levanter_run_m52_h33_baseline_safe_recheck.log`
  - Key evidence:
    - Round boundary remains healthy:
      - `Round 0: generate done`
      - `Round 1: generate start`
      - `Generate reset: start/done` on both hosts
    - Deferred tracker path flushes at end:
      - `Deferred tracker log flush start (4 entries)`
      - `Deferred tracker log flush done`
    - Launcher exits cleanly:
      - `Job finished with no error.`
    - No `unexpected peer` / no continuator halt / no slice health failure / no config-validation failure.
    - `Unexpected run_id` teardown warnings still appear in this passing run.
  - Result:
    - **PASSED**.
- Takeaway:
  - Guard remains effective against unsafe legacy mode after refinement, and baseline deferred behavior remains stable.

### 2026-02-06 - Debug Env Attempt AO (H34: lock-in stress test with 20 prompts x 2048 tokens)

- Hypothesis:
  - The refined safe mode should scale to a materially larger real workload without reintroducing the round-boundary failure.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_20prompts_2048_reset_physical_round2_cleanup_none_noop_lockin.yaml`
  - Key settings:
    - `n_rounds: 2`
    - `max_new_tokens: 2048`
    - `prompts: 20`
    - `defer_tracker_logs_until_end: true`
    - `skip_samples_table: true`
    - `trainer.tracker.type: noop`
    - `engine.max_seqs: 20`
    - `engine.max_seqs_in_prefill: 20`
    - `engine.max_pages: 960`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_20prompts_2048_reset_physical_round2_cleanup_none_noop_lockin.yaml 2>&1 | tee /tmp/levanter_run_m52_final_20prompts_2048_lockin.log`
- Log:
  - `/tmp/levanter_run_m52_final_20prompts_2048_lockin.log`
- Key evidence:
  - Round boundary remains healthy at 20-prompt scale:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Generate reset: start/done` on both hosts for round 1
  - Per-round generation target reached:
    - `Assembling outputs: total_generated=40960` (round 0 and round 1)
  - Round 1 completes on both hosts:
    - `Round 1: generate done`
  - Deferred logging flush executes:
    - `Deferred tracker log flush start (2 entries)`
    - `Deferred tracker log flush done`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator-halted signature / no slice health failure / no config-validation failure.
  - `Unexpected run_id` teardown warnings still appear in this passing run.
- Result:
  - **PASSED**.
- Takeaway:
  - The lock-in mode remains stable under larger prompt-count + long-generation stress.

### 2026-02-06 - Debug Env Attempt AP (H35: lock-in stress parity with wandb at 20 prompts x 2048 tokens)

- Hypothesis:
  - The same 20x2048 lock-in workload should remain stable on `wandb` (not just `noop`), confirming that large-scale stability is backend-agnostic under deferred logging.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_20prompts_2048_reset_physical_round2_cleanup_none_lockin_wandb.yaml`
  - Key settings:
    - `n_rounds: 2`
    - `max_new_tokens: 2048`
    - `prompts: 20`
    - `defer_tracker_logs_until_end: true`
    - `skip_samples_table: true`
    - `trainer.tracker.type: wandb`
    - `engine.max_seqs: 20`
    - `engine.max_seqs_in_prefill: 20`
    - `engine.max_pages: 960`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_20prompts_2048_reset_physical_round2_cleanup_none_lockin_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_final_20prompts_2048_lockin_wandb.log`
- Log:
  - `/tmp/levanter_run_m52_final_20prompts_2048_lockin_wandb.log`
- Key evidence:
  - Round boundary remains healthy:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Generate reset: start/done` on both hosts for round 1
  - Per-round generation target reached:
    - `Assembling outputs: total_generated=40960` (round 0 and round 1)
  - Round 1 completes on both hosts:
    - `Round 1: generate done`
  - Deferred tracker flush executes and `wandb` run finalizes:
    - `Deferred tracker log flush start (2 entries)`
    - `Deferred tracker log flush done`
    - `Finishing wandb run...`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator-halted signature / no slice health failure / no config-validation failure.
  - `Unexpected run_id` teardown warnings still appear in this passing run.
- Result:
  - **PASSED**.
- Takeaway:
  - Large lock-in workload stability holds on `wandb` as well; mitigation is not backend-specific at 20x2048 scale.

### 2026-02-06 - Debug Env Attempt AQ (H36: lock-in stress parity with wandb at 20 prompts x 4096 tokens)

- Hypothesis:
  - The same deferred-lock-in recipe should hold when max response length doubles to 4096 tokens at 20-prompt scale.
- Config:
  - `config/sampler/sample_llama8b_multihost_real_20prompts_4096_reset_physical_round2_cleanup_none_lockin_wandb.yaml`
  - Key settings:
    - `n_rounds: 2`
    - `max_new_tokens: 4096`
    - `prompts: 20`
    - `defer_tracker_logs_until_end: true`
    - `skip_samples_table: true`
    - `trainer.tracker.type: wandb`
    - `engine.max_seq_len: 4608`
    - `engine.max_pages: 1600`
    - `engine.max_seqs: 20`
    - `engine.max_seqs_in_prefill: 20`
- Command:
  - `python infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -e JAX_TRACEBACK_FILTERING off -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_20prompts_4096_reset_physical_round2_cleanup_none_lockin_wandb.yaml 2>&1 | tee /tmp/levanter_run_m52_final_20prompts_4096_lockin_wandb.log`
- Log:
  - `/tmp/levanter_run_m52_final_20prompts_4096_lockin_wandb.log`
- Key evidence:
  - Both rounds complete and boundary remains healthy:
    - `Round 0: generate done` on both hosts
    - `Round 1: generate start` on both hosts
    - `Round 1: generate done` on both hosts
  - Per-round generation target reached at new length:
    - `Assembling outputs: total_generated=81920` (round 0 and round 1)
  - Round 1 peak page usage remains below configured cap:
    - `DecodeStats[after_decode]: ... pages_in_use=1300 free=300 ...` (with `max_pages=1600`)
  - Deferred tracker flush executes and run finalizes:
    - `Deferred tracker log flush start (2 entries)`
    - `Deferred tracker log flush done`
    - `Finishing wandb run...`
  - Launcher exits cleanly:
    - `Job finished with no error.`
  - No `unexpected peer` / no continuator-halted signature / no slice health failure / no config-validation failure.
  - `Unexpected run_id` teardown warnings still appear in this passing run.
- Result:
  - **PASSED**.
- Takeaway:
  - Deferred lock-in mode remains stable at 20 prompts with 4096-token responses; the mitigation scales to longer generation length.

## Final Hypothesis (locked after H36)

Current evidence supports:
**the round-boundary crash is triggered by asymmetric in-loop tracker emission at round boundaries (especially leader-only emission paths); payload type and backend are secondary. Deferring tracker logs until after rounds remains the robust mitigation.**

What is now established:

1. In-loop logging fails across variants:
   - metrics-only (`wandb` and `noop`) **FAIL**
   - samples-only (`Table` and forced rows) **FAIL**
   - minimal scalar probe-only (`wandb` and `noop`) **FAIL**
2. Removing in-loop tracker calls passes:
   - skip metrics + skip samples **PASS**
   - skip leader postprocess **PASS**
3. Deferring tracker logs until end passes:
   - deferred + `wandb` **PASS**
   - deferred + `noop` **PASS**
4. Artificial timing skew without tracker logs still passes:
   - 10s leader sleep with logs disabled **PASS**
5. Baseline regression with default-deferred behavior passes:
   - original baseline config, unchanged, **PASS** (run 1, run 2, and post-refinement run 3)
6. Legacy control with deferred disabled fails:
   - baseline-equivalent config + `defer_tracker_logs_until_end: false` **FAIL**
7. Teardown warning correlation:
   - `Unexpected run_id` warnings appear in PASS logs and are absent in crash logs.
8. Minimal probe relocation check:
   - minimal probe + deferred logging **PASS** while minimal probe + in-loop logging **FAIL**
   - validated on both `wandb` and `noop` backends
9. Hardening guard behavior:
   - unsafe multi-round in-loop modes are rejected early with explicit `ValueError` (**PASS**)
   - safe defer=false/no-emission and single-round legacy in-loop modes remain allowed (**PASS**)
10. Regression hardening:
   - local guard decision-matrix tests pass (9/9) and baseline config now explicitly sets deferred logging true.
11. Skip-leader integration:
   - defer=false + skip_leader_postprocess=true remains allowed and stable (**PASS**).
12. Docs/config safety communication:
   - baseline config and reference docs now explicitly encode the safe setting guidance.
13. Symmetric all-host in-loop emission:
   - all-host minimal probe + defer=false + skip_leader_postprocess=true **PASS** on `noop`.
   - all-host minimal probe + defer=false + skip_leader_postprocess=true **PASS** on `wandb`.
14. Teardown warning non-causality remains consistent:
   - `Unexpected run_id` teardown warnings can co-occur with successful completion.
15. Leader-enabled symmetry check:
   - all-host minimal probe + defer=false + skip_leader_postprocess=false **PASS** on `noop`.
16. Guard precision refinement check:
   - narrowed guard (leader-side in-loop emission only) + no-bypass H31 integration **PASS**.
17. Backend parity after guard refinement:
   - leader-enabled all-host-only mode also **PASS** on `wandb` (H32), matching `noop`.
18. Post-refinement regression pair:
   - known-unsafe legacy config still fast-rejected by guard (**PASS** expected reject).
   - baseline deferred config still end-to-end stable (**PASS**).
19. 20-prompt stress lock-in (`noop`):
   - 20 prompts x 2048 tokens x 2 rounds passes end-to-end with deferred logging and scaled engine limits (**PASS**).
20. 20-prompt stress lock-in (`wandb` parity):
   - the same 20 prompts x 2048 tokens x 2 rounds workload also passes end-to-end (**PASS**).
21. 20-prompt long-response lock-in (`wandb`, 4096):
   - 20 prompts x 4096 tokens x 2 rounds also passes end-to-end with deferred logging and scaled pages/seq-len limits (**PASS**).

Interpretation:
- Root mechanism is unlikely to be generic leader timing skew.
- It is strongly tied to asymmetric in-loop tracker emission path and avoided by deferred log emission.
- Backend (`wandb` vs `noop`) and payload shape (table/rows/metrics/minimal scalar) are not the discriminating factors.
- `Unexpected run_id` teardown warnings are secondary noise, not the crash cause.
- Moving the same tracker call out of the round loop removes the failure.
- Symmetric all-host emission appears stable in both skip-leader and leader-enabled tested modes.
- Startup guard now enforces known-unsafe leader-side emission patterns without over-blocking symmetric all-host-only mode.
- Guard behavior is now covered by unit tests and validated by targeted TPU integration controls.
- User-facing docs/config now reflect the safe operating mode.
- The lock-in stress config validates that the mitigation scales to larger prompt count and long generation length.
- Large-scale lock-in parity now holds on both `noop` and `wandb`.
- Longer response length lock-in (4096) also remains stable under the same deferred logging strategy.

## Next Experiments (updated queue)

1. **H37: optional strictness guardrail**
   - Consider warning-level log (not block) when `emit_minimal_tracker_probe_all_hosts=true` with multi-round `defer=false`, documenting that this mode is experimentally verified but still non-default.
2. **H38: optional cleanup/naming pass**
   - Rename `..._h30_..._guard_bypass.yaml` to a neutral name (no bypass wording) to match current code reality.
