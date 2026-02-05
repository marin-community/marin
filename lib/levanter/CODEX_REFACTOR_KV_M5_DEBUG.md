# DO NOT STOP UNTIL THE ROOT CAUSE IS DETERMINISTICALLY IDENTIFIED. RUN AT LEAST TWO FAILURES AND TWO SUCCESSES.
# M5 Debug Notes (Untracked)

Date: 2026-02-05

Purpose: isolate which M5 changes break multihost model loading.

## Baseline context
- Working tree: `origin/fix/simpo-multihost-inference` @ `211b901b83dfdf8162b748e2e6a7d2613fc6e661`.
- Known-good commit: `b00d36d846916b5d14b4ee8684810eb723af1ca8`.
- Command used for all tests (M3 config):
  `python lib/levanter/infra/launch.py --foreground --zone us-central1-a --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- uv run src/levanter/main/sample_lm_multihost.py --config_path config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml 2>&1 | tee <log>`
- TPU: `simpo_worker_2` (v5p-16, us-central1-a).

Baseline results:
- `b00d36...` + M3 command: success (log: `/tmp/levanter_run_m3.log`, ends with `Job finished with no error.`).
- `origin/fix/...` + stash applied: failure during/after shard loading (log: `/tmp/levanter_run_m3_after_stash.log`).
- `origin/fix/...` + stash applied + `git restore lib/levanter/src/levanter/main/sample_lm_multihost.py`: success (log: `/tmp/levanter_run_m3_after_restore.log`).

Conclusion from baseline: regression is inside stash changes to `lib/levanter/src/levanter/main/sample_lm_multihost.py`.

## Isolation steps and results

### Step 1 — logging only (no new barriers, no decode-stats helper)
Changes applied:
- Added leader-only log messages around model load / engine creation, plus per-round “Round X starting...” log.
- No new barriers.
- No `_log_decode_stats` helper or calls.

Result: **SUCCESS**
- Log: `/tmp/levanter_run_m3_logging_only.log`
- Run id: `tbjrfik4`
- W&B run name: `peach-shadow-209`
- Logs include normal decode stats from `InferenceEngine` and end with: `Job finished with no error.`

### Step 2 — add barrier after model load only
Changes applied:
- Same as Step 1 plus: `barrier_sync_with_tag("sample_lm_multihost_after_model_load")`.
- No engine barrier, no decode-stats helper.

Result: **SUCCESS**
- Log: `/tmp/levanter_run_m3_barrier_after_model_load.log`
- Run id: `tuipdzy3`
- W&B run name: `fanciful-rain-210`
- Normal decode iterations; ends with `Job finished with no error.`

### Step 3 — add barrier after engine only
Changes applied:
- Step 1 logging
- Added: `barrier_sync_with_tag("sample_lm_multihost_after_engine")`
- No model-load barrier, no decode-stats helper.

Result: **SUCCESS**
- Log: `/tmp/levanter_run_m3_barrier_after_engine.log`
- Run id: `rp4fgeyi`
- W&B run name: `fanciful-hill-211`
- Normal decode iterations; ends with `Job finished with no error.`

### Step 4 — decode-stats helper + calls (no barriers)
Changes applied:
- Step 1 logging
- Added `_log_decode_stats(engine, stage, is_leader)` helper using `engine.gen_state.decode_state.stats()`.
- Added calls before/after `engine.generate`.
- No new barriers.

Result: **FAILED**
- Log: `/tmp/levanter_run_m3_decode_stats_only.log`
- Run id: `32irhw57`
- W&B run name: `sunny-pond-212`
- Observed behavior:
  - Model shard reads complete (all 4 `model-0000X-of-00004.safetensors` finish).
  - No `RoundStats[...]` lines appear.
  - No decode iterations or `DecodeStats[...]` after reset/prefill are seen.
  - Immediate failure: `##### Command execution on worker {0,1} failed with exit status 1`.
  - `launch.py` raises `CalledProcessError` from the remote docker command.

## Current conclusion
- The only isolated change that fails is the **decode-stats helper + calls**.
- Barriers and extra logging alone are safe.
- Next micro-test (if needed): keep helper defined but remove the calls (to verify call sites are the trigger) OR call the helper only after decode completes.

### Step 5 — helper defined, no calls
Changes applied:
- Kept `_log_decode_stats` helper definition.
- Removed the two call sites around `engine.generate`.
- No barriers.

Result: **SUCCESS**
- Log: `/tmp/levanter_run_m3_helper_no_calls.log`
- Run id: `khoxoevz`
- W&B run name: `amber-surf-213`
- Normal decode iterations and clean exit: `Job finished with no error.`

### Step 6 — decode-stats helper + calls (rerun to test determinism)
Changes applied:
- Same as Step 4: helper defined and calls before/after `engine.generate`.
- No barriers.

Result: **FAILED (reproducible)**
- Log: `/tmp/levanter_run_m3_decode_stats_only_rerun.log`
- Run id: `2pnnfbif`
- W&B run name: `toasty-aardvark-214`
- Observed behavior:
  - Model shard reads complete.
  - No `RoundStats[...]` lines appear.
  - Immediate failure after shard load: `##### Command execution on worker {0,1} failed with exit status 1.`

### Step 7 — helper call before generate only
Changes applied:
- Helper defined.
- **Only** call `_log_decode_stats` before `engine.generate`.
- No barriers.

Result: **FAILED**
- Log: `/tmp/levanter_run_m3_before_only_run1.log`
- Run id: `phrrrdbq`
- W&B run name: `rural-resonance-215`
- Observed behavior:
  - Model shard reads complete.
  - No `RoundStats[...]` lines appear.
  - Failure: `##### Command execution on worker {0,1} failed with exit status 1.`

### Step 8 — helper call after generate only
Changes applied:
- Helper defined.
- **Only** call `_log_decode_stats` after `engine.generate`.
- No barriers.

Result: **HANG/FAIL (manual interrupt)**
- Log: `/tmp/levanter_run_m3_after_only_run1.log`
- Run id: `fkrybtgt`
- W&B run name: `faithful-wildflower-216`
- Observed behavior:
  - Full decode completes; internal `DecodeStats[after_decode]` and `DecodeStats[after_cleanup]` appear.
  - **No** `RoundStats[...]` line appears.
  - Process stops emitting logs after `after_cleanup` and does not exit; interrupted with Ctrl-C.

## Updated conclusion (so far)
- Any **call** to `_log_decode_stats(...)` is suspect.
- Before-only call triggers early failure (exit status 1).
- After-only call appears to **hang after decode cleanup**, likely stuck inside `jax.device_get(engine.gen_state.decode_state.stats())`.
- Helper definition alone is safe.

### Step 9 — call after generate on **all** hosts (remove leader-only guard)
Changes applied:
- Removed early return; `_log_decode_stats` now runs on both leader + follower.
- Kept call **after** `engine.generate`.
- Log format updated to include `role` ("leader"/"follower").

Result: **SUCCESS**
- Log: `/tmp/levanter_run_m3_after_only_allhosts_run1.log`
- Run id: `2gnscxwg`
- W&B run name: `crimson-hill-217`
- Observed behavior:
  - Full decode completes with normal `DecodeStats[iter*]`, `after_decode`, `after_cleanup`.
  - Run finishes cleanly with W&B summary (no exit status 1).
  - No `RoundStats[...]` lines observed in log (logger for this module appears quiet), but the run completes.

## Updated conclusion (after Step 9)
- **Leader-only** `jax.device_get(engine.gen_state.decode_state.stats())` is the likely trigger.
- Running the same stats call on **all hosts** succeeds, which strongly suggests the failure/hang comes from
  reading a globally-sharded JAX array on just one process.

## Latest summary (2026-02-05)
- Before-only call (leader-only) **failed**.
- After-only call (leader-only) **hung** after `after_cleanup`.
- After-only call on **all hosts** **succeeded**.
- Deterministic culprit: `jax.device_get(engine.gen_state.decode_state.stats())` on leader-only.
- Best hypothesis: single-process host read of a **globally sharded** JAX array causes crash/hang.
- Safe workaround confirmed: run the stats call on **all hosts** (even if only leader logs).
