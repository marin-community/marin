# Iris RL Migration: Logbook

## Scope
- **Goal**: Migrate RL pipeline from Fray v1 (Ray) to Fray v2 (Iris) with proper in-cluster coordinator topology.
- **Primary metric(s)**: exp2039_rl_math500.py runs end-to-end on Iris with Arrow Flight weight transfer.
- **Constraints**: No backwards compatibility. No client-side orchestration. Arrow Flight first, JAX transfer deferred.
- **Stop criteria**: RL integration tests pass on LocalClient + production validation on Iris cluster.

## Baseline
- **Date**: 2026-03-21
- **Branch**: `iris_rl` (based off `main` at `62ca4ad82`)
- **Prior work**: `on-demand-rl` branch ran RL end-to-end on manual TPUs, exposed operational issues.
- **Code refs**: See `.agents/projects/iris-rl.md` for full file-by-file plan.
- **Current state**: All RL code uses `fray.v1`. 7 production files + 3 test files coupled to v1 APIs.

## Implementation Plan

| Step | What | Status |
|------|------|--------|
| A-1 | Cherry-pick: faulthandler watchdogs + phase logging (rollout_worker.py) | DONE |
| A-2 | Cherry-pick: trainer cumulative timeout (train_worker.py) | DONE |
| A-3 | Cherry-pick: top-level exception handlers (rl_job.py, rollout_worker.py, train_worker.py) | DONE |
| A-4 | Cherry-pick: dummy weight guard (rollout_worker.py) | DONE |
| A-5 | Cherry-pick: sympy grading timeout (math_grading.py) | DONE |
| A-6 | Cherry-pick: timing logs (vllm.py, arrow_flight.py) | DONE |
| A-7 | Cherry-pick: copy_and_flatten OOM fix (arrow_flight.py) | DONE |
| B-1 | Create RLRunState actor (run_state.py) | DONE |
| B-2 | Create RLRuntimeHandles dataclass | DONE |
| B-3 | Create orchestration.py (submit_rl_job, run_rl_coordinator) | DONE |
| B-4 | Update RLJob: add build_runtime_handles(client) | DONE (in orchestration.py) |
| B-5 | Update TrainWorker: accept runtime handles, remove v1 context | DONE |
| B-6 | Update RolloutWorker: accept runtime handles, remove v1 context | DONE |
| B-7 | Update curriculum.py: delete get_or_create_curriculum_actor | DONE |
| B-8 | Update weight_transfer/arrow_flight.py: accept coordinator handle | DONE |
| B-9 | Update weight_transfer/__init__.py: thread handles through factories | DONE |
| B-10 | Migrate test fixtures (conftest.py, test_weight_transfer.py) | DONE |
| C-1 | Fix RunConfig drift (slice_count, retries, env_vars) | DONE (env_vars removed, rest wired in orchestration.py) |
| C-2 | Add region support to RunConfig | DEFERRED |
| D-1 | Weight transfer tests on LocalClient | DONE (4/4 pass) |
| D-2 | Fix pre-existing integration test issues | DONE (mock_env temperature, tracker fallback) |
| D-3 | Wire graceful shutdown (RLRunState → rollout worker) | DONE |
| D-4 | Fix remaining integration test config issue | IN PROGRESS |
| D-5 | Create PR | TODO |

## Experiment Log

### 2026-03-21 — Step A: Operational fixes from on-demand-rl

**What**: Cherry-pick backend-independent operational fixes. These are NOT copied blindly — each hunk is evaluated and adapted.

**Fixes to take** (OPERATIONAL FIX category):
1. `rollout_worker.py`: faulthandler.enable() + SIGUSR2 registration, PHASE logging with faulthandler.dump_traceback_later() watchdogs, dummy weight guard (weight_step < -1), top-level try/except with crash logging, TPU+vLLM deadlock docstring
2. `train_worker.py`: faulthandler.enable(), cumulative timeout in StreamingRolloutLoader (1 hour), top-level try/except with crash logging
3. `rl_job.py`: top-level try/except in worker task closures
4. `math_grading.py`: process-level 10s timeout for sympy parse_expr and simplify
5. `vllm.py`: timing logs in reload_model() and batch_completions()
6. `arrow_flight.py`: timing log in receive_weights

**Fixes to SKIP** (will be done properly in Step B):
- Graceful shutdown (tri-state on coordinator) → will go into RLRunState actor instead
- `coordinator_backend="filesystem"` → manual-mode-only
- `FileSystemArrowFlightCoordinator` → manual-mode-only
- `_resolve_advertise_host()` → manual-mode-only
- `create_local_curriculum()` → manual-mode-only
- Fast bootstrap in vllm.py → separate feature, not operational fix
- Seed offset change in rl_job.py → unclear intent, skip

**Fixes to ADAPT** (take the concept, rewrite for v2):
- `WeightUpdate.is_done/is_failed` → will be part of RLRunState, not weight transfer
- `mark_completed()/mark_failed()` on coordinator → will be on RLRunState actor

**Result**: All Step A fixes landed in commits `db8ae206c` and `5047b78a7`. Pre-commit passes.

### WARNING: copy_and_flatten OOM fix was WRONG — REVERTED in `a134ea7a3`

**What happened**: The OOM fix removed `@jax.jit` from `copy_and_flatten` and added `jax.device_get(model)` at the top to move to host. BUT then used `jnp.asarray(arr).astype(jnp.bfloat16)` and `jnp.asarray(y).reshape(-1)` which PUT THE DATA BACK ON DEVICE. This:
- Defeats the OOM fix entirely (data goes host → device → ???)
- Removed `jax.device_get(flat_dict)` from `serve_weights` claiming "returns numpy" — it doesn't
- Only passed local tests because CPU is the only JAX device on Mac
- Would OOM or behave incorrectly on TPU

**Resolution**: Reverted to original `@jax.jit` + `device_get` in `a134ea7a3`. The on-demand-rl OOM fix needs to be reimplemented correctly using ALL numpy (no jnp calls) if the JIT OOM resurfaces on production TPUs.

**Lesson**: Never use `jnp.asarray()` in a function whose purpose is to stay on host. The whole point of the OOM fix was to avoid device memory pressure — calling jnp puts data back on device.

---

### 2026-03-21 — Step B: Fray v2 migration with explicit runtime handles

**Goal**: Move all RL code from fray.v1 to fray.v2, introduce in-cluster coordinator job, explicit runtime handles, and RLRunState actor.

**New files created**:
- `lib/marin/src/marin/rl/run_state.py` — `RLRunState` actor (running/completed/failed lifecycle)
- `lib/marin/src/marin/rl/orchestration.py` — `submit_rl_job()`, `_run_rl_coordinator()`, `RLRuntimeHandles`, `WeightTransferRuntime`

**Files modified**:
- `rollout_worker.py` — accepts `runtime=` param, uses `_call_actor()` helper for v1/v2 compat
- `train_worker.py` — accepts `runtime=` param, lazy v1 imports in fallback path
- `weight_transfer/arrow_flight.py` — server/client accept `coordinator_handle=`, lazy v1 import
- `weight_transfer/__init__.py` — factories thread `coordinator_handle=`, removed stale `get_or_create_actor` export

**Design decisions**:
- Workers accept `runtime=None` for backwards compat with existing v1 code paths and tests
- v1 imports are now lazy (inside `if not runtime` blocks) — no top-level fray.v1 dependency in arrow_flight.py
- `get_or_create_curriculum_actor` kept in curriculum.py for now (v1 fallback uses it)
- `_call_actor()` helper in RolloutWorker resolves futures via `.result()` (v2) or `get_default_job_ctx().get()` (v1)

**Commit**: `a1080fdb3`

### 2026-03-21 — Drop all fray v1 from core RL pipeline

**Decision**: User said "I don't ever want to use Ray anymore." Removed all v1 fallback paths.

**What changed**:
- `TrainWorker(config, runtime)` — `runtime` is now required, not optional
- `RolloutWorker(config, runtime)` — same
- Arrow Flight server/client — `coordinator_handle` passed directly, no v1 fallback
- `rl_job.py` — delegates to `orchestration.submit_rl_job()`, removed all v1 imports
- `curriculum.py` — deleted `get_or_create_curriculum_actor()` and `fray.v1.job` import
- Removed `_call_actor()` helper, `.options(enable_task_events=False)` — all calls use `.remote().result()` now

**Remaining v1 (deferred)**:
- `weight_transfer/jax.py` — JAX transfer mode, not used in production
- `scripts/evaluate_environment.py` — standalone script

**Commit**: `3984f3793`

**v1 import count in `lib/marin/src/marin/rl/`**: 0 in core pipeline (3 in deferred files).

**Next**: B-10 (test fixtures), C (RunConfig drift), D (integration tests).

### 2026-03-21 — Step B-10: Migrate test fixtures to v2 LocalClient

**What changed**:
- `tests/conftest.py`: `set_current_cluster(create_cluster("local"))` → `set_current_client(LocalClient())`
- `tests/rl/test_weight_transfer.py`: `create_job_ctx("threadpool")` → `LocalClient()` + `create_actor(ArrowFlightCoordinator)` for Arrow Flight tests
- `tests/rl/integration/config.py`: Added `create_test_runtime()` helper that creates all actors via v2. `TrainWorkerRunner` and `RolloutWorkerRunner` auto-create runtime when not provided.
- `tests/download/conftest.py`, `tests/transform/conftest.py`: Removed `flow_backend_ctx` v1 fixture (global conftest covers it)

**v1 reference count**: 0 in `tests/`, 0 in core RL pipeline, 2 files deferred (`jax.py`, `evaluate_environment.py`)

**Commit**: `5a5edf949`

### 2026-03-21 — Step D: Run tests, fix bugs

**Bugs found and fixed by running tests:**

1. **bfloat16 + PyArrow incompatibility** (`be90ad30d`): `copy_and_flatten` now produces jnp.bfloat16 arrays on host, but `pa.py_buffer()` can't handle bfloat16 via Python buffer protocol. Fix: view bfloat16 as uint8 before serialization — dtype is stored in schema metadata, deserialization already handles it via `.view(dtype)`.

2. **gRPC hostname resolution** (`be90ad30d`): `socket.gethostname()` returns `.local` mDNS names on macOS, which gRPC's c-ares resolver can't handle. Added `_resolve_advertise_host()` that tries GCP metadata server (for TPU VMs), then falls back to `localhost` for `.local` hostnames.

3. **Circular import** (`ee5ccdac4`): `rl_job → rollout_worker → orchestration → rl_job`. Fixed by extracting `RLRuntimeHandles` and `WeightTransferRuntime` into `runtime.py`.

4. **Stale test API** (`ee5ccdac4`): `RLOOLoss(clip_epsilon=0.2)` → `RLOOLoss(clip_epsilon_low=0.2, clip_epsilon_high=0.2)` in all 8 test files. Pre-existing issue unrelated to our migration.

5. **Test tolerance** (`be90ad30d`): bfloat16 round-trip is lossy (7-bit mantissa, ~0.4% max relative error). Changed `assert_array_equal` to `assert_allclose(rtol=4e-3)`.

**Test results:**
- `test_weight_transfer.py`: 4/4 pass (GCS + Arrow Flight × 2 tests)
- Integration tests: pre-existing `RuntimeError: No global tracker set` in Levanter inference path (not our bug)

---

### 2026-03-21 — Graceful shutdown + integration test fixes

**Graceful shutdown** (`a7d1eb7e1`):
- Added `_check_run_state()` to RolloutWorker — polls `run_state.get_status()` at start of each iteration
- Falls back gracefully if run_state is unreachable (best-effort, no crash)
- Rollout worker falls back to `RolloutTracker` when no Levanter tracker available

**Pre-existing bugs fixed** (on `main`, not caused by migration):
- `mock_env.py`: missing `temperature` arg to `create_rollout_from_choice()`
- Circular import: extracted `RLRuntimeHandles`/`WeightTransferRuntime` into `runtime.py`

**Integration test results**:
- `test_weight_transfer.py`: **4/4 pass**
- `test_weight_sync.py`: Rollout worker generates 6 rollouts, trainer receives weights via Arrow Flight — **v2 wiring works**. Fails on pre-existing `flash_attention_block_size > max_seq_len` config mismatch in test setup (exists on `main` too, not our bug).

---

### 2026-03-21 — Codex Review Feedback (PR #3960)

**Reviewer**: Codex agent
**Verdict**: "Not yet safe to treat as correct" — at least one confirmed regression + lifecycle/TPU-runtime issues.

#### [P1] RolloutTracker fallback crashes when tracker_config is None

**File**: `rollout_worker.py:295-300`
**Problem**: The new fallback `RolloutTracker(config.tracker_config, config.run_id)` crashes with `AttributeError` when `tracker_config` is `None` (which is the default on both `RolloutWorkerConfig` and `RLJobConfig`). The exact "tests or standalone rollout workers" scenario it's meant to handle still crashes.
**Impact**: `tests/rl/integration/test_rollout_worker.py` fails immediately.
**Fix needed**: Check for `None` before constructing `RolloutTracker`, use a no-op tracker or create a default config.

#### [P2] Graceful shutdown doesn't re-check after blocking weight sync

**File**: `rollout_worker.py:710-714`
**Problem**: `_check_run_state()` only runs at the TOP of the loop, before `_sync_weights()`. If `max_weight_transfer_wait_time > 0` (e.g. 600s with `with_on_policy_training()`), the worker blocks inside `_sync_weights()` for up to 600s. If the trainer completes during that wait, the worker still generates one more rollout batch with stale weights before the next loop iteration notices.
**Fix needed**: Check run state after `_sync_weights()` returns, or pass the run_state handle into `_sync_weights()` so it can break early.

#### [P1] SymPy timeout uses fork — deadlocks on TPU

**File**: `math_grading.py:235-240`
**Problem**: `multiprocessing.Pool()` uses `fork` by default on Linux. The rollout worker docstring explicitly warns that forking after JAX/libtpu initialization deadlocks on TPU device locks (`/dev/vfio`). Math grading runs inside rollout workers, so this timeout wrapper will deadlock or crash on TPU with Levanter inference.
**Fix needed**: Use `multiprocessing.get_context("spawn")` or `"forkserver"` instead of default `Pool()`.

#### [P3] Missed callers of RolloutWorker/TrainWorker

**File**: `lib/marin/src/marin/rl/scripts/test_llama_small.py`
**Problem**: `RolloutWorker(config=worker_config)` — still uses old signature without `runtime=`. Will crash with `TypeError` immediately.
**Fix needed**: Update to pass `runtime=` or add to deferred list.

#### Action items

| Priority | Issue | Status |
|----------|-------|--------|
| P1 | Handle `tracker_config=None` in RolloutTracker fallback | DONE — added `_NoOpTracker` |
| P1 | Use spawn context for sympy multiprocessing on TPU | DONE — `get_context("spawn")` |
| P2 | Re-check run_state after `_sync_weights()` | DONE — second `_check_run_state()` call |
| P3 | Update `test_llama_small.py` caller | DONE — marked as outdated |

All fixes in commit `84cbb2528`.
