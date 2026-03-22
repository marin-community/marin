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
| B-7 | Update curriculum.py: delete get_or_create_curriculum_actor | DEFERRED (kept for v1 compat) |
| B-8 | Update weight_transfer/arrow_flight.py: accept coordinator handle | DONE |
| B-9 | Update weight_transfer/__init__.py: thread handles through factories | DONE |
| B-10 | Migrate test fixtures (conftest.py, test_weight_transfer.py) | TODO |
| C-1 | Fix RunConfig drift (slice_count, retries, env_vars) | |
| C-2 | Add region support to RunConfig | |
| D-1 | Integration test on LocalClient | |
| D-2 | Integration test on Iris cluster | |

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
- Note: Initially used `np.float16` instead of `jnp.bfloat16` in copy_and_flatten — wrong dtype, fixed in follow-up commit.

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
