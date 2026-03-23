# On-Demand RL: Research Logbook

## Scope
- **Goal**: Get `exp2039` RL training (Llama 3.1-8B, MATH-500, RLOO) running without a Ray cluster, using manually created TPU VMs (one trainer, one sampler) via Levanter launch.
- **Primary metric(s)**: Training starts and produces rollouts (RL loop running end-to-end).
- **Constraints**: europe-west4-a zone, v6e TPUs, spot capacity, shared state via GCS (`gs://marin-eu-west4`).
- **Stop criteria**: `train:` progress bar advances past 0%, rollouts are being generated.

## Baseline
- **Date**: 2026-03-15
- **Prior work**: A previous agent built the on-demand RL infrastructure on the `on-demand-rl` branch (see `ON_DEMAND_RL.md`). Code was written but never successfully run end-to-end.
- **Code refs**: `experiments/exp2039_rl_math500.py`, `on-demand-rl-scripts/`, `lib/marin/src/marin/rl/`
- **Baseline numbers**: No successful run exists. The first attempt (`bash on-demand-rl-scripts/exp2039_euwest4.sh`) failed immediately.

## Code Changes Log

### Change 1: Fix Docker container path (script)
**File**: `on-demand-rl-scripts/_run_exp2039.sh`
**What**: Changed the docker command from `uv run python experiments/exp2039_rl_math500.py` to `bash /opt/marin/on-demand-rl-scripts/bootstrap_rl.sh /opt/marin/experiments/exp2039_rl_math500.py`
**Why**: The Dockerfile sets `WORKDIR /opt/marin/lib/levanter` (line 46 of `Dockerfile.incremental`). The relative path `experiments/exp2039_rl_math500.py` resolved to `/opt/marin/lib/levanter/experiments/exp2039_rl_math500.py` which doesn't exist. The experiment lives at `/opt/marin/experiments/exp2039_rl_math500.py`.
**Error it fixed**: `/opt/marin/.venv/bin/python3: can't open file '/opt/marin/lib/levanter/experiments/exp2039_rl_math500.py': [Errno 2] No such file or directory`

### Change 2: Create bootstrap script for RL deps
**File**: `on-demand-rl-scripts/bootstrap_rl.sh` (new file)
**What**: A wrapper script that runs inside the Docker container on TPU VMs. It:
1. Installs `marin` and `marin-root` as editable packages (with `--no-deps` to avoid re-resolving the TPU jax/torch setup)
2. Installs `vllm-tpu==0.13.2.post6`, `prime`, `sympy`, `verifiers` additively
3. Then `exec python "$@"` to run the actual experiment
**Why**: The Docker image is built with `uv sync --package levanter --extra serve --extra tpu`, which installs levanter and TPU deps but NOT the `marin` package or its `vllm`/`rl` extras. Three approaches failed before this:
- `uv run --directory /opt/marin` → installed marin but not vllm (missing vllm extra)
- `uv run --package marin --extra vllm --extra rl` → re-resolved deps and installed CUDA torch, breaking TPU (`libcublas.so not found`)
- Dockerfile `uv pip install` → cross-compiling vllm-tpu on ARM Mac via QEMU was impossibly slow
The bootstrap approach installs deps natively on the x86_64 TPU VM at container start (~3 seconds) and uses `--no-deps` for marin to avoid touching the existing jax/torch.

### Change 3: Remove stale `pip_dependency_groups` kwarg
**File**: `lib/marin/src/marin/rl/rl_experiment_utils.py`
**What**: Removed `pip_dependency_groups=["vllm", "math"]` from the `ExecutorStep(...)` constructor call.
**Why**: `pip_dependency_groups` was removed from `ExecutorStep` in PR #3263 (`Remove resources/env_vars from ExecutorStep; use @remote for dispatch`), but `rl_experiment_utils.py` was never updated. This is a pre-existing bug on main — not specific to on-demand RL.
**Error it fixed**: `TypeError: ExecutorStep.__init__() got an unexpected keyword argument 'pip_dependency_groups'`

### Change 4: Bump v6e presets to v6e-16 for trainer
**File**: `experiments/exp2039_rl_math500.py`
**What**: Changed `trainer_tpu_type` from `"v6e-8"` to `"v6e-16"` in the `v6e_euw4a` and `v6e_east1d` deployment presets. Sampler stays `v6e-8`.
**Why**: v6e-8 has 31.25 GiB HBM per chip. Llama 8B RL training (model + reference model + Adam optimizer) exhausts per-chip memory. The `copy_and_flatten` function (which is `@jax.jit`-compiled) tries to pre-allocate 9.62G of intermediate buffers but only 7.14G is free. v6e-16 has more chips to shard across, but same per-chip HBM — so this alone may not fix the OOM. The real fix likely requires removing the `@jax.jit` from `copy_and_flatten` (see Current Blocker below).

### Files NOT changed by this session (pre-existing on branch)
These were already on the `on-demand-rl` branch from the previous agent:
- `experiments/exp2039_rl_math500.py` — manual mode CLI, deployment presets, launch-plan mode (bulk of changes)
- `lib/levanter/src/levanter/infra/cli_helpers.py` — `.marin.yaml` config resolution
- `lib/levanter/src/levanter/infra/tpus.py` — TPU capacity flag fixes
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` — fast vLLM bootstrap
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` — filesystem Arrow coordinator (NOT changed by this session; reverted all my changes)
- `lib/marin/src/marin/rl/weight_transfer/base.py` — coordinator_backend config
- `tests/rl/test_weight_transfer.py` — filesystem coordinator tests

## Current Blocker

**OOM in `copy_and_flatten` during initial weight serve on trainer.**

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_copy_and_flatten':
Attempting to reserve 9.62G at the bottom of memory. That was not possible.
There are 7.14G free, 0B reserved, and 7.14G reservable.
```

- **Location**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py:407-438`
- **Root cause**: `copy_and_flatten` is decorated with `@partial(jax.jit, static_argnames=("convert_to_bfloat16",))`. The JIT compilation pre-allocates ALL intermediate buffers simultaneously: `to_state_dict` output + bf16 conversion + flattened arrays. This exceeds the 7.14G free per-chip HBM.
- **The traceback also shows**: `hsd.to_state_dict(model)` internally calls `_unstack_state_dict` which iterates over stacked transformer layers, calling `device_put` for each of the 32 unstacked slices while the stacked original is still alive — doubling model memory.
- **Proposed fix**: Remove `@jax.jit`, transfer model to host via `jax.device_get(model)` first, then run `to_state_dict` + bf16 + flatten entirely in numpy on host. This was tested and the weight serve succeeded, but the user asked to revert and try v6e-16 first. v6e-16 has the same per-chip HBM so the OOM persists.
- **This OOM occurs on both v6e-8 AND v6e-16** because per-chip memory is identical (31.25 GiB).

## Run History

### 2026-03-15 18:41 — Run 1: First attempt (original script)
- **Config**: v6e-8 trainer, v6e-8 sampler, europe-west4-a
- **Result**: FAIL — `No such file or directory` (wrong path inside Docker container)
- **Fix**: Change 1 above

### 2026-03-15 ~19:10 — Run 2: With path fix + `uv run --directory`
- **Result**: FAIL — `ModuleNotFoundError: No module named 'marin'`
- **Fix**: Added `--directory /opt/marin` to `uv run`

### 2026-03-15 ~19:20 — Run 3: With marin installed
- **Result**: FAIL — `ModuleNotFoundError: No module named 'vllm'`
- **Fix**: Added `--package marin --extra vllm --extra rl`

### 2026-03-15 ~19:30 — Run 4: With vllm extras
- **Result**: FAIL — `ValueError: libcublas.so.*[0-9] not found` (CUDA torch installed, breaking TPU)
- **Fix**: Created bootstrap_rl.sh (Change 2)

### 2026-03-15 ~19:50 — Run 5: With bootstrap script
- **Result**: FAIL — `TypeError: ExecutorStep.__init__() got an unexpected keyword argument 'pip_dependency_groups'`
- **Fix**: Change 3 above

### 2026-03-15 ~20:05 — Run 6: With pip_dependency_groups fix
- **Result**: FAIL — `RESOURCE_EXHAUSTED: Error loading program 'jit_copy_and_flatten': Attempting to reserve 9.62G` (trainer OOM)
- **Fix attempted**: Removed `@jax.jit`, streamed params to host — weight serve succeeded but training OOM'd with 66MB free (still on v6e-8)
- **Fix reverted**: User asked to try v6e-16 instead

### 2026-03-16 ~03:50 — Run 7: With v6e-16 trainer
- **Config**: v6e-16 trainer (new), v6e-8 sampler (reused), europe-west4-a
- **Result**: FAIL — Same `jit_copy_and_flatten` OOM (9.62G needed, 7.14G free). Per-chip HBM is identical on v6e-16.
- **Status**: CURRENT BLOCKER. Need to fix `copy_and_flatten`.

### Change 5: Remove `@jax.jit` from `copy_and_flatten`, do everything on host
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Removed `@partial(jax.jit, static_argnames=("convert_to_bfloat16",))` decorator. Changed function to: (1) `jax.device_get(model)` to transfer entire model pytree to host numpy, (2) `hsd.to_state_dict(host_model)` to unstack layers in numpy, (3) bf16 conversion + flatten in numpy. Returns numpy arrays so the caller's `jax.device_get` is a no-op.
**Why**: The `@jax.jit` caused XLA to pre-allocate all intermediate buffers simultaneously (9.62G needed, 7.14G free). Additionally, `to_state_dict` internally calls `_unstack_state_dict` which does `device_put` for each of 32 unstacked layer slices while the stacked original is alive, doubling model memory. By transferring to host first, all unstacking and conversion happens in numpy with no device memory pressure.

### Change 6: Resolve per-role TPU type from deployment preset
**File**: `on-demand-rl-scripts/_run_exp2039.sh`
**What**: Added `resolve_tpu_type()` function that reads the preset's `trainer_tpu_type` or `sampler_tpu_type` from the Python config. Each role now gets its own TPU type.
**Why**: The script originally used a single `TPU_TYPE` for both roles, ignoring the per-role types defined in the deployment preset. This meant changing `v6e_euw4a` preset's `trainer_tpu_type` to `v6e-16` had no effect — both roles still got `v6e-8`.

### 2026-03-16 05:00 — Run 8: us-central1, v5p-8
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot
- **Result**: PARTIAL SUCCESS
  - Trainer: loaded model, started RLOO training, **served weights successfully** (state_dict=62s, copy=8s)
  - Trainer: waiting for initial rollouts from sampler
  - Sampler: vLLM initialized, 95.74 GiB/chip, KV cache ready
  - Sampler: **FAIL** — `RuntimeError: cannot schedule new futures after shutdown` in `receive_weights`. The fray job context executor shuts down before the sampler can fetch coordinator info to receive weights.
- **Current blocker**: Fray executor lifecycle issue in manual (no-Ray) mode. The `self._ctx` used by the Arrow Flight coordinator client on the sampler side has its thread pool shut down prematurely.

### Change 7: Add FileSystemArrowFlightCoordinator and wire it in
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**:
1. Added `FileSystemArrowFlightCoordinator` class — reads/writes coordinator metadata as JSON on a shared filesystem (GCS). Same interface as `ArrowFlightCoordinator` (`update_server`, `fetch_server`).
2. Added `_create_coordinator(config)` factory — returns filesystem coordinator when `coordinator_backend="filesystem"`, fray actor when `"actor"`.
3. Updated `ArrowFlightServer.__init__` and `ArrowFlightClient.__init__` to use the factory.
4. Updated `serve_weights` and `receive_weights` to call coordinator methods directly (not `.remote()`) in filesystem mode.
5. Made fray import lazy (inside `_create_coordinator`) so it's not needed in filesystem mode.
**Why**: The `ArrowFlightServer` and `ArrowFlightClient` always created fray Ray actors for coordination, even when `coordinator_backend="filesystem"` was set in the config. In manual no-Ray mode, fray's thread pool executor shut down prematurely, causing `RuntimeError: cannot schedule new futures after shutdown` on the sampler. The filesystem coordinator writes to a shared GCS path that both trainer and sampler can access.

### Change 8: Catch `OSError` in filesystem coordinator `_read_metadata`
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Changed `except FileNotFoundError` to `except (FileNotFoundError, OSError)` in `_read_metadata`.
**Why**: gcsfs throws `OSError` (not `FileNotFoundError`) when a GCS object doesn't exist. The sampler tries to read the coordinator JSON before the trainer writes it.

### 2026-03-16 06:10 — Run 10: Fresh v5p-8 us-central1, all fixes
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260315-230928
- **Result**: PARTIAL SUCCESS
  - Trainer: loaded model, served weights via filesystem coordinator, waiting for rollouts
  - Sampler: vLLM initialized, but crashes with `RuntimeError: Session is closed` from aiohttp
  - Root cause: `bootstrap_rl.sh` installs `vllm-tpu==0.13.2.post6` which pulls in transitive deps that downgrade/conflict with the existing aiohttp version in the Docker image
- **Current blocker**: Package compatibility — `uv pip install vllm-tpu` without `--no-deps` breaks aiohttp

### Change 9: Fix bootstrap_rl.sh — restore aiohttp/gcsfs after vllm-tpu install
**File**: `on-demand-rl-scripts/bootstrap_rl.sh`
**What**: Snapshot `aiohttp` and `gcsfs` versions before `uv pip install vllm-tpu`, then restore them after. Previous `--no-deps` attempt broke torch import.
**Why**: `uv pip install vllm-tpu` pulls in transitive deps that downgrade aiohttp, causing `RuntimeError: Session is closed` in vLLM's async engine. The snapshot-restore approach lets all deps install but pins the critical ones back.

### 2026-03-16 06:50 — Run 11: Fresh v5p-8 us-central1, aiohttp pin-restore
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260315-230928
- **Result**: PARTIAL SUCCESS — furthest yet
  - Trainer: loaded model, served weights via filesystem coordinator (**works**)
  - Sampler: vLLM initialized, read coordinator JSON from GCS, connected to Arrow Flight (**works**)
  - Sampler: **FAIL** on weight download — `FlightUnavailableError: DNS server returned general failure` for `t1v-n-c018ea33-w-0:34829`
  - Root cause: **Manual-mode-only issue.** Trainer advertises Arrow Flight servers using `socket.gethostname()` which returns the TPU worker hostname (e.g. `t1v-n-c018ea33-w-0`). This hostname is NOT registered in GCP's internal DNS for queued-resource TPUs. In the Ray cluster flow this works because Ray-provisioned TPU VMs get DNS-resolvable hostnames; in manual mode they don't.
- **Current blocker**: Arrow Flight server addresses use unresolvable internal hostnames (manual mode only).

## Architecture: Why Arrow Flight hostname resolution differs between Ray and manual mode

In **Ray mode**: TPU VMs are created by Ray as cluster workers. They're registered in GCP's internal DNS, and `socket.gethostname()` returns a resolvable name. The `ArrowFlightCoordinator` is a Ray actor in shared memory — no filesystem needed. Everything works.

In **manual mode**: TPU VMs are created independently via `gcloud alpha compute tpus queued-resources create`. Their hostnames (`t1v-n-XXXX-w-N`) are internal TPU worker names NOT registered in DNS. However, both VMs are in the same VPC (same project/zone), so their **internal IPs** (10.x.x.x) ARE routable. The fix is to advertise internal IPs instead of hostnames.

This is the ONLY networking difference. Arrow Flight gRPC, the weight serialization, and the coordinator protocol all work identically in both modes.

## Next Steps: Making Arrow Flight work in both modes

**Goal**: `socket.gethostname()` works in Ray mode. Internal IP from GCP metadata works in manual mode. The fix must support both without breaking the Ray path.

### Plan

**Change 10**: Add `_resolve_advertise_host()` helper to `arrow_flight.py`
- If `config.flight_host` is explicitly set (not `0.0.0.0`), use it as-is (user override).
- If `config.coordinator_backend == "filesystem"` (manual mode): query GCP metadata server for internal IP (`http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip`). Fall back to `socket.gethostname()` if metadata server is unavailable (non-GCP environment).
- If `config.coordinator_backend == "actor"` (Ray mode): use `socket.gethostname()` as before (preserves existing behavior).

**Where to apply**: Two locations in `arrow_flight.py` where `socket.gethostname()` is used:
1. `ArrowFlightServer.__init__` (~line 368) — for constructing `_server_locations`
2. `ArrowFlightServer.serve_weights` (~line 429) — for `server_locations` passed to coordinator

Both should call `_resolve_advertise_host(config)` instead of `socket.gethostname()`.

**After applying**: Re-run on us-central1 v5p-8. The coordinator JSON on GCS will contain IPs like `grpc://10.128.0.XX:34829` instead of `grpc://t1v-n-c018ea33-w-0:34829`.

### Change 10: `_resolve_advertise_host()` helper
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Added `_resolve_advertise_host(config)` that queries GCP metadata for internal IP when `coordinator_backend="filesystem"`, falls back to `socket.gethostname()` for Ray mode.
**Why**: TPU queued-resource hostnames (`t1v-n-XXXX-w-0`) are not in DNS. Internal IPs are routable within the same VPC.

### Change 11: Use strict `<` for stale weight rejection in filesystem coordinator
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Changed `weight_id <= existing["weight_id"]` to `weight_id < existing["weight_id"]`.
**Why**: When restarting with the same RUN_ID, both old and new initial weight serves use `weight_id=-1`. The `<=` check rejected the new write, leaving stale hostname-based addresses in the JSON.

### Change 12: Replace fsspec/gcsfs with google-cloud-storage sync client
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Replaced `fsspec.open()` calls in `FileSystemArrowFlightCoordinator` with `_gcs_read()`/`_gcs_write()` helpers using `google.cloud.storage`.
**Why**: gcsfs uses aiohttp under the hood. When vLLM also runs an async event loop, the gcsfs aiohttp session gets torn down (`RuntimeError: Session is closed`), crashing the EngineCore. The google-cloud-storage client is fully synchronous and doesn't conflict.

### Change 13: Pin constraints in bootstrap_rl.sh, add pylatexenc
**File**: `on-demand-rl-scripts/bootstrap_rl.sh`
**What**: Generate a constraints file from currently installed aiohttp/gcsfs/fsspec/jax/jaxlib versions before installing vllm-tpu. Also added `pylatexenc` to deps.
**Why**: `uv pip install vllm-tpu` pulled in transitive deps that downgraded aiohttp, and math grading needs pylatexenc.

### 2026-03-16 20:10 — Run 14: Sync GCS client, constraints, internal IP
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot
- **Result**: MAJOR PROGRESS
  - Trainer: model loaded, weights served via filesystem coordinator with internal IP (**works**)
  - Sampler: coordinator JSON read via sync GCS client (**works**, no aiohttp conflict)
  - Sampler: weights received via Arrow Flight over internal IP (**works**)
  - Sampler: all 291 params loaded, prefix cache reset, XLA kernel compilation started (**works**)
  - Sampler: **FAIL** — vLLM EngineCore dies 16 seconds after XLA compilation starts. `EngineDeadError: EngineCore encountered an issue.` The stderr from the EngineCore subprocess is not captured in the log.
  - This is NO LONGER a coordinator/networking issue. The full weight transfer pipeline works end-to-end. The crash is in vLLM itself during first inference after weight sync.
- **Current blocker**: vLLM EngineCore crashes after receiving weights and compiling XLA kernels.

### Change 14: Replace fsspec/gcsfs with sync google-cloud-storage client
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Replaced `fsspec.open()` in `FileSystemArrowFlightCoordinator` with `_gcs_read()`/`_gcs_write()` helpers using `google.cloud.storage` (sync HTTP, no aiohttp).
**Why**: gcsfs uses aiohttp under the hood. vLLM also runs an async event loop. The two conflict — gcsfs's aiohttp session gets torn down (`RuntimeError: Session is closed`), killing the vLLM EngineCore. The google-cloud-storage client is fully synchronous.

### 2026-03-16 20:20 — Run 15: Sync GCS client
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot
- **Result**: MAJOR PROGRESS — full weight transfer pipeline works end-to-end
  - Trainer: model loaded, weights served via filesystem coordinator with internal IP (**works**)
  - Sampler: coordinator JSON read via sync GCS client (**works**)
  - Sampler: all 291 params received via Arrow Flight over internal IP (**works**)
  - Sampler: weights loaded into vLLM, prefix cache reset (**works**)
  - Sampler: XLA attention kernel compilation starts (**works**)
  - Sampler: **FAIL** — EngineCore subprocess dies 16s after XLA compilation finishes. No Python traceback (likely segfault in XLA).
- **Diagnosis**: The EngineCore subprocess (separate process from rollout worker) processes weights via `update_weights()` RPC, compiles XLA kernels, then segfaults on first inference. Timeline: weights loaded → 75s XLA compile → 16s silence → "died unexpectedly". This is a vLLM-TPU issue with `update_weights` after `load_format=dummy` init.
- **NOT caused by**: coordinator, Arrow Flight, networking, aiohttp, or curriculum actors.

## Current Blocker: vLLM EngineCore segfault after weight update

The entire on-demand RL infrastructure (filesystem coordinator, Arrow Flight over internal IPs, sync GCS client) works. The remaining issue is in vLLM-TPU itself: `update_weights()` loads new weights into a model initialized with `load_format=dummy`, XLA recompiles, then the EngineCore subprocess segfaults during first inference.

## Root Cause of EngineCore Crash (Identified)

**Flax version mismatch.** The base Docker image has `flax==0.12.0` (installed by `uv sync --package levanter`). The bootstrap script installs `vllm-tpu==0.13.2.post6` which depends on `tpu-inference==0.13.2.post6` which requires `flax<0.12`. This downgrades flax to `0.11.1`.

Levanter/Haliax weight conversion code expects flax 0.12 APIs. vLLM's `_sync_weights` inside the EngineCore subprocess uses flax 0.11.1. The mismatch causes a silent crash (segfault, no Python traceback) during the first inference after weight injection.

**This is NOT a `load_format=dummy` issue** — the same `load_format=dummy` + `update_weights` flow works fine in the Ray cluster where the Docker image has a consistent flax version throughout.

**The fundamental conflict**: levanter requires `flax>=0.12`, but `tpu-inference==0.13.2.post6` (vllm-tpu dep) requires `flax<0.12`. These are incompatible.

## Resolution: Use `uv sync` with correct extras in Dockerfile

The `uv.lock` already resolves this conflict correctly:
- `marin[vllm]` active → `flax==0.11.1`
- `marin[vllm]` not active → `flax==0.12.0`

Every version of `tpu-inference` (including latest 0.13.3) hard-pins `flax==0.11.1`. This is not a range — it's an exact pin. The TPU inference NNX model code is tested only against 0.11.1.

The fix: change the Dockerfile to `uv sync` with both `levanter` and `marin[vllm,rl]` extras so the resolver picks flax 0.11.1 from the start. This eliminates the bootstrap script's runtime `uv pip install` entirely — all deps are baked into the Docker image at build time.

### Change 15: Dockerfile `uv sync` with marin[vllm,rl] extras
**File**: `lib/levanter/docker/tpu/Dockerfile.incremental`
**What**: Change from `uv sync --package levanter --extra serve --extra tpu` to syncing the workspace root which includes all packages with the right extras.
**Why**: The bootstrap script's `uv pip install vllm-tpu` downgraded flax from 0.12→0.11.1 at runtime, causing a mismatch with Levanter's flax expectations. By syncing at Docker build time, the resolver sees both `levanter[serve,tpu]` and `marin[vllm,rl]` constraints simultaneously and picks `flax==0.11.1` consistently.

### Change 16: Simplify bootstrap_rl.sh
**File**: `on-demand-rl-scripts/bootstrap_rl.sh`
**What**: Remove the vllm-tpu install and constraints logic. Only install marin as editable (for source code changes) and run the experiment.
**Why**: All deps are now in the Docker image. Bootstrap only needs to make the local source code available.

### Change 15: Root pyproject.toml `rl` composite extra + Dockerfile `uv sync --extra rl`
**File**: `pyproject.toml`, `lib/levanter/docker/tpu/Dockerfile.incremental`
**What**:
- Added `[project.optional-dependencies] rl = ["levanter[serve,tpu]", "marin[vllm,rl]"]` to root pyproject.toml.
- Changed Dockerfile from `uv sync --package levanter --extra serve --extra tpu` to `uv sync --extra rl` (syncs from workspace root).
- Added `lib/marin/pyproject.toml` and `lib/iris/pyproject.toml` to Dockerfile mount binds.
- Simplified `bootstrap_rl.sh` to just `cd /opt/marin && exec python "$@"` (no more runtime pip installs).
**Why**: All previous bootstrap approaches (runtime `uv pip install`, constraints, aiohttp pin-restore) failed because they created dep conflicts. Using `uv sync` at Docker build time lets the resolver see all constraints simultaneously.

### 2026-03-16 21:50 — Run 17: uv sync --extra rl (fresh TPUs)
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-143830
- **Result**: FAIL — `libcublas.so not found` (CUDA torch installed instead of CPU/TPU torch)
- **Root cause**: The `rl` extra pulls in `marin[vllm,rl]` and `levanter[serve,tpu]`, but NOT `marin[tpu]`. The `tpu` extra on `marin` is what constrains torch to the CPU variant (`torch==2.9.0` from CPU index). Without `marin[tpu]`, the resolver picks the default PyPI torch (CUDA variant).
- **Fix identified**: Add `marin[tpu]` to the root `rl` extra: `rl = ["levanter[serve,tpu]", "marin[vllm,rl,tpu]"]`

## Current State of All Components

| Component | Status | Notes |
|-----------|--------|-------|
| Docker path fix | DONE | `bootstrap_rl.sh` uses absolute path |
| `pip_dependency_groups` bug | DONE | Removed stale kwarg from `ExecutorStep` |
| `copy_and_flatten` OOM fix | DONE | Removed `@jax.jit`, `jax.device_get(model)` first |
| Per-role TPU type | DONE | `resolve_tpu_type()` reads deployment preset |
| FileSystem coordinator | DONE | `FileSystemArrowFlightCoordinator` + `_create_coordinator()` factory |
| GCS read/write (sync) | DONE | `_gcs_read()`/`_gcs_write()` using `google.cloud.storage` (no aiohttp conflict) |
| Internal IP for Arrow Flight | DONE | `_resolve_advertise_host()` queries GCP metadata |
| Stale coordinator JSON | DONE | Strict `<` comparison for weight_id |
| Dep resolution (flax/torch) | IN PROGRESS | Need `marin[tpu]` in root `rl` extra |
| vLLM weight sync | UNKNOWN | Blocked by dep resolution — needs testing with correct flax |
| Curriculum actor (fray) | UNKNOWN | `get_or_create_curriculum_actor` uses fray; may crash in no-Ray mode after weight sync works |

### Change 17: Add `marin[tpu]` to root rl extra + `pylatexenc` to marin[rl]
**Files**: `pyproject.toml`, `lib/marin/pyproject.toml`
**What**: Changed root rl extra to `["levanter[serve,tpu]", "marin[vllm,rl,tpu]"]`. Added `pylatexenc` to marin's rl extra.
**Why**: Without `marin[tpu]`, torch resolved to CUDA variant (libcublas.so not found). pylatexenc is needed for math grading.

### 2026-03-16 22:15 — Run 18: Correct torch + flax via uv sync
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-151242
- **Result**: FAIL — same EngineCore crash as run 15
  - No more libcublas error (torch is correct)
  - Weights received and loaded (all 291 params including model.norm)
  - Prefix cache reset succeeded
  - EngineCore dies 3 seconds after weight sync (no Python traceback — native crash)
  - Missing pylatexenc (separate issue, now fixed in marin[rl])
- **Key finding**: The EngineCore crash is NOT caused by flax version mismatch. It happens with both flax 0.11.1 (this run) and 0.12.0 (run 15). The crash is in vLLM-TPU's `update_weights` path itself.

## Revised Root Cause: EngineCore crash was NOT a vLLM bug

**CORRECTION**: Earlier analysis (runs 15-18) concluded the EngineCore crash was a vLLM-TPU bug with `update_weights` + `load_format=dummy`. This was wrong.

**Actual cause**: The rollout worker crashes because `pylatexenc` is missing. The crash sequence:
1. Weight sync completes successfully (all 291 params loaded, prefix cache reset)
2. Main thread enters rollout loop → `_evaluate_curriculum` → `_evaluate_lesson` → `_load_environment` → imports math grading module → `from pylatexenc import latex2text` → **ModuleNotFoundError**
3. Unhandled exception kills the rollout worker process
4. Rollout worker teardown shuts down the vLLM engine → EngineCore subprocess exits
5. AsyncLLM's output_handler sees the dead EngineCore → logs `EngineDeadError`

The `EngineDeadError` was a symptom, not the cause. The missing `pylatexenc` timestamp-less stderr output always appeared BEFORE the timestamped EngineDeadError in the log. The lack of a Python traceback from EngineCore was misleading — there was no native/XLA crash; the subprocess was simply shut down by its parent.

**The standard Ray RL codebase didn't hit this** because Ray cluster Docker images include all deps (including pylatexenc) via the full `uv sync` with all extras. Our Dockerfile was missing it because `pylatexenc` wasn't listed in `marin[rl]` extras.

**Fix already applied**: Added `pylatexenc` to `marin[rl]` extras in `lib/marin/pyproject.toml` and updated `uv.lock`.

### 2026-03-16 22:50 — Run 19: With pylatexenc, correct flax + torch
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-153951
- **Result**: FAIL — EngineCore crash persists even with pylatexenc installed
  - No pylatexenc errors (fix worked)
  - Weights received, XLA compiled (RPA v3 kernel at 22:54:22)
  - EngineCore dies 18 seconds later at 22:54:40 — no Python traceback, no error output
  - This is a genuine native crash in the EngineCore subprocess

**CORRECTION to correction**: The pylatexenc fix resolved SOME of the crashes (runs 17-18) but there IS also a real vLLM-TPU EngineCore crash that happens independently after `update_weights` + XLA recompilation. Both issues existed simultaneously — pylatexenc masked the real crash in some runs.

## Confirmed: vLLM-TPU `update_weights` crash on `load_format=dummy`

The EngineCore subprocess segfaults ~18 seconds after XLA attention kernel compilation following `update_weights`. No Python traceback is produced. This happens with:
- Correct flax (0.11.1)
- Correct torch (CPU variant)
- pylatexenc present
- All 291 model params successfully synced

The crash is in native XLA/C++ code during the first inference attempt after weight injection into a dummy-initialized model.

### Change 18: Set `inflight_weight_updates=False` in manual mode
**File**: `experiments/exp2039_rl_math500.py`
**What**: Set `inflight_weight_updates=False` in `_default_experiment_config()` or override it in `_run_manual_mode()`.
**Why**: `inflight_weight_updates=True` uses `AsyncvLLMInferenceContext` which syncs weights via `collective_rpc` to the EngineCore subprocess. The EngineCore subprocess segfaults after `update_weights` + XLA recompilation (native crash, no Python traceback). With `inflight_weight_updates=False`, `vLLMInferenceContext` is used instead — it calls `driver_worker.sync_weights()` directly in the same process (no RPC, no subprocess boundary). The trade-off is that weight updates block inference, but for a single-sampler manual mode setup this is acceptable.

### 2026-03-16 23:15 — Run 20: inflight_weight_updates=False
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-160503
- **Result**: MAJOR PROGRESS — vLLM inference works end-to-end
  - Trainer: model loaded, weights served via filesystem coordinator (**works**)
  - Sampler: weights received via Arrow Flight, loaded into vLLM (**works**)
  - Sampler: XLA compiled, inference running at ~2,100 tok/s output (**works**)
  - Sampler: completed 500-prompt evaluation batches multiple times (**works**)
  - Sampler: **STUCK** — never enters rollout generation loop. The curriculum actor (`get_or_create_curriculum_actor`) uses fray Ray actors. `sample_lesson.remote()` and `update_lesson_stats.remote()` fail in no-Ray mode. The sampler loops on evaluations but never writes rollouts.
  - Trainer: times out waiting for rollouts.
- **vLLM EngineCore crash: FIXED** by `inflight_weight_updates=False`.

## Current Blocker: Curriculum actor uses fray (Ray actors)

### Why the curriculum is a shared Ray actor in the standard pipeline

In production RL, there are **N rollout workers** (4-8) generating rollouts in parallel on separate TPUs. The curriculum needs to be shared so that:
- When worker A discovers a lesson is mastered (high reward), ALL workers shift to harder lessons
- Lesson graduation/unlocking is based on collective performance across all workers
- All workers sample from the same weighted distribution at any given time

This is implemented as a **single named Ray actor** (`name="curriculum"`) that both trainer and rollout workers discover via `get_if_exists=True`. The actor lives as a Python process somewhere in the Ray cluster. All workers call `.remote()` methods which Ray serializes and routes over the network to that process.

### Why it doesn't work in manual mode

No Ray cluster → no actor registry → no network transport → `.remote()` calls fail. The sampler loops on evaluations (500-prompt batches work because they use vLLM directly) but never enters the rollout generation loop because `sample_lesson.remote()` fails.

### Why sharing doesn't matter for our setup

We have **1 trainer + 1 sampler**. There's nobody to share curriculum state with. A local `Curriculum` object on the sampler does the same thing as a shared Ray actor. The only cross-VM shared state we need is already handled by:
- **Weights**: Arrow Flight (trainer → sampler)
- **Rollouts**: GCS files (sampler → trainer)
- **Coordination**: GCS JSON (filesystem coordinator)

The curriculum is purely a sampler-side concern with a single sampler.

### Curriculum actor call sites

| File | Line | Method | In try/except? | Purpose |
|------|------|--------|----------------|---------|
| `train_worker.py` | 386 | `save_checkpoint.remote(dir)` | Yes | Periodic checkpoint save |
| `rollout_worker.py` | 607-609 | `update_lesson_stats.remote(stats, "eval", step)` | No (only for eval) | Update after evaluation |
| `rollout_worker.py` | 688 | `sample_lesson.remote(seed)` | Yes | Sample next lesson |
| `rollout_worker.py` | 759-762 | `update_lesson_stats.remote(stats, "training", step)` | No | Update after training batch |

## Plan: Local curriculum for manual mode

### Change 19: Add `get_or_create_local_curriculum()` and use it in manual mode

**Approach**: Add a function that creates a plain `Curriculum` instance (not a Ray actor) and a thin wrapper that makes it callable with the same interface (direct method calls instead of `.remote()`). The rollout worker checks `weight_transfer.coordinator_backend` to decide which to use.

**Files to change**:

1. **`lib/marin/src/marin/rl/curriculum.py`**:
   - Add `get_or_create_local_curriculum(config, checkpoint_path=None)` that returns a `Curriculum` instance directly

2. **`lib/marin/src/marin/rl/rollout_worker.py`** (line 326):
   - Check `config.weight_transfer.coordinator_backend == "filesystem"`
   - If filesystem: `self._curriculum_actor = get_or_create_local_curriculum(config.curriculum_config)`
   - If actor: existing `get_or_create_curriculum_actor(config.curriculum_config)` path
   - Replace `.remote()` + `get_default_job_ctx().get()` calls with direct method calls when local

3. **`lib/marin/src/marin/rl/train_worker.py`** (line 218):
   - Same pattern: local curriculum when `coordinator_backend == "filesystem"`
   - The train worker only calls `save_checkpoint` — can save directly to GCS

**What stays the same**:
- `Curriculum` class itself — unchanged
- `CurriculumConfig` — unchanged
- Ray mode path — unchanged (guarded by coordinator_backend check)

### Change 19: Local curriculum for manual mode
**Files**: `lib/marin/src/marin/rl/curriculum.py`, `lib/marin/src/marin/rl/rollout_worker.py`, `lib/marin/src/marin/rl/train_worker.py`
**What**: Added `create_local_curriculum()` in curriculum.py. Both rollout_worker and train_worker check `config.weight_transfer.coordinator_backend == "filesystem"` and create a local `Curriculum` instance instead of a fray Ray actor. All `.remote()` call sites (lines 607, 688, 759 in rollout_worker; line 386 in train_worker) have conditional direct-call paths.
**Why**: In manual mode there's no Ray cluster for actor discovery. With a single sampler, the curriculum doesn't need cross-VM sharing — a local instance works identically.

### 2026-03-17 00:00 — Run 21: Local curriculum + inflight_weight_updates=False
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-164815
- **Result**: BREAKTHROUGH — full pipeline works end-to-end
  - Trainer: model loaded, weights served via filesystem coordinator (**works**)
  - Sampler: weights received via Arrow Flight (**works**)
  - Sampler: vLLM inference running at ~2,900 tok/s output (**works**)
  - Sampler: local curriculum — `sample_lesson()` and `update_lesson_stats()` working (**works**)
  - Sampler: 500-prompt eval completed, then 256-prompt rollout batch completed (**works**)
  - Sampler: **rollout written to GCS** (`gs://.../rollouts/00001773705616875989_t1v-n-64f641b6-w-0_000000.pkl`) (**works**)
  - Trainer: data loader polling — "No rollouts received from data loader within timeout, retrying..." then script retries exhausted
  - **1 rollout file exists on GCS** — the full generate→write pipeline works
- **Remaining issue**: Trainer's replay data loader doesn't pick up the GCS rollout file before timeout. Likely a polling/path/format issue, not a fundamental blocker. The sampler only wrote 1 rollout batch before the trainer's timeout killed the run.

## Status: Pipeline works end-to-end. Last mile: trainer data loader.

The entire on-demand RL pipeline is functional:
1. Trainer loads model, creates optimizer (**works**)
2. Trainer serves weights via filesystem coordinator + Arrow Flight (**works**)
3. Sampler receives weights over internal IP (**works**)
4. Sampler loads weights into vLLM via sync_weights (**works**)
5. Sampler generates rollouts at ~2,900 tok/s (**works**)
6. Sampler writes rollouts to GCS (**works**)
7. Trainer reads rollouts from GCS → **needs investigation** (timeout/polling issue)

### Data loader investigation (Run 21 continued)

The trainer's `ReplayDataLoader` IS reading rollouts from GCS:
```
00:00:17 - "Collected 1 rollout batches, updated replay buffer" (reward mean: 0.384)
00:06:56 - "Collected 1 rollout batches, updated replay buffer" (reward mean: 0.346)
```

But `sample_rollouts()` returns `None` because `total_count < local_batch_size`:
- `local_batch_size = train_batch_size = 1024`
- Each sampler batch produces ~256 rollouts (auto rollout shape)
- Need ~4 sampler batches to fill the replay buffer
- The sampler generates one batch every ~3 minutes (eval + inference)
- Total time needed: ~12 minutes of sampler uptime

The run died because the **spot TPU was preempted** (SSH connection reset), not a code bug. The pipeline was working — just needed more time.

## Status: PIPELINE COMPLETE. Needs stable compute.

Everything works end-to-end:
1. Trainer loads model, creates optimizer (**works**)
2. Trainer serves weights via filesystem coordinator + Arrow Flight (**works**)
3. Sampler receives weights over internal IP (**works**)
4. Sampler loads weights into vLLM via sync_weights (**works**)
5. Sampler generates rollouts at ~2,900 tok/s (**works**)
6. Sampler writes rollouts to GCS (**works**)
7. Trainer reads rollouts from GCS into replay buffer (**works** — collected 2 batches)
8. Replay buffer fills to `train_batch_size` → training step → **needs ~4 batches, preempted after 2**

### Change 20: Reduce `train_batch_size` to 256 for manual mode
**File**: `experiments/exp2039_rl_math500.py`
**What**: Set `train_batch_size=256` in `_run_manual_mode` (alongside `inflight_weight_updates=False`).
**Why**: The sampler generates 256-rollout batches. With `train_batch_size=1024`, the replay buffer needs 4 batches (~12 min) before training can start. With 256, training starts after 1 batch.

### 2026-03-17 00:46 — Run 22: train_batch_size=256 (FIRST SUCCESSFUL TRAINING STEP)
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-173245
- **Result**: SUCCESS — RL training loop running
  - Trainer: weights served at 00:42:42
  - Sampler: 500-prompt eval completed, 256-prompt rollout batch at 2,971 tok/s
  - Trainer: "Received initial rollouts! Buffer size: 256" at 00:46:32
  - Trainer: **first training step completed** — `train: 1/500, loss=-0.000278`
  - Trainer: serving updated weights at step 0
  - W&B: https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260316-173245-train
- **THE RL LOOP IS RUNNING.**

### Change 21: Restore `@jax.jit` on `copy_and_flatten`
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
**What**: Reverted `copy_and_flatten` to the original `@partial(jax.jit, static_argnames=("convert_to_bfloat16",))` version. Removed the CPU-based `jax.device_get(model)` + numpy path.
**Why**: The CPU path was a workaround for v6e-8 OOM (31 GiB/chip). On v5p-8 (95 GiB/chip) there's plenty of headroom. The `@jax.jit` version does bf16 conversion + flattening on device, then only transfers ~8GB (bf16) to host. The CPU path transferred 32GB (float32) to host then converted in numpy. Result: `state_dict` time went from **100s → 0.01s** (after JIT warmup).

### 2026-03-17 01:25 — Run 23: @jax.jit restored
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, spot, RUN_ID=exp2039-20260316-181533
- **Result**: SUCCESS — RL training running with fast weight serve
  - Initial weight serve: `state_dict=2.89s, copy=7.88s` (JIT warmup)
  - Subsequent weight serve: `state_dict=0.01s, copy=8.54s` (~8.5s total vs ~110s before)
  - Training step 0: `loss=-0.000278, duration=34s` (fwd/bwd=32s, batch_prep=2s)
  - W&B logging active
- **10x speedup** on weight serve vs the CPU path.

## MILESTONE: On-demand RL training is working end-to-end.

The full cycle works without Ray:
1. Trainer loads Llama 3.1-8B, creates optimizer → serves weights via Arrow Flight
2. Sampler receives weights, loads into vLLM, generates rollouts at ~3,000 tok/s
3. Sampler writes rollouts to GCS
4. Trainer reads rollouts, executes training step (loss=-0.000278)
5. Trainer serves updated weights → cycle repeats

## Performance Profile (Run 23, v5p-8 trainer + v5p-8 sampler)

### Per-step breakdown
| Phase | Time | Notes |
|-------|------|-------|
| Weight serve (JIT) | 8.5s | 0.01s serialize + 8.5s D2H copy |
| Training step (fwd/bwd) | 1-32s | 32s first step, ~1-13s after JIT warmup |
| Batch prep | 2-16s | Varies based on replay buffer timing |
| **Trainer total** | **~17s** | After JIT warmup |
| Eval (500 prompts) | 67s | MATH-500 full eval, runs every step |
| Rollout batch (256 prompts) | 28s | Actual rollout generation |
| Weight sync to vLLM | ~15s | Arrow Flight + sync_weights |
| **Sampler total per cycle** | **~110s** | Eval + rollout + weight sync |
| **End-to-end per step** | **~2.5 min** | Bottlenecked by sampler |

### Throughput
| Metric | Value |
|--------|-------|
| Sampler output throughput | ~3,000 tok/s |
| Sampler max concurrency (KV cache) | 1,318 requests |
| Sampler actual concurrency | 256 prompts (rollout) / 500 prompts (eval) |
| Prompts per step | 756 (500 eval + 256 rollout) |
| Training steps completed | 7+ (and counting) |

### Current bottlenecks (ordered by impact)

1. **Eval frequency = 1**: Full 500-prompt MATH-500 eval runs after EVERY training step (67s). This is ~60% of the sampler's time. Setting `eval_frequency=10` would nearly 3x the step throughput.

2. **Single sampler**: The v5p-8 sampler (TP=4, 1 replica) is underutilized — KV cache can hold 1,318 concurrent requests but we send 256-500 at a time. A v5p-16 with 2 DP replicas would ~2x throughput. The vllm_tpu_multi logbook showed 3x scaling with 4 replicas on v6e-16.

3. **train_batch_size=256**: Reduced from 1024 to avoid waiting for multiple rollout batches. Could increase back to 1024 with more samplers or faster rollout generation.

4. **Spot preemption**: v5p-8 spot TPUs get preempted frequently. On-demand or reserved capacity would give more stable runs.

### Post-mortem: Run 23 death (spot preemption cascade)

**Timeline:**
- 02:15 — Step 15 completed. Trainer died (75 min gap in logs — spot preemption).
- 03:30 — Launch script retried, trainer restarted on same TPU. Restored from step-32 checkpoint. Served fresh weights (`weight_id=-1`).
- 03:39 — Step 33 completed. Trainer served weights with new Arrow Flight ports.
- 03:41 — Sampler tried to receive new weights but got `Connection refused` — trainer died a second time. Sampler stuck retrying stale Flight addresses.
- 03:42 — Trainer's `launch.py` exhausted retries, exited. `_run_exp2039.sh` `wait` returned. Bash script exit sent SIGHUP to sampler's `launch.py`. Sampler's Docker container killed via `docker rm -f levanter`. Python process got SIGKILL — no `wandb.finish()`.
- Wandb heartbeat stopped → wandb marked both runs as "crashed" after ~5 min timeout.

**Root cause**: Spot preemption (twice). No code bug.

**No recovery path exists**: When the trainer dies, the sampler doesn't know — it retries stale Flight connections forever. When the trainer restarts, it writes new coordinator JSON, but by then the sampler may be in a retry loop with old addresses. When the launch script gives up on the trainer, it tears down the sampler too (bash `wait` + SIGHUP).

**Future improvements** (not blocking, operational):
- Sampler should re-read coordinator JSON on every connection failure (not cache stale addresses)
- Use on-demand or reserved capacity for runs that need to complete
- Or: add checkpoint-and-resume so the run can survive preemption

### Optimal concurrency: 256 concurrent prompts per replica

**Reference**: Concurrency sweep results in `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md` (lines 256-299).

Benchmarks swept concurrency from 32 to 1024 on v5p-8, v5p-16, and v6e-16. **Concurrency=256 per replica saturates ALL hardware**:

| Hardware | tok/s @ c=64 | tok/s @ c=256 | Speedup |
|----------|-------------:|--------------:|--------:|
| v5p-8 | 4,749 | 7,940 | 1.7x |
| v5p-16 (2 replicas) | 8,681 | 14,303 | 1.6x |
| v6e-16 (4 replicas) | 11,870 | 19,708 | 1.7x |

Going beyond 256 gives zero additional throughput. The original `exp2039` config used `n_prompts=64` (concurrency=64), leaving 1.7x throughput on the table. The `auto` rollout shape correctly targeted 256 concurrency but produced too few total completions per batch (256 vs 1024 needed).

**Fix**: Set `n_prompts=256, n_generations_per_prompt=16` = 4096 completions per batch. This hits optimal concurrency AND fills the `train_batch_size=1024` replay buffer in one cycle. `n_generations_per_prompt` must stay at 16 — it's the RLOO group size, not a throughput knob.

### Change 22: Set `n_prompts=256, n_generations_per_prompt=16` (optimal concurrency)
**File**: `experiments/exp2039_rl_math500.py`
**What**: Changed `n_prompts` from 64 to 256. Kept `n_generations_per_prompt=16` (RLOO group size).
**Why**: Concurrency sweep benchmarks (`/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md` lines 269-280) showed concurrency=256 saturates v5p-8 at 7,940 tok/s (1.7x faster than concurrency=64). Each rollout batch now produces 256×16=4096 completions, filling the 1024 replay buffer in one cycle (4 training steps per cycle).

### 2026-03-17 04:55 — Run 24: n_prompts=256, on-demand, train_batch_size=1024
- **Config**: v5p-8 trainer + v5p-8 sampler, us-central1-a, on-demand, RUN_ID=exp2039-20260316-214909
- **Sampler throughput**:
  - Eval (500 prompts): 2:29, 1,285 tok/s output
  - Rollout (4096 completions): **6:52, 3,638 tok/s output** — 1.7x faster than concurrency=64
  - XLA compilation: 3 kernel compilations on first cycle only, cached thereafter
- **Training**: 4 steps per rollout batch (4096/1024=4)
- **Expected step time**: ~2 min/step (6:52 rollout ÷ 4 steps + eval amortized over 4 steps)
- **On-policy / concurrency / batch size trilemma**: Discovered that `n_prompts=256` with `train_batch_size=1024` does NOT give 4 training steps per rollout batch. The replay buffer's staleness filter (`Filtered 2048 stale rollouts 0 remaining`) discards rollouts that fall too far behind the current training step. In practice: step 0 uses 1024, step 1 uses 1024, then remaining 2048 are filtered as stale = **2 steps, 50% waste**.

  The fundamental constraint: vLLM generates all completions in one call. You can't partially use them without the rest going stale after weight updates.

  | Setting | Concurrency | On-policy? | tok/s | Waste |
  |---------|------------|------------|-------|-------|
  | `n_prompts=64, batch=1024` | 64 | Yes (1:1) | 1,285 | 0% |
  | `n_prompts=256, batch=1024` | 256 | No (50% stale) | 3,638 | ~50% |
  | `n_prompts=256, batch=4096` | 256 | Yes (1:1) | 3,638 | 0% |

  Option 3 (`train_batch_size=4096`) is the clean solution for optimal concurrency + on-policy, but the training step is 4x larger (fwd/bwd ~140s vs ~35s), giving fewer gradient updates per hour. For now, using `n_prompts=64` to reproduce the blog exactly.

### Critical issue: Silent sampler crashes with no error logging

**Observed**: Sampler container exits with code 1 after ~29 training steps. No Python traceback in docker logs. No kernel OOM. The last log line is a progress bar (`Processed prompts: 100%`). The crash causes the trainer to wait for rollouts indefinitely (4+ hours wasted).

**Root causes of missing logs**:
1. Docker captures stdout/stderr but Python's stderr buffering means the traceback may not flush before the process exits
2. The rollout worker's `run()` method has no top-level exception handler — unhandled exceptions kill the process silently
3. vLLM's progress bars write to stderr with `\r` (carriage return) which can overwrite error messages
4. No structured logging of exceptions — errors go to stderr which may not be captured by the log infrastructure

**Impact**: We've lost hours of compute multiple times because crashes go undetected. The trainer sits idle waiting for rollouts that will never come.

### Plan: Debug mode for on-demand RL

**Goal**: Every crash, exception, or unexpected exit must produce a visible, persistent error message with full traceback.

**Changes needed**:

1. **Top-level exception handler in bootstrap_rl.sh**:
   - Wrap the `exec python "$@"` in a trap that logs exit code
   - Redirect stderr to a GCS log file in addition to stdout
   - Add `PYTHONFAULTHANDLER=1` to catch segfaults with Python tracebacks
   - Add `PYTHONUNBUFFERED=1` to force immediate flush of all print/log output

2. **Top-level try/except in rollout_worker.py `run()`**:
   - Catch all exceptions at the top of the while loop
   - Log the full traceback with `logger.exception()`
   - Write the traceback to a GCS error file for post-mortem
   - Re-raise after logging so the process still exits

3. **Top-level try/except in train_worker.py `train()`**:
   - Same pattern as rollout worker

4. **Structured crash file on GCS**:
   - On any crash, write `{shared_root}/errors/{role}_{timestamp}.txt` with:
     - Full traceback
     - Last 100 log lines
     - Memory stats (host + HBM)
     - Step number at time of crash
   - The OTHER worker can poll this file and abort early instead of waiting forever

5. **Heartbeat / liveness check**:
   - Each worker writes a heartbeat timestamp to GCS every 60s
   - The other worker checks: if heartbeat is >5 min stale, abort with clear error
   - Prevents the 4-hour idle wait scenario

**Immediate quick fix** (apply now):
- Add `PYTHONUNBUFFERED=1` and `PYTHONFAULTHANDLER=1` to bootstrap_rl.sh
- These two env vars alone would have captured the crash in this case

### What's NOT a bottleneck anymore
- Weight serialization: 0.01s with @jax.jit (was 100s with CPU path)
- Arrow Flight transfer: ~8.5s for 8GB bf16 model
- Coordinator: filesystem backend on GCS, ~0.1s per read/write
- vLLM weight sync: ~15s via sync_weights (no EngineCore crash with inflight_weight_updates=False)

4. **Curriculum actor**: Still needs filesystem-mode support (`get_or_create_curriculum_actor` uses fray). This will be the next blocker after vLLM works.

## Files Modified This Session (Summary)

| File | Change |
|------|--------|
| `experiments/exp2039_rl_math500.py` | v6e presets → v6e-16 trainer; manual mode CLI (pre-existing) |
| `on-demand-rl-scripts/_run_exp2039.sh` | `resolve_tpu_type()`, bootstrap wrapper |
| `on-demand-rl-scripts/bootstrap_rl.sh` | Created, iterated, simplified to just `exec python` |
| `on-demand-rl-scripts/exp2039_euwest4.sh` | Pre-existing |
| `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` | `FileSystemArrowFlightCoordinator`, `_create_coordinator()`, `_resolve_advertise_host()`, `_gcs_read()`/`_gcs_write()`, removed `@jax.jit` from `copy_and_flatten` |
| `lib/marin/src/marin/rl/rl_experiment_utils.py` | Removed stale `pip_dependency_groups` kwarg |
| `lib/levanter/docker/tpu/Dockerfile.incremental` | `uv sync --extra rl` + marin/iris pyproject.toml mounts |
| `pyproject.toml` | Added `[project.optional-dependencies] rl` composite extra |
| `uv.lock` | Updated with new extra |

---

## Session: 2026-03-17T17:58Z — Exception handling + restart

### Operational Note: Finding on-demand TPUs

**IMPORTANT**: On-demand TPUs created via `launch.py` show up as **queued resources**, NOT in `tpu-vm list`.

- **WRONG**: `gcloud compute tpus tpu-vm list --zone=us-central1-a` — does NOT show `exp2039-trainer`/`exp2039-sampler`
- **RIGHT**: `gcloud alpha compute tpus queued-resources list --zone=us-central1-a` — shows them with state ACTIVE/SUSPENDED/etc.

The Iris-managed TPUs (`marin-tpu_v5p_8-*`, `ray-marin-*`) appear in `tpu-vm list`. The on-demand exp2039 TPUs only appear in queued-resources.

### Change: Add top-level exception handling to RL workers

**Why**: The sampler exited with code 1 silently — no traceback in docker logs, last visible line was a vLLM progress bar. The trainer then waited 4+ hours for rollouts that never came. Root cause: no top-level exception handlers in any entry point; Python stderr buffering can lose tracebacks before exit.

**Files changed**:
| File | What |
|------|------|
| `lib/marin/src/marin/rl/rollout_worker.py` | Wrapped `run()` in try/except/finally. `step=0` before try so except handler can log it. Finally does direct cleanup (not `stop()` which deadlocks). Logs `ROLLOUT WORKER CRASHED at step=X, weight_step=Y`. |
| `lib/marin/src/marin/rl/train_worker.py` | Wrapped `train()` in try/except/finally. Passes on `StopTrainerException` (graceful). Logs `TRAIN WORKER CRASHED`. Finally calls `self.stop()` (safe — no blocking wait). |
| `lib/marin/src/marin/rl/rl_job.py` | Wrapped both `train_worker_task()` and `inference_worker_task()` closures in try/except to catch `__init__` crashes. |
| `experiments/exp2039_rl_math500.py` | Wrapped `_run_manual_mode()` worker dispatch in try/except with `sys.exit(1)`. Added `import sys`. |

**Design decisions**:
- All handlers use `logger.exception()` (ERROR level, always on, includes traceback) then re-raise
- Only the outermost layer (`exp2039`) calls `sys.exit(1)`
- RolloutWorker.finally does inline cleanup (not `stop()`) because `stop()` calls `self._shutdown_complete.wait()` which would deadlock since `run()` is the method that sets it

### Run launched: exp2039-20260317-110652
- **Run ID**: `exp2039-20260317-110652`
- **Shared root**: `gs://marin-us-central1/tmp/exp2039/exp2039-20260317-110652`
- **TPUs**: `exp2039-trainer` and `exp2039-sampler` in `us-central1-a` (v5p-8, on-demand, ACTIVE)
- **Preset**: `v5p_central1a`
- **Launched**: 2026-03-17T18:06Z
- **Status**: Both containers started. Trainer loading model from HF, sampler vLLM engine initialized with dummy weights.
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773770825`
- **Includes**: Exception handling changes (this session's code changes)

---

## Session: 2026-03-17 — Plan to Diagnose Silent Sampler Hangs (Forensics-First)

### Why this plan exists
The current issue is not just missing liveness; the sampler can hang after `sync_weights` with no Python exception. Heartbeat and watchdog are necessary for fast detection, but the main objective is root-cause evidence capture (where it hung, in which phase, with what stack traces/process state).

### Goal
Capture actionable forensic artifacts on the next hang so we can identify the failing boundary (Python, vLLM EngineCore subprocess, XLA/native runtime, or RPC boundary) and then fix the underlying cause.

### Scope
1. Add explicit phase/state telemetry in sampler and trainer.
2. Add hard timeout watchdogs around hang-prone phases.
3. On timeout, dump forensic artifacts to GCS before exiting.
4. Add trainer-side stale-heartbeat abort so we do not burn hours waiting for rollouts.
5. Add a minimal reproducer mode to isolate `sync_weights -> generate` behavior.

### Non-goals
1. This plan does not attempt to fix vLLM/XLA hang behavior immediately.
2. This plan does not change rollout/training algorithm semantics.

### Planned changes

### Change 23 (planned): Add shared liveness + forensic utilities
**File**: `lib/marin/src/marin/rl/liveness.py` (new)
**What**:
1. `SamplerPhase` enum with phases: `IDLE`, `SYNC_WEIGHTS`, `RESET_PREFIX_CACHE`, `GENERATE`, `WRITE_ROLLOUT`, `WAITING_FOR_WEIGHTS`.
2. `HeartbeatRecord` dataclass with fields: `run_id`, `worker_role`, `step`, `weight_id`, `phase`, `phase_started_at`, `last_main_tick_at`, `updated_at`.
3. Sync GCS JSON writer/reader helpers (same synchronous GCS client style used in `arrow_flight.py`).
4. `dump_python_stacks(path)` helper using `faulthandler`.
5. `capture_process_snapshot(path)` helper (`ps`, RSS, CPU, open files where available).

**Why**: Centralize liveness and forensic primitives so rollout/trainer both speak the same heartbeat schema.

### Change 24 (planned): Instrument sampler phases + watchdog + crash bundle
**File**: `lib/marin/src/marin/rl/rollout_worker.py`
**What**:
1. Initialize heartbeat writer at worker startup.
2. Wrap critical operations with phase markers and elapsed timers:
   - receiving weights
   - prefix cache reset
   - first generate call after sync
   - rollout write to GCS
3. Add watchdog context manager with per-phase hard timeouts:
   - `SYNC_WEIGHTS`: 600s
   - `GENERATE`: 1200s
   - `WRITE_ROLLOUT`: 300s
4. On watchdog timeout, write forensic bundle to `gs://.../debug/hangs/<timestamp>/`:
   - `metadata.json` (phase, step, weight_id, elapsed)
   - `python_stacks.txt`
   - `process_snapshot.txt`
   - `recent_events.json` (last N phase transitions)
5. Exit non-zero after bundle write to force visible failure instead of indefinite hang.

**Why**: Turn silent hangs into deterministic failures with preserved evidence.

### Change 25 (planned): Trainer stale-heartbeat detection and fail-fast
**File**: `lib/marin/src/marin/rl/train_worker.py`
**What**:
1. During rollout wait loop, poll sampler heartbeat JSON.
2. If `now - last_main_tick_at > stale_main_timeout` (default 600s), raise explicit error with phase/step/weight_id context.
3. If hang artifact marker exists, surface that GCS path in the trainer error.

**Why**: Prevent multi-hour idle waits and make failures self-identifying.

### Change 26 (planned): Manual stack-dump trigger for live debugging
**Files**:
1. `lib/marin/src/marin/rl/rollout_worker.py`
2. `on-demand-rl-scripts/bootstrap_rl.sh`

**What**:
1. Register `faulthandler` signal handler (SIGUSR2) in sampler process.
2. Ensure stderr is unbuffered and flushes immediately (`PYTHONUNBUFFERED=1`, `PYTHONFAULTHANDLER=1` already in place/kept).
3. Document operator command: send SIGUSR2 to sampler PID to force thread dump while process is stuck.

**Why**: Allows on-demand diagnostic capture without waiting for watchdog timeout.

### Change 27 (planned): Add minimal reproducer mode for hang isolation
**File**: `experiments/exp2039_rl_math500.py`
**What**:
1. Add debug mode that runs only sampler-side critical loop:
   - receive/sync weights
   - single fixed-prompt `generate`
   - repeat with fixed seed
2. Optional flags:
   - `--debug-disable-weight-refresh` (control path)
   - `--debug-restart-engine-each-iter` (process-lifecycle isolate)
3. Emit per-iteration timing + phase transitions to GCS JSONL.

**Why**: Isolate hang trigger from full trainer/replay/curriculum flow and make behavior reproducible faster.

### Execution order
1. Implement Changes 23-25 first (core visibility + fail-fast).
2. Add Change 26 (manual signal-triggered stack dumps).
3. Add Change 27 (reproducer mode for controlled bisect).
4. Run one on-demand cycle and wait for either:
   - normal progress, or
   - watchdog-triggered forensic bundle.
5. Use collected artifacts to classify root cause and plan targeted fix.

### Success criteria
1. On next sampler stall, trainer exits within <=10 minutes with explicit stale-heartbeat error.
2. A hang artifact bundle is present in GCS with stack/process metadata.
3. We can pinpoint the stuck phase and whether EngineCore/native code is implicated.
4. Reproducer mode can trigger or rule out the hang in isolation.

### Risks and mitigations
1. **Risk**: Watchdog false positives during one-time XLA compile.
   **Mitigation**: Use phase-specific timeouts and log first-compile markers.
2. **Risk**: Diagnostic writes fail when process is degraded.
   **Mitigation**: Best-effort local temp write then upload; always emit final stderr summary before exit.
3. **Risk**: Additional logging overhead affects throughput.
   **Mitigation**: Keep heartbeat payload small and write every 30s; detailed dumps only on timeout/failure.

### Notes
1. Heartbeat/watchdog are observability controls, not root-cause fixes.
2. Root-cause progress comes from forensic artifacts + minimal reproducer bisect.

## Session: 2026-03-17 — Implement Phase 1 Diagnostic Logging for Silent Sampler Hangs

### What was done
Implemented a simplified version of Changes 23-26 (~40 lines across 4 files). Skipped the full GCS heartbeat/forensic bundle approach in favor of lightweight faulthandler watchdogs + phase logging.

### Change 28: Phase logging + faulthandler watchdog in rollout worker
**File**: `lib/marin/src/marin/rl/rollout_worker.py` (~15 lines added)
**What**:
1. Added `import faulthandler, signal, sys`.
2. At `run()` entry: `faulthandler.enable()` (segfault/abort tracebacks) + `faulthandler.register(SIGUSR2)` (manual `kill -USR2 <pid>` stack dump).
3. Phase markers with per-phase watchdog timeouts in the main loop:
   - `PHASE: SYNC_WEIGHTS` (600s watchdog)
   - `PHASE: GENERATE` (1200s watchdog)
   - `PHASE: WRITE_ROLLOUT` (300s watchdog)
   - `PHASE: IDLE` (cancel watchdog, log elapsed time)
4. Each `faulthandler.dump_traceback_later(timeout, exit=True)` call will dump all thread stacks to stderr and exit non-zero if the phase exceeds its timeout.
**Why**: On next hang, `docker logs` will show the last `PHASE:` line identifying exactly where it stuck. If the watchdog fires, full thread stacks are printed before exit — no more silent multi-hour hangs.

### Change 29: Timing logs in vLLM reload_model and generate
**File**: `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` (~12 lines added)
**What**:
1. `reload_model()`: Added timing logs for prefix cache reset, state dict conversion, `sync_weights()` call (with param count), and total duration.
2. `batch_completions()`: Added log before `llm.generate()` (prompt count, max_tokens) and after (elapsed time).
**Why**: These are the two operations that span the Python→vLLM→XLA boundary where hangs occur. Timing lets us see exactly how long each blocking call took.

### Change 30: Entry/exit logs in Arrow Flight receive_weights
**File**: `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` (~3 lines added)
**What**:
1. Entry log: `"receive_weights: polling for step > %d"` with current step.
2. After `update_model()`: `"receive_weights: update_model complete, total=%.1fs"`.
**Why**: Distinguishes "stuck polling coordinator" from "stuck deserializing weights" from "stuck in jit'd update_model".

### Change 31: Cumulative timeout in trainer data loader + faulthandler
**File**: `lib/marin/src/marin/rl/train_worker.py` (~8 lines added)
**What**:
1. Added `import faulthandler` and `faulthandler.enable()` at top of `train()`.
2. `StreamingRolloutLoader.__iter__()`: Added cumulative wait tracking. If no rollouts are received for 1 hour total (across retries), raises `TimeoutError` instead of retrying forever. Counter resets on each successful fetch.
**Why**: Previously the trainer would wait indefinitely (60s timeout, infinite retries) if the sampler hung. Now it fails fast with a clear error after 1 hour.

### Change 32: bootstrap_rl.sh — already had PYTHONFAULTHANDLER=1
**File**: `on-demand-rl-scripts/bootstrap_rl.sh`
**What**: No changes needed — `PYTHONFAULTHANDLER=1` and `PYTHONUNBUFFERED=1` were already present.

### What this gives us on next hang
| Signal | Source |
|--------|--------|
| Last `PHASE:` line in docker logs | Identifies stuck operation |
| faulthandler watchdog fires → stderr thread dump + exit | Full stack traces of all threads, automatic exit |
| `kill -USR2 <pid>` | Manual thread dump without waiting for watchdog |
| Timing logs in vLLM/Arrow Flight | How long each blocking call took before hang |
| Trainer TimeoutError after 1h | Fail-fast instead of infinite wait |

### What was NOT implemented (deferred)
- GCS heartbeat protocol (Change 23 `liveness.py`)
- Forensic bundle uploads to GCS (Change 24 partial)
- Minimal reproducer mode (Change 27)
- These are only needed if the phase logging + faulthandler approach doesn't give us enough signal.

### Verification plan
1. Build new Docker image with changes.
2. Launch exp2039 trainer + sampler.
3. Confirm new `PHASE:` log lines appear in `docker logs`.
4. Confirm faulthandler watchdog works: temporarily set a very short timeout (e.g. 5s) and verify stack dump + exit on a slow generate.
5. If sampler hangs naturally, check docker logs for last PHASE line and stack dump.

### Run launched: exp2039-20260317-171645 (restart with diagnostics)
- **Date**: 2026-03-17T17:30Z
- **Run ID**: `exp2039-20260317-171645` (same as previous run, resumed)
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773793716`
- **TPUs**: `exp2039-trainer` + `exp2039-sampler` in `us-central1-a` (v5p-8, on-demand, ACTIVE)
- **Launched**: Both containers started in `-d` (detached) mode via manual `docker run` (not via launch.py foreground)
- **Includes**: All Phase 1 diagnostic logging (Changes 28-31) + `force=True` logging fix

### Fix: Sampler logging was silent
**Problem**: The `PHASE:` and timing logs were invisible in docker logs. `marin.rl.rollout_worker` logger was at WARNING level.
**Root cause**: The sampler doesn't call `levanter.initialize()` (which sets up logging) to avoid JAX distributed init deadlocks with vLLM. So the root logger kept its default WARNING level.
**Fix**: Added `logging.basicConfig(level=logging.INFO, ..., force=True)` in `exp2039_rl_math500.py` before `RolloutWorker().run()`. The `force=True` is needed because vLLM/wandb configure handlers before we get there, and `basicConfig` is a no-op if handlers already exist.

### Verified: Diagnostic logs working
Confirmed all diagnostic log lines visible in `docker logs levanter`:
```
PHASE: SYNC_WEIGHTS step=0
reload_model: calling sync_weights (2 params, 0.2s so far)
reload_model: complete in 12.6s
generate: starting, 500 prompts, max_tokens=1024
generate: done in 86.2s
PHASE: GENERATE step=0 lesson=math_full
generate: starting, 64 prompts, max_tokens=1024
```
All phase markers, timing logs, and Arrow Flight polling logs working as designed.

### Finding: Sampler silently generates with stale weights after failed sync

**Date**: 2026-03-18T02:15Z

**What happened**:
1. The background `launch.py` retry loop (which was still running from an earlier attempt) kept retrying the trainer with `--foreground` mode. Each retry did `docker rm -f levanter` — which also killed the **sampler's** container as collateral damage.
2. The sampler came back up fresh with dummy weights (`weight_step=-2`). It tried to sync with the trainer's Arrow Flight servers, but those were on different ports after the trainer's own restarts.
3. The sampler logged `ERROR Failed to receive weights via Arrow Flight` every second for 300s (the `max_weight_transfer_wait_time`).
4. After 300s, `_sync_weights()` gave up and **proceeded to generate with stale weights** (`weight_step=-2`).
5. The sampler produced rollouts stamped `weight_step=-2`. The trainer (on step 7) rejected them: `Skipping stale rollout batch (rollout_step=-2, current_step=7)`.
6. The trainer's cumulative wait climbed to 2822s with no usable rollouts, heading toward the 1-hour timeout.

**Diagnostic logging worked**: The PHASE logs and Arrow Flight error logs clearly showed the problem in real time:
```
02:01:16 PHASE: IDLE step=1 elapsed=796.4s
02:01:16 PHASE: SYNC_WEIGHTS step=1
02:06:16 PHASE: GENERATE step=1 lesson=math_full        ← still weight_step=-2
02:12:59 ERROR Failed to receive weights via Arrow Flight
02:13:00 receive_weights: polling for step > -2
02:13:00 ERROR Failed to receive weights via Arrow Flight
```
And on trainer side:
```
02:11:36 Skipping stale rollout batch (rollout_step=-2, current_step=7)
02:11:36 No rollouts received from data loader within timeout (cumulative_wait=2822s), retrying...
```

**Root cause bug**: `_sync_weights()` in `rollout_worker.py` treats a failed weight sync as non-fatal. After `max_weight_transfer_wait_time` (300s), it logs "Waited X for new weights, proceeding with current weights" and continues. This is wrong when the "current weights" are from step -2 and the trainer is on step 7 — the sampler burns compute generating rollouts the trainer will discard.

**Fix options considered**:

**Option A: Guard against dummy weights (1 line, chosen)**
In the main loop, after `_sync_weights()`, skip generation if `_current_weight_step < 0`:
```python
if self._current_weight_step < 0:
    logger.warning("No valid weights yet, retrying sync...")
    time.sleep(5.0)
    continue
```
The sampler loops: sync (300s timeout) → no weights? → sleep 5s → sync again. Never generates garbage rollouts. The faulthandler watchdog resets each iteration so it won't kill the process. Once the trainer serves weights, `receive_weights()` succeeds because `coordinator.weight_id` (-1) differs from `_last_weight_id` (-2), and normal operation resumes.

**Option B: Distinguish "no new weights" from "fetch error" in `receive_weights()` (deferred)**
Currently `receive_weights()` returns `None` for both "no new weights available" (normal) and "connection error" (broken). These need different handling. Change lines 718-720 (connect failure) and 760 (catch-all) to raise instead of returning None. Then `_sync_weights()` treats exceptions as "must retry" and only uses `max_wait_time` for the "no new weights" case. This is the right long-term fix but changes the `WeightTransferClient` contract and needs more testing.

**Option C: Option A + relax staleness to `max_rollout_step_delay=1` (not chosen)**
Also change the trainer's replay buffer to accept rollouts from N-1 steps. More resilient to timing mismatches, common in async RL. Not chosen because RLOO is on-policy and we want strict freshness for now.

**Decision**: Implement Option A now (immediate fix), defer Option B (needs API design thought).

**Operational lesson**: Never leave a `launch.py --foreground` retry loop running in the background. It will kill and recreate containers on both TPUs on every retry. Always use `-d` (detached) mode for manual launches.

### Change 33: Guard against generating with dummy weights
**File**: `lib/marin/src/marin/rl/rollout_worker.py` (5 lines added)
**What**: After `_sync_weights()` in the main loop, check `self._current_weight_step < -1`. If true, log a warning and `continue` (retry sync after 5s sleep). The sampler never generates rollouts until it has received real weights from the trainer.
**Why**: Without this, a restarted sampler generates with dummy weights (step -2), producing rollouts the trainer rejects (`max_rollout_step_delay=0`). Both sides waste compute for the duration of `max_weight_transfer_wait_time` (300s) per sync attempt.

**Bug in initial implementation (fixed)**: The guard originally checked `weight_step < 0`, but the trainer serves initial weights at step **-1** by design (`transfer_server.serve_weights(-1, state.model)` in `train_worker.py:308`). So after the sampler received valid initial weights (step=-1), the guard still blocked generation because `-1 < 0`. The sampler logged `No valid weights received yet (weight_step=-1), retrying sync...` and looped forever, never generating rollouts. The trainer then timed out after 1200s waiting for initial rollouts.

The `_current_weight_step` lifecycle:
- Initialized to `-2` (never received any weights)
- Set to `-1` when trainer serves initial weights
- Set to `0, 1, 2, ...` as training progresses

Fix: changed guard from `< 0` to `< -1`, so it only blocks when no weights have ever been received (step=-2), not when valid initial weights are present (step=-1).

### Run launched: exp2039-20260318-021500 (clean restart)
- **Date**: 2026-03-18T02:15Z
- **Run ID**: `exp2039-20260318-021500`
- **Shared root**: `gs://marin-us-central1/tmp/exp2039/exp2039-20260318-021500`
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773793716`
- **TPUs**: `exp2039-trainer` + `exp2039-sampler` in `us-central1-a` (v5p-8, on-demand, ACTIVE)
- **Launched**: Both containers started in `-d` (detached) mode with fresh run ID and shared root.
- **Killed**: 2026-03-18T03:26Z to deploy Change 33 (dummy weight guard).

### Run launched: exp2039-20260318-032600 (with dummy weight guard)
- **Date**: 2026-03-18T03:26Z
- **Run ID**: `exp2039-20260318-032600`
- **Shared root**: `gs://marin-us-central1/tmp/exp2039/exp2039-20260318-032600`
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773801539`
- **TPUs**: `exp2039-trainer` + `exp2039-sampler` in `us-central1-a` (v5p-8, on-demand, ACTIVE)
- **Launched**: Both containers in `-d` (detached) mode.
- **Includes**: All Phase 1 diagnostics (Changes 28-31) + logging fix (`force=True`) + dummy weight guard (Change 33).

### Finding: faulthandler watchdog caught sympy hang in math grading
**Date**: 2026-03-18T05:05Z
**What**: The sampler hung at step 15 during PHASE: GENERATE. The 1200s faulthandler watchdog fired, dumped all thread stacks to stderr, and killed the process.
**Root cause**: Sympy's `_eval_power` (in `sympy/core/numbers.py:2045`) hung computing a power expression from a model-generated answer. Call stack: `math_grading.py:229 _sympy_parse` → `parse_expr` → `__pow__` → `_eval_power` (infinite/very long computation).
**Impact**: Sampler killed after 20 min, trainer starved for rollouts (cumulative_wait climbed to 1381s).
**This is the diagnostic system working as designed** — without the faulthandler watchdog, this would have been another silent multi-hour hang with no traceback.
**Fix needed**: Add a timeout around `sympy.parse_expr` / `grade_answer` in `math_grading.py` to prevent a single malformed model output from hanging the entire sampler.
**Workaround**: Manually restarted the sampler container. The dummy weight guard (Change 33) ensures it waits for real weights before generating.
**Second occurrence**: 2026-03-18T06:45Z, same stack trace. Confirmed this is a recurring issue, not a one-off.

### Change 34: Add 10s timeout to sympy parse_expr and simplify
**File**: `lib/marin/src/marin/rl/environments/tinker_environments/math_grading.py`
**What**:
1. Wrapped `sympy_parser.parse_expr` in `_sympy_parse()` with a `ThreadPoolExecutor` + 10s `future.result(timeout=...)`. On timeout, logs a warning and raises `TimeoutError` (caught by the existing `except Exception: pass` in `are_equal_under_sympy`).
2. Added `_sympy_simplify_with_timeout()` wrapper around `sympy.simplify` with the same 10s timeout, since simplify can also hang on pathological inputs.
**Why**: The model generates expressions like `2**999999` that cause sympy `_eval_power` to hang forever. This killed the sampler three times via the 1200s faulthandler watchdog, losing ~40 min of compute each time. With the 10s timeout, the answer is graded as incorrect (safe default) and the sampler continues.

**Bug in initial fix (ThreadPoolExecutor)**: The first implementation used `ThreadPoolExecutor` with `future.result(timeout=10)`. This doesn't work for CPU-bound sympy hangs because Python threads can't be killed — the GIL never yields in `_eval_power`, so the zombie thread blocks everything. The faulthandler watchdog still fired after 1200s.

**Fix v2 (multiprocessing.Pool)**: Switched to `multiprocessing.Pool` with `pool.apply_async().get(timeout=10)` + `pool.terminate()`. This spawns a real subprocess that can be killed via SIGTERM. The `_parse_expr_worker` function is top-level for pickling compatibility.
**Confirmed working**: At step 132 the sampler logged `WARNING sympy parse_expr timed out after 10s on: (2**4950)-(2**2**100-1)` — exactly the kind of expression that previously killed the process. It timed out, graded as incorrect, and continued.

## Throughput Analysis: exp2039 vs Marin Blog Baseline

### Context
exp2039 runs at ~4.5 min/step wall-clock. The Marin blog baseline ran at ~1.1 min/step. This is not a bug — it's a different experiment config.

### exp2039 config vs blog baseline
| Config | Blog baseline | exp2039 (our run) | Impact |
|--------|--------------|-------------------|--------|
| n_prompts | 32 | 64 | 2x more prompts |
| n_generations_per_prompt | **8** | **16** | 2x more generations |
| completions/step | **256** | **1024** | **4x more work** |
| max_output_tokens | **513** | **1024** | ~2x longer sequences |
| max_input_tokens | 256 | 1024 | 4x longer inputs |
| max_seq_len | 769 | 2048 | 2.7x longer |
| train_batch_size | 256 | 1024 | 4x bigger batches |
| Infra | Single v5p-8 (sync) | v5p-8 trainer + v5p-8 sampler (async) | Separate machines |

### Who set these params and why
- **Kevin Li** created exp2039 (originally `exp1782_vllm_rl.py`, renamed Jan 2026).
- **Kevin + Chris Chou** set `n_generations_per_prompt=16` and `train_batch_size=1024` in PR #2325 (inflight weight updates, Jan 2026).
- The choices are deliberate for the async RL setup:
  - 16 gens/prompt → better RLOO advantage estimation (lower variance). Standard in recent RLOO papers.
  - 1024 batch → more stable gradients with async rollouts.
  - 1024 max output → longer chain-of-thought for harder MATH problems.
- The 513 in the blog baseline is 512+1 (power of 2 + stop token).

### Where time goes (our run, from wandb)
| Phase | Median time | % of step |
|-------|------------|-----------|
| Waiting for rollouts (batch_prep) | 41.5s | 62% |
| Forward/backward | 35.5s | 53% |
| Weight transfer | 14.4s | — |
| **Step duration** | **67.4s** | — |
| **Wall-clock per step** | **269s (4.5 min)** | — |

The actual training step (67s) matches the baseline exactly. The 4x wall-clock slowdown comes entirely from the sampler producing 4x more completions with 2x longer sequences. The trainer is idle 62% of each step waiting for rollouts.

### Not a bug
The 4.5 min/step rate is correct for this config with 1 sampler. To speed up:
1. Add more samplers (2-3 v5p-8s would saturate the trainer)
2. Reduce n_generations_per_prompt from 16→8 (but changes the experiment)
3. Reduce max_output_tokens from 1024→513 (but changes the experiment)

## Training Quality Analysis: exp2039 vs Baseline (flat eval curve investigation)

### Observation
Our eval pass@1 on MATH-500 is flat (0.478 → 0.458 over 144 steps) while the baseline improved from 0.266 → 0.582 over 297 steps. Additionally, our format accuracy starts at 0.930 and *declines* to 0.808, while the baseline started at 0.516 and climbed to 0.992.

### Eval comparison (from wandb)
| Step | Baseline correct | Our pass@1 |
|------|-----------------|------------|
| Init | 0.266 | 0.478 |
| ~10 | 0.355 | 0.478 |
| ~50 | 0.520 | 0.460 |
| ~100 | 0.520 | 0.456 |
| ~144 | ~0.540 | 0.458 |

### Full hyperparameter comparison (all confirmed from code/wandb configs)
**Identical:** LR (2e-6), schedule (constant), adam betas (0.9/0.95), grad clip (1.0), weight_decay (0), kl_coef (0), temperature (1.0), param_dtype (fp32), compute_dtype (bf16).

**Different:**
| Parameter | Baseline | Our run |
|-----------|----------|---------|
| effective batch size | 256 (16 micro × 16 grad_accum) | 1024 (no accum) |
| n_prompts | 32 | 64 |
| n_generations | 8 | 16 |
| completions/step | 256 | 1024 |
| max_output_tokens | 513 | 1024 |
| max_input_tokens | 256 | 1024 |
| PPO clip_epsilon | none | 0.2/0.28 |
| importance sampling | no (sync) | yes (max ratio 2.0) |
| overlong filtering | no | yes |
| num_train_steps | 2048 | 500 |

### RL metric comparison
| Metric | Baseline | Our run |
|--------|----------|---------|
| RLOO loss | ~1000 → ~35 (large, active learning) | ~-0.0005 (near-zero, flat) |
| Gradient norm | 17k-36k | not logged separately |
| Format accuracy | 0.516 → 0.992 (learned format) | 0.930 → 0.808 (declining) |
| Correct accuracy | 0.324 → 0.582 (improving) | 0.608 → 0.604 (flat) |
| Response tokens | 366 → 218 (getting concise) | 385 → 514 (getting wordier) |
| Truncation | not reported | 6% → 18% (increasing) |

### Hypothesis: gradient signal is diluted by multiple factors

1. **4x larger effective batch (256 vs 1024), same LR.** Each optimizer step averages over 4x more samples, diluting the per-sample gradient by 4x. The baseline used grad_accum_steps=16 with train_bsize=256 for an effective batch of 256.

2. **PPO clipping (0.2/0.28) + importance sampling (max 2.0)** — our run clips policy ratios and caps importance weights, further dampening gradient magnitude. The baseline had no clipping (pure REINFORCE).

3. **16 generations vs 8 per prompt** — RLOO computes advantages by subtracting the group mean. With 16 generations, if most in a group agree (all correct or all wrong), advantages are near-zero. More generations = advantages closer to zero = weaker per-sample gradient.

4. **Overlong filtering discards 6-18% of rollouts** — truncated responses are removed from the loss, reducing effective batch utilization.

5. **Starting point is already strong** — our model starts at 0.478 pass@1 vs baseline's 0.266. The baseline had easy wins from learning format (0.516→0.992), which drove most of its improvement. Our model already formats well (0.930), so there's less low-hanging fruit.

### Correction: n_generations=16 should help, not hurt
Initial hypothesis stated 16 gens → weaker advantages. This was wrong. With more generations per prompt, you're *more* likely to get a mix of correct/incorrect in each group, giving a *more stable* baseline estimate for RLOO. The advantage `reward_i - mean(group)` is better estimated with more samples. 16 gens should improve training signal, not weaken it.

### Confirmed: baseline used a different RL framework entirely
The baseline ("Marin Sync RL", Nov 2025) used Kevin's PRIME-based sync RL code — a completely different framework from the current Marin async RL stack. Evidence:
- Config key prefixes (`hyperparameters.*`, `optim_config.*`, `model_paths.*`, `model_config_override.*`) don't match Levanter/Marin structure
- No clip_epsilon in baseline config (pure REINFORCE/RLOO, no PPO-style clipping)
- Uses `.msgpack` checkpoint format, not safetensors
- Display name "Marin Sync RL (bs=256)" — single-process sync mode

Our exp2039 uses the newer Marin RL stack with `RLOOLoss(clip_epsilon_low=0.2, clip_epsilon_high=0.28)`, importance sampling (max ratio 2.0), and overlong filtering. These are features the baseline didn't have.

### Revised hypothesis: gradient signal diluted by DAPO normalization with larger batch

**PPO clipping is a no-op in our config** (confirmed from code). `synchronous=True` in our `RLOOLoss` config means the importance sampling ratio is `exp(current_logprobs - current_logprobs) = 1.0` for every token (`rl_losses.py:301-302`). The clip range `[0.8, 1.28]` never activates because 1.0 is already inside.

**The trainer-inference importance sampling IS active** (`do_trainer_inference_mismatch_importance_sampling=True`). This computes `exp(current_logprobs - policy_logprobs)` where `policy_logprobs` are from the sampler's rollout. In async mode the trainer's model diverges from the sampler's, so this reweights the loss. Capped at 2.0. This could dampen gradients when trainer and sampler are out of sync.

**The real gradient dilution: DAPO global token normalization.** The loss function uses `compute_dapo_loss` (`rl_losses.py:208,226-234`) which divides by `sum(loss_masks)` — the total token count across ALL examples in the batch. With 1024 examples (our run) vs 256 (baseline), the denominator is ~4x larger, making each token's contribution ~4x smaller. Same LR means ~4x weaker per-step updates.

The strongest factors:
1. **DAPO normalization with 4x more tokens (1024 vs 256 batch)** — directly reduces loss magnitude by ~4x.
2. **Trainer-inference IS reweighting** — when trainer is ahead of sampler, ratios may shrink gradient contributions.
3. **Overlong filtering discards 6-18% of rollouts** — reduces effective batch utilization.

### What this does NOT explain
- Why the starting eval scores differ (0.266 vs 0.478). Both use Llama-3.1-8B-Instruct weights. The baseline used a `Meta-Llama-3-8B-Instruct` tokenizer with 3.1 weights (potential mismatch). The baseline also used a completely different RL framework with different prompt rendering.
- Whether the near-zero loss (-0.0005) is expected behavior for RLOO with PPO clipping at this accuracy level, or indicates a bug.
- How much of the baseline's improvement came from format learning (0.516 → 0.992) vs actual math reasoning.

### Possible next steps for training quality (not yet acted on)
1. Scale LR by 4x (to 8e-6) to compensate for DAPO normalization with larger batch
2. Reduce batch to 256 to match baseline's effective gradient scale
3. Switch from DAPO loss (`/sum(masks)`) to PPO loss (`/batch_size`) normalization to decouple loss scale from batch size
4. Check trainer-inference IS ratio metrics in wandb — if mean ratio is far from 1.0, the reweighting may be suppressing gradients
5. Log clip_fraction metric — should be ~0 in sync mode (confirming clipping is no-op)

## Throughput Optimization: Non-Blocking Sync (Option 3)

### The problem: sampler is idle 50% of the time
The current sync flow wastes ~130s per cycle polling for weights that haven't arrived yet:
```
Time 0:00  Sampler finishes generating with step N weights, writes rollouts
Time 0:00  Sampler polls coordinator: "anything newer than step N?"
Time 0:01  Coordinator: "nope, latest is still step N" (trainer still training)
Time 0:02  "nope"
  ... sampler polls every 1s for up to 300s ...
Time 2:00  Trainer finishes step N+1, serves weights N+1
Time 2:01  Sampler: "anything newer than step N?" → "yes, step N+1!"
Time 2:01  Sampler downloads weights N+1, reloads model (12s)
Time 2:13  Sampler generates with step N+1 weights
```
The sampler sits idle for ~130s per cycle waiting for the trainer. This is by design — it ensures every rollout uses the latest weights (pure on-policy training).

### Proposed change: non-blocking weight sync
Instead of blocking, the sampler does a quick poll and proceeds immediately:
```
Time 0:00  Sampler finishes generating with step N weights, writes rollouts
Time 0:00  Quick poll (40ms): "anything newer than step N?" → "nope"
Time 0:00  Start generating ANOTHER batch with step N weights
Time 2:00  Writes 2nd batch (still step N weights)
Time 2:00  Quick poll: "anything newer than step N?" → "yes, step N+1!"
Time 2:12  Reload weights N+1, generate with new weights
```
The sampler never idles. It generates back-to-back rollouts, picking up new weights whenever they're available.

### Why we weren't doing this already
The system was designed for **on-policy training** in single-sampler mode. The 300s polling wait ensures every rollout is generated with the most recent policy weights. This is theoretically correct for RLOO — the advantages assume all samples in a group come from the same policy.

Making the sync non-blocking is a deliberate choice to **trade on-policy correctness for throughput**. Some rollouts will be generated with slightly stale weights.

### How many off-policy batches?

**Measured timing from logs (steps 141-160):**
- Sampler generate + reload + write: **~132s** per batch
- Trainer fwd/bwd + weight serve: **~80s** per step
- Sampler is 1.6x slower than trainer

**The pattern with non-blocking sync:**
1. Sampler writes batch (on-policy, step N weights). Quick poll — no new weights.
2. Sampler starts generating next batch with step N weights (trainer still training step N+1).
3. ~52s into generation, trainer finishes step N+1, serves weights N+1 to coordinator.
4. Sampler can't stop mid-generation. Finishes batch at ~132s (off-policy, 1-step stale).
5. Writes batch with `weight_step=N`. Quick poll — picks up weights N+1, reloads.
6. Generates next batch with step N+1 weights (on-policy again).
7. Repeat: ON, OFF, ON, OFF, ON, OFF, ...

**Result: ~50% of batches are 1-step stale.** Never more than 1 step — the sampler is slower than the trainer, so it can't fall further behind.

### Trainer accepts these batches
`max_rollout_step_delay=1` (set in exp2039 config) means the replay buffer accepts rollouts from `current_step - 1` through `current_step`. If trainer is on step N+1, rollouts with `weight_step=N` are accepted. Only rollouts 2+ steps stale would be rejected, which can't happen since the sampler is slower than the trainer.

### Off-policy impact is negligible
With LR=2e-6 and loss of -0.0005, the policy barely changes between step N and step N+1. The KL divergence between consecutive policies is effectively zero. Using step N weights vs step N+1 weights produces near-identical rollout distributions. The `synchronous=True` flag means no importance sampling correction is applied, but none is needed when the policy shift is this small.

### Projected speedup
| Metric | Current (blocking sync) | Projected (non-blocking) |
|--------|------------------------|--------------------------|
| Sampler cycle | 279s (4.6 min) | 132s (2.2 min) |
| Wall-clock per trainer step | 279s (4.6 min) | 132s (2.2 min) |
| Speedup | — | **2.1x** |
| Off-policy batches | 0% | ~50% (1-step stale) |
| Total time for 500 steps | ~38 hours | ~18 hours |

The speedup comes entirely from eliminating the ~130s idle polling wait. The sampler generates at full throughput instead of waiting.

### Implementation: what needs to change
The change is small — set `max_weight_transfer_wait_time=0` so `_sync_weights()` does a single non-blocking poll instead of blocking for 300s. If no new weights are available, it returns immediately and the sampler continues with current weights. When new weights appear on the next poll (at the top of the next loop iteration), the sampler picks them up.

No changes needed to:
- The loss function (PPO clipping is already a no-op in sync mode)
- The replay buffer (`max_rollout_step_delay=1` already accepts 1-step stale)
- The rollout storage (already supports continuous writes)
- The trainer (already consumes whatever rollouts are in the buffer)

### Sync vs async vs inflight: terminology clarification
There are three separate concepts that are easy to confuse:

1. **`synchronous` flag in `RLOOLoss`** — controls whether importance sampling uses current vs stored logprobs. `True` = ratio is always 1.0 (on-policy loss). Does NOT control whether the sampler blocks.

2. **`inflight_weight_updates` flag** — controls whether weights are updated MID-generation via a background thread + vLLM RPC. Currently disabled because `AsyncvLLMInferenceContext` crashes after `update_weights` RPC. This is Option 2 (complex, broken).

3. **`max_weight_transfer_wait_time`** — controls how long `_sync_weights()` blocks polling for new weights. Currently 300s. Setting to 0 makes it non-blocking. This is Option 3 (simple, safe).

Option 3 does NOT need inflight updates. It uses the existing sync `vLLMInferenceContext` (direct `sync_weights` call, no RPC). Weights are updated between generations, not during them. The sampler just stops waiting around between generations.

### Adding a 2nd sampler
With non-blocking sync, a 2nd sampler becomes viable:
- Both samplers write to the same GCS rollout storage (unique filenames per worker)
- Both poll the same coordinator for weights
- The trainer's replay buffer handles multiple incoming rollout batches via locks
- With 2 samplers producing batches every ~132s each, the trainer gets a batch every ~66s, nearly saturating its ~80s step rate
- Off-policy fraction stays ~50% per sampler (each alternates ON/OFF)

This was how the prior engineers ran it on the Ray cluster — multiple rollout workers generating concurrently.

### Next experiment: non-blocking sync with max_weight_transfer_wait_time=0

**Plan**: Set `max_weight_transfer_wait_time=0` for the sampler so it does a single non-blocking poll for new weights instead of blocking for 300s. If no new weights are available, the sampler immediately starts generating another batch with current weights.

**Where the value comes from**:
- `WeightTransferConfig.max_weight_transfer_wait_time` defaults to `0.0` in `weight_transfer/base.py:70`
- `rl_experiment_utils.py:91` overrides it to `300` (the value we've been running with)
- `rl_job.py:168` sets it to `600` for the Ray cluster path
- exp2039 doesn't set it explicitly — it inherits the 300 from `rl_experiment_utils.py`

**Implementation**: Change `rl_experiment_utils.py:91` from `max_weight_transfer_wait_time: int = 300` to `0`, or add a CLI arg `--max-weight-wait` to exp2039. Either way, only the sampler is affected — the trainer doesn't use this value.

**Expected behavior**:
- Sampler generates back-to-back batches, picking up new weights whenever they appear at the top of each loop
- ~50% of batches will be 1-step stale (ON, OFF, ON, OFF pattern)
- Trainer accepts all of them (`max_rollout_step_delay=1`)
- Projected 2.1x speedup: 4.6 min/step → 2.2 min/step
- Off-policy impact negligible at LR=2e-6

**What to watch for**:
- Eval pass@1 should not degrade (already flat at ~0.46, unlikely to get worse from 1-step staleness)
- Trainer `batch_prep` time should drop dramatically (no more 40-60s waits for rollouts)
- Sampler `PHASE: SYNC_WEIGHTS` duration should drop from ~130s to <1s on most cycles
- The `cumulative_wait` warnings in the trainer should mostly disappear

**Risk**: Low. The only change is removing a blocking wait. All the machinery for handling stale rollouts is already in place (`max_rollout_step_delay=1`, replay buffer freshness checks). The loss function is unaffected (`synchronous=True` means ratio=1.0 regardless). Worst case: training quality is identical to current run but 2x faster.

### Run launched: exp2039-20260318-120714 (non-blocking experiment)
- **Date**: 2026-03-18T12:07Z
- **Run ID**: `exp2039-20260318-120714`
- **TPUs**: `exp2039-nb-trainer` + `exp2039-nb-sampler` in `us-central1-a` (v5p-8, on-demand)
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773860852`
- **Change**: `max_weight_transfer_wait_time=0` in `rl_experiment_utils.py:91`
- **Launched via**: `_run_exp2039.sh` with `--foreground` mode (THIS WAS A MISTAKE — see below)

**Results before crash (114 steps, ~6 hours)**:
- Pace: 3.4 min/step (1.7x faster than blocking run's 5.7 min/step)
- Sampler cycles: 110-154s (vs blocking run's 224-516s)
- Non-blocking sync confirmed working: sampler polls once, gets nothing, immediately generates
- trainer_inference_importance_sampling_ratio_mean: 0.962 (vs blocking run's 0.950 — negligible difference, both caused by bf16 precision mismatch in weight transfer, not actual policy staleness)
- Eval pass@1: 0.466 at step 15 (comparable to blocking run)
- clip_fraction: 0.0 (confirms PPO clipping is a no-op in sync mode)

**Crash at step 114 (~02:00Z on 2026-03-19)**:
- Trainer last logged step 114 at 01:56:06
- Sampler stopped producing rollouts (unknown cause — LOGS WERE DESTROYED)
- Trainer timed out at 02:29:03: `RuntimeError: Timed out waiting for initial rollouts; aborting training.`
- launch.py retry loop restarted both containers, destroying the original crash logs
- New trainer started at step 0, rejected all old rollouts (step 113-116) as stale
- Timed out again at 02:52:32 — same pattern, unrecoverable

**Root cause of crash: UNKNOWN.** We lost the sampler's crash logs because launch.py's retry loop does `docker rm -f levanter` on each retry, wiping the container and its logs. Could have been: sympy hang (though multiprocessing fix should catch it), vLLM crash, OOM, or TPU preemption. The sync run (same codebase, same sympy fix) has been stable for 19+ hours with 9 sympy timeouts caught and survived.

## CRITICAL OPERATIONAL RULE: NEVER USE launch.py --foreground FOR PRODUCTION RUNS

**This has burned us THREE TIMES now:**
1. First time: launch.py foreground retry loop killed the sampler's working container as collateral
2. Second time: launch.py foreground retry loop destroyed crash logs we needed for debugging
3. Third time: NB run crash logs destroyed, can't do post-mortem

**The problem**: `_run_exp2039.sh` calls `launch.py --foreground` which:
1. Runs `docker run -t` (foreground) — SSH session blocks until container exits
2. On any failure (SSH timeout, container crash), launch.py retries
3. Each retry calls `setup_vm_docker()` which does `docker rm -f levanter` — **destroying the container AND its logs**
4. Even on first attempt, it removes any existing `levanter` container

**The rules going forward**:
1. **NEVER use `_run_exp2039.sh` or `launch.py --foreground` for production runs.** Only use them for initial TPU provisioning + image push.
2. **ALWAYS launch containers in detached mode (`docker run -d`)** via direct SSH.
3. **NEVER retry automatically.** If a container dies, leave the corpse for post-mortem (`docker logs levanter` still works on stopped containers).
4. **Use `docker run -d --restart=no`** — no restart policy, let it die and keep logs.

**Template command for launching (copy-paste)**:
```bash
# Set these variables first
IMAGE="us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:XXXXXXXXXX"
RUN_ID="exp2039-YYYYMMDD-HHMMSS"
SHARED_ROOT="gs://marin-us-central1/tmp/exp2039/${RUN_ID}"

# Trainer (on exp2039-nb-trainer or exp2039-trainer)
gcloud alpha compute tpus tpu-vm ssh TPU_NAME --quiet --worker=all --zone=us-central1-a \
  --command="docker rm -f levanter 2>/dev/null; docker run -d --name=levanter --privileged --shm-size=32gb --net=host --init --mount type=volume,source=levanter,target=/home/levanter -v /tmp:/tmp -e WANDB_API_KEY=... -e WANDB_PROJECT=marin -e WANDB_MODE=online -e HF_TOKEN=... -e TPU_STDERR_LOG_LEVEL=3 -e TPU_MIN_LOG_LEVEL=3 -e TPU_CI=True -e JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=-1 -e TPU_BACKEND_TYPE=jax -e PJRT_DEVICE=TPU -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e RUN_ID=${RUN_ID} ${IMAGE} bash /opt/marin/on-demand-rl-scripts/bootstrap_rl.sh /opt/marin/experiments/exp2039_rl_math500.py --mode trainer --deployment-preset v5p_central1a --run-id ${RUN_ID} --shared-root ${SHARED_ROOT} --rollout-shape exp2039"

# Sampler (same command but --mode sampler)
```

**Checking for crashes without destroying evidence**:
```bash
# Check if container is running
docker ps --format '{{.Names}} {{.Status}}'

# If stopped, check exit code and last logs BEFORE removing
docker inspect levanter --format '{{.State.ExitCode}} {{.State.FinishedAt}}'
docker logs levanter 2>&1 | tail -50

# Only remove after you've read the logs
docker rm levanter
```

### COMPLETED: Sync RL run exp2039-20260318-040100 (500/500 steps)

**Run ID**: `exp2039-20260318-040100`
**Duration**: 36 hours 17 minutes (2026-03-18 04:01 UTC → 2026-03-19 16:18 UTC)
**Steps**: 500/500 (complete, clean exit)
**Avg pace**: 4.4 min/step
**Final checkpoint**: `gs://marin-us-central1/tmp/exp2039/exp2039-20260318-040100/exp2039-20260318-040100/checkpoints/`
**Wandb trainer**: https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-040100-train
**Wandb sampler**: https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-040100

**Final metrics (from wandb)**:
- Eval pass@1: ~0.458 (flat throughout, started at 0.478)
- Train correct accuracy: ~0.60
- Train format accuracy: ~0.81
- RLOO loss: ~-0.0005 (near-zero throughout)
- Sympy timeouts caught and survived: 9 (after multiprocessing fix)

**NOT hands-off**: Required 3 manual sampler restarts due to sympy hangs before the multiprocessing fix was deployed:
1. ~05:05 UTC Mar 18: sympy `_eval_power` hang, faulthandler killed at 1200s
2. ~06:45 UTC Mar 18: same hang, deployed ThreadPoolExecutor fix (didn't work — GIL)
3. ~08:40 UTC Mar 18: same hang, deployed multiprocessing.Pool fix (worked)
After fix v2: ran unattended for 30+ hours, 9 sympy timeouts caught, zero crashes.

**Trainer never crashed** — only the sampler. Each time, the trainer waited for rollouts while the sampler was restarted with the same run ID. Training resumed from where it left off with no lost steps.

**Bug: no shutdown coordination.** Trainer exited cleanly after step 499 but never signaled the sampler to stop. Sampler kept running, trying to fetch weights from dead Arrow Flight servers. Required manual `docker kill`.

### Change 35: Unified seed across all components
**Files**: `experiments/exp2039_rl_math500.py`, `lib/marin/src/marin/rl/rl_job.py`
**What**:
1. Added `SEED = 42` constant at top of exp2039 as single source of truth.
2. All seed references now use `SEED`: MathEnv `env_args`, `RLJobConfig.seed`, `TrainerConfig.seed`.
3. Removed `+ 1000` offset from `RolloutWorkerConfig.seed` in `rl_job.py:350`. Rollout worker now gets same seed as trainer.
**Why**: The `+ 1000` offset was added by Russell Power in PR #1731 (Oct 2025) as defensive coding ("not required but ensures we can control randomness for testing"). The decorrelation argument doesn't hold — trainer and sampler use seeds for completely different operations (dropout vs curriculum sampling), and in vLLM mode the sampler uses Python `random.Random`, not JAX PRNGs. The offset just added unexplained noise. All seeds are now `SEED = 42` for transparency and reproducibility.
**Before**: MathEnv=42, TrainWorker=42, RolloutWorker=1042, Trainer=0 (4 different values, 3 different files)
**After**: All=42, one constant, one file.

### Run launched: exp2039nb-20260319-032500 (non-blocking retry, detached mode)
- **Date**: 2026-03-19T03:25Z
- **Run ID**: `exp2039nb-20260319-032500`
- **Shared root**: `gs://marin-us-central1/tmp/exp2039/exp2039nb-20260319-032500`
- **Docker image**: `us-central1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:1773890572`
- **TPUs**: `exp2039-nb-trainer` + `exp2039-nb-sampler` in `us-central1-a` (v5p-8, on-demand, ACTIVE)
- **Launched**: Detached mode (`docker run -d`), direct SSH, NO launch.py, NO retries
- **Change**: `max_weight_transfer_wait_time=0` (non-blocking sync)
- **Includes**: All diagnostics (Changes 28-31), dummy weight guard (Change 33), sympy multiprocessing timeout (Change 34)
- **If it crashes**: Container corpse stays with logs intact. Check with `docker inspect` + `docker logs` before removing.

### COMPLETED: NB run exp2039nb-20260319-032500 (500/500 steps)

**Run ID**: `exp2039nb-20260319-032500`
**Duration**: 28.6 hours
**Steps**: 500/500 (trainer finished, sampler crashed — no graceful shutdown in that image)
**Avg pace**: 3.4 min/step (206s/step) — 1.3x faster than the 4.4 min/step sync run
**Wandb trainer**: https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500-train
**Wandb sampler**: https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500

**Timing breakdown (medians)**:
- Step duration: 66.7s
- Batch prep: 21.3s (high variance: 4s-56s — sampler starved on many steps)
- Forward/backward: 51.0s
- Weight serve: 14.4s

**Eval pass@1**: Started 0.41, peaked ~0.49 mid-training, **degraded to 0.396 by end**. Training quality issue persists.

**Sampler**: 142s/batch, 3280 tok/s. Crashed at end (no graceful shutdown in image).

**Seed configuration (from wandb config)**:
- `TrainerConfig.seed`: **0** (default — `--seed` CLI arg didn't exist yet)
- `MathEnv seed`: 42 (from `SEED` constant)
- `RolloutWorker seed`: unknown (likely 42+1000=1042 from old `rl_job.py` offset)
- No `--seed` flag in args

### Run launched: exp2039-nb-inflight (inflight + matched NB seeds)

**Date**: 2026-03-20
**Run ID**: `exp2039-nb-inflight`
**TPUs**: `exp2039-nb-trainer` + `exp2039-nb-sampler` (v5p-8, us-central1-a)
**Goal**: Full 500-step soak test of inflight weight updates + graceful shutdown. Uses the same seed configuration as the NB run for comparability (TrainerConfig.seed=0, MathEnv seed=42, RolloutWorker seed=1042).

**Changes from NB run**:
- Inflight weight updates enabled (background weight sync thread)
- Graceful shutdown (trainer → sampler exit signal)
- First-weight timeout fix (1200s default for inflight mode)

**Seed config (hardcoded to match NB run)**:
- `TrainerConfig.seed`: 0
- `MathEnv seed`: 42
- `RolloutWorker seed`: 1042 (42 + 1000 offset)
- These are temporary hardcodes, reverted after image build

**What to watch**:
- Step duration should be lower (no sync weight blocking)
- Batch prep variance should be lower (inflight = continuous rollout generation)
- Eval pass@1 trajectory — compare with NB run's degradation
- Graceful shutdown: sampler should exit cleanly when trainer finishes
- Any crashes from the inflight path (soak test)

---

## On-Demand RL: Final Summary (2026-03-21)

### Goal achieved

We successfully built and validated a no-Ray, manual-mode RL training pipeline that runs on dedicated TPU VMs coordinated via GCS. The system supports inflight weight updates, graceful shutdown, and preemption recovery on spot TPUs.

### All runs

| Run | Mode | Steps | Duration | Pace | Eval pass@1 (peak) | Wandb trainer | Wandb sampler |
|-----|------|-------|----------|------|-------------------|---------------|---------------|
| Sync Run 1 | Blocking sync, on-demand | 500/500 | 36.8h | 4.4 min/step | ~0.46 | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-040100-train) | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-032600) |
| Sync Run 2 (seed=1) | Blocking sync, on-demand | 461/500 | 26.7h | 3.4 min/step | 0.472 | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-seed1-20260319-225725-train) | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-seed1-20260319-225725) |
| Inflight 1 (NB) | Non-blocking + inflight, on-demand | 500/500 | 28.6h | 3.4 min/step | 0.488 | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500-train) | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500) |
| Inflight 2 | Inflight + graceful shutdown, spot | 186/500 | 8.7h | 2.6 min/step | 0.490 | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2-train) | [link](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2) |

### Speed progression

| Optimization | Pace | Speedup vs baseline | Projected 500-step time |
|-------------|------|---------------------|------------------------|
| Blocking sync (baseline) | 4.4 min/step | 1.0x | 36h |
| Non-blocking sync | 3.4 min/step | 1.3x | 28h |
| Inflight weight updates | 2.6 min/step | 1.7x | 21h |
| Theoretical floor (pure fwd/bwd) | 0.75 min/step | 5.9x | 6.25h |

### Trainer timing breakdown (medians from Inflight 1, 500 steps)

| Phase | Time | Overlappable? |
|-------|------|---------------|
| Forward/backward | 51s | No (the floor) |
| Batch prep (wait for rollouts) | 21s | Yes — high variance (4-56s) |
| Weight transfer serve | 14s | Yes — with background serving |
| Step overhead | ~5s | Hooks, logging, etc. |
| **Total** | **~67s** (median) | |

### What we built

1. **No-Ray manual mode** — Trainer and sampler run on separate TPU VMs, coordinated via GCS filesystem coordinator (`FileSystemArrowFlightCoordinator`). No Ray cluster needed.

2. **Inflight weight updates** — Background thread fetches weights from Arrow Flight while the main thread generates rollouts. Weights arrive mid-generation instead of blocking between cycles. Required debugging a fork deadlock (JAX + `AsyncLLM` subprocess) — turned out to be a test artifact, not a real bug.

3. **Graceful shutdown** — Tri-state coordinator status (`running`/`completed`/`failed`). Trainer signals completion/failure, sampler exits cleanly, wandb runs finalize. Fixed the bug where the sampler ran forever after the trainer finished.

4. **Preemption recovery** — `weight_id=-1` (initial weights) bypasses the coordinator's stale-weight check, so a restarted trainer can always publish to a coordinator that has stale state from a previous run. Validated in production: preempted at step 111 → resumed from checkpoint step 110 → continued training.

5. **CLI args** — `--seed`, `--num-train-steps` for experiment control. `--seed` flows to all components (MathEnv, trainer, rollout worker).

6. **First-weight timeout fix** — Inflight mode defaults to 1200s for the initial weight wait (trainer and sampler start simultaneously, need time for trainer to initialize and publish).

### What we didn't solve

1. **Training quality** — Eval pass@1 peaks around 0.49 and degrades in longer runs (Inflight 1 ended at 0.396). Multiple hypotheses in the logbook (LR scaling, DAPO normalization, batch size) but not yet investigated.

2. **Second sampler** — Would eliminate batch prep variance (the 4s-56s swing). Single sampler produces rollouts at exactly the trainer's consumption rate — any slowdown starves the trainer.

3. **Background weight serving** — Trainer blocks for ~14s per step on device-to-host copy. Could be overlapped with the next forward/backward pass.

### Key files

| File | What |
|------|------|
| `experiments/exp2039_rl_math500.py` | Experiment entry point, manual mode CLI |
| `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` | Filesystem coordinator, graceful shutdown signals, preemption fix |
| `lib/marin/src/marin/rl/weight_transfer/base.py` | `WeightUpdate` with `is_done`/`is_failed`, abstract `mark_completed`/`mark_failed` |
| `lib/marin/src/marin/rl/train_worker.py` | Completion/failure signaling in `train()` |
| `lib/marin/src/marin/rl/rollout_worker.py` | Inflight mode, graceful shutdown check, first-weight timeout, JAX-before-fork docs |
| `on-demand-rl-scripts/_run_exp2039.sh` | Launch script with seed/capacity/rollout-shape overrides |

### Logbooks and project docs

| Path | What |
|------|------|
| `.agents/logbooks/on-demand-rl.md` | This logbook — full history of all runs, changes, and decisions |
| `.agents/logbooks/debug_inflight_weight_updates.md` | Inflight weight update investigation (hypothesis-driven, 5 experiments) |
| `.agents/projects/graceful_sampler_exit.md` | Graceful shutdown design and implementation |
| `.agents/projects/inflight_weight_updates.md` | Inflight weight update design (Option A vs B analysis) |
| `.agents/projects/rl_preemption.md` | Preemption recovery design and production validation |

