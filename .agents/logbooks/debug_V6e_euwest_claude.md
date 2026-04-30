# Debug Logbook: v6e-4 eu-west4 `AttributeError: x.coords` in vLLM TPU Mesh Init

**Goal**: Diagnose and fix why `vllm-tpu==0.13.2.post6` on v6e-4 in `europe-west4` crashes with `AttributeError` on `device.coords` during mesh initialization, while the identical config succeeds in `us-east1` and `us-east5`.

**Branch**: `alignment_function`
**Related logbook**: `.agents/logbooks/validate_bloom.md` (EXP-006)

---

## 2026-04-03: Initial Investigation

### Symptom

Job `/ahmed/bloom-eval-dpo-europe-west4-v6e4` failed during vLLM engine core init. The crash is deterministic and occurs before any model loading:

```
(EngineCore_DP0 pid=367) ERROR 04-03 22:05:32 [core.py:866] EngineCore failed to start.
  ...
  File "/app/.venv/lib/python3.11/site-packages/tpu_inference/worker/tpu_worker.py", line 246, in init_device
    self.model_runner = TPUModelRunner(self.vllm_config, self.devices, ...)
  File "/app/.venv/lib/python3.11/site-packages/tpu_inference/runner/tpu_runner.py", line 257, in __init__
    self._init_mesh()
  File "/app/.venv/lib/python3.11/site-packages/tpu_inference/runner/tpu_runner.py", line 302, in _init_mesh
    self.mesh = self._create_2d_mesh()
  File "/app/.venv/lib/python3.11/site-packages/tpu_inference/runner/tpu_runner.py", line 371, in _create_2d_mesh
    return make_optimized_mesh(mesh_shape, ..., devices=self.devices)
  File "/app/.venv/lib/python3.11/site-packages/tpu_inference/utils.py", line 205, in make_optimized_mesh
    devices = sorted(devices, key=lambda x: x.coords)
                                             ^^^^^^^^
AttributeError
```

### What the crashing code does

`make_optimized_mesh()` in `tpu_inference/utils.py:198-256`:
1. Takes a list of JAX devices
2. Sorts them by `.coords` (a `(x, y, z)` tuple on `TpuDevice`)
3. For v6e devices, applies a custom reordering for optimal ICI topology
4. Returns a `jax.sharding.Mesh`

The crash happens at step 2 — `.coords` doesn't exist on the device objects.

### Software versions (confirmed from logs)

| Component | Version | Source |
|-----------|---------|--------|
| vLLM | `0.13.2.post6` | `[api_server.py:1351] vLLM API server version 0.13.2.post6` |
| JAX | `0.8.0` | `JAX version 0.8.0 available.` |
| Python | `3.11.14` | `/uv/cache/python/cpython-3.11.14-linux-x86_64-gnu/` |
| tpu_inference | bundled with vllm-tpu | `/app/.venv/lib/python3.11/site-packages/tpu_inference/` |
| TPU | v6e-4 | `tpu_type=v6e-4 \| worker_id=0 \| num_chips=4 \| num_cores_per_chip=1` |
| TPU node | `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88` | |
| Task image | `ghcr.io/marin-community/iris-task:latest` | Iris default |

### Same config succeeds elsewhere

- **us-east1** v6e-4: SFT inference SUCCEEDED
- **us-east5** v6e-4: Two DPO checkpoints SUCCEEDED
- **eu-west4** v6e-4: FAILED with AttributeError on `.coords`

All three use the same pip packages (`vllm-tpu==0.13.2.post6`, `jax[tpu]==0.8.0`), same task image, same Iris controller.

### Root cause hypothesis

In JAX 0.8.0, the `Device` class is `jaxlib._jax.Device`. The `.coords` attribute is **not** a Python-level property — it's populated at runtime by the XLA/PJRT backend plugin (libtpu). Whether `.coords` exists depends on the libtpu version on the TPU VM host.

**Primary hypothesis**: The TPU VM host in eu-west4 has a different libtpu version than us-east1/us-east5. Google rolls out TPU software updates region by region. A newer or different libtpu may have:
- Removed the `.coords` attribute from the PJRT device API
- Changed it to a different name (e.g., `_coords`, `core_on_chip`)
- Failed to initialize it due to a bug

**Alternative hypotheses**:
1. Bad TPU node — hardware/firmware issue on that specific node causing incomplete device initialization
2. Container/pip environment issue — different libtpu wheel installed due to regional pip mirror differences
3. JAX runtime initialization failure — libtpu loaded but didn't fully initialize devices, leaving them without topology attributes

### Device sourcing

From `tpu_worker.py:205`, in the single-host, no-pipeline-parallel case:
```python
self.devices = jax.devices()[:sharding_config.total_devices]
```
These are the raw `jaxlib._jax.Device` objects returned by the JAX runtime. If the runtime didn't populate `.coords`, every call to it will fail.

---

## Investigation Plan

### Phase 1: Confirm the libtpu version hypothesis (non-destructive)

1. **Retrieve the libtpu version from a working region for comparison**
   - SSH into an Iris worker in us-east1 or us-east5 (via `iris task exec` on a running job, or a dev TPU)
   - Run: `python3 -c "import jax; d = jax.devices()[0]; print(type(d)); print(dir(d)); print(hasattr(d, 'coords')); print(d.coords if hasattr(d, 'coords') else 'NO COORDS')"`
   - Run: `python3 -c "import jaxlib; print(jaxlib.__version__)"` and `pip show libtpu-nightly 2>/dev/null || pip show libtpu 2>/dev/null`

2. **Repeat the same on eu-west4**
   - Submit a minimal diagnostic job to eu-west4 that prints device attributes and libtpu version, then exits
   - Script: print `jax.devices()`, `dir(jax.devices()[0])`, `jax.__version__`, `jaxlib.__version__`, libtpu version, and `device.coords` with try/except

3. **Compare device attributes between regions**
   - If `.coords` is present in us-east but absent in eu-west4 → confirms libtpu version drift
   - If `.coords` is absent in both → the successful runs may be using `_create_new_model_mesh()` path instead (controlled by `envs.NEW_MODEL_DESIGN`)

### Phase 2: Identify the fix

4. **If libtpu version drift confirmed**:
   - Pin the libtpu version in the pip dependency group to match the working regions
   - OR use the `_create_new_model_mesh()` path which calls `mesh_utils.create_device_mesh()` (JAX's own mesh utility that doesn't depend on `.coords`)
   - OR patch `make_optimized_mesh()` to fall back to `jax.make_mesh()` when `.coords` is missing

5. **If bad node**:
   - Retry on a fresh job ID to get a different TPU node
   - If it succeeds, the issue was transient

6. **If env issue**:
   - Pin libtpu wheel version explicitly in pip packages

### Phase 3: Verify

7. Re-run the `beta0.01_lr7.5e-7_seed0` inference in eu-west4 with the fix
8. Confirm prompt-collapsed adherence scores match other regions

---

## Next Action

Submit a minimal diagnostic job to eu-west4 v6e-4 that prints JAX device attributes and libtpu version, and simultaneously submit one to us-east1 for comparison.
