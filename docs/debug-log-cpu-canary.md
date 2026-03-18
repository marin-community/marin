# Debugging log for CPU canary smoke test

Goal: get `uv run scripts/canary/coreweave_multihost.py cpu` to successfully boot a CPU-only
cluster and run `train_tiny_model_cpu.py` on CoreWeave.

## Initial status

- Controller boots in `iris-canary-cpu` namespace but dashboard shows autoscaler tabs
- Config used `worker_provider: {}` — wrong provider type for K8s pod scheduling

## Fix 1: wrong provider type in CPU config

Replaced `worker_provider: {}` with `kubernetes_provider:` and `runtime: kubernetes`.
Also fixed `_validate_worker_defaults` to accept `"kubernetes"` runtime.

## Fix 2: missing controller_address for bundle fetch

Init container needs `controller_address` to download workspace bundle via HTTP.
Added explicit `controller_address` to both CPU and multihost yaml configs.

## Fix 3: missing S3/R2 credentials in task pod

Fixed `_build_secret_env()` to map R2 creds → `AWS_*` env vars and set `FSSPEC_S3`.

## Fix 4: nodeSelector — task pods landing on wrong nodes

Task pods had no nodeSelector, landing on any cluster node. Fixed `KubernetesProvider`
to derive `managed_label` from `platform.label_prefix` and inject it into all pod specs.

**Verified**: new pods have `nodeSelector: {"iris-iris-canary-cpu-managed": "true"}`.

## Fix 5: excessive task retries

Pod application errors (exit code != 0) were misclassified as `WORKER_FAILED` (preemption),
allowing 100 retries. Fixed to distinguish infrastructure failures (OOMKilled, Evicted) from
application errors. App errors now use `TASK_STATE_FAILED` (max_retries_failure=0).

## Fix 6: JAX coordinator_address error on K8s

JAX 0.8+ auto-detects K8s and requires coordinator_address. Fixed `initialize_jax()` to
skip `jax.distributed.initialize()` entirely for single-task jobs.

**Note**: multi-task jobs on K8s may have the same issue — needs investigation.

## Current status (run 5)

- Controller restarted with all fixes
- Job `/power/cpu-canary-cpu-canary-20260318-165716` is RUNNING
- Training sub-job (`tra-003964c8`) is RUNNING on `g505de2` (correct node)
- NodeSelector verified active on all new pods
- Waiting for training to complete

## Changes made

| File | Change |
|------|--------|
| `lib/iris/src/iris/cluster/config.py` | Accept `runtime: kubernetes`; pass `managed_label` to KubernetesProvider |
| `lib/iris/src/iris/cluster/controller/kubernetes_provider.py` | Add `managed_label` field + nodeSelector; fix pod failure classification |
| `lib/iris/src/iris/runtime/jax_init.py` | Skip `jax.distributed.initialize()` for single-task jobs |
| `lib/iris/examples/coreweave-cpu.yaml` | `kubernetes_provider` + `controller_address` + `runtime: kubernetes` |
| `lib/iris/examples/coreweave-canary-multihost.yaml` | Added `controller_address` |
| `scripts/canary/coreweave_multihost.py` | 2-phase runner + monitoring + R2→AWS env mapping |
| `lib/iris/tests/cluster/controller/test_kubernetes_provider.py` | Tests for nodeSelector + failure classification |
| `lib/iris/tests/test_jax_init.py` | Updated single-task test expectations |

## Future Work

- [ ] Fix JAX init for multi-task jobs on K8s (same coordinator_address issue)
- [ ] KubernetesProvider should inject S3 creds from K8s secret automatically
- [ ] Confirm full CPU canary completes end-to-end
