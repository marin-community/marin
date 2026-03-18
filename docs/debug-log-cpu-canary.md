# Debugging log for CoreWeave multihost canary

## Phase 1: CPU canary (DONE)

Goal: get `uv run scripts/canary/coreweave_multihost.py cpu` to successfully boot a CPU-only
cluster and run `train_tiny_model_cpu.py` on CoreWeave.

### Fix 1: wrong provider type in CPU config

Replaced `worker_provider: {}` with `kubernetes_provider:` and `runtime: kubernetes`.
Also fixed `_validate_worker_defaults` to accept `"kubernetes"` runtime.

### Fix 2: missing controller_address for bundle fetch

Init container needs `controller_address` to download workspace bundle via HTTP.
Added explicit `controller_address` to both CPU and multihost yaml configs.

### Fix 3: missing S3/R2 credentials in task pod

Fixed `_build_secret_env()` to map R2 creds → `AWS_*` env vars and set `FSSPEC_S3`.

### Fix 4: nodeSelector — task pods landing on wrong nodes

Task pods had no nodeSelector, landing on any cluster node. Fixed `KubernetesProvider`
to derive `managed_label` from `platform.label_prefix` and inject it into all pod specs.

### Fix 5: excessive task retries

Pod application errors (exit code != 0) were misclassified as `WORKER_FAILED` (preemption),
allowing 100 retries. Fixed to distinguish infrastructure failures (OOMKilled, Evicted) from
application errors. App errors now use `TASK_STATE_FAILED` (max_retries_failure=0).

### Fix 6: JAX coordinator_address error on K8s

JAX 0.8+ auto-detects K8s and requires coordinator_address. Fixed `initialize_jax()` to
skip `jax.distributed.initialize()` entirely for single-task jobs.

### CPU result: PASSED

---

## Phase 2: GPU multihost canary (IN PROGRESS)

Goal: get `uv run scripts/canary/coreweave_multihost.py gpu` to boot a 2-node H100
cluster and run Grug MoE training across 16 GPUs (2 VMs × 8 H100 each).

### Architecture

The GPU canary submits a job to Iris running `executor_main(canary_ferry)`.
The canary_ferry step calls `run_grug_moe` which is a plain (non-`@remote`) function,
so `executor_main` runs it **in-process** inside the Iris task pod.

For multi-host, the `GrugMoeLaunchConfig` specifies:
```python
resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", replicas=2)
```

The `replicas=2` triggers Iris to create 2 task pods. Each task runs the training code.
JAX distributed init (`jax_init.py`) coordinates across tasks via endpoint registry.

### Key config: `coreweave-canary-multihost.yaml`

- `host_network: true` — required for NCCL/IB multi-host traffic
- `h100-16x` scale group: 2 VMs, 8 H100 each, IB-connected (`gd-8xh100ib-i128`)
- `backend.coreweave.cloud/superpod: same-slice` — pod affinity on same spine switch
- `controller_address` set to in-cluster service URL

### Fix 7: R2 credentials expired / controller CrashLoopBackOff

Controller pod crash-looped on startup: `PermissionError: Forbidden` when
`download_checkpoint_to_local` called `fs.exists()` on the S3 state path.

Root cause: R2 API keys (`94763782...`) in the environment were expired/revoked.
Confirmed locally — every s3fs operation (ls, HEAD) returned 403.

Fix: load fresh credentials from `~/.env` at script startup via `os.environ.setdefault`.

### Fix 8: UV cache on ramdisk — No space left on device

GPU task pods failed with `No space left on device` installing torch+CUDA (~15GB).
The hostPath `/cache` mapped to `/dev/ram0` (15GB ramdisk) on CoreWeave GPU nodes.
Meanwhile the NVMe RAID at `/dev/md127` has 28TB.

Fix: set `cache_dir: /mnt/local/iris-cache` in `kubernetes_provider` config so the
hostPath lands on the multi-TB NVMe at `/mnt/local` instead of the 15GB ramdisk.

### Fix 9: ExponentialBackoff crash in JAX coordinator polling

Task 1 (non-coordinator) crashed immediately:
```
ExponentialBackoff(initial=poll_interval)  # poll_interval=2.0, maximum defaults to 1.0
ValueError: maximum must be >= initial
```

Fix: pass `maximum=max(poll_interval, 30.0)` to ExponentialBackoff in `_poll_for_coordinator`.

### Fix 10: NCCL hang — missing RDMA/IB device resources

Multi-host training stuck at `broadcast_one_to_all` for 25+ min. GPUs had
61GB VRAM allocated but 0% utilization. Root cause: pods didn't request
`rdma/ib` resources, so no InfiniBand devices were exposed (`/dev/infiniband/`
missing). NCCL fell back to TCP and hung.

Node has `rdma/ib: 64`. Fix: request `rdma/ib: <gpu_count>` in pod resource
limits when `host_network=true` and GPUs are present.

### Fix 11: cluster-specific managed_label missing from pod/configmap labels

Pods only had generic `iris.managed=true` label, not the cluster-specific
`iris-iris-canary-mh-managed=true`. This meant warm_reboot couldn't clean up
old pods (label selector didn't match), and different clusters in the same
namespace could interfere.

Fix: add `managed_label` to pod and configmap labels in `_build_pod_manifest`
and `_apply_pod`.

### Debugging: NCCL hang at broadcast_one_to_all

RDMA/IB devices now visible in container (`/dev/infiniband/uverbs0-8`), but
`broadcast_one_to_all` still hangs (0% GPU util, 61GB VRAM allocated, 25+ min).
Switching to tiny model (`llama_nano`, 10 steps) to isolate, and adding:
- `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`
- `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=5`
- `JAX_LOG_COMPILES=1`

### Fix 12: NCCL falling back to Socket over IB link-local — missing libibverbs

NCCL debug revealed: `Failed to open libibverbs.so[.1]` → falls back to Socket
transport. Socket uses IB interfaces (`ibs0-7`) with link-local IPv6 (`fe80::`)
which don't route cross-host → hang.

Root cause: task image (`python:3.12-slim`) doesn't include `libibverbs1`.

Fix (both):
1. Add `libibverbs1 ibverbs-providers` to task image Dockerfile
2. Set `NCCL_SOCKET_IFNAME=enp157s0np0` as fallback (routable Ethernet)

### Result: tiny model PASSED on 2x H100 nodes

Tiny model (`llama_nano`, 10 steps) completed in 2m30s on gd927de + gd94886.
Both tasks logged matching loss (`eval/loss 11.98489`). NCCL used Socket over
Ethernet (`enp157s0np0`) — not ideal but functional. W&B run:
https://wandb.ai/marin-community/marin/runs/mh-canary-20260318-195805

Next: switch back to Grug MoE canary for full validation.

### Current status

- CPU canary PASSED
- Tiny multi-host GPU canary PASSED (2x H100, llama_nano, 10 steps)
- Switching to Grug MoE canary for full validation

### Known risks

- [ ] JAX multi-task init on K8s (coordinator_address issue from Fix 6 may apply)
- [ ] Pod affinity/colocation for IB connectivity
- [ ] NCCL environment variables may need explicit configuration
- [ ] executor_main runs `run_grug_moe` in-process — need to verify Fray/Iris
      correctly handles `replicas=2` from ResourceConfig

---

## Changes made (cumulative)

| File | Change |
|------|--------|
| `lib/iris/src/iris/cluster/config.py` | Accept `runtime: kubernetes`; pass `managed_label` to KubernetesProvider |
| `lib/iris/src/iris/cluster/controller/kubernetes_provider.py` | Add `managed_label` field + nodeSelector; fix pod failure classification |
| `lib/iris/src/iris/runtime/jax_init.py` | Skip `jax.distributed.initialize()` for single-task jobs |
| `lib/iris/examples/coreweave-cpu.yaml` | `kubernetes_provider` + `controller_address` + `runtime: kubernetes` |
| `lib/iris/examples/coreweave-canary-multihost.yaml` | Added `controller_address` + `host_network` + h100-16x scale group |
| `scripts/canary/coreweave_multihost.py` | 2-phase runner + monitoring + R2→AWS env mapping |
| `lib/iris/tests/cluster/controller/test_kubernetes_provider.py` | Tests for nodeSelector + failure classification |
| `lib/iris/tests/test_jax_init.py` | Updated single-task test expectations |

## Future Work

- [ ] KubernetesProvider should inject S3 creds from K8s secret automatically
