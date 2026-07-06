# Research: TPU fixed-port (8431) collision

Investigation for weaver #401 (2026-07-06). Live inspection of the marin controller and
the running reserved-2048 slice, plus web research on libtpu internals.

## Incident

`/larry/iris-run-job-20260706-144044/grug-train-moe_67b_…_v4_2048_muon_resume15k_v2_10T`
(256 tasks, one per worker of the single reserved slice
`marin-tpu-v4-reserved-2048-us-central2-b-…`) crash-looped on restart with:

```
add_port.cc:83 Failed to add port to server: No address added out of total 1 resolved
for '[::]:8431' (… '[::]:8431': Address already in use; '0.0.0.0:8431': Address already
in use)
```

Each Iris task had exactly one attempt (KILLED after ~2 min, job tripped
`max_task_failures`); the retry loop is inside the container, not at the Iris level. The
babysitter resubmitted repeatedly (05:05 → 05:27 → 06:08 → 14:40, all FAILED) until
14:54 succeeded. At the time, 27 non-larry tasks (~23 `/michaelryan/*`, 4 `/runner/*`
zephyr) were bin-packed onto workers 40, 43, 183, 186, 244 — still present during this
investigation.

## What binds 8431 (web research)

- 8431 is libtpu's **Runtime Metric Service** (`runtime_metric_service.cc:122`:
  "Successfully started Runtime Metric Service on port: 8431"), a telemetry gRPC server
  started whenever the TPU PJRT plugin (`libtpu.so`) initializes. `add_port.cc` is
  gRPC's. Confirmed to fire for single-host, single-process init (jax-ml/jax#30834), so
  it is unrelated to multi-host coordination.
- It is **not** JAX's distributed coordinator (8482, `cloud_tpu_cluster.py:29`), the
  SliceBuilder worker ports, or `TPU_MESH_CONTROLLER`/`TPU_PROCESS_PORT`.
- **Redirectable:** `LIBTPU_INIT_ARGS="--runtime_metric_service_port=<port>"` (default
  8431). `matomatical/tpus` pins it per-chip (`8431 + chip_id`) with a regression test,
  precisely to stop concurrent processes on one VM colliding on it. No
  `TPU_LIBRARY_INIT_ARGS` env var exists (that name conflates `LIBTPU_INIT_ARGS` with
  `TPU_LIBRARY_PATH`). `TPU_VISIBLE_CHIPS/DEVICES`, `TPU_HOST_BOUNDS` etc. control chip
  topology/visibility, not this port.
- **`JAX_PLATFORMS=cpu` avoids it entirely** — traced through JAX's `xla_bridge.py`: the
  `tpu` PJRT plugin factory is registered but only invoked when `tpu` is in the resolved
  platform list; with `JAX_PLATFORMS=cpu`, `libtpu.so` is never `dlopen`'d. Restricting
  visible chips alone does **not** avoid the bind.

## Live evidence (via `iris task exec`, read-only)

Probes were deliberately non-destructive — no `jax.devices()` (that would itself bind
8431 and could crash larry).

**Co-tenants cannot bind 8431.** On `/michaelryan/*` and `/runner/*` tasks on the larry
workers: no `PJRT_DEVICE`/`JAX_PLATFORMS=tpu` env; **no `/dev/vfio` or `/dev/accel*`**
(non-privileged, `CapEff=0x80000` = CAP_SYS_PTRACE only); and **no `libtpu` in the venv**
(only `jax`/`jaxlib` CPU). Without libtpu or device access they cannot initialize the TPU
plugin, so the "CPU co-tenant grabs 8431 via libtpu" hypothesis is **refuted** for the
observed co-tenants. (`env.py:193-199` sets the TPU env only for tasks whose own resource
spec has a TPU device, so CPU-only co-tenants get none; `docker.py:212` gates
`--privileged`/device access on `is_tpu_run`.)

**larry holds 8431.** On the larry task on worker-40: `PJRT_DEVICE=TPU`,
`JAX_FORCE_TPU_INIT=1`, `/dev/vfio` + `accel0-3` present, `CapEff=0x1ffffffffff`
(privileged). `/proc/net/tcp6` shows `[::]:20EF` (20EF = 8431) in state `0A` = LISTEN,
held for the whole run.

**The enabling misconfiguration.** On the shared host:
`/proc/sys/net/ipv4/ip_local_port_range = 1024 65535` (not the default `32768 60999`) and
`/proc/sys/net/ipv4/ip_local_reserved_ports` is **empty**. So 8431 (and larry's other
live TPU LISTEN ports — enumerated as 8431, 8470, 8471 via `/proc/net/tcp*`) sit inside
the random ephemeral pool. Source:
`worker_bootstrap.py:191` / `docker.py:150` set `ip_local_port_range="1024 65535"` (#3066)
with no reservation.

## Conclusion

The observed crash-loop was not co-tenant libtpu theft. 8431 falls inside Iris' widened
ephemeral range with nothing reserving it, so any co-tenant outbound connection can be
assigned local port 8431 and transiently block larry's libtpu from binding it on restart.
Fix: reserve the fixed TPU/JAX service ports via `ip_local_reserved_ports` where the
range is widened.

## Key code references

- `lib/iris/src/iris/cluster/platforms/gcp/worker_bootstrap.py:188-193` — host sysctl tuning (fix site 1)
- `lib/iris/src/iris/cluster/runtime/docker.py:147-152` — `_NETWORK_SYSCTLS` (fix site 2)
- `lib/iris/src/iris/cluster/runtime/env.py:193-199` — TPU env set only for TPU-granted tasks
- `lib/iris/src/iris/cluster/runtime/docker.py:205-216` — `--privileged`/device access gated on `is_tpu_run`
- `lib/iris/src/iris/cluster/worker/task_attempt.py:160` — TPU tasks run `--network=host`
- `lib/iris/src/iris/cluster/worker/port_allocator.py:16` — Iris allocator range `(30000, 40000)`
- `lib/iris/src/iris/runtime/jax_init.py:225-281` — TPU path uses bare `jax.distributed.initialize()`; honors `IRIS_PORT_jax` only off-TPU
