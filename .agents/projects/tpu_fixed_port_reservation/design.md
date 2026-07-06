# Reserve libtpu's fixed ports from Iris' widened ephemeral range

_Why are we doing this? What's the benefit?_

The production `/larry/…/grug-train-moe_67b … v4_2048` run repeatedly crash-looped on
restart with `add_port.cc:83 Failed to add port to server: … '[::]:8431': Address
already in use`, while ~27 non-larry CPU tasks were bin-packed onto 5 of its 256
reserved-slice worker VMs. Port 8431 is bound by **libtpu's Runtime Metric Service**
whenever the TPU PJRT plugin initializes. This design keeps that port (and the rest of
the fixed TPU/JAX service ports) out of the kernel's ephemeral-allocation pool so a
co-tenant can never be handed it, which is what makes the TPU trainer's bind fail.

## Background

libtpu, not JAX, owns 8431 (JAX's own coordinator is 8482, `cloud_tpu_cluster.py:29`;
none of marin's Python defaults — 8476/8482/8081 — is 8431). Live inspection of the
running slice established the mechanism; the full evidence trail is in
[`research.md`](./research.md). The key facts: larry holds `[::]:8431` in `LISTEN` for
the whole run; the co-tenants have **no libtpu installed and no TPU device access**, so
they cannot bind 8431 via libtpu at all; and Iris widens the host ephemeral range to
`1024 65535` (`worker_bootstrap.py:191`) with an **empty** `ip_local_reserved_ports`.

## Root cause

Iris tunes every worker VM for high connection counts
([`worker_bootstrap.py:188-193`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/platforms/gcp/worker_bootstrap.py#L188)),
lowering `net.ipv4.ip_local_port_range` to `1024 65535`. The default Linux range is
`32768 60999`, which *excludes* 8431 — so on a stock host a random ephemeral bind can
never land on it. With the widened range and no reservations, **8431 is inside the
random-allocation pool.** Any co-tenant's outbound connection (GCS, HF, W&B, controller
RPC) can be assigned local port 8431; while that socket is open, larry's libtpu cannot
bind `[::]:8431` and the trainer crash-loops. This needs no libtpu or TPU device on the
co-tenant — only network activity — which is exactly why the co-tenants we probed (which
have neither) still cause it, and why more co-tenants raise the collision odds. TPU task
containers run `--network=host`
([`task_attempt.py:160`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/worker/task_attempt.py#L160)),
so they share this one host netns and its sysctls.

## Answering the four questions

1. **Can grug/MoE reserve the coordinator port so it doesn't depend on a fixed port?**
   Not for 8431. Iris port reservation (`port_allocator.py`, range `(30000,40000)`) is
   worker-local and only redirects the *JAX coordinator* on the non-TPU path
   (`jax_init.py:203,306`). 8431 is libtpu-internal; JAX ignores Iris ports for it. The
   right lever is not per-job reservation but a host-level OS reservation of the fixed
   ports.
2. **Where does 8431 originate; is it redirectable?** libtpu's Runtime Metric Service
   (`runtime_metric_service.cc:122` — "Successfully started Runtime Metric Service on
   port: 8431"), started at TPU-plugin init, confirmed even for single-host/single-proc.
   It *is* redirectable via `LIBTPU_INIT_ARGS="--runtime_metric_service_port=<port>"`,
   and `JAX_PLATFORMS=cpu` prevents the bind entirely (the plugin never `dlopen`s).
3. **If genuinely fixed, how to declare a host-exclusive port?** It is not genuinely
   fixed, so no scheduler resource is needed. The fixed ports are declared once at the OS
   level via `net.ipv4.ip_local_reserved_ports`, which removes them from automatic
   ephemeral assignment while leaving the TPU runtime's explicit `bind()` working.
4. **Thread a host-port requirement into the scheduler + preemption cleanly?** Not
   worth it. Ports are not a scheduler concept and don't need to become one: the
   collision is an OS ephemeral-allocation artifact, fixable in the same place the range
   is already tuned.

## Design

Add the fixed TPU/JAX service ports to `ip_local_reserved_ports` wherever Iris widens
the ephemeral range. A shared constant `RESERVED_HOST_PORTS = "8081,8431,8470-8482"`
([`docker.py`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/runtime/docker.py)),
covering libtpu's metric service (8431), the Cloud TPU runtime/SliceBuilder block
(8470-8482, which includes JAX's coordinator 8482 and marin's default 8476), and
levanter megascale (8081):

- **Host (primary):** `worker_bootstrap.py` emits
  `sysctl -w net.ipv4.ip_local_reserved_ports="{{ reserved_ports }}"`. This is the one
  that matters, since TPU tasks use `--network=host` and inherit host sysctls.
- **Container:** `docker.py`'s `_NETWORK_SYSCTLS` gains the same key, for the
  private-netns tasks that get per-container sysctls.

The controller VM is deliberately excluded — it runs no libtpu, so it has nothing to
protect. `ip_local_reserved_ports` only blocks *automatic* assignment (`bind(0)`,
`connect()` source ports); an explicit `bind(8431)` by the TPU runtime is unaffected.

## Testing

Unit tests assert the invariant, not the literal string: `_expand_reserved_ports` parses
the spec and checks that the confirmed offenders `{8431, 8470, 8471, 8476, 8482}` are
reserved and that every reserved port lies inside the widened range
(`test_docker_runtime.py`), plus that the rendered worker bootstrap script contains the
reservation (`test_bootstrap.py`). Rollout: the code change only affects VMs created
after it lands; the **live reserved slice needs a one-time `sysctl -w
net.ipv4.ip_local_reserved_ports="8081,8431,8470-8482"` pushed to its 256 workers** (or a
slice recreation) to be protected now. This is a pure sysctl and does not require
bouncing the controller or the run.

## Open Questions

- **Belt-and-suspenders:** should we *also* force `JAX_PLATFORMS=cpu` for non-accelerator
  tasks in `env.py` (candidate fix A)? It closes the libtpu-on-co-tenant vector by
  construction and is correct hygiene, but does not address this incident (the co-tenants
  never loaded libtpu). Cheap and independent — worth a follow-up, not this PR.
- **Port set completeness:** 8081/8431/8470-8482 covers every fixed port we observed
  larry bind plus the documented TPU/JAX block. Is any generation (v5/v6e megascale,
  MXLA) using a fixed port outside this set that should also be reserved?
- **Rollout ownership:** who pushes the one-time sysctl to the existing reserved-2048
  slice, and do we want an `iris cluster` helper to apply node sysctls without
  recreation?
