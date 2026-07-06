# Keep the cluster's fixed ports out of Iris' widened ephemeral range

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

Two layers, both derived from shared constants in
[`docker.py`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/runtime/docker.py)
so host and container settings never drift:

- **Raise the ephemeral floor (primary): `1024` → `11000`.** The lowest fixed port the
  cluster's own services bind is iris' controller/worker RPC on 10000/10001; the
  TPU/JAX runtime ports are all ≤ 8482. Putting the floor at 11000 moves the entire
  ephemeral pool above *every* fixed port we own, in one stroke — not just libtpu's, but
  iris' 10001 too, which the enumerated reservation alone would have missed. It still
  leaves ~54k ephemeral ports (vs ~64k), preserving the high-fan-out headroom the wide
  range was for. Applied at all three sites that widen the range: `worker_bootstrap.py`,
  `controller_bootstrap.py`, and `docker.py`'s `_NETWORK_SYSCTLS`.
- **Reserve the TPU/JAX block (defense-in-depth): `RESERVED_HOST_PORTS =
  "8081,8431,8470-8482"`** in `worker_bootstrap.py` (host, for `--network=host` TPU
  tasks) and `docker.py` (private-netns containers). Redundant given the floor, but it
  pins the specific ports so a future floor change can't silently re-expose them.
  `ip_local_reserved_ports` only blocks *automatic* assignment (`bind(0)`, `connect()`
  source ports); an explicit `bind(8431)` by the TPU runtime is unaffected.

The `3201/3202/9230/11755` LISTENers also seen on the TPU VMs are the GCP image's own
monitoring agents, not iris services, so they are out of scope (and 11755 sits above the
floor regardless).

## Testing

Unit tests assert the invariant, not the literal string: for every fixed service port
the cluster binds (`{8081, 8431, 8470, 8471, 8476, 8482, 10000, 10001}`),
`test_ephemeral_floor_sits_above_fixed_service_ports` checks it is either below the floor
or reserved; a second test pins the TPU block in the reservation; and a bootstrap-render
test checks the emitted worker script sets both the raised floor and the reservation
(`test_docker_runtime.py`, `test_bootstrap.py`). Rollout: the code change only affects
VMs created after it lands; the **live reserved slice needs a one-time `sysctl -w
net.ipv4.ip_local_port_range="11000 65535"` (and, belt-and-suspenders,
`net.ipv4.ip_local_reserved_ports="8081,8431,8470-8482"`) pushed to its 256 workers** (or
a slice recreation) to be protected now. Both are pure sysctls and do not require
bouncing the controller or the run.

## Open Questions

- **Belt-and-suspenders:** should we *also* force `JAX_PLATFORMS=cpu` for non-accelerator
  tasks in `env.py` (candidate fix A)? It closes the libtpu-on-co-tenant vector by
  construction and is correct hygiene, but does not address this incident (the co-tenants
  never loaded libtpu). Cheap and independent — worth a follow-up, not this PR.
- **Floor height:** 11000 clears every fixed port we own (TPU ≤ 8482, iris 10000/10001)
  and the GCP monitoring agents seen ≤ 9230, leaving ~54k ephemeral ports. Is any
  generation (v5/v6e megascale, MXLA) or future iris service binding a fixed port between
  11000 and 65535 where the floor wouldn't help and an explicit reservation would be
  needed? (Iris' own PortAllocator range 30000-40000 is inside the pool but is
  bind-checked, a separate concern.)
- **Rollout ownership:** who pushes the one-time sysctls to the existing reserved-2048
  slice, and do we want an `iris cluster` helper to apply node sysctls without
  recreation?
