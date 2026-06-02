# fray-ray: extending the fakeray shim to Ray actors + placement groups

Status: **Draft / in progress.** Scopes the work to run actor-heavy Ray
workloads (e.g. [SkyRL](https://github.com/novasky-ai/skyrl)) on Iris/Fray.
Date: 2026-06-02. Builds on `20260531_fray_smallpond_fakeray_design.md`.

## TL;DR

- The existing `lib/fakeray` shim covers the **stateless-function** slice of Ray
  Core (`@ray.remote` def, `ObjectRef`, `get`/`wait`/`put`). That was enough for
  smallpond.
- Actor-heavy frameworks (SkyRL, OpenRLHF, verl) need three more things:
  **stateful `@ray.remote` classes**, **`ray.util.placement_group`**, and
  **named actors** (`ray.get_actor`/`ray.kill`). A fail-fast probe confirmed all
  three currently break (see Evidence).
- **Actors map cleanly onto Fray** — `Client.create_actor` already returns a
  handle whose `.method.remote()` yields a future. This is a real but bounded
  build.
- **Placement groups are the open design risk.** Ray's PG model (named bundles,
  `strategy=PACK/SPREAD`, fractional-GPU colocation, `PlacementGroupScheduling
  Strategy` capture) has no clean Iris equivalent. v1 implements a **best-effort,
  non-strict** shim and *documents the gap loudly* rather than faking colocation
  guarantees.
- **GPU collectives are out of scope** (NCCL weight broadcast policy→inference).
  The shim sets actors up; cross-actor GPU comm is the framework's job and needs
  Fray actors to exchange IPs — noted, not built.

## Evidence (fail-fast probe, 2026-06-02)

Real SkyRL idioms run under `fakeray.install()` (LocalClient, no torch/SkyRL
install — just the Ray call patterns):

| SkyRL idiom | result | root cause |
|---|---|---|
| `@ray.remote` def + `get` | **OK** | the smallpond surface |
| `@ray.remote` **class** + `.method.remote()` | FAIL | `'ObjectRef' has no attribute 'add'` — `remote()` treats a class like a function |
| `placement_group([...], strategy="PACK")` | FAIL | `No module named 'ray.util'` |
| `Actor.options(num_gpus=, placement_group=).remote()` | FAIL | same actor gap |
| `ray.get_actor(name)` | FAIL | `module 'fakeray' has no attribute 'get_actor'` |

SkyRL's real usage counts (grep over the repo): `placement_group` 88×,
`PlacementGroup` 67×, `@ray.remote` 11× (on classes: `RegistryActor`,
`InfoActor`, `RayJaxBackendImpl`), `ray.util.placement_group` 14×, `ActorPool`
6×, `ray.get_actor` 3×, `ray.kill` 4×. Pinned `ray==2.51.1`.

## What Ray gives that we must emulate

### 1. Stateful actors
```python
@ray.remote
class Worker:
    def __init__(self, cfg): ...
    def step(self, batch): ...

h = Worker.options(num_gpus=1).remote(cfg)   # construct -> ActorHandle
ray.get(h.step.remote(batch))                # method call -> ObjectRef
```
Ray: `@ray.remote` on a class returns an `ActorClass`; `.remote(*args)`
constructs the actor (a process) and returns an `ActorHandle`;
`handle.method.remote(*a)` enqueues a call and returns an `ObjectRef`;
`ray.get(ref)` awaits it.

### 2. Placement groups
```python
pg = placement_group([{"GPU":1,"CPU":1}] * n, strategy="PACK")
ray.get(pg.ready())
Actor.options(
    num_gpus=1,
    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i),
).remote(...)
```
Ray reserves *bundles* atomically (gang), `PACK` = same node where possible,
`SPREAD` = across nodes; fractional GPU (`num_gpus=0.2`) lets multiple actors
**colocate** on one device. Actors are then pinned to a specific bundle index.

### 3. Named actors
`ray.get_actor("name")` resolves a named/detached actor anywhere in the cluster;
`ray.kill(handle)` tears one down.

## How each maps onto Fray/Iris

### Actors → Fray actors (clean)
Fray's `Client` already has the matching shape:
- `create_actor(actor_class, *args, name=, resources=, actor_config=) -> ActorHandle`
- `handle.method.remote(*a) -> ActorFuture` (`.result()` blocks)
- hosted as an Iris job running `_host_actor` (or LocalActor in-process)

Shim plan:
- `remote()` inspects the target: **class → `ActorClass`**, function → existing
  `RemoteFunction` (unchanged).
- `ActorClass.remote(*args)` / `.options(num_gpus=, resources=, name=, ...)
  .remote(*args)` → `client.create_actor(cls, *args, resources=<mapped>,
  name=<auto or given>)` → wrap the Fray `ActorHandle` in a fakeray `ActorHandle`.
- fakeray `ActorHandle.__getattr__(method)` → `ActorMethodStub`; `.remote(*a)`
  calls Fray `handle.method.remote(*a)` and **adapts the `ActorFuture` into a
  fakeray `ObjectRef`** (wrap a `concurrent.futures.Future`, resolve it from a
  small waiter thread) so `ray.get`/`ray.wait` work uniformly with task refs.
- `.options(num_gpus=, num_cpus=, memory=, resources=, name=, ...)` → build a
  Fray `ResourceConfig` (`with_gpu`/`with_tpu`/cpu) + `ActorConfig`.
- `ray.get_actor(name)` → resolve via the client's actor registry / a
  fakeray-side `name -> handle` map. `ray.kill(handle)` → handle's job
  `terminate()`.

Registry: keep a process-global `{name: fakeray ActorHandle}` in the scheduler
so `get_actor` works within a driver; cross-process named lookup defers to Fray's
Iris resolver where available.

### Placement groups → best-effort, NON-strict (the risk)
There is no Iris primitive for "reserve N GPU bundles atomically with PACK/SPREAD
and pin actor i to bundle j." Options:

- **v1 (this work): capture, don't colocate.** `placement_group(bundles,
  strategy)` returns a lightweight `PlacementGroup` object that just *remembers*
  the bundle list + strategy. `pg.ready()` returns an already-resolved ref.
  `PlacementGroupSchedulingStrategy(placement_group=pg, bundle_index=i)` passed to
  `.options()` is unwrapped to the **i-th bundle's resources** (e.g. `{GPU:1}`)
  and turned into a per-actor `ResourceConfig`. Iris then schedules each actor
  independently with those resources.
  - **What you get:** the right *number and size* of GPU actors.
  - **What you DON'T get:** the gang/atomicity (actors may come up incrementally),
    PACK/SPREAD locality, and fractional-GPU colocation on one device. `region`
    can approximate PACK at coarse grain.
  - **Loudly logged**: `fakeray: placement_group strategy=PACK is advisory; Iris
    schedules bundles independently (no colocation/atomic-gang guarantee)`.
- **v2 (future, if needed):** map a whole PG → one Fray `create_actor_group` with
  `coscheduling` (gang) + region pinning, and hand out bundle indices as actor
  indices. Closer to PACK, still no fractional-GPU colocation. Bigger change;
  only do it if a workload actually depends on strict gang/locality.
- **Fractional GPU colocation** (`num_gpus=0.2`, 5 actors per device) likely does
  **not** work on Iris (device attach is whole-device). Flag as unsupported;
  workloads needing it must drop to 1 actor/GPU.

### GPU collectives → out of scope
SkyRL does NCCL weight broadcast policy→inference across actors. The shim only
provisions the actors; the framework builds the process group from actor IPs.
Fray actors do expose addresses, so it's *possible* the framework's own
rendezvous works unmodified — but verifying that is a separate milestone, not
part of the shim.

## Public surface added

```
fakeray.remote(cls_or_fn)            # now dispatches class -> ActorClass
fakeray.ActorClass.remote/.options   # construct actors
fakeray.ActorHandle.<method>.remote  # actor-method calls -> ObjectRef
fakeray.get_actor(name) / kill(h) / cancel(ref)
fakeray.util.placement_group(bundles, strategy=...) -> PlacementGroup
fakeray.util.scheduling_strategies.PlacementGroupSchedulingStrategy
fakeray.util.scheduling_strategies.NodeAffinitySchedulingStrategy   # accepted, best-effort
# installed as ray.util / ray.util.placement_group / ray.util.scheduling_strategies
```

`install()` must register the `ray.util.*` submodules in `sys.modules` (same
reason `ray.exceptions` is registered — dotted imports bypass `__getattr__`).

## Staging

1. **Actor model** (clean): class detection, `ActorClass`, `ActorHandle`,
   method→ObjectRef adapter, `.options()` resource mapping, `get_actor`/`kill`.
   Make probe checks 2, 4, 5 pass on LocalClient.
2. **`ray.util.placement_group` shim** (best-effort): `PlacementGroup`,
   `pg.ready()`, strategy capture, `PlacementGroupSchedulingStrategy` unwrap,
   `sys.modules` registration. Make probe check 3 pass.
3. **Regression test**: fold the probe into `lib/fakeray/tests`.
4. (Later) Iris-backend validation with a tiny 2-actor GPU job; PG v2 + collective
   verification only if a real SkyRL example demands it.

## Honest scope

- This makes the **API present and locally correct** (LocalClient) for SkyRL's
  Ray surface. It does **not** claim SkyRL trains correctly on Iris — placement
  locality, fractional-GPU colocation, and NCCL rendezvous are explicitly
  best-effort / out of scope and must be validated against a real run.
- Same as before: this is the fakeray side only. SkyRL itself may need its own
  packaging/bundle work to run on Iris (separate from the shim).
