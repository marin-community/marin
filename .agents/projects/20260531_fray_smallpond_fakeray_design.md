# `fakeray`: a Ray-compatible shim for running smallpond on Iris/Fray

Status: **Draft / design sketch.** Not implemented. No code committed.
Author: agent-generated (exploration of "run smallpond on Iris").
Date: 2026-05-31.

## TL;DR

smallpond drives its execution through a tiny slice of the Ray Core API
(`@ray.remote`, `ObjectRef`, `put`/`get`/`wait`). We can satisfy that slice with
a ~10-function shim module (`fakeray`) that:

1. installs in place of `ray` (`import fakeray as ray`),
2. runs a small **ready-queue DAG scheduler** in the smallpond driver process, and
3. dispatches each task to a pool of long-lived **Fray actors** (one Iris job
   each), collecting results over RPC.

The load-bearing fact that makes this cheap: **smallpond intermediate `DataSet`s
are path descriptors, not data.** Bulk data already lands as Parquet on
`data_root`. So the shim's "object store" only ferries kilobyte pickles between
the driver and actors — the terabytes never traverse the shim.

The genuinely hard part is **not** the shim. It is that fan-out requires
`data_root` to be a filesystem every actor can read and write — the role 3FS
plays in the original design. On Iris there is no 3FS, so we either mount a
shared POSIX FS on every actor or make smallpond's file I/O `gs://`-native.
That is the real cost and risk; the shim itself is ~300–400 LOC.

Recommended sequencing: prove the shim against Fray `LocalClient` (one node,
local disk — no shared-FS problem) first, then tackle the shared-FS story for
true Iris multi-node fan-out.

---

## 1. Goal and non-goals

### Goal
Let an unmodified-as-possible smallpond program

```python
import smallpond
sp = smallpond.init(num_executors=0)
df = sp.read_parquet("gs://.../in/*.parquet").repartition(256, hash_by="k")
df = sp.partial_sql("select k, count(*) from {0} group by k", df)
df.write_parquet("gs://.../out/")
```

execute its task DAG **distributed across Iris workers** via Fray, instead of
inside a single embedded Ray cluster in one container.

### Non-goals
- Re-implementing Ray's in-memory distributed object store with zero-copy / cross-node
  shared memory. We exploit smallpond's file-materialized data model to avoid this.
- Supporting Ray APIs smallpond does not use (Ray Data, Datasets, Tune, Serve,
  placement groups, named/detached actors, `ray.remote` *actors*, nested remotes).
- GPU scheduling fidelity. smallpond passes `num_gpus`; v1 treats the pool as
  CPU slots and logs that GPU packing is not honored.
- Replacing the single-container Option A (real Ray inside one Iris task). That
  remains the cheaper path for non-fan-out workloads; see §15.

---

## 2. What smallpond actually requires from Ray

Every Ray call site in smallpond (verified by grep over `smallpond/`):

| Ray surface | Where | Use |
|---|---|---|
| `ray.init(address, num_cpus, _memory, runtime_env, dashboard_*, _metrics_export_port)` → `.address_info["gcs_address"]` | `session.py:79` | start/connect cluster; smallpond reads `gcs_address` |
| `ray.shutdown()` | `session.py:154` | teardown |
| `ray.timeline(path)` | `session.py:258` | dump chrome-trace timeline |
| `@ray.remote` (bare) + `fn._function_name = ...` | `task.py:1001,1039` | wrap a task's exec as a remote function |
| `.options(name=, num_cpus=, num_gpus=, memory=)` | `task.py:1041` | per-task resource + name |
| `RemoteFn.remote(task, *dep_refs)` → `ObjectRef` | `task.py:1052` | dispatch; deps passed as refs |
| `ray.put(dataset)` → `ObjectRef` | `task.py:995` | wrap an already-materialized output (resume / boundary data) |
| `ray.ObjectRef` (type/annotation) | `task.py:601,982` | the handle type stored on each task node |
| `ray.get(refs, timeout=)` | `dataframe.py:131,182,261` | block for results; surface exceptions |
| `ray.wait(refs, num_returns=, timeout=0, fetch_local=False)` → `(ready, not_ready)` | `dataframe.py:173,243` | non-blocking progress poll |
| `ray.exceptions.RuntimeEnvSetupError` | `dataframe.py:262` | caught around dispatch |

That is the **entire** contract. Note the DAG is built lazily and recursively:
`Task.run_on_ray()` calls `dep.run_on_ray()` for each input first (getting their
`ObjectRef`s), then `exec_task.options(...).remote(task, *dep_refs)`. Real Ray:
(a) returns each ref immediately, (b) treats `ObjectRef` arguments as
dependency edges, (c) auto-dereferences them to the actual `DataSet` values
when finally invoking `exec_task`. The shim must reproduce exactly those three
behaviors.

### 2.1 The fact that makes this tractable

`smallpond/logical/dataset.py`: the `DataSet` hierarchy is **path-based**.
`ParquetDataSet`/`CsvDataSet`/`JsonDataSet`/`FileSet` carry `paths` + `root_dir`
(via `__slots__`), not data. `PartitionedDataSet` holds a list of such
descriptors. The actual bytes are Parquet files on `data_root`, written
atomically by `task.exec()` and read back by the next task via DuckDB/pyarrow.

Consequences:
- An `ObjectRef` only needs to carry a **pickled descriptor** (sub-KB to KB).
- Passing a dep's value to a downstream actor over RPC is cheap.
- The shim **never moves bulk data**; it moves filenames. Workers read/write the
  bulk directly against shared storage.

The two exceptions are boundary sources `ArrowTableDataSet` (`.table`) and
`PandasDataSet` (`.df`), which hold in-memory data from `sp.from_arrow/from_pandas`.
These do travel over RPC if used as a remote input. They are user-supplied
constructor inputs (typically small); we accept this and note it (§13).

---

## 3. Why Iris/Fray don't provide this directly

Fray's `Client` protocol (`lib/fray/src/fray/client.py`) offers:
- `submit(JobRequest) -> JobHandle` — coarse: one job == one container == one
  `uv sync` + process. Too heavy for per-DAG-node dispatch.
- `create_actor_group(actor_class, *, name, count, resources) -> ActorGroup` with
  `wait_ready() -> [ActorHandle]`; `handle.method.remote(*args) -> ActorFuture`;
  `future.result()`. Actors are long-lived Iris jobs hosting an `ActorServer`;
  methods are RPCs (args cloudpickled). `ActorConfig(max_concurrency=N)`.
- `wait_all([JobHandle])`, `JobHandle.wait/status`.

What Fray/Iris **lack** vs Ray: (1) an object store (`put`/`get` of live values
addressable cluster-wide), and (2) a fine-grained futures-DAG scheduler. The
shim supplies both — cheaply, because (1) reduces to "pickle a descriptor" and
(2) is a textbook ready-queue over a fixed actor pool.

Mapping the actor calling convention is a near-perfect fit: Fray's
`handle.method.remote(*args) -> ActorFuture; .result()` is exactly the
future semantics the shim's scheduler needs.

---

## 4. Architecture

```
            Iris job: smallpond driver  (user's `iris job run`)
            ┌───────────────────────────────────────────────┐
            │  import fakeray as ray                          │
            │  smallpond.Session  (builds logical plan)       │
            │                                                 │
            │  fakeray.RemoteFn.remote(task, *dep_refs)       │
            │        └─► register node in DAG, return ref     │
            │                                                 │
            │  fakeray Scheduler (ready-queue, in-process)    │
            │   • node ready when all dep refs resolved       │
            │   • assign to a free actor slot                 │
            │   • actor.run_task.remote(task, *dep_values)    │
            │   • on result: mark ref ready, unblock children │
            │                                                 │
            │  Fray ActorGroup  (current_client())            │
            └─────────┬───────────────┬───────────────┬───────┘
                      │ RPC            │ RPC           │ RPC
              ┌───────▼──────┐ ┌───────▼──────┐ ┌──────▼───────┐
              │ Iris job:    │ │ Iris job:    │ │ Iris job:    │
              │ Executor #0  │ │ Executor #1  │ │ Executor #M-1│
              │ run_task():  │ │              │ │              │
              │  task.exec() │ │   ...        │ │   ...        │
              └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                     │ read/write bulk Parquet         │
                     ▼                ▼                ▼
            ┌───────────────────────────────────────────────┐
            │   SHARED data_root  (gs:// or mounted FS)       │  ← the hard part
            └───────────────────────────────────────────────┘
```

Three pieces:
1. **`fakeray` module** — the Ray API surface (§6). Pure glue + the scheduler.
2. **`Scheduler`** — in the driver; ready-queue over actor slots (§7).
3. **`SmallpondExecutorActor`** — a Fray actor that runs one `task.exec()` (§8).

The driver is the user's Iris job (the "head"). It owns the DAG, the scheduler,
and the `ActorGroup`. Actors are dumb executors.

---

## 5. `ObjectRef` and the "object store"

`ObjectRef` is a driver-local handle backed by a `concurrent.futures.Future`:

```python
@dataclass(eq=False)
class ObjectRef:
    id: str                       # uuid; identity-based equality (eq=False)
    _fut: "Future[Any]"           # resolves to the task's return value (a DataSet)
```

- `ray.put(value)` → a ref whose future is already resolved to `value`. Used for
  resume (`load(ray_dataset_path)`) and boundary datasets.
- A `.remote(...)` call → a ref whose future the scheduler resolves when the
  actor returns the output `DataSet`.
- `ray.get(ref)` → `ref._fut.result()` (driving the scheduler meanwhile);
  re-raises the task exception if the future failed.
- Dependency dereferencing: when the scheduler dispatches a node, it replaces
  each `ObjectRef` argument with `ref._fut.result()` (already resolved, since the
  node is only ready once deps are done) before sending to the actor. This
  reproduces Ray's auto-deref.

Because values are descriptors, the "store" is just the set of resolved futures
held in the driver. Optional **persistence (B2):** also write each descriptor to
`{data_root}/_fakeray/objs/{id}.pkl`. smallpond *already* writes
`ray_dataset_path` per task and checks it on re-run, so B2 is largely redundant
with smallpond's own resume; v1 keeps refs in-memory and leans on smallpond for
idempotency/resume.

---

## 6. The `fakeray` API surface

```python
# fakeray/__init__.py   (sketch — illustrative, not final)

_SCHED: "Scheduler | None" = None

class RuntimeEnvSetupError(RuntimeError): ...
class exceptions:                      # smallpond does `ray.exceptions.RuntimeEnvSetupError`
    RuntimeEnvSetupError = RuntimeEnvSetupError

def init(address=None, *, num_cpus=None, _memory=None,
         runtime_env=None, **_ignored):
    """Start the scheduler + Fray actor pool. Ignores Ray-specific kwargs."""
    global _SCHED
    cfg = FakeRayConfig.from_env()           # pool_size, ram per actor, image/extras
    client = current_client()                # IrisClient in a job, else LocalClient
    _SCHED = Scheduler(client, cfg, runtime_env=runtime_env)
    _SCHED.start()                           # create_actor_group + wait_ready
    return _InitResult(address_info={"gcs_address": "fakeray://local"})

def shutdown():
    global _SCHED
    if _SCHED: _SCHED.shutdown(); _SCHED = None

def put(value) -> ObjectRef:
    return _SCHED.put(value)

def get(refs, timeout=None):
    scalar = isinstance(refs, ObjectRef)
    out = _SCHED.get([refs] if scalar else list(refs), timeout=timeout)
    return out[0] if scalar else out

def wait(refs, *, num_returns=1, timeout=None, fetch_local=True):
    return _SCHED.wait(list(refs), num_returns=num_returns, timeout=timeout)

def timeline(path):                          # best-effort; Iris has its own UI
    _SCHED.dump_timeline(path)

def remote(fn=None, **opts):
    # supports bare @remote and @remote(**opts)
    def wrap(f): return RemoteFunction(f, opts)
    return wrap(fn) if fn is not None else wrap

class RemoteFunction:
    def __init__(self, fn, opts): self._fn, self._opts = fn, dict(opts)
    def options(self, **opts):               # name, num_cpus, num_gpus, memory
        merged = {**self._opts, **opts}; return RemoteFunction(self._fn, merged)
    def remote(self, *args, **kwargs) -> ObjectRef:
        return _SCHED.submit_task(self._fn, args, kwargs, self._opts)
    def __setattr__(self, k, v):             # smallpond sets `_function_name`
        object.__setattr__(self, k, v)
```

Install it (primary): edit the three `import ray` lines in the patched fork to
`import fakeray as ray`. Alternative (zero fork edits): `sys.modules["ray"] =
fakeray` before `import smallpond` — fragile w.r.t. import order; documented but
not preferred, consistent with the repo's "no ad-hoc compatibility hacks" rule.

---

## 7. The scheduler (ready-queue over an actor pool)

```python
class Scheduler:
    def __init__(self, client, cfg, runtime_env=None):
        self._client, self._cfg = client, cfg
        self._nodes: dict[str, _Node] = {}          # ref.id -> node
        self._free: queue.SimpleQueue[ActorHandle] = queue.SimpleQueue()
        self._lock = threading.Lock()

    def start(self):
        env = create_environment(workspace=os.getcwd(),
                                 extras=self._cfg.extras)     # smallpond + duckdb etc
        self._group = self._client.create_actor_group(
            SmallpondExecutorActor, name="smallpond-exec",
            count=self._cfg.pool_size,
            resources=ResourceConfig(cpu=self._cfg.cpu, ram=self._cfg.ram),
            actor_config=ActorConfig(max_concurrency=1))      # 1 task per actor
        for h in self._group.wait_ready(count=self._cfg.pool_size):
            self._free.put(h)

    # called from RemoteFunction.remote(); returns immediately
    def submit_task(self, fn, args, kwargs, opts) -> ObjectRef:
        dep_refs = [a for a in args if isinstance(a, ObjectRef)]
        ref = ObjectRef(id=uuid4().hex, _fut=Future())
        self._nodes[ref.id] = _Node(ref, fn, args, kwargs, opts,
                                    pending={d.id for d in dep_refs})
        self._maybe_ready(ref.id)
        return ref

    # block until all `targets` resolved, pumping the scheduler
    def get(self, targets, timeout=None):
        deadline = _deadline(timeout)
        while not all(t._fut.done() for t in targets):
            self._tick(deadline)                  # dispatch ready, reap finished
        return [t._fut.result() for t in targets] # re-raises on failure

    def _tick(self, deadline):
        # 1. dispatch every ready node to a free actor
        for node in self._ready_nodes():
            try:
                actor = self._free.get(timeout=_left(deadline))
            except queue.Empty:
                break
            values = [self._deref(a) for a in node.args]      # ObjectRef -> DataSet
            fut = actor.run_task.remote(node.fn_payload, values, node.kwargs)
            self._inflight[node.ref.id] = (actor, fut, node)
        # 2. reap any finished inflight calls
        for rid, (actor, fut, node) in list(self._inflight.items()):
            if fut_done(fut):
                self._free.put(actor)
                del self._inflight[rid]
                try:
                    node.ref._fut.set_result(fut.result())
                except Exception as e:
                    node.ref._fut.set_exception(e)
                self._unblock_children(rid)
```

Notes:
- **Ready** = all `pending` dep ids resolved. `_unblock_children` removes the
  finished id from dependents' `pending` sets.
- `run_task` receives a cloudpickled callable payload + already-dereferenced
  input `DataSet`s. The fn payload is smallpond's nested `exec_task` closure; we
  pickle `(fn, task)` — `task` is the real payload, `fn` is generic.
- v1 is **slot-based**: `pool_size` actors, `max_concurrency=1`, so ≤ pool_size
  concurrent tasks. `opts["num_cpus"]/["memory"]` are recorded for the actor's
  `ResourceConfig` at pool-creation time, not used for per-task bin-packing
  (§13).
- `ray.wait(..., timeout=0)` → one non-blocking `_tick` then partition targets
  into `(ready, not_ready)`. This matches smallpond's progress-poll usage.
- The scheduler is single-threaded (driven by `get`/`wait` calls); `ActorFuture`
  polling uses Fray's RPC futures. No background thread needed for v1, which
  keeps failure semantics simple.

---

## 8. The executor actor

```python
class SmallpondExecutorActor:
    """Long-lived Fray actor. One method: run a single smallpond task."""
    def __init__(self):
        # imports happen once per actor process; warms duckdb, arrow, smallpond
        import smallpond.execution.task  # noqa: F401

    def run_task(self, fn_payload: bytes, input_datasets: list, kwargs: dict):
        import cloudpickle
        fn, task = cloudpickle.loads(fn_payload)
        # reproduce what smallpond's exec_task body does:
        task.input_datasets = list(input_datasets)
        status = task.exec()
        if status != WorkStatus.SUCCEED:
            raise task.exception or RuntimeError(f"task {task.key} failed: {status}")
        dump(task.output, task.ray_dataset_path, atomic_write=True)  # smallpond resume
        return task.output                                           # descriptor over RPC
```

This is essentially smallpond's existing `exec_task` closure body (`task.py:1002`)
lifted into an actor method. We can literally reuse that code. The actor:
- reads inputs' bulk data lazily inside `task.exec()` (DuckDB opens the Parquet
  paths from the input descriptors — against shared storage);
- writes its output Parquet to `data_root` (shared storage);
- returns only the output descriptor.

Pool warmth: because actors persist, the per-task cost is one RPC + one
`task.exec()`, **not** a container build / `uv sync`. The `uv sync` happens once
when each actor job starts.

---

## 9. Integration with smallpond

1. **Use `num_executors=0`.** This stops `Session.__init__` from calling
   `platform.start_job(worker.py ...)` (which would `ray start` real workers).
   The shim owns parallelism; pool size comes from `FakeRayConfig` (env/arg), not
   `num_executors`. Document this clearly — it is the one behavioral wart.
2. **`import fakeray as ray`** in `session.py`, `task.py`, `dataframe.py`
   (3 lines in the fork) — or `sys.modules` injection (§6).
3. **`data_root` must be shared** (§10). The driver and every actor resolve the
   same `data_root`; smallpond already threads it through `RuntimeContext`.
4. Boundary I/O (`read_parquet`/`write_parquet`) already accepts arbitrary paths;
   `gs://` works iff smallpond's reader/writer honor fsspec for those (DuckDB
   `httpfs`/pyarrow do; the descriptor `dump`/`load` and marker files are the gap).
5. `ray.init`'s `runtime_env` (smallpond sets `LD_PRELOAD`, malloc tuning,
   `ARROW_*`, thread caps) → the shim maps these into the actor `JobRequest`
   `environment.env_vars` so executors inherit the same tuning.

Resulting user code is unchanged except `init(num_executors=0)` + an
environment setting for pool size:

```python
import fakeray; fakeray.configure(pool_size=64, data_root="gs://marin-eu-west4/tmp/ttl=7d/rav/sp")
import smallpond
sp = smallpond.init(num_executors=0, data_root="gs://marin-eu-west4/tmp/ttl=7d/rav/sp")
```

---

## 10. The shared-filesystem problem (the actual hard part)

smallpond was built for **3FS**: a POSIX-mounted, cluster-shared filesystem. Every
worker does ordinary `open()` on a 3FS path and sees the same bytes. The Ray
shim does nothing to change smallpond's assumption that `data_root` is shared.
On Iris there is no 3FS, so we must provide an equivalent. Options, worst-to-best
for a first cut:

- **(FS-1) Per-actor local disk — does NOT work for fan-out.** Each Iris task has
  its own ephemeral disk; actor A cannot read actor B's Parquet. Only viable for
  the single-node staging milestone (§11), where one box's local disk is shared
  among threads/processes.

- **(FS-2) `gs://` via fsspec, end to end.** Make `data_root` a GCS path and
  ensure *all* of smallpond's I/O goes through fsspec/`gcsfs`:
  - Bulk Parquet read/write: DuckDB (`httpfs`/`gcs` secret) and pyarrow already
    support `gs://`. Needs config wiring + a credentials story in the actor.
  - **Gap:** smallpond's `io/filesystem.py` `dump`/`load` and the marker/status
    files use local `open()` (verified: `open(path, "wb")`). These must become
    fsspec-backed for `gs://`. This is the bulk of the real work — a focused but
    non-trivial edit to smallpond's I/O layer, plus correctness review of every
    `os.path`/`os.makedirs`/`os.path.exists` on `data_root` (markers, atomic
    rename — GCS has no atomic rename; `atomic_write` via temp+rename needs an
    object-store-aware implementation).
  - Cost driver / policy: keep `data_root` in the **same region** as the workers
    (per repo rules on cross-region egress); use a `tmp/ttl=7d/...` prefix.

- **(FS-3) Mounted shared POSIX FS (gcsfuse / Filestore / Lustre).** If a shared
  mount is available on all Iris workers at a common path, smallpond runs almost
  unmodified (it just does `open()`), exactly like 3FS. This is the **lowest-code,
  highest-ops** option: it needs the mount provisioned on the worker image /
  pod spec, which is an Iris/infra change, not a smallpond change. gcsfuse has
  weak POSIX semantics (no atomic rename, slow metadata) that may collide with
  smallpond's atomic-write assumptions; Filestore/Lustre are stronger but cost
  more and are regional.

**Recommendation:** FS-3 if a shared mount can be provisioned (cheapest code,
mirrors 3FS); otherwise FS-2 with a careful audit of smallpond's atomic-write /
rename / existence checks. Either way, **this — not the shim — is where the
effort and risk concentrate.** The shim is inert without it.

---

## 11. Staging plan (each stage independently testable)

1. **`fakeray` over Fray `LocalClient`, single box.** No Iris, no shared FS
   (local disk is shared among in-process actors). Proves the scheduler,
   `ObjectRef`/`get`/`wait`/`put` semantics, and auto-deref against the *real*
   smallpond DAG. Acceptance: the prices.parquet quickstart and a repartition→sql
   job produce byte-identical results to stock single-node smallpond. Zero
   infra. This is the milestone that de-risks the shim itself.
2. **`fakeray` over Fray `IrisClient`, single worker, local `data_root`.** Proves
   actor hosting, RPC payload sizes, env/`LD_PRELOAD` propagation, pool warmup.
   Still no fan-out (one actor). Acceptance: same job green as an Iris job.
3. **Shared-FS spike (FS-2 or FS-3).** Smallest possible: two actors on two
   workers, a 2-partition repartition, `data_root=gs://...`. This is the
   go/no-go for multi-node. Acceptance: partition written by actor A is read by
   actor B; final output correct.
4. **Scale + fidelity.** Pool of N, GraySort-style shuffle on real data; add
   resource-aware admission (§13), speculative-exec passthrough (§12), timeline.

Stop after stage 1–2 if the goal is only "validate the approach"; stages 3–4 are
the real-fan-out investment.

---

## 12. Failure handling & retries

- **Task failure:** actor raises → `ActorFuture` errors → scheduler sets the
  ref's future exception → `ray.get` re-raises. This matches Ray and lets
  smallpond's `dataframe.py` error path run.
- **smallpond's own retry:** `exec_task` already counts retries via
  `ray_marker_path` files and resumes finished tasks via `ray_dataset_path`. With
  a shared `data_root` this keeps working unchanged — a re-dispatched task sees
  its marker/finished state. This is a reason to prefer leaning on smallpond's
  fault tolerance over inventing new shim-level logic.
- **Actor/worker death (preemption):** Fray `ActorGroup.is_done()` /
  `ActorConfig(max_restarts, max_task_retries)` cover actor restarts. The
  scheduler must detect an actor that vanished mid-task (RPC error), return its
  slot, and re-dispatch the node (its inputs are still resolved). v1: on RPC
  failure, re-queue the node up to `max_task_retries`, then fail the job.
- **Speculative execution:** smallpond supports it; the shim does **not** drive
  it in v1 (no duplicate dispatch). Acceptable — it's an optimization, off by
  default in single-cluster mode. Note as a future item.

---

## 13. Fidelity & known limitations (state them, don't hide them)

- **Resource packing:** v1 is slot-based (N actors × 1 concurrent task). Ray
  packs by `num_cpus`/`memory` per task; we don't. A job with heterogeneous
  per-task memory may under/over-subscribe. Mitigation later: weighted admission
  (track free cpu/mem per actor; `max_concurrency>1`; bin-pack). Log the
  discrepancy so it's visible.
- **GPU tasks:** `num_gpus` ignored beyond optionally sizing the actor pool's
  `ResourceConfig`. Out of scope for the CPU-shuffle target.
- **Boundary data over RPC:** `ArrowTableDataSet`/`PandasDataSet` carry real
  bytes; large `from_pandas` inputs cross RPC. Mitigation: spill these to a
  Parquet file in `data_root` at `put` time and replace with a `ParquetDataSet`
  descriptor (keeps the "only descriptors on the wire" invariant).
- **`ray.timeline`:** best-effort or no-op; Iris dashboard is the real
  observability surface.
- **Single-driver scheduler throughput:** one Python thread dispatching. Fine for
  thousands of tasks; for the 50-node/100k-task GraySort regime, the dispatch
  loop and RPC fan-out may need a background pump thread or batched dispatch.
  Measure in stage 4.
- **No atomic rename on object stores:** smallpond's `atomic_write` (temp +
  rename) must be reimplemented for `gs://` (write temp object + server-side
  copy/compose, or per-task unique names + manifest). Tracked under FS-2.

---

## 14. Testing

- **Unit (LocalClient):** scheduler ready-queue ordering; `get`/`wait`/`put`
  semantics; auto-deref; exception propagation; a diamond DAG (A→{B,C}→D) yields
  correct topo execution. No mocks beyond the Fray `LocalClient` boundary.
- **Integration (LocalClient, real smallpond):** quickstart + repartition + join;
  assert output Parquet equals stock smallpond (externally-observable behavior,
  not log strings — per repo testing policy).
- **Integration (IrisClient):** stage-2/3 jobs as `requires_cluster` tests;
  assert final dataset correctness and that work landed on ≥2 workers (via Iris
  job summary task→worker mapping).
- **Fault injection:** kill an actor mid-task; assert re-dispatch + correct
  result (reuses smallpond's `fault_inject_prob`).

---

## 15. Alternatives considered

- **Option A — real Ray inside one Iris task (single-node).** ~30 lines of
  packaging fixes (already prototyped); only blocker is the container's 64 MB
  `/dev/shm` starving Ray's object store (Iris sets `--shm-size=100g` for
  TPU/GPU tasks only — `lib/iris/.../runtime/docker.py`). Cheaper and correct for
  non-fan-out workloads. The fakeray shim is only worth it when you specifically
  want smallpond to distribute across Iris workers. **A and the shim are
  complementary, not competing** — A is the fast path, the shim is the scale path.
- **Real Ray multi-node on Iris** (Iris launches a real Ray head + `ray start`
  workers, smallpond unchanged). Reproduces smallpond's stock multi-node design,
  but means running a second scheduler/cluster inside Iris (double scheduling,
  the head-in-a-task awkwardness, the shm issue ×N) and a Ray-version treadmill.
  The shim avoids standing up Ray entirely.
- **Full in-memory object store** (true `ray.put` cross-node value transfer).
  Rejected: that is rebuilding Ray, and unnecessary given file-materialized
  DataSets.

---

## 16. Effort & risk

- **Shim + scheduler + actor (stages 1–2):** small and well-bounded, ~300–400
  LOC, a few days including tests. Low risk — the API contract is tiny and the
  Fray actor future maps cleanly.
- **Shared-FS for fan-out (stage 3, the real cost):** medium-to-high risk,
  depends entirely on FS-3 availability vs FS-2 effort. The atomic-write/rename
  and existence-check audit of smallpond's I/O layer is the most likely source of
  subtle correctness bugs. Budget the majority of the project here.
- **Scale/fidelity (stage 4):** optional, demand-driven.

Net: the shim is the easy, fun 20%; the shared filesystem is the load-bearing
80%. Don't start the shim expecting fan-out for free — fan-out is bought with the
filesystem.

---

## 17. Open questions / decisions for a human

1. Is a shared POSIX mount (gcsfuse/Filestore/Lustre) provisionable on Iris CPU
   workers? If yes → FS-3, smallpond nearly unmodified. If no → commit to FS-2
   (fsspec-native smallpond I/O).
2. Is multi-node smallpond actually wanted, or is single-node (Option A, fix
   `/dev/shm`) sufficient for the real use case? The shim only pays off for #1=yes
   and this=multi-node.
3. Fork policy: explicit `import fakeray as ray` edits in the smallpond fork
   (clean, debuggable) vs `sys.modules` injection (zero fork edits, fragile)?
4. Where does `fakeray` live — a module inside the smallpond fork, or a small
   standalone package that depends on `fray`? (Dependency direction allows
   smallpond→fray; fray must not depend on smallpond.)
```
