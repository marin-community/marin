# Resilient TPU Training Design Sketch

## Problem

Current multislice training in Fray is still gang-scheduled. One slice loss cancels the entire cohort, and the existing "flex" path only accepts exact sizes from a fixed list. That is better than a single hard size, but it is not truly elastic and it does not satisfy the desired property that the run should keep making progress with whatever slice count is currently available.

## Recommendation

Build elasticity as a controller-driven restart system, not as in-place world-size mutation inside a running JAX job.

The core observation is:

- JAX process groups appear static once formed.
- Levanter checkpoints already support restore on a different host count.
- Iris already has a persistent controller model and slice reconciliation logic.
- JAX's official fault-tolerance docs are still experimental and explicitly not fully ready for TPU.

That combination suggests a clean architecture:

1. A persistent controller owns run identity, membership policy, and last committed state.
2. TPU workers are launched as short-lived fixed-mesh cohorts.
3. Membership changes trigger a controlled stop at the next safe point, followed by relaunch on the currently available slice count.

## Proposed Components

### 1. Elastic policy

Replace `num_slices: int | Sequence[int]` as the high-level user API with a range and policy object.

```python
@dataclass(frozen=True)
class ElasticSlicesConfig:
    min_slices: int = 1
    max_slices: int | None = None
    target_slices: int | None = None
    scale_up_cooldown_seconds: int = 1800
    resize_mode: Literal["checkpoint_boundary", "manual"] = "checkpoint_boundary"
```

This makes the contract explicit:

- any slice count in `[min_slices, max_slices]` is admissible;
- progress is required at `min_slices`;
- scale-up is opportunistic, not mandatory.

### 2. Persistent elastic controller

Add an `ElasticTrainingController` that runs on non-preemptible CPU infrastructure. It should not be a TPU worker or a best-effort Ray actor.

Responsibilities:

- track run epoch, current slice set, last committed checkpoint, and restart reason;
- decide when to admit scale-up or trigger scale-down;
- coordinate safe-point barriers;
- relaunch cohorts with the new world size;
- publish status for observability.

Suggested control loop:

```python
while not terminal:
    available = slice_manager.available_slices()
    wanted = policy.choose_slice_count(available)

    if cohort is None:
        cohort = launch_cohort(checkpoint=state.latest_checkpoint, num_slices=wanted)
        continue

    event = controller.wait_for_event()

    if event.kind in {"slice_lost", "slice_gained"}:
        if policy.should_reconfigure(event, available):
            cohort.request_stop_at_safe_point(reason=event.kind)
            checkpoint = cohort.await_checkpoint()
            cohort = launch_cohort(checkpoint=checkpoint, num_slices=policy.choose_slice_count(available))
```

### 3. Fixed-mesh training cohorts

Each cohort is a normal Levanter run:

- fixed JAX process group,
- fixed device mesh,
- fixed sharding for the duration of the cohort epoch.

This keeps the training code close to current semantics. Elasticity only happens at safe points between cohorts.

### 4. Safe-point protocol

Add a lightweight safe-point callback in Levanter:

- every `N` steps, or on controller request, all hosts enter a barrier;
- if no resize is requested, continue immediately;
- if resize is requested, flush async checkpoint writes, write a small manifest, and exit cleanly.

The manifest needs only a few fields:

```python
@dataclass(frozen=True)
class ElasticCheckpointManifest:
    run_id: str
    epoch: int
    step: int
    checkpoint_path: str
    slice_count: int
    mesh_shape: dict[str, int]
    batch_tokens: int
```

### 5. Batch and optimizer semantics

Keep per-device microbatch fixed. Adjust global effective batch by changing gradient accumulation after each restart.

Rules:

- if slice count drops, increase accumulation so optimizer semantics stay close to the original target batch;
- if slice count rises, reduce accumulation before increasing batch directly;
- record the effective batch in the checkpoint manifest and logs.

This avoids coupling correctness to a specific world size.

### 6. State transport abstraction

Use durable checkpoint restore as the correctness path.

Add an optional fast path:

```python
class ElasticStateRelay(Protocol):
    def publish(self, state_ref: str, state: PyTree) -> None: ...
    def fetch(self, state_ref: str, exemplar: PyTree) -> PyTree: ...
```

Implementations:

- `TensorStoreRelay`: always available, durable, slower.
- `ArrowFlightRelay`: existing, practical fallback for fast host-to-host state movement.
- `JaxTransferRelay`: only enable when the runtime proves the API is available and stable.

This is where TransferServer fits. It should accelerate warm restarts, not define the whole system's correctness story.

## Why not just extend Fray's list-of-sizes model?

Because the list-of-sizes workaround solves only admission. It does not solve runtime membership change.

Current Fray behavior on slice failure is still:

1. cancel the cohort,
2. count it as preemption,
3. relaunch the whole job.

That is fundamentally gang behavior. A more generic `num_slices` parser does not change the underlying lifecycle model.

## Why not lead with DiLoCo?

DiLoCo-like training is worth a separate track, especially if the goal is to use many unreliable slices with low communication overhead. There is now enough evidence to treat it seriously: the original DiLoCo paper reports robustness to workers appearing and disappearing, OpenDiLoCo reports multi-continent replication with `90-95%` utilization, and INTELLECT-1 reports a `10B` / `1T` token run built on a hybrid DiLoCo system.

But it still changes the optimization algorithm and only helps if the full model fits on the minimum local slice footprint.

That makes it a good research branch, not the baseline fix for Levanter's current synchronous training stack.

Recommended sequencing:

1. ship elastic restart for synchronous training;
2. benchmark optional state relays;
3. evaluate DiLoCo or similar as an alternative training mode for models that fit on a single slice.

## Main Risks

- Restart latency may be dominated by compile time rather than checkpoint read time.
- Some models cannot actually make progress on one slice because model-parallel state does not fit.
- Input pipeline determinism must survive world-size changes.
- Optimizer-state restore across arbitrary slice counts must be validated, not assumed.
- JAX TransferServer may remain too experimental for production use.

## Phased Rollout

### Phase 0: prove prerequisites

- verify checkpoint restore across arbitrary slice counts;
- measure compile/cache reuse after resize;
- measure checkpoint barrier latency.

### Phase 1: elastic scale-down only

- trigger restart when slices are lost;
- continue on any admissible lower slice count;
- no opportunistic scale-up yet.

### Phase 2: opportunistic scale-up

- detect additional healthy slices;
- restart at a checkpoint boundary onto a larger cohort when worth it.

### Phase 3: fast-state relay

- benchmark Arrow Flight and JAX TransferServer as resume accelerators;
- keep TensorStore as fallback and source of truth.

## Immediate Next Steps

1. Prototype a controller-owned safe-point request path.
2. Add an elastic slice policy config at the Levanter launcher layer.
3. Run checkpoint restore tests across changing slice counts.
4. Decide whether the first controller implementation lives in Levanter, Fray, or Iris-backed launcher code.
