# Run the executor inside the training job

_Why are we doing this? What's the benefit?_

Today an executor entrypoint runs in one Iris job, walks the full DAG (tokenize → train), and pins itself and every downstream job to whichever GCP region the entrypoint happened to land in. When the training step is preempted, Iris may schedule it in a different region than the one its data was created in — at which point we either crash or silently pay cross-region egress to copy tens to hundreds of GB. We want training data, by definition, to live in the region the training job runs in.

The fix is to invert the launch model: the entrypoint is just an `IrisClient.submit(...)` call — region-agnostic, no reservation, fire-and-forget. The training job, when it runs, calls `Executor.run(...)` on its own upstream deps inside its assigned region. Region pinning becomes a property of training, not of the entrypoint.

## Background

Each `@remote` step is already its own Fray/Iris job (`lib/marin/src/marin/execution/step_runner.py:345`, `remote.py:74-85`); only the *DAG walk* runs in-process inside `executor_main`. The executor already handles concurrent execution of the same step via a distributed lock + status file (`step_runner.py:341`, `executor_step_status.py`); peers and sweep members converge naturally with no extra coordination. Per-region prefix resolution already works via `rigging.filesystem.marin_prefix()` (`lib/rigging/src/rigging/filesystem.py:132-147`), which reads GCS metadata and returns `gs://marin-{region}` automatically. The executor's GCS-path-based region inference (`lib/marin/src/marin/execution/executor.py:561-631`) stays — it's how a step whose outputs land in `gs://marin-us-east5/...` ends up pinned to `us-east5`. The piece that goes away is Iris's parent-task → child-job region inheritance, currently implemented at `lib/iris/src/iris/cluster/worker/task_attempt.py:696` by stamping `IRIS_WORKER_REGION` into the child task's environment.

## Design

Two changes.

**1. Entrypoint stops calling `executor_main`. It calls the iris client directly.** The pipeline file's `__main__` block becomes a submission of the training step's `@remote` callable:

```python
if __name__ == "__main__":
    training_step.fn(training_step.config)   # @remote → IrisClient.submit
```

`@remote` already submits via `IrisClient.submit` and blocks on the result (`lib/marin/src/marin/execution/remote.py:74-85`); the entrypoint inherits that — submit and wait for completion is the default, matching today's `executor_main` behaviour. A non-blocking variant for users who want fire-and-forget can be added later if a real caller needs it. The entrypoint Iris job is CPU-only, region-agnostic, no reservation, short-lived. For sweeps, the entrypoint submits N training jobs (sequential `submit + wait`, or parallel via threads / a small concurrency helper) — same DAG, different configs.

**2. Training functions call `Executor.run(upstream_deps)` at the top of their bodies.** The training step's `@remote` function (e.g. `run_grug_base_trial`) gains a small wrapper that, before training, materialises its data dependencies:

```python
def run_grug_base_trial(config: GrugBaseLaunchConfig):
    Executor().run(upstream_steps(config))   # tokenize, etc.
    train(config)                             # JAX init + training loop
```

`upstream_steps(config)` is a new public helper in `marin.execution` that recursively walks any object and returns the `ExecutorStep` instances embedded in it — the same traversal `Executor` already does privately in `_build_dependency_graph` (`executor.py:418, 462`), exposed as a top-level function (mirrors `collect_gcs_paths` in `rigging/filesystem.py:300-313`). The training function only receives `config`, not the `ExecutorStep` itself, and the upstream tokenize steps are nested inside `config.data.components` etc. (see `lm_varying_mixture_data_config(components=...)` in `experiments/references/reference_training_pipeline.py:68-75`). The `ExecutorStep` objects are serialised into the `JobRequest` via the `@remote` decorator's pickle path.

`Executor.run` walks this DAG locally:

- Sub-jobs (tokenize etc.) are submitted as `@remote` Fray/Iris jobs from inside the training job. Their output paths resolve under `marin_prefix()`, which is `gs://marin-{R}` for the training job's region R; the executor's path-based inference (`executor.py:561-631`) then stamps `regions=[R]` onto each sub-job's `ResourceConfig`, keeping data co-located with training.
- Multi-VM TPU jobs: every task calls `Executor.run`. The existing `step_lock` (`step_runner.py:341`) ensures exactly one task per step actually runs the work; the rest read `STATUS_SUCCESS` and skip. No leader logic, no sentinels, no `task_index == 0` branch.
- Sweeps: every member calls `Executor.run` on its own DAG view. Shared upstream steps are computed exactly once across the whole sweep — `step_lock` is the cross-process coordinator.

**3. Iris change.** Remove the implicit `IRIS_WORKER_REGION` injection at `lib/iris/src/iris/cluster/worker/task_attempt.py:696`. After this, a parent task in `us-central1` submitting a child job with no explicit region produces a region-unconstrained child. Region pinning is now only via (a) explicit `regions=[...]` on `ResourceConfig`, or (b) the executor's path-based inference firing because outputs land under a region-specific prefix. `JobInfo.worker_region` (`lib/iris/src/iris/cluster/client/job_info.py:122`) stays available for callers that explicitly want to know where they're running.

## Migration

`grep -rl executor_main experiments/` returns 89 files. Per-file changes are mechanical. Concretely, the reference pipeline (`experiments/references/reference_training_pipeline.py:112-113`) goes from:

```python
# Before — entrypoint walks the DAG, pinning everything to one region
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# ... (model, data, training_step definitions unchanged) ...

if __name__ == "__main__":
    executor_main(steps=[training_step])
```

to:

```python
# After — entrypoint just submits the training job
from marin.execution.executor import ExecutorStep, this_output_path

# ... (model, data, training_step definitions unchanged) ...

if __name__ == "__main__":
    training_step.fn(training_step.config)   # @remote → IrisClient.submit, blocks
```

Everything between the imports and `__main__` (model config, tokenize step construction, `lm_varying_mixture_data_config(components=...)`, training_step definition) stays identical. The upstream tokenize `ExecutorStep` objects are still embedded in `training_step.config.data.components`; they are now materialised inside the training job rather than from the entrypoint. See `spec.md` for the full rewritten pipeline.

The training-side change is a single line per training launcher (one place per launcher — `run_grug_base_trial`, `train_lm`, etc., not 89 places):

```python
def run_grug_base_trial(config: GrugBaseLaunchConfig):
    Executor().run(upstream_steps(config))   # new — materialise deps
    train(config)                             # existing body unchanged
```

The bulk of the 89 files are pipeline definitions like the one above; the per-experiment migration is the entrypoint one-liner.

This design doc covers the framework changes (entrypoint pattern, `upstream_steps` helper, Iris inheritance removal). The actual per-experiment migration sweeps are deferred to follow-up PRs:

- **PR 1 (this design)**: framework changes + migrate `experiments/references/reference_training_pipeline.py` as the proof-of-concept.
- **PR 2..N**: sweep the remaining 88 callers in batches, grouped by training launcher (so each PR touches one launcher + its experiments).
- **Final PR**: delete `executor_main` once nothing references it.

We do not maintain dual paths; once a launcher is migrated all of its experiments migrate with it.

## Costs / Risks

- **Storage duplication.** Training data is re-tokenized in each region a job lands in. Accepted.
- **Cold-start cost on first run in a new region.** A preempted training job restarting in a fresh region pays the full tokenize time before training resumes. Acceptable — replaces a worse failure mode (crash or large egress bill).
- **New pattern: an Iris job submitting other Iris jobs.** Not used elsewhere in `experiments/`; we are the first consumer. Constraint plumbing (env, packages, bundle propagation through the `@remote` payload) needs care.
- **Migration burden.** A grep for `executor_main` in `experiments/` returns 89 callers. We migrate the reference pipeline first; the rest sweep in a follow-up PR series. We do not maintain dual paths.

## Testing

- **Unit**: training-function wrapper invokes `Executor.run(deps)` then `train(config)`; concurrent calls with overlapping DAGs converge via `step_lock` (already covered by executor tests, but extend with a multi-process case).
- **Iris integration test** on the dev cluster: 2-task fake training job whose "executor.run" sleeps; assert exactly one task does the work and both reach the post-executor code.
- **End-to-end on the reference pipeline**: launch via the new entrypoint, verify (a) entrypoint job has no region constraint, (b) tokenize sub-jobs land in the same region as the training job, (c) data is written under that region's `marin-*` bucket, (d) `TransferBudget.bytes_used` (`rigging/filesystem.py:444`) is near zero.
- **Cross-region preemption smoke test**: kill training mid-run; if Iris reschedules it to a different region, executor runs again locally and training resumes without cross-region reads.
- **Iris regression test** for the inheritance removal: a parent task in `us-central1` submits a child with no explicit region; child must be region-unconstrained.
- **Sweep coordination**: launch a small sweep where members share an upstream tokenize step; verify the shared step runs exactly once across the sweep.

## Open Questions

- **Non-training executor pipelines.** Are there pipelines today that use `executor_main` for non-training work (eval-only, dataset-only) where there's no obvious "leaf training step" to wrap with the new pattern? If so, we either keep `executor_main` available for those or come up with an analogous "leaf step calls executor on its own deps" wrapper.
- **Sweep concurrency in the entrypoint.** A sweep entrypoint that loops `submit + wait` runs sweep members serially. Most users will want them parallel. Do we ship a small concurrency helper (`submit_all([step1, step2, ...])` that submits all then awaits) in the same PR, or punt until the first sweep caller migrates?
- **Reservations interaction.** This design works whether `--reserve` exists or not. #4631 (replace reservations with allocations) is parallel work; mention it for context but not blocking.
