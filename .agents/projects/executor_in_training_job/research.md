# Research: executor inside the training job

## In-repo findings

### Current launch flow

- Users invoke `iris job run --reserve <tpu> experiments/.../pipeline.py`. The entrypoint is the script itself, and it terminates by calling `executor_main(steps=[...])` (`lib/marin/src/marin/execution/executor.py:1628`). `executor_main` constructs an `Executor` and calls `Executor.run()` (`:1275`).
- `Executor.run()` walks the DAG and dispatches each `ExecutorStep` to a `StepRunner`. For `@remote`-wrapped fns, `StepRunner._do_launch` (`lib/marin/src/marin/execution/step_runner.py:217`) routes to `_run_remote_step` (`:345`), which submits the step as a Fray job via `RemoteCallable.__call__` (`lib/marin/src/marin/execution/remote.py:74-85`, calling `fray_client.submit(JobRequest(...))`).
- **Each `@remote` step is already its own Fray/Iris job today.** The inversion in this design is at the orchestrator level (where the DAG walk happens), not at the step level.

### Region pinning in the executor

- `_maybe_attach_inferred_region_constraint` (`lib/marin/src/marin/execution/executor.py:561-631`) infers a region from any GCS path in a step's config/deps/output_path via `_infer_gcs_regions` (`:201-264`), and injects `regions=[pinned_region]` into the step's `ResourceConfig`. Cross-region deps within a single step raise (`:244-246`).
- This stays in the new design: a step that explicitly references a `gs://marin-us-central1/...` path still pins to `us-central1`. We are *not* removing this.

### Region pinning in Iris

- Iris's `IrisClient.submit(...)` accepts `region_constraint(...)` (`lib/iris/src/iris/client/client.py:41`).
- When a job submitted *from inside another Iris task* lands at the controller, the child inherits the parent's `worker_region` as an automatic region constraint. This is the implicit pinning the user wants removed: an entrypoint job that lands in `us-central1` (because of where Iris ran its scheduler-of-the-day) should not force every downstream training job to also be `us-central1`.
- `worker_region` comes from `IRIS_WORKER_REGION` (`lib/iris/src/iris/cluster/client/job_info.py:122`); inheritance happens during child submission.

### Multi-VM coordination — already handled by `step_lock`

- `step_runner.py:341` wraps step execution in `with step_lock(output_path, step_label) as status_file:` — a distributed lock with heartbeat that blocks until either the lock is obtained or the step is observed complete. `step_runner.py:124`: "steps (STATUS_SUCCESS on disk) are skipped automatically."
- This means concurrent calls to `Executor.run` on the same DAG converge: all peers in a multi-VM training job (and all members of a sweep) call `Executor.run`; `step_lock` ensures exactly one wins per step, the rest read SUCCESS and skip.
- No leader / sentinel / barrier logic is needed at the design layer — it would duplicate `step_lock`.
- `JobInfo.task_index` (`lib/iris/src/iris/cluster/client/job_info.py:73`) and `JobInfo.worker_region` (`:122`) are still useful for diagnostics / explicit region-aware code paths, but they are not required for the design's correctness.

### Per-region prefix resolution

- `marin_prefix()` (`lib/rigging/src/rigging/filesystem.py:132-147`) already does the right thing: env `MARIN_PREFIX` first, else GCS metadata server → `gs://marin-{region}`, else `/tmp/marin`. Region overrides exist for europe-west4. So a training job that lands in any supported region resolves the correct bucket without configuration.
- `marin_region()` (`:150-157`) gives the current region from metadata or `MARIN_PREFIX`.
- `mirror://` and `CrossRegionGuardedFS` (`:551`, `:723`) handle the rare case where a job legitimately needs to read a remote-region file.
- All canonical region buckets exist already (`REGION_TO_DATA_BUCKET`, `:72-79`).

### Reservations

- `--reserve` and `IrisClient.submit(reservation=...)` (`client.py:565`) pre-provision workers but do not pin region. Issue #4631 ("get rid of reservations") is open and the user has confirmed the feature will be deleted in a follow-up PR.

### Reference pipeline

- `experiments/references/reference_training_pipeline.py:81-110` defines a single training `ExecutorStep` whose config references tokenize `ExecutorStep` objects (DCLM, Dolmino, SmolTalk). When `executor_main` runs in the entrypoint today, those tokenize steps are walked first and run as Fray sub-jobs before training launches.
- Under the new design the entrypoint never walks the DAG. The training job receives the same root step and walks the DAG itself, running the upstream tokenize steps (now as children of the training job's region) before training.

### Existing patterns that change

- No examples in `experiments/` of an Iris job using `IrisClient.submit()` to launch sibling top-level jobs. This is a new pattern; we'll need a small launcher helper alongside the executor.
- `mirrored()` wrapper (`executor.py:948-955`, `mirror_budget` at `filesystem.py:491`) remains available for opt-in cross-region access during the transition (e.g. if a user wants to read an already-tokenized dataset that only exists in `us-central1`).

### Related design docs

- `.agents/projects/20260302_iris_reservation_design.md` — reservations are demand pre-provisioning; orthogonal to region pinning.
- `.agents/projects/iris-log-store-gcs-fallback.md` — unrelated.

### Related issues

- #4631 (open) — get rid of reservations. Adjacent; this design doc clears one major use case.
- #4969, #3981 — root-region prefix leakage into child jobs. Removing implicit region inheritance in Iris should fix or simplify these.
- #4428 — move TPU region inference behind a public Iris API. Compatible with this design (executor still does its own inference).
- #4223 — `.mirrored()` for cross-region reads. Stays available.

## User decisions (from Q&A)

1. **Multi-VM**: every task calls `Executor.run`; the existing `step_lock` (`step_runner.py:341`) is the coordinator. No leader logic, no sentinel files.
2. **Per-region prefix**: already done — `rigging.filesystem.marin_prefix()`.
3. **Scope of "remove implicit region pinning"**: the *Iris* parent → child region inheritance is removed (`task_attempt.py:696`). The *executor*'s GCS path-based region inference (`executor.py:561-631`) stays.
4. **Reservations**: deleted in a follow-up PR. Out of scope here.
5. **Existing data**: re-tokenized in the region where the job lands. We accept the storage cost.
6. **Entrypoint**: just an `IrisClient.submit(...)` call — no `executor_main`, no new helper, no DAG walk in the entrypoint.
7. **Sweeps**: same path — entrypoint submits N training jobs; members compete for shared upstream steps via `step_lock`.

## Web prior-art pass

Skipped. This is an internal orchestration refactor (move the DAG walk from outer to inner job, drop one source of constraint inheritance), not a new category of system.
