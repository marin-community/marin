# Research: Per-User GCS Storage Prefix

Research digest for the `per_user_gcs_prefix` design doc. Findings as of `main` @ `ea00d29f1`.

## In-repo: current state

### `MARIN_PREFIX` resolution

`marin_prefix()` at [`lib/rigging/src/rigging/filesystem.py:136-151`](../../lib/rigging/src/rigging/filesystem.py) is the central helper:

```
1. MARIN_PREFIX env var
2. GCS instance metadata → gs://marin-{region}
3. /tmp/marin (local fallback)
```

Regional buckets defined in `REGION_TO_DATA_BUCKET` at the same file (lines 61-68): `marin-us-central1`, `marin-us-central2`, `marin-us-east1`, `marin-us-east5`, `marin-us-west4`, `marin-eu-west4`.

Temp prefix scheme: `gs://marin-{region}/tmp/ttl={N}d/` with `N` in `(1,2,3,4,5,6,7,14,30)` (lines 78-83).

### Executor path computation

The chain from config → final output path:

1. `ExecutorMainConfig.prefix: str | None = None` — `lib/marin/src/marin/execution/executor.py:1697`
2. `executor_main()` resolves `prefix = config.prefix or marin_prefix()` — line 1734
3. `Executor(prefix=prefix, ...)` — line 1742
4. `Executor.compute_version(step)` builds `output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)` — line 1504
5. `step.override_output_path` honored if set — line 1516
6. `self.output_paths[step] = output_path` — line 1541
7. Public utility `compute_output_path()` at lines 1778-1808 takes an optional `prefix=` (falls back to `marin_prefix()` at 1795)

### Executor cache check

Single-root, single-token status file:

- Check location: [`executor.py:1448-1454`](../../lib/marin/src/marin/execution/executor.py)
- `StatusFile(info.output_path, worker_id="check")` returns one of `SUCCESS`, `FAILED`, `DEP_FAILED`, `RUNNING`
- Status file lives at `{output_path}/.executor_status` — `lib/marin/src/marin/execution/executor_step_status.py:44`, reader at 74-93
- No multi-root lookup exists today; cache hit is strictly `{prefix}/{step}-{hash}/.executor_status == SUCCESS`

### Step-level path knobs

- `StepSpec.override_output_path` — `lib/marin/src/marin/execution/step_spec.py:59-60` — absolute or prefix-relative override
- `StepSpec.output_path_prefix` — same file, lines 50-51 — falls back to `MARIN_PREFIX` env if unset
- **No** existing `shared`, `published`, `canonical`, or `global` step flag

### Iris user identity

The chain is already designed and partially implemented per `.agents/projects/20260301_iris_user_job_names.md`:

- `IrisClient.submit(..., user: str | None = None)` — `lib/iris/src/iris/client/client.py:540-594` (param at line 554, docs at 577)
- `iris job run --user` — `lib/iris/src/iris/cli/job.py:818`
- `get_job_info().user` for in-job lookup — `lib/iris/src/iris/cluster/client/job_info.py:73-74` (function at lines 85-93, env var fallback at 96-127)
- Resolution order per design: explicit override → `get_job_info()` context → `getpass.getuser()`

### Zephyr output paths

Decoupled from `marin_prefix()`:

- `lib/zephyr/src/zephyr/dataset.py:671-722` — writers accept `output_pattern` as caller-supplied string/callable
- Zephyr imports `marin_temp_bucket` only for shuffle intermediates: `lib/zephyr/src/zephyr/execution.py:44`
- **Implication:** user-prefix routing applies automatically only to zephyr steps invoked *through marin executor*; standalone zephyr scripts construct paths manually

### Bucket lifecycle (`infra/configure_buckets.py`)

- `build_ttl_rules()` at lines 76-84 — only manages `tmp/ttl=Nd/` deletes
- `ALLOWED_TTL_DAYS = (1,2,3,4,5,6,7,14,30)`
- **Foreign-rule preserving merge** at lines 87-103, 111-112 — safe to add per-user TTL rules alongside existing ones

### Existing per-user accounting

- `scripts/ops/egress_report.py` lines 11-12, 30-31 — tracks per-user cross-region egress
- `LARGE_TIER_USER_THRESHOLD = 10` constant — implies operational awareness of per-user behavior
- No storage-side per-user tracking exists yet

### Ferry behavior

- `experiments/ferries/datakit_ferry.py:152-155` — sets `os.environ["MARIN_PREFIX"]` dynamically to `marin_temp_bucket(ttl_days=1)`
- Open question for design: when MARIN_PREFIX is set non-default, do we still layer `users/{user}/` on top?

## Prior designs

### `20260301_iris_user_job_names.md` — direct precedent

Already establishes:
- Canonical user resolution order (explicit override → context → OS user)
- The user-identity surface (`IrisClient.submit(user=)`, `iris job run --user`, `get_job_info().user`)
- Explicitly names "future per-user resource tracking, scheduler guidance, and quota/accounting features" as motivation

This GCS-prefix proposal is the storage-side application of that work.

### `20260312_iris_auth_design.md`

Sibling design on auth surface — relevant for future hard-enforcement but not v1 since v1 is convention + reporting.

## Related GitHub issues

- **#3876** — `[iris] Bind per-user service account at login for agent-driven job isolation` — closely aligned with per-user resource boundaries; relevant for v2 enforcement
- **#3461** — `Unify worker storage prefix for profiles, logs, and checkpoints` — precedent for unifying storage prefix semantics
- **#3721** — `GCS cleanup and purge` — general storage housekeeping
- **#5744** — `Clean up non-current Datakit/normalized datasets from GCS` — dataset lifecycle parallel

## Design decisions confirmed via Q&A

| Decision | Choice | Source |
| --- | --- | --- |
| User identity | Iris `--user` flag plumbed through, fallback to `getpass.getuser()` | user confirmed |
| Path layout | `users/{user}/` inside existing regional bucket (no new bucket) | user confirmed |
| Promotion mechanism | Step-level annotation in experiment code (no separate CLI for v1) | user confirmed |
| Cache mechanism | Personal → shared root fallback; promotion is the only cross-user reuse path | user confirmed |
| Scope (v1) | Executor outputs + zephyr pipeline outputs invoked through marin steps | user confirmed |
| Enforcement (v1) | Directory convention + reporting only; no write-time quotas | user confirmed |

## Open / unanswered

- `born_shared` step flag: write-directly-to-shared vs. write-personal-then-copy on completion
- MARIN_PREFIX explicit-override behavior: layer user prefix or respect the override as-is
- Lifecycle policy on `users/*`: default TTL vs. pure reporting for v1
- Cache fallback semantics for non-shared steps: probe shared root as fallback, or strict personal-only?
