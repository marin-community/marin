# Iris Operations

All subcommands have `--help`. Use it.

Connection selectors:

- `--cluster=NAME` (preferred for known clusters): resolves a named config.
- `--config=PATH`: pins an exact YAML config file.
- `--controller-url=URL`: connects to an explicit URL.

How the controller is reached depends on the cluster. **IAP-fronted clusters
(marin, marin-dev) are reached directly over their IAP HTTPS URL — there is no
SSH tunnel**; `require_controller_url` returns the IAP URL and every request is
authenticated at the edge (see [Authentication](#authentication-headless--ci) and
`docs/iap-gclb.md`). Non-IAP clusters open an SSH tunnel to the controller VM
automatically.

Use `iris cluster list` to see named clusters. Use `--config` when you mean a custom or pinned file path.

## Authentication (headless / CI)

For an IAP-fronted cluster the CLI authenticates every request at the IAP edge;
the controller mints no token. Two ways to get an edge token:

- **Interactive** — `iris --cluster=NAME login` runs the desktop OAuth browser
  flow and caches a refresh token. Needs a browser; not usable headless.
- **Headless / CI / agent** — do **not** run `iris login` (it needs a browser).
  Instead give the process credentials for an *IAP-allowlisted service account*.
  The keyless way is to point your ADC at one by impersonation — no browser, no
  key file; iris reads it through the standard resolver with no flag or env:

  ```bash
  gcloud auth application-default login \
    --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com
  iris --cluster=marin-dev cluster status
  ```

  Needs `roles/iam.serviceAccountTokenCreator` on the SA (to impersonate) and
  `roles/iap.httpsResourceAccessor` on the cluster backend for that SA (the IAP
  allowlist — the impersonated SA email is the identity IAP authorizes). On a GCE
  VM whose service account is already allowlisted none of this is needed: ambient
  metadata credentials are used automatically.

See `docs/iap-gclb.md` ("The three caller paths") for the audience-vs-identity
model behind this.

## Cluster Lifecycle

```bash
iris cluster start|stop|restart|status
iris cluster dashboard              # open tunnel, print URL, block
iris cluster dashboard-proxy        # local proxy to remote controller (no tunnel needed)
```

### Controller Restart

`iris cluster controller restart` restarts the controller only (seconds of downtime, workers unaffected).
`iris cluster restart` tears down **everything** — controller + all workers. All jobs die. **Never run the full `iris cluster restart` without explicit user approval.**

Workflow: confirm the tree holds exactly the code to ship (`git status`, `git log -1`) -> capture baseline (`iris cluster status`) -> restart -> verify.

`iris cluster controller serve --dry-run` is not a restart-validation step: it boots a full local controller that serves until killed (task dispatch, VM changes, and checkpoint writes suppressed) for interactive state inspection — e.g. replaying a checkpoint to debug scheduling. Rely on the unit suite / CI on the tree as the pre-restart gate.

If checkpoint times out: `iris cluster controller restart --skip-checkpoint` (restores from last periodic checkpoint; some recent state may be lost).

**Restart builds and deploys your local working tree.** `iris cluster controller restart` builds fresh controller/worker/task images from your **current checkout — HEAD plus any staged/unstaged changes** (`get_git_sha()` is a tree-content hash), pushes them (`:<hash>` and `:latest`), pins the deploy to `:<hash>` in memory, and restarts the container in place. So the restart ships whatever code is in your tree; there is no separate image-rebuild step. To deploy a merged controller fix: update your checkout (`git pull`, or check out the fix) **then** restart — restarting from a stale checkout ships that stale code. Always confirm the controller is running the `:<git-short-hash>` you expect (`iris cluster status`), not just that it came back up; a stale-checkout deploy once cost ~5 red-canary days (`.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md`).

Restarts default to the fast Rust profile, which skips LTO and reduces native link time, and amd64+arm64 images. For an amd64-only dev cluster:

```bash
iris --cluster=marin-dev cluster controller restart \
  --image-platform linux/amd64
```

Pass `--cargo-profile release` for an LTO build. Keep the default multi-platform build when the deployed cluster needs arm64 images.

**Rollout state is recorded automatically.** Each `controller restart` writes a rollout record to `gs://…/<cluster>/state/rollout-record.json` — the image it deployed, the image it replaced, the pre-deploy checkpoint it took, and a phase (`pending` → `committed` for a forward deploy; `rollback_requested` → `rolled_back` for a revert). The rollback coordinates are captured as part of the deploy, so you never track them by hand. A forward restart also **health-checks the new controller and auto-rolls back** to the previous image + its pre-deploy checkpoint if the deploy fails to come up. (The *first* deploy after this landed has no prior record, so there is nothing to auto-roll back to — recover a failed first deploy by checking out known-good code and restarting forward, or use the on-VM procedure below.)

**A failed SSH leg aborts the restart safely.** On a GCE cluster the restart drives the VM over `gcloud compute ssh --tunnel-through-iap`; if that SSH fails (it retries 3×), the CLI prints `Rollback restart failed: Command failed after 3 attempts: SSH exit code 255` and exits — but the running controller was never touched. Confirm with `iris cluster status`: the old version still healthy means nothing deployed and nothing needs rolling back; fix SSH and retry the restart.

**GCE controller SSH auth is per-username, and agent/headless sessions may lack it.** `gcloud compute ssh` connects as your *local OS username*; `Permission denied (publickey)` right after the IAP tunnel opens means the VM refused that username+key pair, not that the tunnel failed. The controller VM does not necessarily honor every key visible in project/instance `ssh-keys` metadata for your username, so a key that "should" work from metadata inspection can still be refused — and adding new keys (metadata or OS Login) from an unattended session is exactly the kind of credential change an operator should approve first. If the restart's SSH leg is refused from your session, run the restart from a session that already has working SSH to the VM rather than minting access. Note this only gates GCE clusters (`marin`, `marin-dev`); CoreWeave controller restarts go through the Kubernetes API (kubeconfig at `~/.kube/coreweave-iris`, context pinned per cluster config) and need no SSH.

### Rolling back a controller deploy (migration-aware)

**Roll back the last deploy.** `iris cluster controller restart --rollback` reads `rollout-record.json`, then redeploys the previous image and restores its pre-deploy checkpoint — no coordinates to look up. Run it while the controller is still reachable so it takes the in-place path.

```bash
iris --cluster=marin cluster controller restart --rollback
```

**Why it restores a checkpoint, not just the old image.** A restart runs forward-only migrations in place on the on-VM state DB (`schema_migrations` tracks applied stems; there is no down-migration), and some are destructive — e.g. `0039_drop_api_keys`, `0040_drop_users`. Redeploying the old image alone would leave it loading a schema it does not understand, hitting missing-table errors at runtime. So a correct rollback must **also restore the pre-deploy (pre-migration) checkpoint** — the one taken while the old code was still running. `--rollback` does both from the record: it writes `rollback_requested` and restarts the previous image; on boot the controller restores that checkpoint over its migrated local DB, then marks the record `rolled_back`. That consume-once step is a one-shot — a later crash or VM reboot reuses the restored DB instead of rewinding to the checkpoint again.

For a wedged/unreachable controller, or a deploy with no prior rollout record, use the fully-manual on-VM procedure below instead, which never risks recreating the VM.

### Controller Checkpoint Rollback (wedged / OOM recovery)

**When.** The controller is wedged by a bloated local DB — typically a controller-VM OOM after a large job backlog: RPCs hang and the healthcheck times out. A plain restart does **not** help: startup reuses the local DB whenever it is present (`download_checkpoint_to_local` only runs when the db dir is absent — see `controller/main.py`), so `docker restart` / `gcloud compute reset` just reload the same bloated DB and re-wedge.

The fix is to roll the local DB back to a pre-spike checkpoint by hand. Run the steps below on the controller VM. **Do this only when the user has asked you to recover a wedged controller.**

Definitions used below — read them from the cluster config (`config/marin.yaml`):

- `STATE_DIR` — controller local state dir, default `/var/cache/iris/controller` (override: `storage.local_state_dir`). The DB lives in `$STATE_DIR/db`.
- `REMOTE` — `storage.remote_state_dir` (e.g. `gs://marin-us-central2/iris/state`). Checkpoints live at `$REMOTE/controller-state/<epoch_ms>/{controller.sqlite3.zst,auth.sqlite3.zst}`.

```bash
# 0. SSH to the controller VM (the GCE instance labelled iris-<prefix>-controller=true),
#    then set STATE_DIR/REMOTE from the cluster config so the commands below resolve.
gcloud compute ssh iris-controller-marin --zone <zone> --tunnel-through-iap
export STATE_DIR=/var/cache/iris/controller
export REMOTE=gs://<bucket>/iris/state

# 1. Pick a pre-spike checkpoint. The DB size is a good proxy for backlog/health:
#    a checkpoint much larger than its neighbours was already bloated — pick an
#    earlier, smaller one. Each subdir is named with its epoch_ms.
gcloud storage ls --long --readable-sizes "$REMOTE/controller-state/**/controller.sqlite3.zst"

# 2. Stop the controller (frees the RAM the bloated DB is consuming).
sudo docker stop iris-controller

# 3. Move the bloated DB ASIDE — never delete it. Startup reloads $STATE_DIR/db
#    if present, so this is what forces a fresh restore; keeping it makes the
#    rollback reversible.
sudo mv "$STATE_DIR/db" "$STATE_DIR/db.bloated.bak.$(date +%s)"

# 4. Restore the chosen checkpoint into $STATE_DIR/db using the controller image's
#    own download_checkpoint_to_local (handles the GCS pull, zstd decompress, and
#    the paired auth DB). Run it in a one-shot container so it reuses the VM's
#    ambient GCS credentials. Substitute <epoch_ms> from step 1.
IMAGE="$(sudo docker inspect --format='{{.Config.Image}}' iris-controller)"
sudo docker run --rm --network=host -v /var/cache/iris:/var/cache/iris "$IMAGE" \
    .venv/bin/python -c "from pathlib import Path; \
from iris.cluster.controller.checkpoint import download_checkpoint_to_local as restore; \
ok = restore('$REMOTE', Path('$STATE_DIR/db'), checkpoint_dir='$REMOTE/controller-state/<epoch_ms>'); \
raise SystemExit(0 if ok else 1)"

# 5. Confirm the restore actually produced a DB BEFORE starting (if it didn't, the
#    controller would reload the latest — often still-bloated — checkpoint on start).
test -f "$STATE_DIR/db/controller.sqlite3" || echo "RESTORE FAILED — do not start; move the backup back"

# 6. Start and verify it serves.
sudo docker start iris-controller
curl -sf http://localhost:10000/health && echo " controller healthy"
```

**Rollback cost.** Jobs and state created *after* the chosen checkpoint are dropped. Workers on separate VMs and other infrastructure are unaffected — they re-register with the recovered controller.

**If it goes wrong.** The previous DB is preserved at `$STATE_DIR/db.bloated.bak.<ts>`. To undo the rollback, `docker stop`, `rm -rf $STATE_DIR/db`, `mv` the backup back, and `docker start`.

## Job Management

```bash
iris job run -- python train.py         # submit + stream logs
iris job list --state running           # filter by state
iris job logs /user/job-name -f         # follow job + child logs
iris job stop /user/job-name            # exact job name + its children
iris job stop --prefix /user/job-prefix # all jobs with this ID prefix
iris job summary /user/job-name         # per-task state, exit, duration, peak memory
```

For machine-readable job data, use the Iris Python client (`IrisClient`) directly.

### `job run` gotchas

- **Remote jobs only see env vars you put in the job spec.** The submitter's
  shell env is not copied into the container. Pass required values explicitly:
  `iris job run -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" -- python train.py`.
- **`--memory` not `--ram`** — unrecognized flags silently pass through to the command string.
- **`-e KEY VALUE`** uses two positional args. If `$VALUE` is unset, the parser eats the next token. Always quote: `-e KEY "${VALUE}"`.
- **`--gpu` requests hardware; `--extra gpu` requests the Python dependency extra.** Need both for GPU JAX jobs.
- **A job that dies in BUILDING with a `uv sync` error is failing the default full-workspace sync, not your command.** Scope it with `EnvironmentSpec(sync_packages=[...])`, or skip setup entirely with `EnvironmentSpec(setup_scripts=[])` (bring-your-own image). The build log labels each step (`[iris setup] step N/M`) so you can tell which script failed. See "Task Setup" in `AGENTS.md`.
- **Use `--gpu` or `--tpu` to request accelerators, instead of `--region` or `--zone`.** Let Iris handle scaling group constraints. Use `--region` or `--zone` when you are trying to pin data to a particular location.
- **`--reserve`** is a hard zone constraint: it confines the job to a zone where the named accelerator has actually been obtained (empirically — a live, non-erroring slice in the region), and the job waits if none exists yet (an availability probe meanwhile scales the accelerator up). It does not hold capacity and does not attach accelerator devices. Use `--tpu`/`--gpu` on the task that needs hardware.
- **`executor_main` parent jobs** (e.g., canary ferries) submit GPU sub-tasks via Fray. The parent must be CPU-only (`--cpu 1 --memory 2g`), otherwise it hogs the GPU node and deadlocks. Memory at or above 4 GB requires `--enable-extra-resources` (see "Validator opt-in" below).

## Task Operations

```bash
iris task exec /user/job/0 -- bash          # shell into running container
iris task exec /user/job/0 -- python -c "import jax; print(jax.devices())"
```

Default timeout is 60s. Use `--timeout 300` for slow commands, `--timeout -1` for no timeout (last resort).

The exec session is non-interactive and buffers output. To run a command that survives disconnect, wrap with `nohup` + `&`:

```bash
iris task exec /user/job/0 -- bash -c "nohup bash -c 'your-command > /tmp/out.log 2>&1' &"
iris task exec /user/job/0 -- cat /tmp/out.log   # check later
```

### Kicking a wedged task (emergency override)

When a scheduling bug or stuck node strands a task on a machine, force its
current attempt terminal without touching the rest of the job:

```bash
iris job kick /user/job/0                       # preempt task 0 (reschedules if budget remains)
iris job kick /user/job/0 --state failed        # fail task 0 with no retry
iris job kick /user/job/0:3                      # only if attempt 3 is still current (guards against a race)
iris job kick /user/job --reason "stuck node"   # kick every active task in the job
```

The kick is queued on the controller and applied on the next control tick
through the same finalization path the scheduler's preemptions use, so it shares
one write transaction with the scheduler instead of racing it. Only tasks
running on a worker (ASSIGNED / BUILDING / RUNNING) can be kicked; pending or
already-terminal tasks are rejected with a reason. `preempted` charges the
preemption budget; `failed` is terminal with no retry.

`kick`, `stop`, and `kill` also read ids from **stdin** (`--stdin`, or a literal
`-` target) and take `--dry-run`. This is the query→act bridge: select the
targets with SQL, preview, then fire. See "Bulk actions: query → act" below.

### Recovering a stuck terminating Kubernetes pod

Use [the `recover-stuck-k8s-pod` skill](../../.agents/skills/recover-stuck-k8s-pod/SKILL.md)
when a CoreWeave pod remains after its Kubernetes deletion deadline. The Grafana
**K8s control plane** dashboard classifies overdue pods; its alert fires only for
node-bound, nonterminal GPU pods without finalizers.

The recovery order is safety-critical: record the node's existing cordon state,
cordon it, quiesce the exact Iris attempt and every sibling workload, then use a
CoreWeave force reboot if targeted graceful deletion still cannot stop the pod.
Never force-delete the pod object while the old process may still be running.
Kubernetes does not wait for kubelet confirmation, so replacement work can start
while the old process still owns the GPU. Force-delete a stale object only after
CoreWeave confirms the reboot completed (or process death is otherwise proven).

## Process Inspection & Profiling

```bash
iris process status                         # controller resource usage
iris process status -t /system/worker/<id>  # worker process status
iris process logs -f                        # follow controller logs
iris process logs --level WARNING           # filter by level
iris process profile threads                # thread dump (prints to stdout)
iris process profile cpu -d 10              # 10s CPU profile (writes .speedscope.json)
iris process profile mem                    # memory flamegraph (writes .html)
iris process profile cpu -t /user/job/0     # profile a running task container
```

**Prefer `iris process profile` over SSH** for profiling — it uses the `/system/process` RPC and avoids direct VM access. SSH is a fallback only when the RPC doesn't cover your needs.

## Scheduler & Autoscaler

```bash
iris rpc controller get-scheduler-state        # pending queue, resource constraints, priority bands
iris rpc controller get-autoscaler-status       # per-group demand, backoff, failures, quota
iris rpc controller get-provider-status         # scheduling events, cluster capacity
iris cluster vm status                          # scale groups with slice counts
```

Priority bands: `PRIORITY_BAND_INTERACTIVE` (default), `PRIORITY_BAND_PRODUCTION` (can preempt interactive), `PRIORITY_BAND_BATCH` (preemptible). See [`docs/priority-bands.md`](docs/priority-bands.md) for the user-facing guide on when to pick each band.

`get-scheduler-state`'s `running_buckets` is a **live DB projection** (tasks where
`state=RUNNING AND current_worker_id IS NOT NULL`), not an independent in-memory set.
It is self-consistent within a single call but **skews across separate RPC calls** on
a busy cluster — a task can move workers between two calls seconds apart. Do not
diagnose a "worker running a task the tasks-table doesn't show" by diffing
`running_buckets` against a *separately-timed* `iris query`; that mismatch is snapshot
skew, not a leak. To check for a genuine leak, use one atomic query (e.g. RUNNING
tasks whose `current_worker_id` is absent from `workers`).

## SQL Queries

The controller exposes its SQLite DB via RPC:

```bash
iris query "SELECT state, count(*) FROM jobs GROUP BY state"
iris query "SELECT state, count(*) FROM tasks GROUP BY state" -f csv
```

**Never modify the controller database** without explicit user approval — read-only queries only, even on offline checkpoints.

State codes: 1=PENDING, 2=BUILDING, 3=RUNNING, 4=SUCCEEDED, 5=FAILED, 6=KILLED, 7=WORKER_FAILED, 8=UNSCHEDULABLE, 9=ASSIGNED (tasks only), 10=PREEMPTED (tasks only), 11=COSCHED_FAILED (tasks only — a coscheduled sibling bounced when its gang-mate went down; terminal, not charged preemption budget), 12=MISSING.

### Sharp edges

- **Active states**: 2 (BUILDING), 3 (RUNNING), **and 9 (ASSIGNED)** — not just RUNNING. Forgetting ASSIGNED causes resource attribution misdiagnosis.
- **Committed resources**: `workers` has `committed_cpu_millicores`, `committed_mem_bytes`, etc. Total capacity is in `metadata_proto` (serialized protobuf). Available = capacity - committed.
- **`request_proto`**: serialized protobuf in `jobs.request_proto`. You need protobuf to decode — plain SQL cannot inspect task constraints.

### Useful queries

```sql
-- Failed jobs with errors
SELECT job_id, error, exit_code FROM jobs WHERE state=5 ORDER BY submitted_at_ms DESC LIMIT 10;

-- Quota-blocked scale groups
SELECT name, consecutive_failures, quota_reason FROM scaling_groups
WHERE consecutive_failures > 0 OR quota_reason != '';

-- Active slices (GCP)
SELECT slice_id, lifecycle, scale_group, worker_ids FROM slices WHERE lifecycle='ready';

-- Task attempt history (debugging retries)
SELECT task_id, attempt_id, state, exit_code, error FROM task_attempts
WHERE task_id LIKE '%<job_fragment>%' ORDER BY attempt_id;
```

Controller audit events (`event=<action> entity=<id> trigger=<trigger> <k=v ...>`)
are emitted as structured `logger.info` lines — query them through
`iris process logs` with its **built-in `--substring` filter**, not via SQL.

**`process logs` has no `--since` flag** (its only options are `-t/--target`,
`--level`, `-f/--follow`, `--max-lines`, `--substring`). Do **not** pipe the raw
output through `grep` — an unrecognized `--since` is dropped and a post-hoc `grep`
over the default window silently returns nothing. Filter server-side instead:

```bash
iris process logs --substring='event=worker_failed' --max-lines 200
iris process logs --substring='<slice-or-worker-or-job-id>' --max-lines 40   # trace one entity's whole lifecycle
```

Useful event names (the `action` passed to `log_event`): `worker_registered`,
`worker_failing`, `worker_pruned`, `assignment_queued`, `task_preempted`,
`task_unschedulable`, `task_timeout`, `job_submitted`, `slice_ready`,
`slice_pruned`, `reconcile_rpc_failed`. `task_preempted` records
`reason=Preempted by <preemptor-task-id>`, so substring-tracing a victim shows
exactly which higher-priority job evicted it.

Full table list: `iris query "SELECT name FROM sqlite_master WHERE type='table'"`.

### Bulk actions: query → act

`iris query` is admin-only and read-only, so it is the safe surface for *finding*
the exact set of tasks/jobs you want to act on. `iris job kick`, `iris job stop`,
and `iris job kill` read ids from **stdin** (`--stdin`, or a literal `-`), so a
query pipes straight into an action — no hand-copying ids. Stdin parsing is
CSV-tolerant: it takes the first field of each line and keeps only ids (leading
`/`), so a `-f csv` header row and trailing columns are dropped automatically.

**Always `--dry-run` first** to confirm the set, then re-run without it:

```bash
# Drain everything EXCEPT one protected job off a slice, so it can bind its ports.
SLICE=marin-tpu-v4-reserved-2048-us-central2-b-...
SEL="SELECT t.task_id FROM tasks t JOIN workers w ON t.current_worker_id=w.worker_id
     WHERE w.slice_id='$SLICE' AND t.state IN (2,3,9) AND t.job_id NOT LIKE '/larry/%'"

iris query -f csv "$SEL" | iris job kick --stdin --dry-run          # preview
iris query -f csv "$SEL" | iris job kick --stdin --reason "drain slice for /larry"
```

`--state preempted` (default) reschedules the kicked tasks elsewhere; `--state
failed` does not retry. Prefer kicking **tasks** (`t.task_id`, task index kept)
over whole jobs when you only need to clear specific workers — a job target
kicks *all* its active tasks, including ones on other slices.

Canonical joins (the schema doesn't pre-wire these, so keep them here):

```sql
-- Which scale group is the size-N slice? (find the slice_id to target)
SELECT scale_group, device_variant, count(*) AS workers, count(DISTINCT slice_id) AS slices
FROM workers WHERE device_type='tpu' GROUP BY scale_group, device_variant;

-- Everything occupying a slice's workers, by job and task state.
SELECT t.job_id, t.state, count(*) FROM tasks t
JOIN workers w ON t.current_worker_id=w.worker_id
WHERE w.slice_id='<slice_id>' AND t.state IN (2,3,9) GROUP BY t.job_id, t.state;

-- Co-tenants sharing a worker VM with a given job (CPU tasks bin-packed onto
-- TPU hosts show up here — a common source of host-global port collisions).
SELECT t.job_id, t.task_id, w.md_tpu_worker_id FROM tasks t
JOIN workers w ON t.current_worker_id=w.worker_id
WHERE w.worker_id IN (
  SELECT current_worker_id FROM tasks WHERE job_id LIKE '/larry/%' AND state IN (2,3,9)
) AND t.job_id NOT LIKE '/larry/%' AND t.state IN (2,3,9);
```

To *dump* rather than act, feed the same selection to `iris job logs` /
`iris job summary` per id, or read the task rows directly with a wider `SELECT`.

### Offline checkpoint analysis

For slow queries, query offline. **Never run expensive queries against the live DB** — they stall the controller.

```bash
# Download the checkpoint file (path printed by command above)
sqlite3 /tmp/controller.sqlite3 "SELECT ..."
```

Prefer to use the last checkpoint from GCS. Only take a new controller checkpoint if this is too old:

```bash
iris cluster controller checkpoint
```

## Stats Namespaces

Time-series measurements live in finelog stats namespaces, not the controller SQLite DB (see `AGENTS.md` "Decisions vs measurements"). The controller bundles a StatsService alongside its log server (started by `_start_local_log_server` in `controller/controller.py`); both are mounted on the same uvicorn app and reachable at the `/system/log-server` endpoint advertised by `cluster_config.endpoints` (or, in fallback mode, at the URL printed as `Local log server ready at <addr>` on controller startup).

Namespaces:

- `iris.worker` — per-tick host utilization (cpu, mem, disk, running task count, net bps), keyed by `ts`.
- `iris.task` — per-attempt task resource snapshots, keyed by `ts`.
- `iris.task_state` — controller-emitted (every 30s) task counts by state per root job, plus `oldest_pending_age_ms` / `oldest_building_age_ms` wait ages, keyed by `root_job_id`. The `root_job_id=""` row is the per-cluster rollup, written even when idle — its absence means the controller is down. Feeds fleet-wide stuck-BUILDING alerting and queue-depth history.
- `iris.admission_probe` — on Kubernetes clusters, the outcome (every 60s) of a `dryRun=All` canary pod apply that traverses the full admission chain, keyed by `outcome` (`ok`/`failed` with `error_class`, latency, truncated message). `failed` rows (or silence) detect fail-closed admission webhooks before any task pod exists.
- `iris.profile` — per-capture profile blobs (cpu/memory/thread, periodic or on-demand), keyed by `source` so the dashboard's per-source list query prunes via parquet row-group min/max. Filter on `source` (a task path like `/user/job/.../<index>`, `/system/worker/<id>`, or `/system/controller`) and `type` (`cpu`/`memory`/`thread`). `format` is the blob encoding — the GCE/TPU worker's periodic CPU captures are py-spy **speedscope** JSON; the k8s backend's periodic captures are py-spy **thread dumps** (`type=thread`), since a hung collective samples no CPU but a thread dump pinpoints where every rank is blocked. `vm_id` is the writer VM (worker id, `controller-self`, or `k8s/<node-or-pod>`). To find a hang, read the last periodic `thread` capture per `source` before the freeze.

Retention is finelog segment-based. Target for `iris.profile` is 7 days.

Get a profile for a task — open the dashboard task page and use the "Profile history" panel; rows are CPU captures from the worker's 10-minute periodic loop plus any on-demand captures, click to download. To capture on demand, hit the "Profile now" button on the task page, the worker page (`/system/worker/<id>`), or the controller status page (`/system/controller`).

Profiles are written by the worker (periodic CPU + on-demand all types), by `K8sTaskProvider` (periodic thread dumps of every running pod + on-demand all types), and by the controller for `/system/controller` self-captures. The k8s backend has no per-node worker daemon, so its `PeriodicProfiler` runs the equivalent 10-minute loop controller-side (`profile_poll_interval`), dumping each running pod's threads off the reconcile path.

Query the namespace directly with the finelog CLI (opens a tunnel to the cluster's finelog deployment named by `finelog.config`):

```bash
cd lib/finelog
uv run finelog query marin "SELECT source, type, format, count(*) FROM \"iris.profile\"
  WHERE source LIKE '/user/job/%' AND type='cpu' GROUP BY 1,2,3"
```

To aggregate a whole job's CPU profiles into a per-worker-sub-job breakdown + merged
flamegraph, use `scripts/job_profile_summary.py` — it resolves the cluster's finelog
deployment, pulls every CPU capture under a job (and its descendant sub-jobs), parses the
speedscope stacks, and reports where CPU is spent:

```bash
uv run python scripts/job_profile_summary.py /user/job/id          # per-sub-job + top leaves
uv run python scripts/job_profile_summary.py <dashboard-url>       # accepts iris.oa.dev URLs
uv run python scripts/job_profile_summary.py /user/job/id --subjob <name> --show-stacks
uv run python scripts/job_profile_summary.py /user/job/id -o merged.folded --svg flame.svg
```

## Users & Auth

```bash
iris login                            # IAP clusters: cache the IAP edge refresh token locally
iris rpc controller list-users        # active users with task/job counts
iris user budget list                 # per-user budget limits
```

Users authenticate **only through IAP**: the GCLB validates an OIDC token at the edge
and forwards a signed assertion the controller verifies; the controller mints **no**
user token. `iris login` runs the browser desktop-OAuth flow once and caches the IAP
edge refresh token (each RPC silently re-mints the short-lived edge token from it).
Authorization is **config-driven** — roles are resolved per request from an in-memory
`RolePolicy` built from the cluster config at controller start (admins from
`auth.admin_users`, the IAP `unprovisioned_role` for everyone else). There is no
`users` table and no reconciliation: config is the sole source of truth. To deprovision
a user, remove them from `auth.admin_users` and reload/restart the controller; the
rebuilt policy resolves them to the non-admin default on their next request (no token to
revoke — the role is resolved per request). The only fleet-wide credential kill switch
is rotating the cluster signing key (`iris cluster init-keys` + redeploy), which
re-auths every worker.

### Calling the IAP endpoint with `curl`

The built-in Marin desktop OAuth client is configured as an IAP programmatic
client. The first command opens a browser and caches a long-lived refresh token
in `~/.config/marin/credentials/marin.json`:

```bash
uv run iris --cluster marin login
```

Mint a short-lived IAP ID token from the cached credentials and send it in
`Proxy-Authorization`:

```bash
IAP_TOKEN="$(uv run python -c 'from rigging.credentials import iap_edge_provider; print(iap_edge_provider("marin").get_token())')"
curl --fail-with-body \
  --header "Proxy-Authorization: Bearer ${IAP_TOKEN}" \
  https://iris.oa.dev/proxy/system.log-server/health
```

`Proxy-Authorization` is reserved for IAP. Keep `Authorization` available for
an Iris JWT when a controller route requires one. When
`auth.iap.signed_header_audience` is configured, the controller accepts the
identity assertion added by IAP and resolves the caller's Iris role by email.

The path proxy encodes `/` in an endpoint name as `.`. The finelog endpoint
`/system/log-server` is therefore `system.log-server` in the public URL.
`/proxy/system/finelog` addresses an endpoint named `/system` with a `finelog`
subpath and does not reach the controller's finelog server.

## Troubleshooting

| Symptom | Diagnostic |
|---------|-----------|
| Job stuck PENDING | `iris rpc controller get-scheduler-state` for constraints. Check quota: `iris query "SELECT name, consecutive_failures, quota_reason FROM scaling_groups WHERE quota_reason != ''"` |
| Workers not joining (GCP) | `iris cluster vm status` for slice lifecycle. SSH to VM, check bootstrap logs. |
| Autoscaler not scaling | `iris rpc controller get-autoscaler-status` — check `backoff_until_ms`, `consecutive_failures`. |
| Task retrying | `iris job summary /user/job` — per-task state and exit codes; `iris job logs /user/job` for the per-attempt errors. |
| Task failed with exit 137 / suspected OOM | `iris job summary /user/job` — per-task peak memory + exit code. If most shards peak near the container memory limit, raise `--memory` on resubmit. |
| Dashboard unreachable | Verify tunnel is alive. `curl -sf http://localhost:10000/health`. |

## GCP (TPU) Operations

### Connecting

```bash
# SSH tunnel (IAP)
gcloud compute ssh iris-controller-marin --zone=us-central1-a \
  --project=hai-gcp-models --tunnel-through-iap -- -L 10000:localhost:10000 -N

# Then: iris --controller-url=http://localhost:10000 ...
# Or preferred named-cluster auto-tunnel: iris --cluster=marin ...
# Exact-file form for custom or pinned configs: iris --config=lib/iris/config/marin.yaml ...
```

Configs: `marin.yaml` (production), `marin-dev.yaml` (dev, smaller scale caps).

### GCP Resources

```bash
# Controller VM
gcloud compute instances list --project=hai-gcp-models \
  --filter="labels.iris-marin-controller=true" --format="table(name,zone,status)"

# Iris-managed worker VMs
gcloud compute instances list --project=hai-gcp-models \
  --filter="labels.iris-marin-managed=true" --format="table(name,zone,status)"

# TPU VMs (all zones)
gcloud compute tpus tpu-vm list --project=hai-gcp-models --zone=- \
  --format="table(name,zone,state,acceleratorType)" | head -30
```

### TPU Bad-Node Recovery

**Trigger patterns** (bad node, not a code bug):
- `RuntimeError: No accelerator found. Please run on a TPU or GPU.`
- `FAILED_PRECONDITION`
- `Device or resource busy`

**Recovery:** extract worker IP from logs -> map to VM name (`gcloud compute tpus tpu-vm list --zone <ZONE> --format="table(name,networkEndpoints[0].ipAddress)"`) -> delete bad node (`gcloud compute tpus tpu-vm delete <NAME> --zone <ZONE> --quiet`) -> resubmit job.

Only delete the specific bad node. If multiple nodes fail simultaneously or the same node fails again, escalate to the user.

### GCP State

State dir: `gs://marin-us-central2/iris/<cluster>/state/` — contains `bundles/` (code packages) and `controller-state/` (SQLite checkpoints). Per-task log parquet segments are shipped separately by finelog under `<finelog.remote_log_dir>/log/` (see `lib/finelog/config/<cluster>.yaml`).

### GCP Gotchas

- **Quota is the primary scaling bottleneck.** The autoscaler backs off exponentially per scale group. Check with `iris rpc controller get-autoscaler-status`.
- **Stuck TPU VMs.** Occasionally a TPU VM gets stuck in DELETING for days. Check: `gcloud compute tpus tpu-vm list --project=hai-gcp-models --zone=- --filter="state=DELETING"`.

---

## CoreWeave (GPU) Operations

Always read [`docs/coreweave.md`](docs/coreweave.md) before operating a
GPU/CoreWeave cluster. Use `lib/iris/config/coreweave-*.yaml` for CoreWeave
cluster configs.

## CI Workflows

| Workflow | Trigger | What |
|----------|---------|------|
| `marin-canary-ferry.yaml` | Daily 6AM UTC | TPU canary on GCP (`marin-dev.yaml`) |
| `marin-canary-ferry-coreweave.yaml` | Daily 10AM UTC | GPU canary on CW — shares `iris-ci` controller + H100 nodepool with `iris-smoke-coreweave.yaml` (concurrency group `iris-coreweave-ci-shared`) |
| `iris-smoke-gcp.yaml` | PRs touching `lib/iris/` | GCP smoke test (ephemeral cluster) |
| `iris-smoke-coreweave.yaml` | PRs touching `lib/iris/` | CW integration tests (warm cluster) |
| `ops-docker-images.yaml` | `workflow_dispatch` / Sun 02:00 UTC | Rebuilds + pushes `iris-{controller,worker,task}:latest` to GHCR (see Controller Restart) |

```bash
# Trigger manually
gh workflow run "<workflow name>" -R marin-community/marin --ref main
# View failed run
gh run view <run-id> -R marin-community/marin --log-failed | tail -50
```
