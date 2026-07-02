# Iris Operations

All subcommands have `--help`. Use it.

Connection selectors:

- `--cluster=NAME` (preferred for known clusters): resolves a named config and auto-tunnels.
- `--config=PATH`: pins an exact YAML config file and auto-tunnels.
- `--controller-url=URL`: connects to a manually established tunnel.

Use `iris cluster list` to see named clusters. Use `--config` when you mean a custom or pinned file path.

## Cluster Lifecycle

```bash
iris cluster start|stop|restart|status
iris cluster dashboard              # open tunnel, print URL, block
iris cluster dashboard-proxy        # local proxy to remote controller (no tunnel needed)
```

### Controller Restart

`iris cluster controller restart` restarts the controller only (seconds of downtime, workers unaffected).
`iris cluster restart` tears down **everything** — controller + all workers. All jobs die. **Never run the full `iris cluster restart` without explicit user approval.**

Workflow: dry-run locally (`iris cluster controller serve --dry-run`) -> capture baseline (`iris cluster status`) -> restart -> verify.

If checkpoint times out: `iris cluster controller restart --skip-checkpoint` (restores from last periodic checkpoint; some recent state may be lost).

**Shipping a code change ≠ restarting.** marin pins `iris-controller:latest` (`config/marin.yaml:33`), so a restart only re-pulls whatever `:latest` currently is. To deploy a merged controller fix you must first rebuild the image (`gh workflow run "Ops - Docker Images"`, or wait for the Sunday build) and *then* restart — restarting against a stale `:latest` ships nothing. Confirm the controller is running the `:<git-short-hash>` you expect, not just that it came back up. Skipping the rebuild cost ~5 red-canary days (`.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md`).

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
iris job stop /user/job-name            # kill job + children
iris job summary /user/job-name         # per-task state, exit, duration, peak memory
iris job summary /user/job-name --json  # same, machine-readable
iris job bug-report /user/job-name      # structured diagnostic dump
```

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

## SQL Queries

The controller exposes its SQLite DB via RPC:

```bash
iris query "SELECT state, count(*) FROM jobs GROUP BY state"
iris query "SELECT state, count(*) FROM tasks GROUP BY state" -f json
```

**Never modify the controller database** without explicit user approval — read-only queries only, even on offline checkpoints.

State codes: 1=PENDING, 2=BUILDING, 3=RUNNING, 4=SUCCEEDED, 5=FAILED, 6=KILLED, 7=WORKER_FAILED, 8=UNSCHEDULABLE, 9=ASSIGNED (tasks only), 10=PREEMPTED (tasks only).

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

Controller audit events (`event=<kind> action=<action> entity=<id> ...`) are
emitted as structured `logger.info` lines — query them through
`iris process logs` with a substring filter, not via SQL. Example:

```bash
iris process logs --since 24h | grep 'event=worker_failed'
```

Full table list: `iris query "SELECT name FROM sqlite_master WHERE type='table'"`.

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
- `iris.profile` — per-capture profile blobs (cpu/memory/thread, periodic or on-demand), keyed by `source` so the dashboard's per-source list query prunes via parquet row-group min/max. Filter on `source` (a task path like `/user/job/.../<index>`, `/system/worker/<id>`, or `/system/controller`) and `type` (`cpu`/`memory`/`thread`). `format` is the blob encoding — periodic CPU captures are py-spy **speedscope** JSON. `vm_id` is the writer VM (worker id, `controller-self`, or `k8s/<node-or-pod>`).

Retention is finelog segment-based. Target for `iris.profile` is 7 days.

Get a profile for a task — open the dashboard task page and use the "Profile history" panel; rows are CPU captures from the worker's 10-minute periodic loop plus any on-demand captures, click to download. To capture on demand, hit the "Profile now" button on the task page, the worker page (`/system/worker/<id>`), or the controller status page (`/system/controller`).

Profiles are written by the worker (periodic CPU + on-demand all types), by `K8sTaskProvider` (on-demand only), and by the controller for `/system/controller` self-captures.

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
iris login                            # authenticate, store JWT locally
iris rpc controller list-users        # active users with task/job counts
iris user budget list                 # per-user budget limits
iris key create --name ci-bot         # create API key
iris key list / iris key revoke       # manage API keys
```

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
| Task retrying | `iris job bug-report /user/job` — full attempt history with per-attempt errors. |
| Task failed with exit 137 / suspected OOM | `iris job summary /user/job` — per-task peak memory + exit code. If most shards peak near the container memory limit, raise `--memory` on resubmit. |
| Dashboard unreachable | Verify tunnel is alive. `curl -sf http://localhost:10000/health`. |

## Known Bugs

1. **Committed resource leak** (`transitions.py`): `_decommit_worker_resources()` can miss certain task termination paths, leaving stale committed resources on workers. Symptom: workers show high committed CPU/memory/TPU with zero active tasks. Detect by joining `workers` against active tasks in `task_attempts`.

2. **Worker-failure thread stall on gcloud subprocess** (#3678): The reaper thread calls `notify_worker_failed` -> `scale_down` -> `terminate` which runs a synchronous `gcloud compute tpus tpu-vm delete`. If the gcloud API hangs, worker removals queue up. Symptoms: tasks stuck in ASSIGNED (9), stale `last_heartbeat_ms`. Diagnose with `py-spy dump` — look for `subprocess.run` -> `terminate` on the reaper thread. Kill the stuck gcloud process to unblock.

---

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
