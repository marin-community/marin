# Iris Operations

All subcommands have `--help`. Use it.

Two connection modes: `--config=PATH` (auto-tunnels) or `--controller-url=URL` (manual tunnel).

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

## Job Management

```bash
iris job run -- python train.py         # submit + stream logs
iris job list --state running           # filter by state
iris job logs /user/job-name -f         # follow logs
iris job stop /user/job-name            # kill job + children
iris job summary /user/job-name         # per-task state, exit, duration, peak memory
iris job summary /user/job-name --json  # same, machine-readable
iris job bug-report /user/job-name      # structured diagnostic dump
```

### `job run` gotchas

- **`--memory` not `--ram`** — unrecognized flags silently pass through to the command string.
- **`-e KEY VALUE`** uses two positional args. If `$VALUE` is unset, the parser eats the next token. Always quote: `-e KEY "${VALUE}"`.
- **`--extra gpu`** installs CUDA jaxlib but does NOT request GPU hardware. Need both `--gpu H100x8 --extra gpu`.
- **`--reserve`** holds capacity for scheduling only — does not attach accelerator devices. Use `--tpu`/`--gpu` on the task that needs hardware.
- **`executor_main` parent jobs** (e.g., canary ferries) submit GPU sub-tasks via Fray. The parent must be CPU-only (`--cpu 1 --memory 16g`), otherwise it hogs the GPU node and deadlocks.

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

Priority bands: `PRIORITY_BAND_INTERACTIVE` (default), `PRIORITY_BAND_PRODUCTION` (can preempt interactive).

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

-- What the controller has been doing
SELECT kind, count(*) FROM txn_log GROUP BY kind ORDER BY count(*) DESC LIMIT 10;
```

Full table list: `iris query "SELECT name FROM sqlite_master WHERE type='table'"`.

### Offline checkpoint analysis

For slow queries, trigger a checkpoint and query offline. **Never run expensive queries against the live DB** — they stall the controller.

```bash
iris cluster controller checkpoint           # trigger fresh checkpoint
# Download the checkpoint file (path printed by command above)
sqlite3 /tmp/controller.sqlite3 "SELECT ..."
```

## Users & Auth

```bash
iris login                            # authenticate, store JWT locally
iris rpc controller list-users        # active users with task/job counts
iris user budget list                 # per-user budget limits
iris key create --name ci-bot         # create API key
iris key list / iris key revoke       # manage API keys
```

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

2. **Heartbeat thread stall on gcloud subprocess** (#3678): The heartbeat loop calls `notify_worker_failed` -> `scale_down` -> `terminate` which runs a synchronous `gcloud compute tpus tpu-vm delete`. If the gcloud API hangs, **all task dispatch stops** because dispatches are delivered via heartbeats. Symptoms: `dispatch_queue` growing, tasks stuck in ASSIGNED (9), stale `last_heartbeat_ms`. Diagnose with `py-spy dump` — look for `subprocess.run` -> `terminate` on the heartbeat thread. Kill the stuck gcloud process to unblock.

---

## GCP (TPU) Operations

### Connecting

```bash
# SSH tunnel (IAP)
gcloud compute ssh iris-controller-marin --zone=us-central1-a \
  --project=hai-gcp-models --tunnel-through-iap -- -L 10000:localhost:10000 -N

# Then: iris --controller-url=http://localhost:10000 ...
# Or config-based auto-tunnel: iris --config=lib/iris/examples/marin.yaml ...
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

State dir: `gs://marin-us-central2/iris/<cluster>/state/` — contains `bundles/` (code packages), `controller-state/` (SQLite checkpoints), `logs/` (Parquet).

### GCP Gotchas

- **Quota is the primary scaling bottleneck.** The autoscaler backs off exponentially per scale group. Check with `iris rpc controller get-autoscaler-status`.
- **Stuck TPU VMs.** Occasionally a TPU VM gets stuck in DELETING for days. Check: `gcloud compute tpus tpu-vm list --project=hai-gcp-models --zone=- --filter="state=DELETING"`.
- **Reservation system.** Accelerator jobs create `:reservation:` sub-jobs that hold slices. View with `iris query "SELECT * FROM reservation_claims"`.

---

## CoreWeave (GPU) Operations

### Connecting

```bash
# Port-forward (keep terminal open)
kubectl --kubeconfig ~/.kube/coreweave-iris \
  port-forward -n iris svc/iris-controller-svc 10000:10000

# Then: iris --controller-url=http://localhost:10000 ...
# Or config-based: iris --config=lib/iris/examples/coreweave.yaml ...
```

Namespaces: `iris` (main dev), `iris-ci` (persistent CI), `iris-canary` (GPU canary, ephemeral).
Configs: `coreweave.yaml`, `coreweave-ci.yaml`, `coreweave-canary.yaml`.

### KubernetesProvider vs Worker Daemons

On CW, there are **no persistent worker daemons**. The controller dispatches tasks directly as K8s pods. `list-workers` returns empty. The `workers` SQL table is empty. Use `iris rpc controller get-kubernetes-cluster-status` for pod/node status.

### kubectl Operations

```bash
kci get pods -n iris -l iris.managed=true     # task pods
kci get nodepools                             # all nodepools (cluster-scoped)
kci get events -n iris --sort-by=.lastTimestamp | tail -30
kci logs -n iris deployment/iris-controller -f        # controller logs
```

(`kci` = `kubectl --kubeconfig ~/.kube/coreweave-iris`)

### NodePool Management

```bash
# Check status (columns: TARGET, QUEUED, INPROGRESS, CURRENT, CAPACITY, QUOTA)
kci get nodepools

# Scale — do NOT use `kubectl scale --replicas`, that's the wrong field
kci patch nodepool <name> --type=merge -p '{"spec":{"targetNodes":N}}'

# Delete
kci delete nodepool <name>
```

**Stuck deletion** (autoscaler fights deletion or node mid-delivery):

```bash
kci scale deployment iris-controller -n iris --replicas=0   # stop autoscaler
kci patch nodepool <name> --type=merge -p '{"spec":{"autoscaling":false,"targetNodes":0}}'
# If still stuck (mid-delivery): remove finalizer
kci patch nodepool <name> --type=json -p '[{"op":"remove","path":"/metadata/finalizers"}]'
kci delete nodepool <name>
```

### CW Teardown

`iris cluster stop` deletes pods but **NodePools survive** (scale to zero via CW autoscaler). To avoid lingering GPU costs:

```bash
iris cluster stop
kci delete nodepool -l iris-<label_prefix>-managed=true
```

### CW Gotchas

- **NodePools survive `cluster stop`.** Delete explicitly to avoid lingering GPU costs.
- **`list-workers` returns empty.** KubernetesProvider dispatches pods directly — no persistent workers. Use `iris rpc controller get-kubernetes-cluster-status`.
- **`list-tasks` requires `job_id`.** Calling without it throws `ConnectError: job_id is required`.
- **`cluster start` always rebuilds+pushes images.** Needs `docker login ghcr.io` with `write:packages` PAT.
- **Konnectivity agent.** `kubectl port-forward` returns 500 until `konnectivity-agent` pods are running (~18-30s after node provisions).
- **`kubectl scale --replicas` is wrong for NodePools.** Use `kci patch nodepool ... '{"spec":{"targetNodes":N}}'`.

### GPU-canary pod stuck Pending, `NotTriggerScaleUp: 2 max node group size reached`

- Check **account-wide** H100 contention, not just `iris-canary`: `kci get nodepools -A`. If `iris-ci-h100-8x` (or any other pool) is already holding the zone's H100 quota at `maxNodes=1`, the canary's `iris-canary-h100-8x` cannot scale up — CW account caps total H100 in US-WEST-04A.
- Workaround: reuse the CI nodepool (point `coreweave-canary.yaml` `h100-8x` selector at `iris-iris-ci-managed=true` and coordinate with iris-ci) or scale `iris-ci-h100-8x` to 0 before the canary runs.
- Root fix: CW support ticket to raise `gd-8xh100ib-i128` account quota ≥2 in US-WEST-04A.

---

## CI Workflows

| Workflow | Trigger | What |
|----------|---------|------|
| `marin-canary-ferry.yaml` | Daily 6AM UTC | TPU canary on GCP (`marin-dev.yaml`) |
| `marin-canary-ferry-cw.yaml` | Daily 10AM UTC | GPU canary on CW — shares `iris-ci` controller + H100 nodepool with `iris-coreweave-ci.yaml` (concurrency group `iris-coreweave-ci-shared`) |
| `iris-cloud-smoke-gcp.yaml` | PRs touching `lib/iris/` | GCP smoke test (ephemeral cluster) |
| `iris-coreweave-ci.yaml` | PRs touching `lib/iris/` | CW integration tests (warm cluster) |

```bash
# Trigger manually
gh workflow run "<workflow name>" -R marin-community/marin --ref main
# View failed run
gh run view <run-id> -R marin-community/marin --log-failed | tail -50
```

## Cold-Start Timings

| Resource | Time |
|----------|------|
| CW CPU node | ~14 min |
| CW H100 bare-metal | ~20 min |
| CW first training step (from zero) | ~25-30 min |
