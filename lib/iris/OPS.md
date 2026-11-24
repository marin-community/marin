# Iris Operations

Use `iris <command> --help` for flags. Examples omit the connection selector:
prefer `--cluster=NAME`, use `--config=PATH` for an exact config, or
`--controller-url=URL` for an explicit endpoint. `iris cluster list` lists names.
Cluster coordinates and storage paths come from [config/](config/README.md).

Start read-only. Cluster/controller restarts, workload changes, node reboots,
and direct Kubernetes mutations require authorization for the target. Never
modify the controller database to repair state. Avoid `kubectl describe pod`
on task Pods: it can print credentials in environment values.

This file is an entry point and command reference. Keep incident measurements,
host identities, and chronology in [Echo](https://marina.oa.dev/echo/); use
[use-echo](../../.agents/skills/use-echo/SKILL.md) to find or update a record.

| Topic | Reference |
| --- | --- |
| Controller deployment gates | [Controller rollout](../../.agents/skills/use-iris/references/controller-rollout.md) |
| CoreWeave access, storage, networking, node recovery | [CoreWeave](docs/coreweave.md) |
| GCE ingress and authentication | [IAP and GCLB](docs/iap-gclb.md) |
| Credentials delivered to tasks | [Secret delivery](docs/secrets.md) |
| Sandbox isolation and egress | [Container profiles](docs/container-profiles.md) |
| Federation and shared budgets | [Federation](docs/federation.md) |
| Task results and retries | [Task states](docs/task-states.md), [output archives](docs/task-outputs.md) |
| Logs, telemetry, SQL semantics | [Finelog operations](../finelog/OPS.md), [query-finelog](../../.agents/skills/query-finelog/SKILL.md) |

## Authentication (headless / CI)

IAP-fronted clusters use their HTTPS endpoint directly; other GCE clusters open
an SSH tunnel automatically. CoreWeave uses a context-bound Kubernetes port-forward.

Interactive callers use `iris --cluster=NAME login`. Headless callers use ADC
for an IAP-allowlisted service account; do not start a browser login. An
authorized caller can impersonate the account:

```bash
gcloud auth application-default login --impersonate-service-account=<service-account>
iris --cluster=NAME cluster status
```

Impersonation needs `roles/iam.serviceAccountTokenCreator`; the account needs
`roles/iap.httpsResourceAccessor` on the backend. Allowlisted GCE service
accounts use ambient metadata credentials. See [IAP access control](docs/iap-gclb.md#controller-access-control).

## Cluster Lifecycle

```bash
iris cluster status
iris cluster dashboard                # print URL and keep the connection open
iris cluster dashboard-proxy          # local proxy
```

### Controller Restart

Follow the [rollout procedure](../../.agents/skills/use-iris/references/controller-rollout.md).
A controller restart deploys the current working tree, including uncommitted
files, and leaves workers running. A full `iris cluster restart` destroys
workers and jobs; never run it without express approval.

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=<name> cluster controller restart
```

Use the `controller` extra for Kubernetes preflight dependencies. Verify the
deployed tree/image hash and health against the baseline. Keep multi-architecture
task images for mixed fleets; see [image deployment](docs/image-push.md).
Preflight resolves controller secrets and principal references before changing
the deployment; see [cluster configuration](config/README.md).

If checkpointing times out, inspect logs and thread stacks before retrying:
an RPC timeout does not cancel the server-side backup. `--skip-checkpoint`
risks losing newer state on restore. `controller serve --dry-run` starts a
local inspection server; it is not a deployment validation command.

### Rolling back a controller deploy (migration-aware)

With rollback authorization, `cluster controller restart --rollback` uses the
rollout record to restore both the previous image and its pre-deploy checkpoint.
An image-only rollback can leave an incompatible migrated schema. Jobs created
after the checkpoint are lost. See the [rollout procedure](../../.agents/skills/use-iris/references/controller-rollout.md).

### Controller Checkpoint Rollback (wedged / OOM recovery)

An unreachable controller may need [manual checkpoint recovery](docs/controller-recovery.md).
Restarting alone reuses the local database. Preserve the existing database and
obtain recovery authorization before restoring older state. For Kubernetes,
check [controller storage and placement](docs/coreweave.md#controller-state).

## Job Management

```bash
iris job run -- python train.py
iris job list --state running
iris job describe /user/job
iris job logs /user/job -f
iris job cancel /user/job              # exact job and children
iris job cancel --prefix /user/prefix
iris job complete /user/job            # records unfinished descendants as successful
```

| Resource | Inspection | Actions |
| --- | --- | --- |
| Job | `list`, `describe`, `logs`, `wait` | `run`, `cancel`, `complete` |
| Task | `list`, `describe`, `events`, `logs`, `wait` | `exec`, `profile`, `preempt`, `fail` |
| Attempt | `describe`, `events`, `logs`, `wait` | `profile`, `preempt`, `fail` |

Task actions target the current attempt. Attempt actions require
`/user/job/task:attempt` and reject stale attempts. Workload logs share filters;
`process logs` is for controller, worker, and runtime diagnostics.

Logs default to the last 1,000 lines. Search older output server-side:

```bash
iris job logs /user/job --max-lines 400000 --no-tail --substring 'Saving checkpoint'
```

Tag each root job you submit with what it is for: `--project <effort>`,
`--category <kind>` and `--description "<what it does>"` (SDK:
`IrisClient.submit(tags=job_pb2.JobTags(...))`). Suggested projects and
categories are in "Iris Job Tags" in the root `AGENTS.md`. On a child job, each
tag it leaves unset comes from the parent. The tags go to Finelog `iris.controller_db__jobs.tags`.
A controller older than job tags accepts a tagged submission and drops the
tags, so the job runs untagged; the job and its command are otherwise unchanged.
Iris options go before the command: from its first word on, every argument,
`--project` included, belongs to the command, with or without `--`.

Kubernetes rejects `EXISTING_JOB_POLICY_RECREATE` for an active job. Cancel it
and verify its Pods are gone before submitting the replacement; a terminal job
state alone does not prove process death.

The `old-iris-jobs` Loom watch lists root jobs active for more than a week in a
Google Sheet, with owner and the resources their active tasks hold, and cancels
the jobs that owners mark `Kill` there (see the
[watch description](../../infra/loom/README.md#scheduled-watches)). To print the
rows of the next run without a change to the sheet or to a job:

```bash
uv run python scripts/ops/old_iris_jobs.py --min-age-days 7 --cluster marin \
  --sheet-id <sheet-id> --impersonate <service-account-with-sheet-access> --dry-run
```

The `marin` hub covers the jobs it federates. For jobs submitted directly to a
CoreWeave controller, use `--cluster <cw-cluster>`; this needs CoreWeave
kubeconfig access.

### Submitting a GPU gang from a workstation

Submit through the federation hub if your CoreWeave token cannot port-forward:

```bash
iris --cluster=marin job run --no-wait --target-cluster=<cw-cluster> \
  --cpu 2 --memory 2GB --disk 32GB --timeout 7200 --job-name my-coordinator \
  -e IRIS_PORT_JAX 32731 -- python -m experiments.<launcher>
```

Size this CPU coordinator separately from the accelerator child jobs launched
through Fray. Use a unique `IRIS_PORT_JAX` per concurrent gang. The timeout
includes queueing, and stopping the coordinator stops its children.
`SchedulingGated` means Kueue has not admitted the gang. See [CoreWeave](docs/coreweave.md).

### Sandbox prepared-image workdirs

The Kubernetes backend advertises `sandbox-image-workdir` as a task capability.
SANDBOX pods without a bundle, inline workdir files, or workdir file references
preserve the prepared image's `/app`; an empty workdir volume would hide those
files. Explicitly staged inputs and non-sandbox jobs retain their workdir mounts.
Harbor requires this capability before submitting prepared-image tasks. Older
controllers need a separately authorized rollout before submitting prepared-image tasks.

### Sandbox disk limits

Kubernetes sandbox pod manifests set CPU and memory container limits. A nonzero
`ResourceSpec.disk` also sets the pod storage request and limit in exact bytes;
omitting disk sets neither. Ordinary jobs retain their existing rounded storage
requests and limits. Sandbox tasks mount no writable hostPath. Kubelet accounts
for runtime-reported writable layers, logs and disk-backed emptyDir volumes;
read-only image layers are excluded. Eviction is asynchronous.
`RESOURCE_LIMIT_DISK` in the task error and attempt terminal reason identifies a
newly launched sandbox pod evicted for exceeding its own storage limit. This
charges the failure retry budget. Ordinary pods retain their existing eviction retry behavior.

Disk accounting depends on the container runtime. gVisor writable-layer accounting
has not been qualified with a sandbox overflow run, so Kubernetes backends do not
advertise a `sandbox-storage-limit` capability.

Inspect the pod's status reason/message as well as the container termination:
exit 137 and the word `ephemeral-storage` alone do not establish a task limit.

### `job run` gotchas

- The current working tree is bundled, including uncommitted code.
- Use `--memory`, not `--ram`; quote both values in `-e KEY "VALUE"`.
- `--gpu` requests hardware; `--extra gpu` selects Python dependencies.
- Select accelerators with `--gpu`/`--tpu`; use region/zone only for placement.
  `--reserve` constrains placement to obtained capacity; it does not attach devices.
- A `uv sync` failure in BUILDING is setup failure. Use `--sync-package` to scope
  setup or `--no-sync` for a prepared image; see [Task Setup](AGENTS.md#task-setup).
- Supply credentials through [secret delivery](docs/secrets.md). CoreWeave tasks
  need the submitter's storage keys; operator keys stay on infrastructure Pods.
- `executor_main` coordinators must remain CPU-only so they do not hold the
  accelerator capacity their children need. Memory at or above 4 GB needs
  `--enable-extra-resources`.

## Task Operations

```bash
iris task describe /user/job/0
iris task events /user/job/0
iris attempt events /user/job/0:2
iris task exec /user/job/0 --timeout 300 -- python -c 'import jax; print(jax.devices())'
```

`task events` is the first stop when a pod or Kubernetes Event has already been
garbage-collected. It queries `iris.task_event` across every retained attempt in
the task's current job incarnation in one call and shows both backend
observations (`k8s/kueue`, `k8s/container`) and controller decisions
(`iris/controller`) in chronological order. Events are retained for up to 30
days. Kubernetes `DisruptionTarget` conditions are recorded as
`k8s/disruption` events with the pod and node identity plus a snapshot of the
node's taints, conditions, hardware/topology labels, and CoreWeave health
annotations.

For an evaluator that retries after a `Completed` container event and a later
`PodDeleted` worker-failure verdict, inspect its durable result archive before
rerunning or accepting the retry. The evaluator may already have written a
complete, sealed result even though its parent lacks a canonical record. See
[Auditing a curated run](../../docs/tutorials/run-lm-evals.md#audit-a-curated-run-before-its-record-appears)
for the archive checks. The Iris verdict alone does not establish whether the
application result is complete.

Default timeout is 60s. Use `--timeout 300` for slow commands, `--timeout -1` for no timeout (last resort).

The exec session is non-interactive and buffers output. To run a command that survives disconnect, wrap with `nohup` + `&`:

```bash
iris task exec /user/job/0 -- bash -c "nohup bash -c 'your-command > /tmp/out.log 2>&1' &"
iris task exec /user/job/0 -- cat /tmp/out.log   # check later
```

### Task with no logs or W&B progress

Check `task describe`, `task events`, and `job logs` first. PENDING and BUILDING
tasks may have no training process; a RUNNING process may still be compiling or
blocked on memory. Record the attempt, exit reason, and resource request before
changing anything.

### Log filtering

The dashboard full-log filter accepts regex and searches beyond loaded lines.
The find box searches only browser-loaded lines. Fix invalid regex before
interpreting empty results.

### Acting on a wedged Task or Attempt

With workload authorization:

```bash
iris task preempt /user/job/0           # retries within preemption budget
iris task fail /user/job/0             # terminal, no retry
iris attempt preempt /user/job/0:3      # only if attempt 3 remains current
```

Actions apply on the next control tick and accept ASSIGNED, BUILDING, or RUNNING
tasks. A successful RPC means queued; confirm the resulting state. Federation
forwards the action to the execution peer. `job complete` records success;
`job cancel` records KILLED. See [bulk actions](#bulk-actions-query--act).

### Recovering a stuck terminating Kubernetes pod

Follow [stuck-pod recovery](../../.agents/skills/use-iris/references/stuck-pod.md).
Never force-delete the API object before process death is proven: a replacement
may otherwise run while the old process still owns the GPU. Preserve the node's
pre-existing cordon and quiesce sibling workloads before a reboot.

## Process Inspection & Profiling

```bash
iris process status
iris process logs --level WARNING --substring '<entity-or-error>' --max-lines 200
iris process profile threads
iris process profile cpu -d 10
iris process profile mem
iris process profile threads -t /user/job/0
```

Prefer RPC profiling over SSH. If RPCs hang while `/health` works, inspect
controller thread stacks through the configured Kubernetes context. Repeated
`BundleStore.write` or `_content_lock` stacks suggest upload/lock contention.

For GPU hangs, compare JAX compile fingerprints and collect native thread
profiles for divergence inside XLA. Keep normal NCCL logging at WARN; enable
INFO with a focused subsystem only for diagnostic jobs. See the
[training-stall contract](../../docs/ops/training-stall-alert-contract.md#nccl-ras-snapshots)
for persisted communicator snapshots. A bounded, read-only check from a RUNNING
task is:

```bash
iris task exec /user/job/0 --timeout 15 -- \
  python -m rigging.telemetry.probes.nccl_client --timeout 8
```

## Scheduler & Autoscaler

```bash
iris rpc controller get-scheduler-state
iris rpc controller get-autoscaler-status
iris rpc controller get-provider-status
iris cluster vm status
```

See [priority bands](docs/priority-bands.md). Scheduler `running_buckets` and
separate SQL reads are different snapshots; use one atomic query to diagnose
assignment leaks. For Kubernetes capacity use `rpc controller list-backends`;
there is no Iris worker daemon or Iris autoscaler.

## SQL Queries

The controller query RPC is read-only and admin-only. Discover tables before
assuming schema; keep expensive analysis off the live controller:

```bash
iris query "SELECT name FROM sqlite_master WHERE type='table'"
iris query 'SELECT state, count(*) FROM tasks GROUP BY state' -f csv
```

### Sharp edges

- Active resource holders include ASSIGNED (9), BUILDING (2), and RUNNING (3).
- `workers.committed_*` measures allocations; capacity is in `metadata_proto`.
- Decode the serialized protobuf in `jobs.request_proto` to inspect job constraints.
- Use [task states](docs/task-states.md) and [schema.py](src/iris/cluster/controller/schema.py)
  for state codes and columns.

### Useful queries

```sql
SELECT job_id, error, exit_code FROM jobs WHERE state=5
ORDER BY submitted_at_ms DESC LIMIT 10;

SELECT task_id, attempt_id, state, exit_code, error FROM task_attempts
WHERE task_id LIKE '/user/job/%' ORDER BY attempt_id;

SELECT name, consecutive_failures, quota_reason FROM scaling_groups
WHERE consecutive_failures > 0 OR quota_reason != '';
```

Read audit events with `process logs`. Filter server-side
with `--substring='event=worker_failed'` or an entity ID. `process logs` has no
`--since` flag; a grep over the default tail can miss older evidence.

### Bulk actions: query → act

Task preemption/failure and job cancellation accept `--stdin` and `--dry-run`.
CSV headers and extra columns are ignored. Select exact IDs and preview before
an authorized action:

```bash
SEL="SELECT t.task_id FROM tasks t JOIN workers w ON t.current_worker_id=w.worker_id
     WHERE w.slice_id='<slice-id>' AND t.state IN (2,3,9)"
iris query -f csv "$SEL" | iris task preempt --stdin --dry-run
iris query -f csv "$SEL" | iris task preempt --stdin --reason 'authorized slice drain'
```

### Offline checkpoint analysis

Use an existing checkpoint from `storage.remote_state_dir` for slow analysis.
Take a new one with `iris cluster controller checkpoint` only when needed.
Restore/download details are in [checkpoint recovery](docs/controller-recovery.md).
Keep offline databases read-only too.

## Stats Namespaces

Measurements live in Finelog, separate from controller decisions. Discover
schemas and bound the time window before querying; see [query-finelog](../../.agents/skills/query-finelog/SKILL.md).

| Namespace | Meaning |
| --- | --- |
| `iris.worker` | Host utilization per tick, keyed by `ts` |
| `iris.task` | Per-attempt resource samples; Kubernetes CPU needs consecutive samples, memory peaks reset with the agent, and disk usage is unavailable |
| `iris.task_event` | Backend verdicts and controller actions, retained up to 30 days; use `task events` first |
| `iris.task_state` | Task counts and wait ages per root job every 30 seconds; empty `root_job_id` is the cluster rollup, emitted even when idle |
| `iris.controller_db__jobs`, `__job_config`, `__tasks`, `__task_attempts` | Durable copies of the controller DB tables of the same names; keys, `cluster` values, which columns are copied, joins, their cost to the controller and how to turn them off (`finelog.controller_rows: false`) are in [observability rows](docs/observability-rows.md) |
| `iris.admission_probe` | Kubernetes admission canary outcomes every 60 seconds; failure or silence can identify webhook outages |
| `iris.profile` | CPU/memory/thread captures; filter by `source`, `type`, and time; retention target is seven days |
| `telemetry_v1.node_agent` | Physical-node gauges, including runtime-filesystem capacity |

Disruption events identify the machine with `node_system_uuid`/provider ID and
the reboot with `node_boot_id`. Group nearby events from the same machine and
boot before counting separate hardware failures.

Workers capture periodic CPU profiles; Kubernetes captures periodic thread
dumps. The dashboard's Profile history and Profile now controls retrieve or
create captures. For a job-wide CPU breakdown:

```bash
uv run python lib/iris/scripts/job_profile_summary.py /user/job
uv run python lib/iris/scripts/job_profile_summary.py /user/job -o merged.folded --svg flame.svg
```

## Users & Auth

```bash
iris rpc controller list-users
iris user budget list
```

IAP supplies user identity; controller config supplies roles. See
[IAP access](docs/iap-gclb.md#controller-access-control), [cluster auth config](config/README.md),
and [shared budgets](docs/federation.md#shared-budgets). Never print bearer tokens.

### Worker credential renewal

Workers renew JWTs before expiry and cache them at
`<cache_dir>/credentials/worker.jwt` with mode `0600`. If task state advances
but logs report UNAUTHENTICATED, compare endpoint discovery and direct Finelog
access using expiry metadata only. Expired bootstrap and cached credentials
need an operator-issued replacement; preserve running containers when restarting
the worker/node agent. Deploy controller renewal support before its callers.

### Calling the IAP endpoint with `curl`

Use `rigging.credentials.iap_edge_provider(<cluster>).get_token()` with
`Proxy-Authorization`; reserve `Authorization` for Iris JWTs. Endpoint names
replace `/` with `.` in proxy URLs: `/system/log-server` becomes
`/proxy/system.log-server/health`. See [IAP ingress](docs/iap-gclb.md).

## Troubleshooting

| Symptom | First check |
| --- | --- |
| PENDING job | Scheduler constraints, autoscaler demand/backoff/quota |
| GCP workers absent | `cluster vm status`, then worker bootstrap logs |
| Repeated retries | Task events, attempt exit reason, application result archive |
| Exit 137 | Container memory limit and peak; confirm OOM before resubmitting |
| RPC timeout with healthy `/health` | Controller thread profile |
| Dashboard unavailable | Configured IAP route or Kubernetes port-forward |
| Instant exit 255 / exec format error | [Image architecture mismatch](#image-architecture-mismatch) |

### Image architecture mismatch

`ArchMismatchImageExecuted` evidence `message` confirms an exec-format error;
`signature` is only an instant-exit heuristic. Check the registry's platform
index before removing any working node cache. See
[image diagnostics](docs/image-push.md#image-architecture-mismatch).

## GCP (TPU) Operations

### Connecting

Prefer named configs (`marin`, `marin-dev`); use the configured project/zone
for direct GCP access. IAP-fronted controllers do not need SSH tunnels.

### GCP Resources

Use the cluster's `iris-<label_prefix>-controller` and
`iris-<label_prefix>-managed` labels to find controller and worker instances.
`gcloud compute tpus tpu-vm list --zone=<zone>` lists TPU VMs.

### TPU Bad-Node Recovery

For `No accelerator found`, `FAILED_PRECONDITION`, or `Device or resource busy`,
map the worker IP to the exact TPU VM and verify a host fault first. With recovery
authorization, replace only that node and resubmit the affected work. Escalate
simultaneous or repeated failures; see [infrastructure troubleshooting](../../.agents/skills/use-iris/references/troubleshooting.md).

### GCP State

Use `storage.remote_state_dir` for bundles and controller checkpoints.
Finelog segments use its separately configured `remote_log_dir`.

### GCP Gotchas

Quota failures back off per scale group. Check autoscaler status before adding
capacity. Inspect DELETING TPU VMs through GCP before retrying provisioning.

## CoreWeave (GPU) Operations

Use the kubeconfig, context, and namespace from the named config; an ambient
`KUBECONFIG` can override the configured path. See
[access and read-only status](docs/coreweave.md#access-and-read-only-status)
for credential-safe Pod/Kueue/NodePool projections and override diagnosis.

```bash
iris --cluster=<name> cluster status
iris --cluster=<name> rpc controller list-backends
```

### CPU topology admission

IaC gives each CoreWeave cluster's physical CPU pool a hostname-only TAS
admission path: the `iris-cpu` ResourceFlavor and `iris-cpu-hostname` Topology,
placed first in `iris-cq` with zero GPU/RDMA quota. CoreWeave's
`node-metadata-controller.compute.coreweave.com` owns the fabric,
superpod and leafgroup labels; their live values on CPU Nodes are empty.
Kueue v0.18.3 accepts empty topology label values;
missing keys exclude a Node. Do not patch those provider-owned labels.

`provisioning.coreweave.kueue.cpu_scale_group` names the CPU pool per cluster:
`cpu-erapids` on 08a, `cpu-genoa` on 02a and `cpu-turin` on rno2a. That cluster's
CPU NodePool stops declaring the four synthetic physical
labels; capacity, instance type and taints are unchanged.

With approval for a cluster, review a targeted Pulumi preview and apply
only these four resources: `topology-cpu`, `resource-flavor-cpu`,
`cluster-queue`, and that cluster's CPU NodePool: `nodepool-cw-use08a-cpu-erapids`
on 08a, `nodepool-cw-use02a-cpu-genoa` on 02a, `nodepool-cw-rno2a-cpu-turin`
on rno2a. Require no replacement,
deletion, capacity, Helm-release or GPU-flavor change. The ClusterQueue update
can admit existing queued work and changes the preferred pool for new CPU-only
admissions. It does not rewrite admitted Workloads. This procedure requires no
Iris or Kueue controller restart.

From the repository root, with stack KMS access and the configured kubeconfig:

```bash
cw08_kueue_urn='urn:pulumi:cw-us-east-08a::marin-iac::marin:coreweave:KueueAddon$kubernetes:kueue.x-k8s.io/v1beta1:'
cw08_cpu_pool_urn='urn:pulumi:cw-us-east-08a::marin-iac::marin:coreweave:CoreweaveCluster$kubernetes:compute.coreweave.com/v1alpha1:NodePool::nodepool-cw-use08a-cpu-erapids'
cw08_targets=(
  --target "${cw08_kueue_urn}Topology::topology-cpu"
  --target "${cw08_kueue_urn}ResourceFlavor::resource-flavor-cpu"
  --target "${cw08_kueue_urn}ClusterQueue::cluster-queue"
  --target "$cw08_cpu_pool_urn"
)
PULUMI_PYTHON_CMD="$PWD/.venv/bin/python" KUBECONFIG="$HOME/.kube/coreweave-iris" \
  pulumi -C infra/pulumi preview --stack cw-us-east-08a "${cw08_targets[@]}"
```

After approval of that exact preview, run the same command with `up` in place
of `preview`. Do not use an untargeted stack update: other changes may be pending.
The stack name and NodePool URN change per cluster; the four targets do not.

Roll out one cluster at a time, starting with 08a, and apply the next cluster
only after the previous cluster's admission canary passes. rno2a has a single
CPU node, so check its capacity headroom before applying there.

After applying, confirm `iris-cpu` references `iris-cpu-hostname`, that this
Topology has only the hostname level, that `iris-cq` is Active, and that GPU
and RDMA quotas on `cw-tas` retain their prior values. Keep
`TASHandleOverlappingFlavors` enabled: both topologies end at hostname, allowing
Kueue to account for transient overlap during rollout. Verify a separately
authorized CPU SANDBOX canary receives an `iris-cpu` assignment, starts its
process and exits successfully; admission alone does not verify the runtime.

### CoreWeave sandbox RuntimeClass RBAC

Before deploying this Iris revision to any CoreWeave cluster, apply the
Pulumi-owned controller ClusterRole rule granting `get` on
`runtimeclasses.node.k8s.io`. This includes clusters using the default `gvisor`
class. `IrisRbac` in `infra/pulumi` owns the ClusterRole; the controller rollout
does not update it. For each stack, review a targeted preview containing only
this rule, then replace `preview` with `up` after approval:

```bash
stack=cw-us-east-08a # repeat for each CoreWeave stack before its controller rollout
role_urn="urn:pulumi:${stack}::marin-iac::marin:coreweave:IrisRbac\$kubernetes:rbac.authorization.k8s.io/v1:ClusterRole::cluster-role"
PULUMI_PYTHON_CMD="$PWD/.venv/bin/python" KUBECONFIG="$HOME/.kube/coreweave-iris" \
  pulumi -C infra/pulumi preview --stack "$stack" --target "$role_urn"
```

The deployment preflight reads the RuntimeClass before creating the controller
Deployment. A 403 on this GET raises a `KubectlError` with the API status and
reason; only a 404 is reported as a missing class.

### Public LoadBalancer reachability

A working port-forward does not prove federation ingress works. Test controller,
in-cluster ingress, and external reachability separately; see the
[LoadBalancer runbook](docs/coreweave.md#public-loadbalancer-reachability).

### CoreWeave root filesystem full

`NodeRuntimeFilesystemAlmostFull` reports `/` or separately mounted `/tmp`
below 5% free for five minutes. It needs deployed node agents and Grafana;
the NVMe `node_disk_*` gauge alone does not cover root. Missing/stale samples
do not establish pressure; see [alert provisioning](../../infra/grafana/README.md).

If exec reports `write /tmp/runc-process…: no space left on device`, check host
root bytes/inodes even when Ready and DiskPressure look normal. Inspect
`/var/log/runsc`; use a node-debug container's main command when exec cannot
start. See [runtime storage](docs/coreweave.md#runtime-storage) for automatic
NVMe log placement and diagnosis. Task/Shellbox `/tmp` uses private Kubernetes
storage, separate from host `/tmp`. Historical evidence is in [Echo](https://marina.oa.dev/echo/wiki/947).

### CoreWeave kernel deadlock pending reboot

Check `KernelDeadlock`, `PendingPhaseState`, `CWActive`, and bound workloads.
Preserve provider cordons and follow the [node-reboot runbook](docs/coreweave.md#coreweave-kernel-deadlock-pending-reboot);
do not overlap stateful controllers or force-delete live GPU processes.

### Rolling back the egress allowlist

Changing defaults affects new jobs; stored policies and existing Pods have
separate lifetimes. Follow the [allowlist rollback order](docs/coreweave.md#rolling-back-the-egress-allowlist)
and [container profile semantics](docs/container-profiles.md).

## CI Workflows

| Workflow | Purpose |
| --- | --- |
| `marin-canary-ferry.yaml` | TPU canary on GCP |
| `marin-canary-ferry-coreweave.yaml` | GPU canary on the shared `iris-ci` fleet |
| `iris-smoke-coreweave.yaml` | CoreWeave integration; shares the CI controller and capacity with the GPU canary |
| `ops-docker-images.yaml` | Rebuild and publish pinned Iris images |

Inspect workflow source before dispatching or retrying: CoreWeave CI resets
workers and redeploys its shared controller. Those actions need the same target
authorization as manual operations. Read failures with `gh run view <id> --log-failed`.
