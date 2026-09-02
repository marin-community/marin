# grafana

The Marin infra dashboard, as an IAP-gated Cloud Run service: Grafana plus a bridge
that fronts five sources for its Infinity datasource — finelog SQL, the live Iris
controller, the GitHub API, public W&B report data, and the CoreWeave k8s API servers. One instance serves
both GCE clusters, reaching `finelog-marin` / `finelog-marin-dev` and each cluster's
Iris controller on their internal IPs over Direct VPC egress, and polls the public
CKS API servers of the CW clusters read-only. `marin` is the federation hub (the
CoreWeave clusters forward their rows to it), so its finelog datasource sees the
whole fleet; `marin-dev` sees only itself.

Dashboards and datasources are provisioned from the files in this directory. Grafana's
state — users, stars, preferences, alert state, and UI-created dashboards — lives in the
shared `marin-metadata` Postgres (`infra/cloudsql`), so UI edits persist across redeploys.
The provisioned dashboards under `dashboards/` are still code: change the JSON and redeploy
to update them.

## Why Cloud Run and not an Iris job

A service that monitors X should not run on X: Grafana on Iris would serve the
dashboards you need *during* an Iris incident from the thing that is down. Cloud Run
reaches the finelog and controller internal IPs over
`--vpc-egress=private-ranges-only` without living on the cluster it watches.

## The bridge

Grafana's Infinity datasource fetches JSON over loopback from the bridge, which fronts
the upstreams and returns flat JSON rows. It runs beside Grafana; backend datasources
fetch server-side, so nothing outside the container reaches it.

```
GET /finelog/{cluster}/query?sql=&from=&to=      finelog SQL
GET /finelog/marin/fleet_health                  main query probe + k8s mirror readiness
GET /finelog/marin/alerts/fleet_health           alert rows: server labels + value(0|1)
GET /finelog/marin/alerts/training_stalls        active jobs + stalled-progress value(0|1)
GET /finelog/marin/alerts/loss_spikes            active hero runs + loss-spike value(0|1)
GET /finelog/marin/alerts/training_telemetry     watched hero runs + silent-telemetry value(0|1)
GET /finelog/marin/alerts/training_optimizer     watched hero runs + optimizer-fault value(0|1)
GET /finelog/marin/alerts/training_health        watched hero runs + degraded-signal value(0|1)
GET /finelog/marin/alerts/zephyr_stalls          active pipelines + stalled-progress value(0|1)
GET /iris/{cluster}/job_counts | jobs | workers | health
                                                    live controller RPCs
GET /iris/{cluster}/query?sql=                    ad-hoc SELECT (admin/null-auth)
GET /github/ferries | builds | nightlies          GitHub REST / GraphQL
GET /wandb/report/{train-loss,paloma-macro-loss,mfu}
                                                    public report runset and sampled history
GET /wandb/history?run=&metric=&project=          one run's whole logged history for one metric
GET /wandb/activity?run=&project=                 one run's active, wall, and downtime seconds
GET /k8s/control_plane | crashloops | pending     CW control-plane state, all clusters
GET /k8s/termination_candidates | kueue | events | health
                                                    ... one response, `cluster` column
GET /k8s/workloads                                live Iris task placement and requested resources
GET /k8s/nodes                                    node inventory, allocatable resources, readiness, lifecycle state
GET /k8s/node_pools                               NodePool capacity, autoscaling policy, conditions
GET /k8s/finelog | finelog_events                 mirror pods/PVCs and matching warnings
GET /k8s/overview                                 explicit pending/crashloop counts
GET /k8s/gpu_racks                                GPU nodes grouped by physical rack: trays total/ready
GET /k8s/arch_mismatch                            containers killed by an exec-format failure on a non-amd64 node
GET /k8s/alerts/{unreachable,crashloops,          alert rows: string labels + one
     webhook_ready,degraded,node_deadlocks,       numeric; gpu_rack_trays omits rows for
     stuck_gpu_pods,gpu_rack_trays,               a cluster it cannot reach, others zero
     arch_mismatch}
GET /health                                       bridge liveness
```

finelog: a panel sends SQL and a window; the bridge substitutes the `{{from}}` / `{{to}}`
macros, runs it against finelog's `Query` RPC (SELECT-gated and deadline-bounded there),
turns the Arrow result into JSON, and caches per (cluster, SQL, window bucket) so a
relative range keeps one cache key as its edges drift. It calls only `Query`, avoiding the
`WriteRows` / `DropTable` a direct Grafana-to-finelog datasource would also expose.
Timestamps come back as epoch milliseconds, so a panel selects a raw or `date_bin`-ned
time column without casting. finelog has JSON SQL UDFs, so a panel groups by a label in SQL
— `json_get(labels,'region')`; the bridge also flattens a `labels` column into
`label_<key>` fields. Jobs uses the hub datasource for a bounded recent view of
`iris.task_event`, beside Clusters' live API server events.

`v1/vllm/overview` backs the inference dashboard with a bounded, entity-scoped
Finelog query. Operators select a job, root effort, or execution and a raw time
window to compare token throughput, request pressure, KV-cache use,
latency/outcomes, and worst-replica freshness after a serve exits; reset-aware
deltas preserve replica identity, and an explicit no-data row distinguishes
missing telemetry from healthy application silence.

`fleet_health` reads one row from `finelog-marin`'s `log` namespace and combines that
result with the three CoreWeave mirror Deployments' HTTP-readiness state. A hub query
at or above 5 seconds is slow. Clusters' finelog row adds effective pod
resources, restart history, probe presence, node placement, PVC class/capacity, and
recent matching Kubernetes Warning events.

Iris: the bridge owns each query behind a fixed endpoint and returns flat rows, so the
dashboard never sends raw admin SQL. `jobs` (root jobs by state — in-flight plus 24h
terminal) and `query` use the controller's `ExecuteRawQuery`; `workers` aggregates
`ListWorkers` (worker liveness is in-memory, not SQL); `health` is the controller
`/health`. These rely on the marin controller's null-auth mode — `ExecuteRawQuery` is
admin-only — so an authed controller would break `jobs` and the ad-hoc `query`.

GitHub: `ferries`, `builds`, and `nightlies` fan out over the Actions REST and GraphQL
APIs with a server-side token (the rate-limit shield), cached, panel fields precomputed.
`nightlies` fetches each configured nightly workflow (across the marin repo and the fork
repos), classifies each (lane, day) cell server-side — health, overdue, and duration state —
and serves one linked, duration-aware row per lane and UTC day. The internal panel plugin
groups those rows into the compact trailing-week matrix.

W&B: the bridge reads public W&B anonymously, so Grafana never needs a W&B key. It
serves three shapes. `/wandb/report/{chart}` follows the runset pinned in the public
hero-training report spec and samples train cross-entropy, Paloma macro loss, and MFU
against cumulative training tokens. `/wandb/history` samples one metric across one
named run, keyed on W&B's own `_step`, which is the Levanter step because Levanter
logs through `wandb.log(..., step=<training step>)`. `/wandb/activity` reads the same
run's clocks out of `summaryMetrics`, one small request rather than a history
download: `_runtime` is the seconds the training process was alive, which
`resume="allow"` restores at every restart, so it is the run's active execution time
across every attempt. Wall time runs from the run's creation to its last heartbeat,
which makes the remainder downtime. `_runtime` advances only when an attempt logs, so
the total holds still while a restart initializes rather than counting the wait as
work. Without an explicit `project` the bridge searches `RUN_HISTORY_PROJECTS` in
order and fails with a 404 when no project holds the run.

k8s: the bridge polls the three production CoreWeave clusters' public CKS API servers with plain
httpx GETs (paginated LISTs, bounded timeouts, one 429 retry) and a single org-wide CW
read-role bearer token from `CW_READ_TOKEN` — genuine read-only kubectl, no Secrets, no
writes. Each response aggregates every cluster with a `cluster` column: watched
control-plane components (a config constant: kueue-controller-manager, iris-controller,
traefik, cert-manager) with ready/desired/restarts/waiting state, admission-webhook
ready-endpoint counts from `discovery.k8s.io` EndpointSlices, backoff pods, pending and
scheduling-gated pods, the unadmitted Kueue backlog per queue, and recent Warning
events. It also reports pods still present at least two minutes after their API
deletion deadline, classified as node cleanup, finalizer cleanup, terminal cleanup,
unbound cleanup, or invalid timestamp. Those rows include the assigned node, GPU
request, canonical Iris task-attempt id from `IRIS_TASK_ID`, priority class, and
finalizers. The pod-level scans skip provider-managed namespaces (`cw-*`, `kube-*`):
CoreWeave's per-node daemons are thousands of pods of someone else's infrastructure,
while the namespaces we operate hold about a hundred. These are current-state reads —
the bridge stores no history; trends come from the finelog-backed rows.

`nodes` reports Kubernetes readiness and schedulability together with CoreWeave's
node-pool, GPU, rack/slot, InfiniBand, driver, and lifecycle labels. It retains
`node.coreweave.cloud/cordonReason`, `KernelDeadlock`, and `PendingPhaseState`.
`CoreWeaveNodeKernelDeadlock` pages when `KernelDeadlock=True` persists for five
minutes. The alert labels carry the node and structured condition reason; the
dashboard retains the condition message and pending lifecycle phase for diagnosis.
The route accepts comma-separated `cluster` and `node` filters after reading its
cached fleet snapshot.

`node_pools` reads the cluster-scoped `compute.coreweave.com/v1alpha1` NodePool
objects. Each row includes current/target/min/max nodes, in-progress/queued/prefill
work, autoscaling and scale-down policy, and the `Validated`, `AtTarget`, `Capacity`,
`Quota`, and `NodeReconfigurationRequired` conditions. `missing_nodes` is the
positive target-current gap; `problems` includes failed validation/capacity/quota
conditions and pending node reconfiguration. Normal scaling leaves `problems`
empty even while `AtTarget=False`. This is current API state; the bridge retains no
NodePool history.

`gpu_racks` lists every GB200 NVL72 node (`nvidia.com/gpu` capacity present and
`node.kubernetes.io/instance-type` containing `gb200`), grouped by its CoreWeave
`node.coreweave.cloud/rack` label, with the rack's full name
(`ds.coreweave.com/physical-topology.rack-name`), instance type, and how many of
its trays are registered vs. Ready. The instance-type filter matters: other GPU
node pools carry a CoreWeave rack label too, but not the 18-node shared-rack
topology the 16/18 thresholds assume — `cw-us-east-02a`'s H100 fleet
(`gd-8xh100ib-i128`) has 29 racks, 26 of them a single standalone node, and
without the filter every one read as "1 of 18 trays." A tray that never
re-registers with the k8s API — the common failure mode after hardware
maintenance — is invisible here, so a GB200 rack short of 18 trays is a floor
on what's down, not a guarantee.

The `/k8s/alerts/*` routes exist for Grafana's table-alert contract: string label
columns plus exactly one numeric column, and always at least one row per cluster — an
explicit zero when healthy — so an alert rule can never enter NoData. A cluster the
bridge cannot read becomes labeled rows (its error class: auth, network, timeout, http)
rather than an empty result: `unreachable` reports 1, the count-style routes report
zero (the unreachable rule pages instead of fabricating counts), and `webhook_ready`
reports 0 ready endpoints — which also fires the webhook rule, deliberately, since
unknown admission state is the failure class it watches. A missing `CW_READ_TOKEN`
reads as an auth failure on every cluster rather than failing the boot, which would
take Grafana down with it.

The controller and finelog IPs are resolved from GCE labels and refreshed after a
connection failure. A dead controller or GitHub returns 5xx (not empty rows) and the
failure is not cached, so a panel shows an error rather than blank data; `iris/.../health`
is the exception — it returns `reachable=false` so the panel can render the outage.

## Layout

```
src/server.py          the bridge routes (Starlette): finelog SQL, Iris, GitHub, k8s
src/finelog_source.py  finelog query over its internal IP (LogClient)
src/hero_runs.py       hero-run enrollment from Iris state and Levanter telemetry
src/hero_health.py     run-health signal scan and the telemetry/optimizer/health projections
src/iris_source.py     live controller RPCs: jobs, workers, health, federation peers, ad-hoc query
src/github_source.py   ferry runs and CI build rollup, precomputed
src/wandb_source.py    public W&B report runset, and whole-run history for one metric
src/k8s_source.py      CW k8s API reads + the per-cluster fan-out and alert rows
src/discovery.py       GCE label -> internal IP
src/config.py          cluster targets, watched components, and bridge settings
src/cache.py           TTL cache with in-flight coalescing
src/errors.py          UpstreamError -> 5xx
src/dashboard_stitch.py  resolves dashboards/*.json panelRef markers into full panel bodies
provisioning/          datasources (finelog, iris, github, k8s), dashboards, alerting
dashboards/            dashboard JSON source — reviewed like code; see "Adding a dashboard"
dashboards/panels/     panel bodies shared across dashboards, referenced by panelRef
marin-infra-panel/     internal React status page and its reusable dense views
Dockerfile             Grafana + bridge venv + pinned Infinity and internal panel plugins
entrypoint.sh          runs both; if either dies the container dies
__main__.py            Pulumi entry point — the Cloud Run service (iac.gcp.cloud_run)
Pulumi.yaml            Pulumi project, run on the shared repo venv
```

The infrastructure hierarchy is fleet → cluster → NodePool → node → GPU. Jobs and
runs cross that hierarchy: a job asks a cluster for capacity, and its tasks occupy
nodes from one or more NodePools. Dashboard titles name the view and scope instead
of repeating the Kubernetes object name.

| Scope | Dashboard title | Source file | Question | Selectors |
|---|---|---|---|---|
| Fleet | Home | `home.json` | Is anything wrong right now? | none |
| Fleet | Fleet health | `clusters.json` | Are the cluster control planes, nodes, racks, and telemetry paths healthy? | cluster |
| Fleet | Fleet accelerators | `accelerators.json` | Where is GPU power going, and are the GPUs doing work? | cluster |
| Cluster | Cluster capacity | `cluster_capacity.json` | What jobs and requests occupy one cluster and its nodes? | cluster |
| Cluster | Node pools | `node_pools.json` | Is CoreWeave capacity at target? | cluster |
| Node | Node details | `nodes.json` | What is happening on one physical GPU node? | cluster, node |
| Workload | Jobs | `jobs.json` | What is running, queued, and stuck? | cluster, job |
| Workload | Runs | `runs.json` | How is each Levanter training run doing? | cluster, run |
| Workload | Training run | `training.json` | Is one training run on track? | run |
| Workload | Inference telemetry | `inference.json` | How did one vLLM serve behave? | identity kind, serve |
| Services | Infra | `infra.json` | Are nightly runs, main CI, workers, and hero training healthy? | none |

`home.json` is provisioned as the default home dashboard
(`GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/dashboards/home.json`,
the stitcher's output path) — everyone who opens grafana.oa.dev without a
specific dashboard in mind lands here instead of Grafana's stock welcome page.
It leads with a native `alertlist` panel (every rule currently Alerting or
Pending, across every group), then a stat strip led by fleet GPU power, the
fleet pulse charts, and the control-plane and GB200 rack inventories — the
stats and inventories are shared `panelRef` fragments, so none of it drifts
independently of the dashboards those fragments also serve.

`accelerators.json` is the GPU fleet view: total watts per cluster, the same watts
attributed to the training run occupying each node, utilization against
tensor-core activity, HBM, temperature, and the hardware-fault counters
(XID, row remap, PCIe replay). The SM-utilization raster keeps one lane per GPU,
ordered by cluster, node, and device; hover identifies the device and exact value.
It fixes the query at 100 time buckets, keeping the default fleet-wide
result below the bridge's 200,000-row limit. The temperature heatmap retains the
fleet distribution without carrying per-device labels. Both read the
`iris-node-agent` telemetry stream, which each CoreWeave node's agent fills from
that cluster's `dcgm-exporter`.
TPU hosts report no power, so this dashboard covers the GPU clusters only.
Power is attributed to a run by joining the node agent's `node_name` to the
`node_name` on Levanter's resource attributes, per time bucket — the residue is
`(idle / unattributed)`, which is the number worth driving down.

`cluster_capacity.json` is the quick occupancy view for one cluster. It combines
live `/k8s/workloads` placement and scheduler requests with `/k8s/nodes`
allocatable CPU, memory, and GPU capacity. The custom panel rolls tasks up by
root job and renders each GPU node as a block-packing card, with unallocated GPU
slots left visible. Recent `iris.task` samples supply per-job CPU and working-set
memory; `iris-node-agent` telemetry supplies host CPU/memory and aggregate GPU/HBM
pressure. Scheduler requests and observed utilization are labeled separately
because neither is a substitute for the other. The default cluster is
`cw-us-east-02a`; the selector is single-valued to keep placement legible.

`nodes.json` is the selected-node view. Its live row reads `/k8s/nodes`; bounded
Finelog panels retain per-GPU utilization, SM/tensor activity, HBM, core/HBM
temperatures, board power, NVLink/PCIe throughput, inventory, and faults beside
host CPU, memory, disk, and network history. The node selector comes from recent
GPU `hardware_inventory` rows, so a node that stops reporting stays selectable
until it falls outside the dashboard window. The profiling panels depend on the
matching DCGM fields being enabled by the cluster's exporter; a missing field leaves
only that series empty.

`node_pools.json` is the live CoreWeave capacity view. Its stat strip sums current,
target, missing, and off-target counts across the selected clusters. The table keeps
each pool's scaling work, bounds, policy, conditions, and problem reasons visible.
GCE clusters have no CoreWeave NodePool objects and return no rows.

`jobs.json` reads the `iris.task_state` finelog namespace on the marin hub — one
row per active root job every 30s per cluster-view (CoreWeave) controller,
carrying waiting/running task counts and the oldest PENDING and stuck-in-BUILDING
wait ages, plus a `root_job_id=''` per-cluster rollup — grouped by the forwarded
origin `cluster`. Fleet tasks in flight and the stuck-jobs count are `panelRef`
fragments shared with `home.json`. The active panels pin a fixed two-minute window
(`timeFrom: 2m`) so a finished job — which stops emitting with no final zero row —
ages out rather than lingering as active. GCE controllers (marin, marin-dev) emit
no `iris.task_state` (their DB is directly `ExecuteRawQuery`-able), so their
job-state counts come from the live `/iris/{cluster}/job_counts` endpoint in its own row,
and their per-task resource history sits in a collapsed row below. The `$job`
selector scopes the active-jobs table and the waiting-task series; with every job
selected the latter is the fleet backlog broken out by job, and narrowed to one
job it is that job's queue over time.

`training.json` shows whether one run is on track. `runs.json` compares runs. The
single-value selector puts the newest hero run first. It uses `run_id` across
clusters and discovers runs from a fixed 12-hour window, so widening the graph
range does not turn the selector into a whole-store scan. A `run_id` supplied in
the dashboard URL remains available for older graph windows. The status strip
uses one 15-minute query over the semantic `levanter.metrics` table for ten
fields. The strip includes the two hero
alert inputs: time
since the last completed step and
train loss, plus step time, throughput, schedule progress, and token count.
Active execution and active share come from `/wandb/activity`, which makes
the strip a mixed-datasource panel. Those two totals describe the whole run, and the
eviction that keeps the step-axis loss panel on W&B bounds any finelog answer to the
retained window. On 2026-08-24 `hero-12d8b6f0-dee637` read 93.5 hours active against
105.0 hours of wall clock, an 89 percent active share, while finelog retained its
last three days. A mixed panel names each frame after its refId and prefixes every
field label with it, which is why the strip's defaults set a display name of
`${__field.name}`: a field with a display name of its own keeps the prefix off, while
a `renameByRegex` transformation cannot, because it reads the raw field name and the
prefix is added later. The strip stands ten grid rows tall because the stat layout
picks its tile grid from the aspect ratio, and twelve tiles in a shorter panel land in
one unreadable row.

The Attempts table carries the recent detail behind that total: one row per Iris
execution over a fixed seven-day window, newest first, running or not, so the top row
stays the last attempt after that attempt fails. Its job cell links to the attempt in
the Iris dashboard. That link has to come from finelog, because W&B records neither
the cluster nor the job root. It interpolates the job cell itself plus a cluster
column the table hides rather than draws. Leave the job root unencoded in SQL:
Grafana percent-encodes an interpolated data-link value, which is what makes the root
one path segment for the Iris route, and a pre-encoded path arrives there as a
literal `%2F` that `ListTasks` rejects. Hide the cluster column with
`custom.hideFrom.viz`: Grafana 13's table ignores the legacy `custom.hidden`, and a
transformation that dropped the column would take half the link with it.

The execution-health strip shows the current attempt age, Iris task counts,
task-state age, and retained retry events. Initialization age appears only before
training progress. Its orange 45-minute band matches the initialization alert.
Its red 60-minute band matches the GPU startup supervisor. The Iris values match
the current execution cluster and job root. The JAX process-zero phase heartbeat
selects one stable replica for the attempt age.
The Token drops and Router health panels show the MoE signals that the hero
monitor uses. The dashboard uses a 7% drop limit. The router limits are 5.92
entropy and 400 bias.

The two loss panels read different stores on purpose. The step-axis panel reads
W&B through `/wandb/history`, because finelog evicts telemetry segments once
`levanter.metrics` passes its storage policy and a finelog query only scans locally
resident segments. A run that outlives that window therefore has no step 0 left in
finelog, while its W&B run keeps the whole history and spans every restart under
one id. W&B samples the series and the bridge caches it for a minute, so this
panel lags the wall-clock panel and ignores the Grafana time range.

The wall-clock loss panel stays on finelog for live detail. It separates each
`execution_uid` and disconnects gaps longer than five minutes. Iris forms this
identity from the controller-minted `attempt_uid`. Thus, a new controller process
cannot join loss from a prior process with the same numeric task attempt.

## Alerting

Grafana unified alerting, provisioned entirely from the files under
`provisioning/alerting/` — contact points, the notification policy tree, and the rules.
File provisioning owns that tree: UI edits to provisioned alerting resources are
rejected by Grafana and would be overwritten by the files anyway. Change the YAML and
redeploy.

Critical rules notify operators immediately: an unreachable cluster or
federation peer, a crash-looping watched component, an admission webhook with no ready endpoints, a
dead production Iris controller, an unhealthy finelog hub or mirror, CoreWeave
storage above 80 percent of quota, or stalled training, a loss spike, silent
telemetry, or an unstable optimizer on an enrolled hero run.
Warning rules remain in Grafana's home alert list without sending email, Slack,
or Loom notifications: a degraded component, a GPU pod that stays node-bound and
nonterminal without finalizers for five minutes after the bridge's two-minute
overdue threshold, and a GB200 rack with fewer than 16 trays Ready for five
minutes (the NVL72 rack spec is 18; a floor rather than an outright outage —
see `gpu_racks` above). The hero training rule selects fresh running
`iris.task_state` roots named `hero-*-coord` or `hero-*-coord-<retry>`. It derives
their run IDs and reads exact matches from structured `service=levanter` telemetry. It waits 15 minutes for
training progress or 45 minutes for initialization, then remains pending for five
minutes. Its `notification=hero-run`
route uses the critical receiver and groups notifications by logical run. It does not
require task-to-node GPU attribution. The run ID before `-coord` and the Levanter
trainer ID must match; zero eligible roots produce an explicit healthy
row. A second rule over the same enrolled roots watches their `train_loss`, firing
when the lowest loss of the last five minutes clears the mean plus six standard
deviations of the preceding 55, or when the loss stops being finite. Six sigma is
the band Levanter's skip-step optimizer rejects a step on, and reducing the recent
window to its floor is what separates a divergence from the single excursion
skip-step already absorbs. It takes the `notification=hero-run` route as well: a
hero run diverging unwatched costs more than a false page, which a silence
answers. Both hero rules share one enrolment query per cache interval.

Three more rules carry the rest of the hero on-call policy, the checks the
standalone Pushover monitor applies — see
[the run-health contract](../../docs/ops/hero-run-health-alerts.md).
`TrainingTelemetryGone` and `TrainingOptimizerUnstable` page on the same
`notification=hero-run` route: telemetry silent for ten minutes, labelled
`telemetry_gone` while Iris still counts the tasks and `run_down` when it no
longer does, a loss floor a whole unit above its trailing floor that the
six-sigma band did not catch, a gradient norm above 2, or three skipped steps in
fifteen minutes. `TrainingRunHealthDegraded` takes the announce-only
`notification=slack` exception for token drops, router collapse, a throughput or
MFU floor, a worse evaluation, a stale `iris.task_state` row, and retries.

These three watch a wider enrolment: a run that either the Iris rollup or fresh
Levanter `phase` telemetry reports. The stall and loss rules enroll from
`iris.task_state` alone, so a break in that path stops them watching a training
run with no signal that it happened, which is what `iris_state_stale` reports.
One `levanter.metrics` scan per cache interval feeds all three, reduced over the
newest execution process zero reports so a retry cannot mix two attempts; the
loss-jump check filters its two windows to that execution for the same reason.
`TrainingProgressStalled` labels a silent run `telemetry_gone` and emits a zero
rather than firing beside `TrainingTelemetryGone`, so one outage stays one page.
A warning-only Zephyr rule reads fresh
`progress_time_seconds` rows from `service=zephyr` telemetry. It waits 45 minutes after a
stage start or shard completion, then remains pending for five minutes. The
execution ID separates concurrent pipelines under one root job. The stuck-pod
rule groups by node and links the cordon-first
recovery skill; terminal, unbound, and finalizer-held pods stay dashboard-only.
The CoreWeave storage-telemetry freshness warning and `TrainingRunHealthDegraded`
carry the explicit `notification=slack` exception, so they announce without
launching an ops agent.
Other workload-tier signals (gated pods, Kueue backlog, workload crashloops) are
dashboard panels rather than alert rules because they have expected benign
causes. `severity=critical` routes to `ops-critical` (email
ops@openathena.ai, and the bridge, which announces the alert in Slack and opens a
Loom triage session on that thread). `severity=warning`
matches the always-active `dashboard-only` mute timing: Grafana continues
evaluating and displaying the alert, but creates no notification. Every rule sets
`noDataState: Alerting` and `execErrState: Alerting`, and the alert endpoints return
explicit zeros when healthy, so monitoring-path failures use the same
critical or warning handling as the rule.

Federation peer reachability comes from `ListPeers` on the Marin controller.
The controller heartbeat traverses production DNS, TLS, Traefik, the source-IP
allowlist, and the Iris RPC path from Marin's static egress address. Grafana
reads the resulting state over its existing VPC path to Marin, so Grafana's
Cloud Run egress does not need admission to the CoreWeave federation ingress.

Alert state — pending (`for`) timers, notification dedup, silences — lives in the
shared `marin-metadata` Postgres with the rest of Grafana's state (see Deploy), so it
survives redeploys. `min=max=1` keeps a single alert evaluator.

Email is optional. SMTP is plain Gmail submission (`smtp.gmail.com:587`, STARTTLS),
sending as grafana@openathena.ai with an app password from Secret Manager; the app
sends mail itself, so deliverability (SPF, spam filtering) rests on the sending
account. The deploy enables SMTP only when the `marin-grafana-smtp-credentials`
secret exists — without it the service still deploys, the email receiver fails
silently, and critical alerts still reach Slack and Loom. After changing contact
points or their credentials, send a test notification to all receivers (Alerting
→ Contact points → Test) rather than trusting config presence.

Slack is deliberately *not* a Grafana receiver at all. A Slack incoming webhook
answers with a bare `ok` and never reveals the message timestamp, so an alert
Grafana posts cannot be joined by anything else — and the timestamp is exactly
what Loom needs to route the thread to the triage session. The bridge posts
instead, which also means one message per alert rather than two side by side.

Both receivers post through the bridge, so every alert lands in one channel with
one credential and one rendering, and alert text is escaped the same way either
way. They differ only in what follows: `ops-critical` opens a Loom triage run on
the thread it announced, while `ops-slack` — the fallback for malformed or
unlabeled alerts — announces and stops, because an alert carrying no severity to
route on carries no incident to triage. The tradeoff is that a fallback
notification does not reach Slack while the bridge is down; email remains an
independent path for critical alerts, and the fallback sees almost no traffic
because every alert rule sets a severity.

The Loom receiver posts to the bridge on `127.0.0.1`; it is not exposed through
Grafana or IAP. For each firing group the bridge first posts the alert to
`slack_alerts_channel` with `chat.postMessage`, keeping the returned message
timestamp. It then asks the Cloud Run metadata server for a Google-signed
identity token for `https://loom.oa.dev`, exchanges it for a short-lived `ops`
Loom token, and creates an idempotent run for `marin-community/marin` on the
`operator` channel, naming that thread. Loom routes the thread to the triage
session, so the session answers in the thread and an `@russbot` reply there
reaches it instead of launching a second session — see [Loom's
slack-trigger docs](https://github.com/marin-community/loom/blob/main/docs/slack-trigger.md).
The session link is threaded under the announcement.

Generic firing groups feed the live `Grafana operator` session on the `operator`
channel. `TrainingProgressStalled`, `TrainingLossSpike`,
`TrainingTelemetryGone`, and `TrainingOptimizerUnstable` also carry the trusted
`operator_behavior=hero` label, which selects the `operator:hero` channel. Loom
therefore keeps a separate durable coordinator for Hero while both behaviors use
the same four-session `ops` policy pool. Every nonterminal session explicitly
using `ops` counts toward that pool; an operator-launched child with no explicit
profile uses Loom's `default` profile. The extra `ops` capacity covers handoffs or
explicit same-profile delegation but is shared, not reserved per channel.
Repeated notifications for the same alert fingerprint and start time reuse the
same Loom run, and thread a
short "still firing" note under the original announcement. The Slack thread key is
the Grafana notification group key. A replacement alert instance in the same group
therefore keeps the thread even when the prior instance first resolves. Resolved
notifications create no run and are noted on that same thread.
Ordering is deliberate: Slack first, so an alert reaches people even when Loom is
unreachable, and that failure is reported into the thread instead of only into
Grafana's notification history. A Slack failure is logged and still opens the
triage session.

The Hero behavior receives static discovery and query guidance instead of a
precomputed evidence snapshot. It tells the coordinator how to follow the alert's
logical run across execution and coordinator retries, then query current
`telemetry_v1`, `iris.task_state`, `iris.task_event`, and `log` rows. The guidance
uses exact IDs and Finelog's literal `prefix` predicate, includes the deduplicated
event `count`, and asks the coordinator to compare tasks and ranks rather than
grep a fixed error-signature list. Live evidence stays out of the webhook path, so
query latency, truncation, redaction, or an unknown cluster label cannot silently
turn a partial snapshot into the operator's starting facts.

Open threads are tracked for six hours in the bridge's memory, which is sound
because Cloud Run runs this service at `min=max=1`. A revision rollover forgets
them: the next notification for a still-firing alert announces afresh, and a
resolution for an alert this revision never announced is dropped rather than
posted bare.

The Loom Pulumi stack binds the exact `marin-grafana` service-account email and
numeric subject to the `ops` profile. The Grafana stack reads the Loom URL and
profile from that stack's `workloadClients` output, so
the caller and verifier cannot drift through duplicated configuration.

## Secrets and rotation

All secrets live in Secret Manager, hand-placed, and reach the container as env
vars via the `CloudRunService` `secrets` field; the values never enter Pulumi or
git. The deploy account is fail-closed on secret creation, so the program only
references secrets — each must exist and be declared in
`infra/pulumi/src/iac/gcp/grafana.py` before the `marin` stack grants access.

Loom itself needs no secret: the bridge authenticates with the Cloud Run service
account and short-lived Google/Loom tokens, and Pulumi owns the
identity-to-profile binding. The Slack bot token is the one alerting credential.

| Env var | Secret | Feeds |
|---|---|---|
| `GITHUB_APP_PRIVATE_KEY` | `marin-grafana-github-app-private-key` | ferry/build/nightly panels |
| `GF_DATABASE_PASSWORD` | `cloudsql-grafana-password` | Grafana's Postgres state (see Deploy) |
| `CW_READ_TOKEN` | `marin-grafana-cw-read-token` | k8s source (all CW clusters) |
| `SLACK_ALERTS_BOT_TOKEN` | `marin-grafana-slack-bot-token` | the bridge's alert announcements |
| `GF_SMTP_PASSWORD` | `marin-grafana-smtp-credentials` | Grafana SMTP (email alerts, optional) |

`GF_DATABASE_PASSWORD`, `CW_READ_TOKEN`, and `SLACK_ALERTS_BOT_TOKEN` must exist
before a deploy — Cloud Run fails to start a revision that references a missing
secret. `GF_SMTP_PASSWORD` and `GITHUB_APP_PRIVATE_KEY` are optional: `__main__.py`
probes for each and wires it only when the secret exists (the GitHub App also
needs its `github_app_client_id` config). Unset, the GitHub panels deploy
unauthenticated and the build panel shows no data.

`CW_READ_TOKEN` is an org-wide CoreWeave API token minted with only the `read` role
(CKS binds it to the built-in `view` ClusterRole): read-only kubectl across every
cluster in the org, no Secrets, no writes. The built-in role omits Nodes and the
NodePool CRD, so the CoreWeave Pulumi stacks bind each exact Managed Auth username
to the read-only `marin-grafana-node-reader` role for those resources. The usernames
live under `provisioning.coreweave.grafana_observer_rbac` in each Grafana cluster
config so both tokens can retain access during a rotation.

Rotation is overlap-safe:

1. Mint a second read-role token in the CoreWeave console. Save it as a single
   line with no CR/LF; Secret Manager preserves trailing newlines.
2. Use the token against one cluster's `SelfSubjectReview` to get its
   `cwtoken-…` username. Append it to `grafana_observer_rbac.usernames` in
   `cw-us-east-02a.yaml`, `cw-us-east-08a.yaml`, and `cw-rno2a.yaml`, retaining
   the old username during the handoff.
3. Preview and update the three CoreWeave Pulumi stacks. Verify both tokens can
   list Nodes and NodePools, while pod creation, Secret reads, and impersonation
   remain denied.
4. Add the new token as a `marin-grafana-cw-read-token` version, deploy a fresh
   Grafana revision, and verify every k8s bridge route.
5. Remove the old username from the three configs and update the stacks again.
   Then disable the old secret version and revoke the old CoreWeave token.

The same Secret Manager overlap pattern applies to the Slack bot token and SMTP
password: add a version, redeploy, then retire the old credential. Write the
payload with `printf '%s'`, not `echo` — a trailing newline reaches the
`Authorization` header, though the bridge strips it defensively.

Creating the secrets:

1. CoreWeave console → API access → new token (e.g. `grafana-observer`) with only the
   `read` role, then
   `echo -n "<token>" | gcloud secrets create marin-grafana-cw-read-token --project=hai-gcp-models --data-file=-`
2. The `@russbot` bot-user token (`xoxb-…`) the bridge announces alerts with —
   the same Slack app Loom posts as, so the announcement and Loom's replies come
   from one identity. Reuse the token from Loom's `LOOM_DOTENV`
   (`LOOM_SLACK_BOT_TOKEN`; see `infra/loom`), then
   `printf '%s' "xoxb-..." | gcloud secrets create marin-grafana-slack-bot-token --project=hai-gcp-models --data-file=-`
3. Apply the `marin` GCP stack, which grants the deploy account IAM management on
   that secret and the runtime account access to it (declared in
   `infra/pulumi/src/iac/gcp/grafana.py`). Without it the Grafana deploy fails
   Cloud Run's secret-access validation, even once the value exists.
4. Confirm `slack_alerts_channel` names the channel you want and that `@russbot`
   is in it. It is `#marin-alerts` (`C0BN20081CH`); to move it, take the id from
   the channel's Copy link and `/invite @russbot` there. `pulumi up` fails naming
   the key if it is ever unset.
5. (optional, enables email) Gmail app password for grafana@openathena.ai, then
   `printf '%s' "<app-password>" | gcloud secrets create marin-grafana-smtp-credentials --project=hai-gcp-models --data-file=-`
6. Send a test notification to `ops-critical` and confirm email arrives, one alert
   message lands in the channel, and its thread gains a triage-session link. Test
   `ops-slack` too: it should post one message and no session link.

## Develop

```bash
uv run pytest                     # bridge unit tests
cd marin-infra-panel
npm ci
npm run typecheck && npm run lint && npm run test:ci && npm run build
docker build -t marin-grafana .
docker run --rm -p 3000:8080 -e PORT=8080 marin-grafana
# → http://localhost:3000 (without an IAP identity header, anonymous Viewer; panels need VPC access to finelog)
```

No alerting credentials are needed to start: both contact points are loopback
webhooks, so Grafana boots without them and the bridge answers 503 on the alert
routes until `LOOM_ALERT_URL` and the Slack settings are present.

Panels only render against the real VPC: querying needs credentials that list the
finelog VMs and a network path to them. Locally you get Grafana, the provisioned
dashboards, and a bridge that 500s on query — enough to confirm every dashboard
parses and its variables survive provisioning:

```bash
curl -s 'http://localhost:3000/api/search?type=dash-db'
curl -s http://localhost:3000/api/dashboards/uid/marin-accel
```

## Deploy

Pulumi owns the deploy: the runtime service account, Artifact Registry repo and image, the
Cloud Run service, and the IAP settings. The `marin` infrastructure stack separately owns the
runtime account's project and secret grants plus the Cloud Run and IAP grants. The service
and its image build come from the reusable `iac.gcp.cloud_run.CloudRunService` component
(`infra/pulumi`); this directory is its own Pulumi project. It runs on the shared repo venv
and shares `infra/pulumi`'s state backend.

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev  # once: let buildx push to Artifact Registry
uv run --all-packages --extra deploy marin-deploy grafana rollout
```

The deploy command loads the Cloudflare provider token from Secret Manager and
Pulumi previews the update before asking for confirmation.

Add IAP viewers through `infra/pulumi/src/iac/gcp/grafana.py`, referencing encrypted principals
from `iam_data.yaml`, and apply the `marin` stack. The Grafana stack does not own IAM grants.

Production reads the `grafana-alerts` URL and profile from the `marin-loom`
stack. Apply that stack before rolling a Grafana revision with
`marin-grafana:loom_alerts` enabled. For the first deployment in a new project,
disable Loom alerts to create the Grafana service account, apply Loom, then
enable the integration and deploy Grafana again.

The stack uses the shared `marin-iac-key` KMS secrets provider. The operator needs
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key; no passphrase is used.

The rollout builds the Dockerfile with buildx, pushes it digest-pinned to Artifact
Registry, and rolls the service to that digest. `min` and `max` instances are both 1: one
warm instance serves this internal dashboard, min 1 keeps alert evaluation warm and first
paint off a cold start, and max 1 avoids duplicate alert notifications from parallel
evaluators.

Grafana's state is the `grafana` database on the shared `marin-metadata` Cloud SQL Postgres
(`infra/cloudsql`). `__main__.py` reads the instance connection name from a
`pulumi.StackReference` to the `marin-cloudsql` stack, mounts the Cloud SQL connector socket
under `/cloudsql`, and hands the socket directory to `entrypoint.sh` as
`DATABASE_SOCKET_DIR`, which composes `GF_DATABASE_URL` from it (Grafana's host:port
settings reject the colons in a connection name). `GF_DATABASE_PASSWORD` comes from the
`cloudsql-grafana-password` secret. Prerequisite: bring up the `marin-cloudsql` stack and
create the `grafana` SQL user + its secret version (see `infra/cloudsql/README.md`) before
the rollout, or Grafana fails to reach its database.

IAP is the outer gate. Its `X-Goog-Authenticated-User-Email` header becomes a Grafana
auth-proxy account. The container's nginx listener adds a fixed `Editor` role for those
accounts, which lets every admitted person create and expire alert silences. Grafana syncs
the role on authenticated requests. Before Grafana starts, `python -m grafana_migrations`
applies pending `src/grafana_migrations/migrations/mNNNN_*.py` files and records each version
in `marin_schema_migrations`. Migration `m0001` changes memberships created as `Viewer` by
older revisions to `Editor`; Grafana's normal role synchronization rejects that change when
the organization has no `Admin`. Requests without IAP's identity header remain anonymous
`Viewer`.

Cloud Run grants `run.invoker` only to the IAP service agent, and direct Cloud Run IAP
[routes every ingress path through IAP](https://cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run).
IAP [strips client-provided `X-Goog-*` headers](https://cloud.google.com/iap/docs/signed-headers-howto)
before forwarding a request, so the presence of `X-Goog-Authenticated-User-Email` safely
distinguishes authenticated users in this deployment. To allow anonymous viewing later, grant
IAP's `roles/iap.httpsResourceAccessor` role to `allUsers`. IAP then admits ordinary requests
[without checking authentication credentials](https://cloud.google.com/iap/docs/force-login),
so they have no identity claim and remain Grafana `Viewer`. A sign-in link to
`?gcp-iap-mode=FORCE_LOGIN` lets a visitor authenticate through IAP; subsequent requests carry
the email header and become `Editor`.

The OAuth consent screen is project-level and shared across the project's IAP services. The
central IAM config admits the OpenAthena Workspace domain, Loom VM service account, and
service-specific viewers. The Cloud Run component registers the Marin desktop OAuth client as
a programmatic audience. IAP admission is independent of the Grafana organization role.

The ferry, build, and nightly panels read the GitHub API, which gates the GraphQL
build query behind auth even for public repos. The bridge authenticates as the
"Marin Ops Agent" GitHub App (`src/github_app.py`): it signs a JWT with the app's
private key, looks up its installation, and mints a read-only token scoped to the
repos the panels read (the main repo and every nightly lane repo, all under
`marin-community`), refreshing it before expiry. A static token that expired is
what blanked the build panel.

The app's client id is committed as `github_app_client_id` (not secret); the
private key is hand-placed. `__main__.py` wires the app only when both are present,
so the merge-triggered deploy never blocks. The client id is already set, so
enabling auth is one step (plus its permissions grant):

```bash
# Its grants are declared in infra/pulumi/src/iac/gcp/grafana.py, so:
gcloud secrets create marin-grafana-github-app-private-key \
  --project=hai-gcp-models --data-file=key.pem
```

Install the app on `marin-community` with access to the main repo and every
nightly lane repo (`evalchemy`, `harbor`, `MarinSkyRL`, `vllm`, `tpu-inference`),
read-only on Contents, Metadata, Commit statuses, Checks, and Actions. The minted
token is attenuated to that subset even if the app holds broader grants.

## Adding a dashboard

Drop JSON in `dashboards/` and redeploy. Panels use the Infinity datasource with
`url: /query` and an `sql` param, plus `from`/`to` set to `${__from}`/`${__to}`.
Write the window into the SQL as `{{from}}` / `{{to}}`, and bin the time axis with
`date_bin(INTERVAL '${__interval_ms} milliseconds', ts)` so Grafana sizes the
buckets to the panel — see `dashboards/jobs.json`.

Dashboards address the `finelog-marin` hub directly rather than through a
datasource variable. The hub is the fleet view: the CoreWeave clusters forward
into it and their rows carry an origin `cluster` column, while `finelog-marin-dev`
forwards nowhere and sees only itself. A hub selector on a fleet dashboard empties
every panel and silently changes what `cluster=''` means, so `finelog-marin-dev`
stays available in Explore instead.

`$cluster` selects clusters, not datasources: a multi-select custom variable
listing the clusters in `src/config.py`, filtered in SQL with
`COALESCE(NULLIF("cluster", ''), 'marin') IN (${cluster:sqlstring})`. It is
custom rather than a query so a cluster that has gone silent — exactly when you
want to select it — is still in the dropdown; a test asserts the list matches the
config. `$run`, `$job`, `$execution` and `$identity` are Infinity query variables
(`queryType: "infinity"` wrapping a normal `infinityQuery`; the first returned
field becomes both text and value). The first three filter on `${cluster:sqlstring}`
themselves, so narrowing the cluster narrows the list below it.

Four things about the data will bite you:

- **Quote `cluster`.** finelog's SQL parser rejects a bare `cluster` identifier
  anywhere but the first select-list position — `SELECT ts AS t, cluster FROM …`
  fails to parse. Qualifying or wrapping it is enough for the parser, but panels
  always write `"cluster"`, and a test rejects every other spelling so nobody has
  to remember which positions are safe.
- **Bound every telemetry query, and fold the boundary.** Write
  `timestamp_ms >= CAST(EXTRACT(EPOCH FROM {{from}}) * 1000 AS BIGINT)`, never
  `timestamp_ms >= {{from}}`, which cannot prune segments. A test asserts that no
  panel is exempt. An unbounded scan is both slow and a lie about coverage: on
  2026-08-21 the full retained `hero-12d8b6f0-dee637` loss series took 3.785s
  from a new job and still began at step 1050, because eviction had already taken
  everything before it. Whole-run history belongs to `/wandb/history`.
- **Cost tracks the selected window, not the returned rows.** Sampling after a
  window function limits the response and the Grafana render, but it does not
  lower the finelog scan. Prefer one scan with conditional aggregation to several
  tidy single-metric queries.
- **Clusters forward in bursts.** A cluster minutes behind makes the right edge of
  a fleet chart dip. That is why `accelerators.json` carries a freshness panel, and
  why the power stat reduces to the latest sample per GPU rather than a bucketed sum.

A cluster keeps one colour on every panel of every dashboard — the categorical
palette's first six slots, in the cluster order in `src/config.py`. Colour follows
the entity, not its rank, so filtering `$cluster` down to two clusters does not
repaint the survivors; a test enforces it. Grafana's semantic green/yellow/orange/red
stay reserved for thresholds and outcome states and are never an entity's colour.

## Sharing a panel across dashboards

A panel that belongs on more than one dashboard (e.g. the k8s workload-issue
tables that also appear on the `infra.json` cockpit) is a single fragment file
under `dashboards/panels/<name>.json` — the panel's full body (type, title,
description, datasource, fieldConfig, options, targets) with `id` and `gridPos`
omitted, since those two fields are the only ones that legitimately vary by
placement. Each dashboard references it with a stitch marker instead of the full
body:

```json
{ "id": 4, "gridPos": { "h": 8, "w": 24, "x": 0, "y": 7 }, "panelRef": "control_plane_components" }
```

`src/dashboard_stitch.py` resolves every `panelRef` marker into its fragment body
at image build time (Dockerfile), the same way the `marin-infra-panel` build above
resolves TSX into JS — the fragment is the reviewed source, the merged dashboard
JSON `/etc/grafana/dashboards` ships to the container is derived, not committed.
`uv run pytest` runs the same resolution before asserting datasource UIDs, filter
expressions, and stat-panel schemas, so a stitching mistake fails locally, not
after a deploy. This was the fix for `infra.json`'s k8s panels drifting from the
bridge's actual field names (`phase`/`count`/`involved_object` that the routes no
longer return) — a panel shared this way can only go stale in one place.

Deliberately not a Grafana library panel: those live in Grafana's Postgres state,
not git, and only sync through the Library Elements HTTP API — no file-based
provisioning exists for them as of Grafana 13.x. A `panelRef` fragment stays
100% file-provisioned like everything else here, at the cost of only resolving
at build time rather than being editable through the Grafana UI.
