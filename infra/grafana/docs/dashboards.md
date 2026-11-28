# Grafana dashboards

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
dashboards/<stem>/     one dashboard's single-use fragments, named by panel or section title
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
| Workload | RL Post-training | `rl_runs.json` | How is one RL run doing, sync or async? | training type, cluster, run, job, execution |
| Workload | Training run | `training.json` | Is one training run on track? | run |
| Workload | Inference overview | `inference_overview.json` | Is inference progressing, and are responses slow or queues growing? | identity kind, serve |
| Workload | Inference diagnostics | `inference.json` | Which engines, request stages, or workload changes explain the slowdown? | identity kind, serve |
| Services | Infra | `infra.json` | Are nightly runs, main CI, workers, and hero training healthy? | none |

The RL Post-training pickers and panels describe themselves on the board. MarinSkyRL documents
telemetry export at `docs/grafana-rl-runs.md`.

RL panels have a 360-point budget. Overview, GPU and execution requests pass
Grafana's interval. Execution panels use a 30-second minimum; run-wide charts use one minute.
The bridge raises either interval as needed to keep each trend within 360 points.

Execution panels require an Iris `job_id`. Runs without one show missing-job errors on the
execution panels; the run-wide panels still render.

The Run, Job and Execution pickers scan the producer rows defined in
`src/rl_producer_policy.sql`. Dashboard stitching expands that same policy into
all three picker queries. Engine counters sum clamped growth between ordered
samples, preserving origin cluster and full series identity before aggregation.
The scan projects narrow keys and hashes resource/attribute labels to limit the
window's memory. Finelog enforces a 30s query deadline; a timeout is not a partial
counter total.

Node-GPU detail uses `/v1/rl/gpu` for ranges of 7h or less. Longer ranges show
"Zoom to 7h or less for GPU detail." without querying Finelog. The other run-wide
panels use `/v1/rl/overview` and keep the selected window, up to seven days.

The two inference dashboards keep the selected identity and time range when
linked. The existing `marin-inference` UID now opens diagnostics, preserving old
links and panel IDs; `marin-inference-overview` is the entry point from Home.
Shared charts use the existing panel-fragment stitcher. Both dashboards read
the cached `/v1/vllm/overview` result. Their identity picker reads only
`num_requests_running` rows from Finelog's `session-discovery` projection,
including MarinSkyRL rows whose `metric_source` is `vllm`. These current-state
samples preserve the selected time-window semantics and include long-lived
sessions observed during the window. No new collector or metric is required.

Inference rates describe the observed engines, not upstream demand. Running
requests are admitted to scheduling, not a GPU batch size; iteration tokens mix
prefill and decode work. Histogram counts show the available observations, and
request-completion histograms exclude unfinished requests. Quantiles are bucket
upper bounds, not interpolated percentiles. Missing observations stay missing,
and freshness covers only producers that emitted records. Use RL phase timing
and the accelerator dashboards for context before diagnosing starvation or GPU
inefficiency. Older runs may lack engine-resolved histograms or publication health.

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
single-value selector puts the newest hero run first. It uses the process-zero
phase heartbeat to discover `run_id` values across clusters from a fixed 12-hour
window, so widening the graph range does not turn the selector into a
whole-store scan. A `run_id` supplied in
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

Projected finish comes from `/wandb/activity` as well, extrapolating the same run-own
accounting to the stop step: this run's steps since its first sampled one, over the wall
clock since that sample, carried forward to the step the run stops at, which `_step /
run_progress` recovers because Levanter logs progress as the global step over the stop
step. Measuring from the first sample means a run resumed from a checkpoint is not
credited with the steps it inherited, while every restart, checkpoint, and eval since
then slows the rate. The averaging window is the run's whole life under this id, the
steadiest one there is, and also the one that lags a throughput change longest; hero ids
rotate at each restore, so in practice it is the time since the last one. The wall clock
ends at the last heartbeat, so an outage in progress moves the date only once the run
resumes. On 2026-09-10 `hero-ragged_a2a-nccl2307-ep-step81k`, resumed at step 81,740 and
at 85,956 twenty-three hours later against a stop step of 390,244, projected 2026-11-18;
a finelog query over the same day agreed to within a day, and crediting the run with the
inherited steps would have put the finish three days out.

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

## Adding a dashboard

Drop JSON in `dashboards/` and redeploy. A one-off panel can use the Infinity
datasource with `url: /query`, an `sql` param, and `from`/`to` set to
`${__from}`/`${__to}`. Write the window into that SQL as `{{from}}` / `{{to}}`,
and bin the time axis with
`date_bin(INTERVAL '${__interval_ms} milliseconds', ts)`.

When two or more panels scan the same identity, namespace, and window, add a fixed
shared-dataset endpoint instead. Put the bounded source query and fixed local views
in a domain module, use `DashboardDataset` for its caps and cache key, and have each
panel pass the same query-affecting parameters plus its `view`. Do not accept SQL or
metric names from the request. Keep separate sources when namespaces or cardinality
profiles differ; Jobs deliberately uses five sources, while Node uses one. Add a
page-level test that requests every view and asserts the cold Finelog call count.

Set the same explicit `maxDataPoints` on panels sharing a dataset (360 for
Training, Node, Jobs, Runs, RL, and Inference; 100 for Accelerator). Grafana's
automatic point count depends on panel width and can otherwise send different
`${__interval_ms}` values for the same window, defeating shared computation.
Panel time-range overrides remain independent. For dataset requests ending within
60 seconds of server time, the bridge floors both boundaries to 15-second buckets
before generating SQL and cache keys. This coalesces relative overrides that
Grafana resolves milliseconds apart. It can move either boundary back by less
than 15 seconds; the existing result-cache TTL is additional. Older historical
windows and ranges shorter than 15 seconds retain exact boundaries. Original
input limits are validated before alignment. Inference uses the same alignment.
The bridge also coalesces identical
`SourceQuery` requests across datasets and retains their Arrow results for the
20-second cache TTL, capped at 128 MiB across source results. Exact SQL and time
bounds after alignment remain part of the source key.

Training metric views query only the visible metrics window. Attempt history and
execution-health views load independently, so a loss chart does not wait for Iris
task state or seven-day retry history. Attempts and execution age share one
seven-day ordered aggregate, independent of the panel's graph range. The run
picker refreshes on dashboard load; reload the dashboard to discover a new run.
Dataset completion logs include the cache key, execution-slot queue time, and
projection time (including its lock wait). `dashboard source query` logs record
actual Finelog calls rather than cache hits. Compare these before proposing
rollups: repeatedly loading an unrelated source is a query-routing problem;
repeatedly aggregating the same required historical rows is a rollup candidate.

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

The stitcher also expands `panelGroupRef` into an ordered list of panels, so a
large dashboard keeps its sections in bounded fragments. A dashboard's single-use
fragments live in a directory named after the dashboard (`dashboards/rl_runs/`
for `rl_runs.json`), with files named by panel or section title instead of
numeric ids; bodies shared across dashboards stay in `dashboards/panels/`.
Picker SQL uses the `{{rl_producer_row_names}}` token, which the stitcher
expands from the Python-defined producer names in `_SHARED_SQL` and raises on
any other `{{token}}` — Grafana's own `{{from}}`/`{{to}}` macros pass through.
Grafana receives only resolved JSON.
