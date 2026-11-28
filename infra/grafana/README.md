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
GET /finelog/{cluster}/v1/{node,training,runs,rl,accelerator,jobs}/overview
                                                    bounded shared dashboard datasets
GET /finelog/{cluster}/v1/zephyr/overview        bounded ranked shuffle snapshot
GET /finelog/{cluster}/v1/rl/recent              bounded recent RL runs
GET /finelog/{cluster}/v1/rl/execution           selected job's executions
GET /finelog/{cluster}/v1/rl/gpu                 node-GPU detail for ranges up to 7h
GET /finelog/marin/fleet_health                  main query probe + k8s mirror readiness
GET /finelog/marin/relay_status                  direct regional relay heartbeats
GET /finelog/marin/alerts/fleet_health           alert rows: server labels + value(0|1)
GET /finelog/marin/alerts/relay_status           stale relay/table rows + value(0|1)
GET /finelog/marin/alerts/training_stalls        active jobs + stalled-progress value(0|1)
GET /finelog/marin/alerts/loss_spikes            active hero runs + loss-spike value(0|1)
GET /finelog/marin/alerts/training_telemetry     watched hero runs + silent-telemetry value(0|1)
GET /finelog/marin/alerts/training_optimizer     watched hero runs + optimizer-fault value(0|1)
GET /finelog/marin/alerts/training_health        watched hero runs + degraded-signal value(0|1)
GET /finelog/marin/alerts/zephyr_stalls          active pipelines + stalled-progress value(0|1)
GET /finelog/marin/alerts/b200_utilization       GB200 clusters + state + idle value(0|1)
GET /iris/{cluster}/job_counts | jobs | workers | health
                                                    live controller RPCs
GET /iris/{cluster}/query?sql=                    ad-hoc SELECT (admin/null-auth)
GET /github/ferries | builds | nightlies          GitHub REST / GraphQL
GET /wandb/report/{train-loss,paloma-macro-loss,mfu}
                                                    public report runset and sampled history
GET /wandb/history?run=&metric=&project=          one run's whole logged history for one metric
GET /wandb/activity?run=&project=                 one run's active, wall, and downtime seconds, and projected finish
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

The inference pages open on five minutes and refresh every two minutes. Ranges of
seven hours or less fetch the selected series' samples with one bridge query.
That query scans the vLLM and MarinSkyRL namespaces once each. The bridge
computes all panels locally with DuckDB and shares the result across both pages.
Every dashboard panel requests the same 15-second bucket;
the bridge raises it as needed to cap each trend at 360 points. The fetch retains
a three-minute counter lookback and rejects
more than one million samples or 50,000 series. Local calculation uses one thread,
a 512 MB memory limit, no disk spill, and a lock because Grafana and the bridge
share one CPU. Dense short ranges may still hit the sample limit.

Longer ranges, up to seven days, fetch only the signals used by the summary
with the same two bounded namespace scans. The bridge aggregates these samples
locally. The summary
shows reset-aware token and outcome totals, an hourly generated-token timeline,
observed waiting and KV-cache peaks, and native ITL from coherent counter pairs.
The same sample and memory limits apply. It omits latency
tails, output-length distribution, per-engine detail, and client or proxy time.
Zoom to seven hours or less for those server details. Run triage separates
query health (detail, summary-only, empty, timeout, or sample limit) from
selected-range run attention. When first-token observations outnumber
`request_success_total` finishes, it shows both counts and points to the
evaluator in Iris. TTFT is counted when a first token appears; the finish
counter changes later. Separate vLLM failure and timeout counters are not in
this comparison; inspect the outcome panel for them.
An in-progress or partial window can have a gap without a failed request. A
missing counter or a query failure gives no run conclusion. The cue does not
identify client or proxy timeouts. Other panels point to query health when
they have no data. Missing historical ITL stays unavailable rather than being
inferred from TPOT.

The identity picker retains request-state discovery. For ranges longer than
seven hours, it scans the first seven hours to stay within Finelog's deadline.
Paste an exact job, run, or execution ID into **Manual serve ID** when the
desired session began later. The diagnostics page links the selected or manual
serve to Iris; follow its evaluator child to inspect client results, retries,
and proxy logs. vLLM counters do not measure that time.

To replay both pages against the existing four-node data in
[Marin #8929](https://github.com/marin-community/marin/issues/8929), run from the
repository root (the replay uses the repository's Finelog connection helpers):

```bash
PYTHONPATH=infra/grafana/src uv run python infra/grafana/benchmarks/inference_query_cost.py \
  --identity snowball-throughput-20260905-four-node \
  --from-ms 1788576120000 --to-ms 1788576150000 \
  --output /tmp/inference-query-after
```

The replay requests the selector, sends each page's panels in parallel, then
repeats both pages with a warm cache. It saves SQL, Arrow inputs, responses,
upstream call counts and durations, and page latency. `--refresh-after 120` also
waits two minutes and advances the window. To compare the original draft, extract
`infra/grafana/{src,dashboards}` from commit
`4ace1ba5eb91cd184dec17c704ec7aa5962a9ac3` with `git archive`, point `PYTHONPATH` at
that extracted `src/`, and pass its parent as `--grafana-dir`. Keep the identity,
window, and request sequence the same, and use a separate output directory.
These are read-only live queries; repeat only the windows needed for comparison.

The Node Details, Zephyr, Training, Runs, RL Post-training, Accelerators, Jobs,
and Home dashboards use the same shared-dataset contract. Each endpoint validates every
identity and time input, loads the source group needed by the requested view,
and projects related panel views locally. Concurrent requests share the group
cache and identical source queries share their Arrow inputs across groups.
Runs separates metrics from power attribution; Jobs separates its five namespaces;
RL separates phase, Ray, engine, and execution-detail dependencies. Its GPU
utilization chart has a narrow source; the other GPU views share detailed input.
Training already separates metrics, attempts, task state, and retry events.
Node and Zephyr retain one shared source because their views consume the same
bounded input.

Accelerator power/GPU/node tiles read one compact latest-power aggregate. Power
and utilization charts share a compact history query; fleet, device, attribution,
and fault-table views load their own groups. Home therefore does not fetch device
rasters, training attribution, provisioning, or task/worker resource charts.
The full Accelerator page also requests the compact history alongside its broader
fleet charts; this adds a narrow scan but keeps Home responses small.
Device histories use packed Arrow series, preserving every point and enforcing
the sample limit before local expansion.

`src/dashboard_dataset.py` owns only cache-independent execution rules: declared
row/sample caps, one-thread DuckDB projection, a 512 MB memory cap, and no disk
spill. The domain modules own metric names, source SQL, windows, cardinality caps,
and counter/latest-state semantics. Fleet accelerator raster data is additionally
limited to 60 time slices; aggregate charts retain up to 100. A budget breach is a
400 asking the operator to narrow the filters or window, never an unbounded retry.
Selectors remain independent `/query` requests because discovery has different
indexing and window requirements from panel calculation.

For a cross-dashboard cost comparison, use
`benchmarks/dashboard_query_cost.py` with an existing Finelog address. Supply a
JSON object mapping dashboard filenames to selector names and values, such as
`{"training.json": {"run": "selected-run"}}`, then run from the repository root:

```bash
PYTHONPATH=infra/grafana/src:infra/grafana uv run python -m benchmarks.dashboard_query_cost \
  --grafana-dir infra/grafana --dashboard training.json \
  --address http://127.0.0.1:19001 --to-ms <settled-end-ms> \
  --variables variables.json --output <new-evidence-directory> --live-window
```

Repeat `--dashboard` or omit it to inspect every dashboard. Each iteration uses
empty bridge caches, dashboard time defaults, visible panels, and panel overrides;
`--warm` also records reuse. The harness records SQL, Arrow inputs, responses,
elapsed time, actual SQL calls, and namespace-discovery calls. It excludes
selectors, collapsed rows, external APIs, and browser rendering. `--live-window`
holds the bridge wall clock near the fixed data window and adds millisecond skew
to relative overrides; query timing remains real. For baseline comparisons,
point both `PYTHONPATH` and `--grafana-dir` at the same archived Grafana revision.

`fleet_health` reads one row from `finelog-marin`'s `log` namespace and combines that
result with the three CoreWeave mirror Deployments' HTTP-readiness state. A hub query
at or above 5 seconds is slow. Clusters' finelog row adds effective pod
resources, restart history, probe presence, node placement, PVC class/capacity, and
recent matching Kubernetes Warning events.

`relay_status` comes from a complete snapshot each regional Finelog sends directly to
the hub every 30 seconds. It does not travel through the row-forwarding path. The alert
fires when a regional heartbeat is two minutes old, the required node-agent namespace
is absent, or a nonzero publication/forwarding lag has made no progress for ten minutes.

`NodeRuntimeFilesystemAlmostFull` pages critical ops when a node's `/` or
separately mounted `/tmp` stays below 5% free space for five minutes. It reads
the node-agent `node_filesystem_available_bytes` and `node_filesystem_total_bytes`
gauges, preserving cluster, node UID, and mountpoint identity. Only the latest
samples from the last five minutes participate. Missing data stays normal while
node agents are upgraded; Finelog fleet and relay alerts cover query outages.
Deploy the updated Iris node agents along with Grafana. The existing NVMe
`node_disk_*` metrics do not measure the root RAM disk; see the
[root-filesystem runbook](../../lib/iris/OPS.md#coreweave-root-filesystem-full).

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
work. The same request's `_step` and `run_progress`, with the first point of a
sampled history, give the projected finish (see the training dashboard below). Without
an explicit `project` the bridge searches `RUN_HISTORY_PROJECTS` in order and fails
with a 404 when no project holds the run.

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
expected failures are cached as message-only errors without traceback or response
objects, so retries still show an error rather than blank data; `iris/.../health`
is the exception — it returns `reachable=false` so the panel can render the outage.

Pod scans process one API page at a time. The control-plane crashloop alert
queries only the watched deployments using their label selectors; broad pod
scans remain for workload, termination, and architecture views. Malformed JSON
is reported as an HTTP upstream failure without retaining its response body.

Dashboard dataset loading and projection allow at most two computations at a
time, including vLLM views. Waiting requests do not load Arrow inputs. The shared
Arrow source cache has a 128 MiB referenced-buffer budget; this is separate from
active query and projection memory. Expected cached errors retain only their
message and HTTP classification. Grafana's Cloud Run container has 4 GiB.

## Layout

See [Layout](docs/dashboards.md#layout).

## Adding a dashboard

See [Adding a dashboard](docs/dashboards.md#adding-a-dashboard).

## Sharing a panel across dashboards

See [Sharing a panel across dashboards](docs/dashboards.md#sharing-a-panel-across-dashboards).

## Alerting

See [Alerting](docs/alerting.md#alerting).

## Secrets and rotation

See [Secrets and rotation](docs/operations.md#secrets-and-rotation).

## Develop

See [Develop](docs/operations.md#develop).

## Deploy

See [Deploy](docs/operations.md#deploy).

