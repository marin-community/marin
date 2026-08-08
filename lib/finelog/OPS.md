# Finelog Operations

## Access through the Iris IAP endpoint

The Iris controller exposes its finelog server as the `/system/log-server`
endpoint. For `iris.oa.dev`, the public path prefix is
`https://iris.oa.dev/proxy/system.log-server/`.

Authenticate once with the built-in Marin desktop OAuth client (it is
registered as an IAP programmatic client):

```bash
uv run iris --cluster marin login
```

The command caches a refresh token in `~/.config/marin/credentials/marin.json`.
That refresh token mints a short-lived ID token without opening the browser
again:

```bash
IAP_TOKEN="$(uv run python -c 'from rigging.credentials import iap_edge_provider; print(iap_edge_provider("marin").get_token())')"
curl --fail-with-body \
  --header "Proxy-Authorization: Bearer ${IAP_TOKEN}" \
  https://iris.oa.dev/proxy/system.log-server/health
```

IAP consumes `Proxy-Authorization`. Use `Authorization` separately if the
target Iris route also requires an Iris JWT.

The endpoint proxy replaces `/` in endpoint names with `.`. Use
`system.log-server` for `/system/log-server`; `/proxy/system/finelog` addresses
a different endpoint and does not reach finelog.

The finelog CLI uses the same cached credentials when its deployment config
sets `client_url`:

```bash
uv run finelog query marin 'SELECT * FROM "iris.profile" LIMIT 10'
```

## Diagnosing query latency

Inspect the namespace before changing its policy or resetting it. Record its row
count, bytes, segment count, key column, and policy from `ListNamespaces`; a
correct key with many segments points to a different problem than a
misconfigured key. Do not reset a shared namespace such as `telemetry_v1` without
checking which producers use it.

Use native timestamp comparisons for timestamp columns. For telemetry's epoch-millisecond column, keep the predicate numeric:

```sql
WHERE ts >= now() - INTERVAL '5 minutes'
-- telemetry_v1
WHERE timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '5 minutes') * 1000 AS BIGINT)
```

DataFusion folds `now()` to a literal and can push the resulting range into
Parquet pruning. `epoch_ms` is a column in Finelog's `log` namespace, not a
timestamp conversion function.

For a bounded query that is still slow, run `EXPLAIN ANALYZE` and compare
`row_groups_pruned_statistics`, `bytes_scanned`, `metadata_load_time`, and
`time_elapsed_opening`. High metadata/opening time with few scanned bytes means
row-group pruning worked and file metadata is the remaining cost. The
`*_eval_time` metrics are accumulated elapsed time across concurrent per-file
tasks, not CPU time, so they overlap and do not sum to wall clock — treat a large
one as a place to look, not as a measured cost.

An unbounded substring query (`col LIKE '%…%'`) prunes only when that column
carries a trigram index; otherwise it decodes the column for every row in the
namespace. `ListNamespaces` reports which columns are indexed. How much it prunes
depends on the pattern's literal runs: `%CUDA_ERROR%` only requires `CUDA` and
`ERROR`, so any row group holding both survives, where `%CUDA\_ERROR%` — or
`contains(data, 'CUDA_ERROR')` — gives the index the whole string. Escape the
underscores when you mean them literally. Adding one is a
`RegisterTable` away and does not need a reset, but the index backfill runs a
few segments per namespace per 30 s tick, so a large namespace speeds up over
tens of minutes rather than at once. Enabling a column supersedes the whole
`.fidx` policy, so every segment's bundle is rebuilt rather than extended: budget
one core across the namespace's full segment count.

An unindexed substring predicate spends its cost in the `LIKE` kernel, not in
IO. `bytes_scanned` stays small while `pushdown_rows_pruned` reaches the
namespace's row count. Read both.

For repeated equality families, declare the hot string values in
`ColumnIndex.exact_values`. Finelog stores exact source-row postings in the
segment's `.fidx` bundle. The planner attaches them for `=` and same-column
`IN`/`OR` predicates when they retain at most 25% of the segment; denser matches
keep the contiguous source scan.

Use a named `Schema.projections` entry when the recurring query also benefits
from a compact physical copy. Each projection declares one predicate and an
explicit included-column list. Covered segments substitute the narrow Parquet
file while uncovered segments use postings or source Parquet, so partial
backfill is useful. `telemetry_v1` has one `training-status` projection for the
three dashboard metric names.

Change a projection in place; do not version its name. Re-registering a name
with a different predicate or column list supersedes the registered definition:
new segments build the new one, existing segments stay queryable under the
definition they were written with (each `.fidx` section carries its own
coverage), and the backfill rebuilds them a few per tick and deletes the
superseded Parquet files. A widened copy under a second name leaves both being
built for every new segment forever.

For broad low-cardinality summaries, set `ColumnIndex.value_counts`.
Unfiltered `SELECT col, count(*) FROM table GROUP BY col` and `count(col)` then
rewrite to a `FinelogIndexAggregate` node that combines exact per-segment
summaries without opening Parquet. `EXPLAIN` shows the rewrite. It is
all-or-nothing and limited to one grouping column; filters, joins, multiple
aggregates, a per-segment column above 4,096 distinct values, or a combined
result above 16,384 values use DataFusion.
`telemetry_v1` enables this for `service`, `kind`, and `name`, while its
training-status metric names also use an exact filtered projection.

`GET /api/segments?namespace=telemetry_v1&physical=true` reports each local
segment identity and `.fidx` section directory. Use it to distinguish incomplete
backfill from a planner miss. `GET /api/server` reports corrupt bundle and
section counters; either condition is a safe scan fallback but should trigger a
local rebuild investigation. A time bound remains the fastest containment:
`telemetry_v1` is keyed on `timestamp_ms`, so bounded queries can prune before
any secondary method runs.

`finelog query` applies a client deadline just past the server's own 10s one.
Raise both with `--timeout` and `FINELOG_QUERY_TIMEOUT_MS` if a query genuinely
needs longer.

Row groups are sized to hold a fixed number of *encoded* bytes, so a namespace of
narrow rows gets far fewer of them than one of wide log lines. Encoded rather
than in-memory bytes is what matters: a telemetry row compresses to ~8 bytes
against a log line's hundreds, so an in-memory target under-sizes worst exactly
where the fix is needed.

Each segment's footer carries the layout revision it was written with. Since the
terminal level is never re-compacted, a maintenance pass re-encodes segments
still on an older revision, a couple per namespace per 30 s tick — otherwise a
namespace's bulk would keep its old row groups until eviction aged it out, which
for `telemetry_v1`'s 15 GiB is about four days and for `log`'s about eight. The
rewrite keeps the filename and preserves the rows and their order, so it costs no
remote bandwidth: the archive keys objects by basename and only uploads segments
still marked `Local`. A rewritten segment's remote copy keeps the old layout
while holding the same rows.

Watch it with the `rewrote segment layout` events, which report the before and
after byte size per segment. Confirm the era split before concluding a layout
change did or did not land — compare footer bytes for segments modified before
and after the deploy, since a whole-namespace average is dominated by whatever
has not been rewritten yet.

`EXPLAIN ANALYZE` reports `row_groups_pruned_statistics` as `<total> total`,
which is the count for the segments a query touched *after* any injected access
plan, so it doubles as the check on whether trigram pruning fired.

`query_metadata_cache_mb` in a deployment config overrides DataFusion's
process-wide Parquet metadata cache limit. Leave it unset to retain DataFusion's
default. Finelog logs the effective limit at query-engine startup; every slow
query warning also includes `metadata_cache_limit_bytes`,
`metadata_cache_size_bytes`, `metadata_cache_entries`, and
`metadata_cache_hits`. Compare warm-query latency and those fields before
retaining or increasing an override.

`query_index_cache_mb` bounds decoded `.fidx` headers and sections. Cache
entries are keyed by segment identity and section ID, charged by decoded heap,
and invalidated when backfill publishes a replacement bundle. Size it for the
active trigram, posting, and value-count working set rather than source Parquet
bytes.

`StoragePolicy` controls eviction of eligible uploaded segments from Finelog's
local cache. It is not a row-age retention guarantee and does not delete objects
from the remote archive.

## Onboarding a cluster onto the forwarding hub

`marin` is the hub: every other cluster's finelog forwards its rows there, so a
job federated to CoreWeave reads back from `iris.oa.dev`. A sender authenticates
with its own Ed25519 keypair — private half in Secret Manager, public half inline
in the hub's `jwt` auth layer.

Mint the keypair. The private half never touches the repo; the public half is not
secret and belongs in version control.

```bash
CLUSTER=cw-rno2a
openssl genpkey -algorithm ed25519 -out "/tmp/$CLUSTER.pem"
openssl pkey -in "/tmp/$CLUSTER.pem" -pubout          # -> paste into marin.yaml

gcloud secrets create "finelog-$CLUSTER-signing-key" \
  --project=hai-gcp-models --replication-policy=automatic \
  --labels=component=finelog,purpose=forwarding
gcloud secrets versions add "finelog-$CLUSTER-signing-key" \
  --project=hai-gcp-models --data-file="/tmp/$CLUSTER.pem"
shred -u "/tmp/$CLUSTER.pem"
```

Then wire both ends. In `config/marin.yaml`, add a `jwt` key entry naming the
cluster and its public key. In `config/$CLUSTER.yaml`, add `forwarding:` with the
hub, the cluster name, and the pinned secret version:

```yaml
forwarding:
  target: https://finelog.oa.dev
  cluster: cw-rno2a
  signing_key: gcp-secret://projects/748532799086/secrets/finelog-cw-rno2a-signing-key/versions/1
```

The public key in `marin.yaml` must be the public half of `forwarding.signing_key` —
that pairing is what authenticates the sender, and a wrong key is a 401 on every
push. `forwarding.cluster` is the origin name the sender stamps on every forwarded
row; keep it equal to the hub key entry's `cluster` label so reads line up.

Roll the **hub first** (a sender whose key the hub does not yet trust gets 401),
then the sender. `deploy up` resolves `signing_key` from Secret Manager on the
operator's machine and projects it into the pod's `<name>-env` Secret, so whoever
runs it needs `roles/secretmanager.secretAccessor` on that secret.

```bash
uv run finelog deploy restart marin              # hub: gcp backend, in-place
export R2_KEY_ID=... R2_KEY_SECRET=...
uv run finelog deploy up "$CLUSTER" --no-build   # sender: k8s, applies Secret + env
```

Forwarding starts at the sender's current watermark: rows already in its store
stay there and stay queryable, but they do not backfill into the hub.

Confirm the hub is receiving. The sender stamps the `cluster` column, and a row
only lands once its token verified, so a row carrying the sender's name is proof
forwarding reached the hub. Bound the scan by time — an unbounded `GROUP BY` over
the whole `log` namespace will time out. An empty `cluster` is the hub's own rows;
a sender missing from this list is a sender whose rows are not arriving.

```bash
uv run finelog query marin --format table \
  'SELECT cluster, count(*) AS rows FROM "log"
   WHERE epoch_ms > (extract(epoch from now()) * 1000 - 600000)
   GROUP BY cluster'
```

### Distinguishing missing regional logs from delayed hub forwarding

The regional Finelog is the record; the `marin` hub is an asynchronous copy. If
logs for a federated Iris task are absent from the hub, query the exact task key
on both stores before diagnosing the pod-side shipper. Iris task keys include the
attempt suffix, such as `:0`:

```bash
CLUSTER=cw-us-east-08a
KEY=/user/job/task:0

uv run finelog query marin --format table \
  "SELECT seq, epoch_ms, source, data, cluster FROM \"log\"
   WHERE key = '$KEY' AND cluster = '$CLUSTER' ORDER BY seq"
uv run finelog query "$CLUSTER" --format table \
  "SELECT seq, epoch_ms, source, data FROM \"log\"
   WHERE key = '$KEY' ORDER BY seq"
```

Interpret the pair as follows:

- Regional rows present and hub rows absent or only a prefix: forwarding is
  delayed. Repeat the exact hub query; do not treat an immediate empty result as
  loss.
- Rows absent regionally but present in `kubectl logs <pod> -c task`: inspect
  `kubectl logs <pod> -c log-shipper` and the regional Finelog ingest path.
- Rows absent from both Finelog stores and the container runtime: the task did
  not emit the expected output or its runtime logs are already unavailable.

The forwarder gives every live namespace one batch-sized turn per round and
starts another round immediately while work remains. A large telemetry backlog
therefore does not monopolize forwarding ahead of new log rows. Hub or network
failures can still delay a turn because forwarding is best effort.

Inspect the sender's forwarder messages without changing the deployment. Read
the deployment name and Kubernetes connection details from
`lib/finelog/config/$CLUSTER.yaml`:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> -n iris \
  logs deployment/<finelog-name> --since=30m --timestamps=true | \
  rg 'finelog forwarder'
```

Warnings name the affected namespace. `backlog exceeds the warning threshold`
reports pressure but does not change the forwarding cursor; the sender continues
draining every locally retained row. `rows evicted before they were forwarded`
means that local retention has already made source sequence positions unreadable.
The cumulative `skipped_seqs` progress counter also includes permanently rejected
malformed batches and does not by itself prove that `log` rows were dropped.

To rotate a key, add the new Secret Manager version, add its public key alongside
the old one under the same `keys[].cluster` (the hub accepts either), roll the
hub, re-pin the sender's `signing_key` to the new version, roll the sender, then
drop the old public key and roll the hub again.

## Checking that a server is ingesting

`/health` answers 200 whenever the process is listening, and it is also the
Kubernetes liveness, readiness, and startup probe, so it cannot fail on a
condition a restart will not clear. The body carries the verdict: `ok`, or
`degraded: <namespace>: registration failed: <reason>`.

`telemetry_v1` must be registered before it accepts a row, and the registration
is re-driven from the catalog's persisted schema on every boot. When the
binary's schema and the catalog disagree in a way no merge can reconcile (a
column type change), every write to that namespace fails until one of them
changes, across restarts.

```bash
curl -sf http://<host>:<port>/health          # ok | degraded: ...
curl -sf http://<host>:<port>/api/server | jq .ingest
```

`/api/server`'s `ingest` block names each namespace, its state, the error, when
it first failed, and how many attempts have been made since. The dashboard's
System page shows the same under **Ingest**. `deploy up`, `deploy restart`, and
`safe_deploy` gate on the body, so a deploy that wedges ingest fails and rolls
back.

## Deciding a deploy before it touches a host

A wedged registration is only visible after the image is running, and a restart
does not clear it. Both checks below run against a candidate image on your own
machine, decide the deploy, and never write to a deployment.

### Schema pre-flight

Whether an image's `log` and `telemetry_v1` schemas merge against a
deployment's catalog is a pure function of two schemas, and both are cheap to
reach. `safe_deploy preflight` reads the registered side over `ListNamespaces`
and runs it through `finelog-server check-schema` inside the image being
deployed, so the merge rules under test are the rules that ship:

```bash
uv run lib/finelog/scripts/safe_deploy.py preflight --all
uv run lib/finelog/scripts/safe_deploy.py preflight marin --image <ref-or-digest>
```

Each deployment keeps its own catalog and they can disagree, so decide them
together. `rollout` runs the same check before it writes a bootstrap and
refuses on a failure. A deployment the tunnel cannot reach falls back to its
checked-in golden under `lib/finelog/deploy/registered_schemas/`, and reports
`UNKNOWN` if there is none.

`preflight::tests` re-decides every golden on each pull request that touches
the server or a golden, so a merge that would wedge production fails at PR
time. A rollout refreshes the golden it just deployed; a stale golden can only
fail a change production would have accepted.

Namespaces a client registers — `iris.worker`, zephyr's tables — are reported
as unchecked. They are not the server image's to decide.

### Shadow boot against a snapshot

The pre-flight cannot see catalog adoption, `.fidx` section format revisions,
Parquet layout revisions, or planner regressions in projection substitution.
Booting the candidate image against a copy of a real store can:

```bash
uv run finelog deploy snapshot marin /tmp/marin-store
uv run finelog deploy shadow-check /tmp/marin-store --image <digest>
```

`snapshot` copies the catalog plus the newest few segments per namespace and
their `.fidx` sidecars, round-robin so a byte budget is not spent entirely on
`telemetry_v1`; bound it with `--segments-per-namespace` and `--max-bytes`. It
reads the **local store dir** over SSH or `kubectl exec`, never the
`gs://`/`s3://` archive — that would be a cross-region read, and the archive is
not on the startup path being rehearsed anyway, since the remote reconcile is
backgrounded and never blocks the bind. A segment left behind is not a problem:
a `LOCAL` catalog row whose file is gone is dropped at boot and a `BOTH` row
collapses to `REMOTE`. The GCE path stages the archive under `/var/tmp` on the
VM, so keep `--max-bytes` inside the boot disk's headroom.

`shadow-check` asserts the store opens, every namespace in the catalog
rehydrates, the server-owned namespaces register, and every checked-in Grafana
dashboard query runs green. A query over a namespace this deployment does not
have is reported as not run rather than counted either way.

The rehearsal cannot touch what it was snapshotted from: `--mode shadow`
refuses a `gs://`/`s3://` remote or a forwarding target at startup, and never
starts maintenance, whose boot reconcile redundancy-drops covered segments and
deletes the archived objects. Use the same mode for any local benchmark over a
copied store.

## Diagnosing Kubernetes mirror readiness

Use the kubeconfig and context from `config/<cluster>.yaml`; do not rely on the
file's current context. Inspect the deployment, termination reason, probe events,
and persistent cache before changing resources:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris --context <context> -n iris \
  describe pod -l app=finelog-<cluster>
kubectl --kubeconfig ~/.kube/coreweave-iris --context <context> -n iris \
  logs deployment/finelog-<cluster> --previous --tail=300 --timestamps=true
kubectl --kubeconfig ~/.kube/coreweave-iris --context <context> -n iris \
  exec deployment/finelog-<cluster> -- cat /sys/fs/cgroup/memory.events
kubectl --kubeconfig ~/.kube/coreweave-iris --context <context> -n iris \
  exec deployment/finelog-<cluster> -- df -h /var/cache/finelog
kubectl --kubeconfig ~/.kube/coreweave-iris --context <context> -n iris \
  logs deployment/finelog-<cluster> --timestamps=true | \
  rg 'finelog (catalog sqlite ready|local segment adoption complete|namespace startup complete|store startup complete|remote reconcile complete)'
```

Exit 137 is ambiguous by itself. A nearby `Killing ... failed liveness probe`
event with zero `oom_kill` events means kubelet terminated an unresponsive
process; it was not a memory-limit OOM. Compare `memory.current` and
`memory.peak` with the configured limit, and compare cache use with the PVC
capacity before raising either. Slow `WriteRows` calls coincident with large
compactions indicate ingest pressure; tune `cpu_request`, `cpu_limit`,
`memory_request`, and `memory_limit` in the cluster's finelog config. Every
Kubernetes deployment also has a five-minute startup probe so reopening an
existing network-backed store does not feed a liveness restart loop.

The startup events carry millisecond timings for SQLite open, one-time catalog
adoption, local directory discovery, catalog reads, Parquet footer reconciliation,
batched catalog refresh, namespace rehydration, and total store open. The catalog
event also reports the effective SQLite journal and synchronous modes. Remote
reconcile runs after the listener binds and reports object listing, footer fetch,
catalog update, and delete timings separately; a slow remote phase cannot explain
pre-bind readiness delay.
