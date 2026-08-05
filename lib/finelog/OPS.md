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
`RegisterTable` away and does not need a reset, but the sidecar backfill runs a
few segments per namespace per 30 s tick, so a large namespace speeds up over
tens of minutes rather than at once. A time bound is the faster answer in the moment: `telemetry_v1` is keyed
on `timestamp_ms` and a 10-minute window answers in about a second where the same
query unbounded takes 30.

`finelog query` applies a client deadline just past the server's own 60s one.
Raise both with `--timeout` and `FINELOG_QUERY_TIMEOUT_MS` if a query genuinely
needs longer.

Row groups are sized to hold a fixed number of *bytes*, so a namespace of narrow
rows gets far fewer of them than one of wide log lines. This only applies to
segments written since the change: existing segments keep the row groups they
were written with, and their footers shrink as compaction and eviction turn them
over. `EXPLAIN ANALYZE` reports `row_groups_pruned_statistics` as
`<total> total`, which is the count for the segments a query touched.

`query_metadata_cache_mb` in a deployment config overrides DataFusion's
process-wide Parquet metadata cache limit. Leave it unset to retain DataFusion's
default. Finelog logs the effective limit at query-engine startup; every slow
query warning also includes `metadata_cache_limit_bytes`,
`metadata_cache_size_bytes`, `metadata_cache_entries`, and
`metadata_cache_hits`. Compare warm-query latency and those fields before
retaining or increasing an override.

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

Warnings name the affected namespace. `backlog exceeds the lag cap` or `rows
evicted before they were forwarded` means that namespace skipped source sequence
positions; the cumulative `skipped_seqs` progress counter alone does not prove
that `log` rows were dropped.

To rotate a key, add the new Secret Manager version, add its public key alongside
the old one under the same `keys[].cluster` (the hub accepts either), roll the
hub, re-pin the sender's `signing_key` to the new version, roll the sender, then
drop the old public key and roll the hub again.

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
