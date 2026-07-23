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
export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
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
