# Finelog Pulumi deployment

This project builds Finelog and deploys one server to each active Kubernetes
cluster. Each `Pulumi.<cluster>.yaml` stack loads the matching configuration from
`lib/finelog/config/<cluster>.yaml`. The reusable resource component lives in
`iac.kubernetes.finelog`.

Pulumi owns the image, Deployment, Service, and, for persistent caches, the
PersistentVolumeClaim. The `<config.name>-env` Kubernetes Secret (for example,
`finelog-cw-use02a-env`) stays outside Pulumi state. Create or rotate it with
`finelog deploy sync-secret` before updating a stack.

## Update a server

Authenticate to GHCR and select the target cluster explicitly:

```bash
uv sync --all-packages --extra deploy
pulumi login gs://marin-iac-state
export KUBECONFIG=~/.kube/coreweave-iris
export R2_KEY_ID=...
export R2_KEY_SECRET=...
docker login ghcr.io

uv run finelog deploy sync-secret <cluster>
uv run marin-deploy finelog rollout <cluster>
```

`marin-deploy finelog rollout` captures the active Kubernetes Deployment revision before
running `pulumi up` for the matching stack. Pulumi builds
`lib/finelog/deploy/Dockerfile`, pushes the configured image tag, rolls the
Deployment to the returned digest, and verifies that the new server accepts
writes. If the update or verification fails after changing the Deployment, the
wrapper restores the captured ReplicaSet, verifies it, and refreshes Pulumi
state to match the restored workload. Pass `--yes` to skip Pulumi's confirmation.

ReplicaSet rollback assumes the prior binary can read every remote table state
the candidate published. It cannot cross an object-state format change. In
particular, current object references omit the former `sha256` field; a binary
that requires that field fails recovery after the current server publishes a
table revision. Before registering the first object-native table on a cluster,
treat the candidate as roll-forward-only. If verification later fails, deploy a
corrected descendant against the same remote root. Do not use
`finelog rollback` to restore a SHA-validating image.

The Deployment uses `Recreate` because the Finelog store permits only one
writer. It retains ten ReplicaSets for rollback. The stack derives its rollout
identity from the checkout's content-addressed Git tree SHA and stamps the tree
SHA, base commit, and dirty status into the image. Run the deploy from the
intended checkout; there is no rollout counter in Pulumi configuration.

`deployment.k8s.cache_storage` selects the cache lifetime:

- `persistent-volume` is the default. Pulumi creates a protected PVC, or adopts
  the existing claim named by `cache_pvc_name`.
- `node-local` mounts an `emptyDir` backed by the node's ephemeral disk. Pulumi
  requests and limits ephemeral storage to `storage_gb`, and normally creates
  no PVC.
  It survives container restarts but not pod replacement. All bundled regional
  senders use this mode because object storage is their durable spool and the
  hub becomes the read source for rows it settles.

## Forwarding relay lifecycle

A deployment with `forwarding` configured runs relay maintenance. It preserves
the table schema sent to the hub, but does not compact or build local query
indexes, projections, placement rewrites, or encoding rewrites. Unsettled L0
objects are the durable spool.

The downstream cursor is the logical deletion boundary. Each forwarding turn
publishes the cursor and removes whole segments at or below the minimum cursor
for all configured targets in one table-state mutation. A retryable forwarding
failure does not advance the cursor, so it cannot delete an unforwarded segment.
Settled segments leave the live relay snapshot immediately; pinned queries and
rollback-visible table states retain their immutable objects until the exact
release deadline expires. Unknown orphan uploads use a separate grace period.

The cache storage selects write acknowledgement durability. A node-local cache
acknowledges object-backed rows only after their objects and table-state HEAD are
published remotely, so deleting the pod cannot lose an acknowledged tail. A
persistent-volume cache acknowledges after the local commit and publishes the
same state asynchronously. Object uploads use a separate bounded async pool and
do not occupy Parquet flush capacity. SIGTERM does no special forwarding drain;
the replacement resumes from the published state and downstream cursor.

The first transition from a version-0 node-local store needs an operator drain
because version-0 tables have no object-state acknowledgement boundary. Before
replacing it, confirm every namespace's forwarding cursor has reached its
persisted high water. Then shorten local retention and let legacy archive
maintenance upload `LOCAL` segments and evict only `BOTH` segments. Register
the version-1 table specifications after the archive-only history is evicted:
migration excludes `REMOTE` rows and rewrites only the small local tail. Wait
for every table to reach `RETIRED` before treating object state as the recovery
authority. The old
flat archive objects may remain until a separate inventory-backed cleanup.

With `persistent-volume`, set `deployment.k8s.cache_pvc_name` to adopt and mount
an existing replacement claim. Enable the stack's `import` option when Pulumi
first adopts that claim.

Changing an existing Pulumi-managed `persistent-volume` stack directly to
`node-local` is intentionally not automatic: Pulumi refuses to remove the
protected claim. Decide whether to recover or retain that data, then explicitly
remove the claim from Pulumi state before updating the Deployment. The bundled
regional stacks were adopted in `node-local` mode, so their prior claims never
entered the new stack state.

Do not mount an object store as the cache filesystem. Finelog's SQLite catalog
and active segments require local filesystem semantics; object storage belongs
behind `remote_log_dir` until the server owns that storage path natively.

For a read-only preview, run `pulumi preview --stack <cluster>` from
`infra/finelog`. Running `pulumi up` directly bypasses the wrapper's automatic
rollback.

After rotating the environment Secret without changing the image or resource
configuration, replace the pod so the new process reads the updated values,
then run the same ingest-health check used by Pulumi updates:

```bash
# Use `kube_context`, `namespace`, and `name` from lib/finelog/config/<cluster>.yaml.
kubectl --context "<kube-context>" -n "<namespace>" delete pod -l app="<deployment-name>"
kubectl --context "<kube-context>" -n "<namespace>" rollout status deployment/<deployment-name>
uv run --frozen --package marin-finelog finelog deploy verify <cluster>
```

To restore the next older retained ReplicaSet, run:

```bash
uv run marin-deploy finelog rollback <cluster>
```

Pass `--to-revision N` to select an exact revision shown by `kubectl rollout
history deployment/<name>`. The command waits for the exact revision created by
the rollback, verifies ingest health, and refreshes Pulumi state. If the target
fails verification, it restores and verifies the revision that was serving when
the command started. This rolls back the Pod template and image only. It does
not restore PVC contents, a discarded node-local cache, or an older value of
the out-of-band environment Secret. A later deploy from a newer checkout rolls
forward again.

## Adopt an existing server

Bootstrap the empty stack with the shared KMS provider, then run one import pass:

```bash
pulumi stack init <cluster> \
  --secrets-provider="gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
pulumi config set finelog:cluster <cluster>
pulumi config set finelog:import true
pulumi preview
pulumi up
pulumi config rm finelog:import
```

The import pass always adopts the existing Deployment and Service. A
`persistent-volume` deployment also adopts and protects its existing PVC;
inspect the preview before applying to ensure that claim imports without
replacement. A `node-local` deployment does not adopt an old PVC.

The stack does not own the `iris-system` PriorityClass. The cluster substrate
stack in `infra/pulumi` remains its canonical owner.
