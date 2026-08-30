# Finelog Pulumi deployment

This project builds Finelog and deploys one server to each active Kubernetes
cluster. Each `Pulumi.<cluster>.yaml` stack loads the matching configuration from
`lib/finelog/config/<cluster>.yaml`. The reusable resource component lives in
`iac.kubernetes.finelog`.

Pulumi owns the image, PersistentVolumeClaim, Deployment, and Service. The
`<config.name>-env` Kubernetes Secret (for example, `finelog-cw-use02a-env`)
stays outside Pulumi state. Create or rotate it with `finelog deploy sync-secret`
before updating a stack.

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

The Deployment uses `Recreate` because the Finelog store permits only one
writer. It retains ten ReplicaSets for rollback. The stack derives its rollout
identity from the checkout's content-addressed Git tree SHA and stamps the tree
SHA, base commit, and dirty status into the image. Run the deploy from the
intended checkout; there is no rollout counter in Pulumi configuration.

Set `deployment.k8s.cache_pvc_name` only when incident recovery moves a server
to an existing replacement claim. Pulumi keeps the canonical `<name>-cache` PVC
protected but mounts the named claim in the Deployment; create and verify that
claim before applying the stack. Removing the override remounts the canonical
claim on the next `Recreate` rollout.

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
the command started. This rolls back the Pod template and image only; it does
not restore PVC contents or an older value of the out-of-band environment
Secret. A later deploy from a newer checkout rolls forward again.

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

The import pass adopts the existing PVC, Deployment, and Service. Inspect the
preview before applying: the PVC must import without replacement. The component
protects the PVC after adoption, so a later `pulumi destroy` cannot delete the
cache accidentally.

The stack does not own the `iris-system` PriorityClass. The cluster substrate
stack in `infra/pulumi` remains its canonical owner.
