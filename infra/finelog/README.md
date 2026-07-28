# Finelog Pulumi deployment

This project builds Finelog and deploys one server to each active Kubernetes
cluster. Each `Pulumi.<cluster>.yaml` stack loads the matching configuration from
`lib/finelog/config/<cluster>.yaml`. The reusable resource component lives in
`iac.kubernetes.finelog`.

Pulumi owns the image, PersistentVolumeClaim, Deployment, and Service. The
`finelog-<cluster>-env` Kubernetes Secret stays outside Pulumi state. Create or
rotate it with `finelog deploy sync-secret` before updating a stack.

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
cd infra/finelog
pulumi stack select <cluster>
pulumi preview
pulumi up
```

`pulumi up` builds `lib/finelog/deploy/Dockerfile`, pushes the configured image
tag, and rolls the Deployment to the returned digest. The Deployment uses
`Recreate` because the Finelog store permits only one writer.

After rotating the environment Secret without changing the image or resource
configuration, increment `finelog:deploy_generation` to restart the pod:

```bash
pulumi config set finelog:deploy_generation "$(date -u +%Y%m%d%H%M%S)"
pulumi up
```

To roll back an image, check out the last known-good commit and run `pulumi up`
for the affected stack. The image build publishes that source and the
Deployment records its returned digest.

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

After adoption, use the manual `Ops - Finelog` GitHub workflow to update a
sender. The workflow refreshes the out-of-band Secret, builds and pushes the
image, and sets a new deployment generation. It remains manual so a merge cannot
race a stack's one-time import. Apply the `infra/permissions` stack once before
the first dispatch so its deployment identity can read the three Finelog signing
keys.
