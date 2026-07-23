# Loom production deployment

The `marin-loom` Pulumi stack manages `loom.oa.dev`: its GCE host, persistent
data disk, Artifact Registry repository, Secret Manager access, Cloudflare DNS,
and scheduled disk snapshots. The runtime is built from the operator's local
Loom worktree and runs as a Docker Compose application on the GCE host.

## Prerequisites

Use the shared Marin state backend and KMS provider:

```sh
pulumi login gs://marin-iac-state
pulumi stack select marin-loom --cwd /path/to/marin/infra/loom
```

The `loom-oa-dev` GitHub App must be installed on the repositories Loom serves.
Its private key, webhook secret, and client secret belong only in the
`LOOM_DOTENV` Secret Manager secret. The App callback and webhook URL use
`https://loom.oa.dev`.

Authenticate Pulumi's providers and the local Docker client:

```sh
export CLOUDFLARE_API_TOKEN="$(gcloud secrets versions access latest \
  --project=hai-gcp-models --secret=cloudflare-oa-dns-token)"
gcloud auth configure-docker us-central1-docker.pkg.dev
```

The local Docker builder must support `linux/amd64`.

## Deploy

Set `LOOM_SOURCE` to the Loom worktree to build. Pulumi builds that tree during
preview to catch image failures without pushing it. `pulumi up` rebuilds and
pushes the image, places the provider-produced digest in VM metadata, and waits
for `https://loom.oa.dev/api/ready` after activation.

```sh
cd /path/to/loom
export LOOM_SOURCE="$(git rev-parse --show-toplevel)"
pulumi preview --cwd /path/to/marin/infra/loom --stack marin-loom --diff
pulumi up --cwd /path/to/marin/infra/loom --stack marin-loom
curl -fsS https://loom.oa.dev/api/ready
```

The build includes tracked and untracked files allowed by the Loom worktree's
`.dockerignore`. Review the local diff before deployment.

Pulumi renders the Compose and Caddy configuration into VM metadata. The GCE
startup unit mounts the persistent disk, reads one numbered `LOOM_DOTENV`
version, pulls the digest-pinned image, runs `docker compose up -d`, applies the
configured Loom deployment policy, and checks readiness. It does not clone a
repository or build images on the VM.

## Update secrets

Do not put secret values in Pulumi configuration or state. Upload a reviewed
dotenv payload to Secret Manager, record the returned numeric version, and pin
that version in the stack:

```sh
gcloud secrets versions add LOOM_DOTENV \
  --project=hai-gcp-models --data-file=/path/to/reviewed.env
pulumi config set --cwd /path/to/marin/infra/loom --stack marin-loom \
  dotenvSecretVersion "$SECRET_VERSION"
```

Delete the local payload after upload. The startup script never reads `latest`,
so uploading another secret version does not change the running service.

## Automation identities

Runtime profiles and workload federation mappings live in
`Pulumi.marin-loom.yaml` and are applied through Loom's deployment API during
activation. The `grafana_alert` profile is restricted to the Google identity of
the existing `marin-grafana` Cloud Run service account. Pulumi resolves that
account's email and immutable numeric subject; it does not create or copy a Loom
token.

At runtime, the Grafana bridge gets a Google-signed ID token from the Cloud Run
metadata server, exchanges it at `/api/auth/federate`, and uses the resulting
short-lived, profile-scoped token to create the alert session. No long-lived
Loom credential belongs in the Grafana stack or Secret Manager.

Apply the Loom stack before deploying a Grafana revision that enables a new
federated caller. This ensures the identity mapping and profile exist before the
contact point begins sending alerts. The Grafana stack consumes the URL and
profile from this stack's `workloadClients` output. The `marin-grafana` service
account already exists in the production Grafana stack. In a new environment,
deploy Grafana once with `marin-grafana:loom_alerts` set to `false`, deploy Loom
to bind the new service account, then enable Loom alerts and redeploy Grafana.

## Restart and rollback

Each Loom session supervisor runs in a separately labeled Docker container.
Recreating the control-plane service preserves those containers, and the new
control plane discovers and adopts them. Do not run `docker compose down` while
sessions are live because it removes their shared network.

To roll back, check out the prior Loom tree, restore its numbered
`dotenvSecretVersion` when necessary, and run the normal preview and update.
The persistent data disk and its scheduled snapshots are protected Pulumi
resources.
