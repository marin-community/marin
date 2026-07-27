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

By default, Pulumi resolves the HEAD of Loom's default branch to its full commit
SHA and uses that immutable Git context as the image input. When the resolved
commit changes, preview reports an image update. `pulumi up` builds and pushes
the changed image, places the provider-produced digest in VM metadata, and waits
for `https://loom.oa.dev/api/ready` after activation.

```sh
pulumi preview --cwd /path/to/marin/infra/loom --stack marin-loom --diff
pulumi up --cwd /path/to/marin/infra/loom --stack marin-loom
curl -fsS https://loom.oa.dev/api/ready
```

Set `buildContext` to a Loom worktree to deploy local changes instead. The local
build includes tracked and untracked files allowed by that worktree's
`.dockerignore`; review its diff before deployment. Pulumi saves `-c` values in
the stack configuration, so remove the override to return to the remote HEAD.

```sh
pulumi up --cwd /path/to/marin/infra/loom --stack marin-loom \
  -c buildContext=/path/to/loom
pulumi config rm --cwd /path/to/marin/infra/loom --stack marin-loom buildContext
```

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
activation. The `grafana-alerts` federation mapping authorizes the Google
identity of the existing `marin-grafana` Cloud Run service account to select
only the `ops` profile. Pulumi resolves that account's email and immutable
numeric subject; it does not create or copy a Loom token.

The Pulumi declaration is authoritative at activation time. An unchanged
profile keeps its database revision; a changed declaration overwrites the
current row and advances the revision. UI or API edits persist only until the
next activation. Deployment pruning is enabled, so a profile or federation
removed from `Pulumi.marin-loom.yaml` is removed from new selection on the next
activation. Weaver's stock `default`, `github_comment`, and `watch` profiles are
not deployment-managed and are not pruned.

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

## VM permissions

The Loom VM service account runs interactive agent sessions. Keep its ambient GCP
permissions in `Pulumi.marin-loom.yaml` instead of adding one-off project bindings:

- `vmProjectRoles` grants named predefined or project-custom IAM roles on the
  configured GCP project.
- `vmPulumiKmsKeys` grants encrypt/decrypt access only on the listed crypto keys. This
  lets the VM read and update Pulumi stacks that use those keys as secrets providers.

These lists are additive and reviewed as code. They do not register Cloud SQL database
users or grant PostgreSQL table privileges; the owning service stack must do both.
Echo owns the `loom-vm` Cloud SQL principal, login roles, and table grants in
`infra/echo`.

The production declaration includes the shared Marin Pulumi key. An identity that
already has access must apply this change once because a stack cannot bootstrap access
to its own secrets-provider key.

The VM currently has enough access to read Echo and the shared Pulumi state. An Echo
preview additionally needs the KMS grant above. A future Echo deploy should add
mutating permissions only after reviewing its preview; the resource graph currently
requires Cloud Run, Cloud Scheduler, Cloud SQL, Artifact Registry, service-account,
project-IAM, Secret Manager IAM, and IAP IAM administration, plus access to
`cloudsql-pulumi-admin-password`. Prefer the existing project custom IAP IAM role and
secret-level access over project-wide `roles/iap.admin` or
`roles/secretmanager.admin`.

## Restart and rollback

Each Loom session supervisor runs in a separately labeled Docker container.
Recreating the control-plane service preserves those containers, and the new
control plane discovers and adopts them. Do not run `docker compose down` while
sessions are live because it removes their shared network.

To roll back, check out the prior Loom tree, restore its numbered
`dotenvSecretVersion` when necessary, and run the normal preview and update.
The persistent data disk and its scheduled snapshots are protected Pulumi
resources.
