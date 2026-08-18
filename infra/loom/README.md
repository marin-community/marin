# Loom production deployment

The `marin-loom` Pulumi stack manages `loom.oa.dev`: its GCE host with a
retained Hyperdisk root disk, Artifact Registry repository, Secret Manager
access, and Cloudflare DNS. The production host is an on-demand C4D VM with an
AMD Turin CPU. It uses NVMe for its boot disk and gVNIC for networking, as
required by the machine family. Hyperdisk performance is pinned to the included
3,000 IOPS and 140 MB/s baseline. The runtime is built from the operator's local
Loom worktree and runs as a Docker Compose application on the GCE host.

## Prerequisites

`Pulumi.yaml` selects the shared Marin state backend. Select the production
stack before deploying:

```sh
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
startup unit stores Docker state on the persistent root disk, reads one numbered
`LOOM_DOTENV` version, pulls the digest-pinned image, runs
`docker compose up -d`, applies the configured Loom deployment policy, and
checks readiness. It does not clone a repository or build images on the VM.

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

The `fork-ferry` mapping accepts OIDC tokens only from the `marin` repository's
`ops-fork-ferry.yaml` workflow on `main`. It authorizes the dedicated
`fork-ferry` automation profile. Loom brokers short-lived `loom-oa-dev` GitHub
App tokens for the profile's Marin fork repositories, with contents, issues,
and pull-request write access. The App key remains in `LOOM_DOTENV`; the profile
does not store a GitHub token or grant Actions access. The GitHub Pulumi stack
reads the mapping's profile from this stack's `githubFederationProfiles` output
and publishes it as the workflow's `LOOM_FORK_FERRY_PROFILE` repository variable.

Organization prompt policy lives beside the runtime profiles in
`profiles/<name>/AGENTS.md`. A profile's `instructionsFile` is resolved below
`infra/loom`, read by Pulumi, and reconciled into Loom's visible profile
`instructions` field. Loom therefore does not need access to this checkout at
runtime, and the effective text remains inspectable in Settings. The production
`slack.profile` and `github.profile` settings select their dedicated profiles;
ordinary sessions use the deployment-managed `default` profile, while workload
and future GitHub Actions callers select the automation profile authorized by
their federation mapping.

The `default`, `github`, and `slack` profiles allowlist the repositories where
interactive sessions may fall back to the `loom-oa-dev` GitHub App. Loom stamps
only the session's current repository and brokers a short-lived installation
token when the launching user has not stored a personal token. A personal token
continues to take precedence. Keep these lists aligned with the repositories
registered in production and the App's installations.

The Pulumi declaration is authoritative at activation time. An unchanged
profile keeps its database revision; a changed declaration overwrites the
current row and advances the revision. UI or API edits persist only until the
next activation. Deployment pruning is enabled, so a deployment-managed
setting, profile, or federation removed from `Pulumi.marin-loom.yaml` is removed
from its deployment layer on the next activation. Stock profiles omitted from
the declaration remain unmanaged and are not pruned; production intentionally
manages `default` so interactive instruction and runtime policy are reviewed in
this repository.

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

A stack cannot bootstrap access to its own secrets-provider key. An identity that
already has key access must apply any new `vmPulumiKmsKeys` grant.

Previewing Echo requires read access to its resources, Pulumi state objects, and
secrets-provider key. Deploying Echo also requires mutation access for Cloud Run,
Cloud Scheduler, Cloud SQL, Artifact Registry, service accounts, project IAM, Secret
Manager IAM, and IAP IAM, plus payload access to
`cloudsql-pulumi-admin-password`. Prefer the existing project custom IAP IAM role and
secret-level access over project-wide `roles/iap.admin` or
`roles/secretmanager.admin`.

## Restart and rollback

Each Loom session supervisor runs in a separately labeled Docker container.
Recreating the control-plane service preserves those containers, and the new
control plane discovers and adopts them. Do not run `docker compose down` while
sessions are live because it removes their shared network.

To roll back an application release, check out the prior Loom tree, restore its
numbered `dotenvSecretVersion` when necessary, and run the normal preview and
update. The separately managed root disk is protected, retained if removed from
Pulumi, and not auto-deleted with the VM. A replacement root disk must use an
explicit `bootDiskSnapshot`; keep that source snapshot until a newer rollback
point has been verified.
