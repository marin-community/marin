# Loom production deployment

This Pulumi stack adopts and extends the existing `loom.oa.dev` GCE deployment
in `hai-gcp-models`. Pulumi owns durable infrastructure and non-secret runtime
placement. The complete rendered Loom environment remains a Secret Manager
version and never enters Pulumi configuration or state.

## External prerequisites

Use the shared Marin state backend and KMS provider:

```sh
pulumi login gs://marin-iac-state
pulumi stack select marin-loom --cwd infra/loom
```

The `loom-oa-dev` GitHub App is owned by `marin-community` and installed for all
organization repositories. The App cannot be created completely through the
Pulumi GitHub provider: its private key, webhook secret, and client secret have
separate lifecycles. Verify the public configuration and installation before
an infrastructure update:

```sh
infra/loom/preflight.py
```

Store those private values only in the rendered `LOOM_DOTENV` Secret Manager
payload. The App callback and webhook URL use `https://loom.oa.dev`.

## Phase A: adopt without rolling the VM

`runtimeRolloutEnabled: false` makes the VM ignore metadata and startup-script
changes. Imports therefore adopt the live host without changing the process it
runs. These production resources predate Pulumi and must be imported exactly
once:

```sh
export PROJECT=hai-gcp-models REGION=us-central1 ZONE=us-central1-a
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:serviceaccount/account:Account loom-vm \
  "projects/${PROJECT}/serviceAccounts/loom-vm@${PROJECT}.iam.gserviceaccount.com"
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:compute/firewall:Firewall loom-web \
  "projects/${PROJECT}/global/firewalls/loom-allow-web"
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:compute/firewall:Firewall loom-ssh \
  "projects/${PROJECT}/global/firewalls/loom-allow-ssh"
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:compute/address:Address loom-address \
  "projects/${PROJECT}/regions/${REGION}/addresses/loom-ip"
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:compute/disk:Disk loom-data \
  "projects/${PROJECT}/zones/${ZONE}/disks/loom-data"
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:secretmanager/secret:Secret loom-dotenv \
  "projects/${PROJECT}/secrets/LOOM_DOTENV"
pulumi import --cwd infra/loom --stack marin-loom \
  cloudflare:index/dnsRecord:DnsRecord loom-dns-address \
  169959d6aafcbfd77764b8efafa3a509/9d4fb362c09efd0688bec9a5015af6b4
pulumi import --cwd infra/loom --stack marin-loom \
  gcp:compute/instance:Instance loom \
  "projects/${PROJECT}/zones/${ZONE}/instances/loom"
```

Run `pulumi preview --diff` and require zero replacements. Phase A should show
the existing resources as unchanged or adopted and only additive hardening:
the image repository, backup bucket, snapshot policy, least-privilege runtime
IAM, uptime check, alert policy, and dashboard.

Do not import a Secret Manager version. Render a candidate environment in a
temporary directory, compare only sorted key names and per-value hashes with
the active version, then add the reviewed payload out of band:

```sh
gcloud secrets versions add LOOM_DOTENV \
  --project=hai-gcp-models --data-file=/path/to/reviewed.env
```

Delete the temporary payload immediately after upload. Never pass it to
`pulumi config set`, write it to this repository, or print it in a terminal
transcript. Record the numeric version returned by `gcloud` and set
`dotenvSecretVersion` to that exact value. The startup script never reads
`latest`, so a later secret upload cannot change a release implicitly.

## Phase B: immutable release rollout

Pulumi builds the release on the operator's Docker daemon from the exact remote
Git commit, pushes it to Artifact Registry, and records the resulting digest.
It does not build on the production VM or depend on a GitHub Actions publisher.
Before building, authenticate the host Docker client and make sure its buildx
builder supports the VM's architecture:

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
docker buildx inspect --bootstrap
```

The selected builder must advertise `linux/amd64`; set `buildxBuilder` when a
named local or remote builder should be used. Build first while runtime metadata
remains ignored:

```sh
pulumi config set --cwd infra/loom --stack marin-loom dotenvSecretVersion "$SECRET_VERSION"
pulumi config set --cwd infra/loom --stack marin-loom buildCommit "$COMMIT_SHA"
pulumi up --cwd infra/loom --stack marin-loom
```

`build_on_preview` is disabled, so preview never performs a build. The first
update builds `linux/amd64` on the operator host, pushes a full-SHA immutable
tag, and stores the digest output without changing the running VM. Review that
release, then arm it in a separate update:

```sh
pulumi config rm --cwd infra/loom --stack marin-loom buildCommit
pulumi config set --cwd infra/loom --stack marin-loom gitRef "$COMMIT_SHA"
pulumi config set --cwd infra/loom --stack marin-loom runtimeRolloutEnabled true
pulumi preview --cwd infra/loom --stack marin-loom --diff
```

The program rejects branch/short-SHA inputs and refuses to build and activate
the same commit in one update. Activation resolves the retained full-SHA tag
from Artifact Registry and feeds its digest reference directly into VM metadata;
there is no manual digest configuration. Pulumi validates that the real provider
returns the expected repository's `@sha256:` reference before updating the VM.
The first rollout contains a large metadata diff because Phase A deliberately
ignored the whole legacy map. It must contain no replacement, deletion,
boot-disk, data-disk, or network-interface change. The imported `ssh-keys` and
`enable-osconfig` metadata remain ignored and are not removed.

After reviewing that metadata-only change, run `pulumi up`, then:

```sh
infra/loom/post_up.py --stack marin-loom
```

The helper refuses to run while rollout is disabled, inventories live
supervisors and DockerRunner containers, triggers exactly one startup script
generation, and checks public liveness, database/migration readiness, container
identity preservation, and ACP reattachment logs.

The runtime deployment manifest defaults to `pruneDeployment: false`. Enable
pruning only with a reviewed, non-empty set of profiles, workload identities,
or GitHub federation mappings; the program rejects an empty pruning manifest.

## Restart behavior and rollback

The initial adoption preserves the current Compose topology. The planned Loom
runtime change moves each session supervisor into a separately labeled Docker
container. Recreating the control-plane service will then preserve session
container IDs and let the restarted Loom process discover and adopt them.

Until that release lands, a control-plane container replacement can still end
sessions. Take an online SQLite backup before the first Phase B rollout and
inventory the legacy supervisor names. The activation helper refuses that
cutover by default; after accepting the one-time interruption, run it once with
`--allow-legacy-cutover`. Later deployments must omit that flag so missing
supervisors, DockerRunner containers, or ACP reattachments fail verification.
The first cutover skips the ACP reattachment-log gate because legacy ACP sockets
are expected to disappear with the old control container.

Roll back by setting `gitRef` to a retained prior commit and pairing it with the
prior numbered secret version. Pulumi resolves the existing tag to its immutable
digest without rebuilding it. Service-only
restarts preserve DockerRunner containers. Do not run `docker compose down`
while sessions are live because it also removes the shared network they use.

Do not add an Artifact Registry cleanup policy to the `loom` repository without
replacing this rollback mechanism: retained commit tags are the rollback index.
After rotating a credential, upload a new `LOOM_DOTENV` version, set its exact
`dotenvSecretVersion`, and run the normal activation and verification cycle so
the running control service actually consumes it.

The data disk, reserved address, secret shell, backup bucket, and Cloudflare
record are protected Pulumi resources. Removing protection or destroying this
stack requires a separate, explicit teardown decision.
