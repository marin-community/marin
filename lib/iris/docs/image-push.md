# Image Push Architecture

## Overview

Iris images are pushed to **GHCR** (GitHub Container Registry) as the single source of truth.
GCP VMs pull from **Artifact Registry remote repositories** that act as pull-through caches
for GHCR. This gives fast, low-cost pulls within GCP without requiring multi-region push
infrastructure.

## Architecture

```
Developer → docker push → ghcr.io/marin-community/iris-worker:v1
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            us-docker.pkg.dev              europe-docker.pkg.dev
            /hai-gcp-models                /hai-gcp-models
            /ghcr-mirror/...               /ghcr-mirror/...
                    │                               │
                    ▼                               ▼
             US GCP VMs                     Europe GCP VMs
```

### How it works

1. **Push**: Images are pushed only to `ghcr.io/marin-community/`.
2. **Pull**: When a GCP VM pulls from `us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/...`,
   the AR remote repo transparently fetches from `ghcr.io` on first access and caches it.
3. **Rewrite**: The autoscaler, controller bootstrap, and worker task image resolver
   rewrite image tags per the cluster's `platform.gcp.registry_mirrors` map
   (upstream registry → zone prefix → mirror repo prefix). On `marin`,
   `marin-dev`, and `ci-gcp-smoke`:
   - `ghcr.io/...` in `us-*` zones → `us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/...`
   - `ghcr.io/...` in `europe-*` zones → `europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/...`
   - Zone prefixes absent from the map (`asia-*`, `me-*`, …) → pull directly from upstream
   - Non-GCP (CoreWeave) → pulls directly from `ghcr.io`

### Docker Hub task images

The same pull-through mechanism caches Docker Hub base images used as task images
(e.g. harbor sandbox images like `ubuntu:24.04`). A separate AR remote repo,
`docker-mirror`, proxies `registry-1.docker.io`, routed by the `docker.io` key of
`registry_mirrors`. The worker's image resolver rewrites Docker Hub references to
it — bare names (`ubuntu:24.04` → `library/ubuntu:24.04`, applying Docker's
implicit `library/` namespace), namespaced names (`bitnami/redis:latest`), and
explicit `docker.io/...` / `index.docker.io/...`. References that name a registry
absent from the map (`gcr.io`, Artifact Registry, a private host) pass through
unchanged.

Every named repo must exist and be enabled in each mapped multi-region, or those
pulls fail. A cluster with no `registry_mirrors` pulls everything straight from
upstream.

### Cost

- Multi-region → same-continent region egress is **free** per
  [AR pricing](https://cloud.google.com/artifact-registry/pricing).
- GHCR/Docker Hub → AR cache miss incurs internet egress, but only on the first
  pull per image/tag.

## Authentication

To push images to GHCR, log in with a **classic** personal access token (PAT) that has the
`write:packages` scope:

```bash
echo $GH_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

Fine-grained tokens do not support the Container Registry; use a classic token.

In CI (GitHub Actions), use the automatic `GITHUB_TOKEN` secret instead — see
`.github/workflows/marin-canary-ferry-coreweave.yaml` for an example. The workflow needs
`packages: write` permission.

## Infrastructure Setup

The mirror repos and their cleanup policies are Infrastructure-as-Code: the
`registries:` block of the `provisioning:` section in
[`lib/iris/config/marin.yaml`](../config/marin.yaml) declares each repo (name,
Docker upstream, multi-regions), and `infra/pulumi/src/iac/gcp/registries.py`
(`GcpArtifactRegistries`) turns it into `google_artifact_registry_repository`
resources with `mode=REMOTE_REPOSITORY` and the cleanup policies. `ghcr-mirror`
uses the default 30d-delete / keep-16 (sized for the versioned iris image
stream); `docker-mirror` overrides it with a plain 7-day TTL, since base-image
packages hold too few versions for a keep floor to leave anything deletable.
Both repos are declared there, so a stack bring-up provisions them together:

```bash
# Recon against the live repos (imports, never plans a destructive create):
cd infra/pulumi && pulumi stack select marin
pulumi config set marin-iac:import true && pulumi preview   # adopt existing repos
# Steady state:
pulumi config set marin-iac:import false && pulumi up
```

New mirrors are added by editing the `registries:` block, not by hand. The
gcloud equivalent of one repo (for reference / a one-off outside the stack) is:

```bash
gcloud artifacts repositories create docker-mirror \
  --project=hai-gcp-models --repository-format=docker --location=us \
  --mode=remote-repository --remote-docker-repo=DOCKER-HUB
```

Then route the cluster's pulls through it by adding the repo to
`platform.gcp.registry_mirrors` (already set on `marin`, `marin-dev`, and
`ci-gcp-smoke`).

### Verify

```bash
# List remote repos
gcloud artifacts repositories list --project=hai-gcp-models --filter="mode=REMOTE_REPOSITORY"

# Test pull-through
docker pull us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:latest
docker pull us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04
```

## Code

- **Rewrite logic**: `lib/iris/src/iris/cluster/platforms/gcp/worker_bootstrap.py`
  - `upstream_registry()`: canonical registry key for an image reference (Docker Hub aliases collapse to `docker.io`)
  - `docker_hub_repo_path()`: Docker Hub repository path with the implicit `library/` namespace applied
  - `rewrite_image_to_mirror()`: applies the `registry_mirrors` map for a zone
- **Task image resolver**: `GcpWorkerProvider.resolve_image()` applies the map for the worker's zone
- **Autoscaler**: `_per_group_bootstrap_config()` rewrites the worker image per scale group
- **Controller bootstrap**: `build_controller_bootstrap_script_from_config()` rewrites the controller image
- **Bootstrap scripts**: Already detect `-docker.pkg.dev/` and configure `gcloud auth` automatically

## Troubleshooting

- **Slow first pull**: Expected — the AR remote repo fetches from GHCR on cache miss.
  Subsequent pulls from the same continent are fast.
- **Auth errors**: GCP VMs need access to the AR repo. The bootstrap scripts handle
  `gcloud auth configure-docker` automatically.
- **Missing image**: Check that the image exists on `ghcr.io` first. The AR remote repo
  cannot serve images that don't exist upstream.
