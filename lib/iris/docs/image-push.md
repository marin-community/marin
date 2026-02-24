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
   automatically rewrite GHCR image tags to the appropriate AR remote repo based on
   the VM's zone → continent mapping:
   - `us-*` zones → `us-docker.pkg.dev`
   - `europe-*` zones → `europe-docker.pkg.dev`
   - `asia-*` / `me-*` zones → **not supported** (raises error; provision AR remote repo first)
   - Non-GCP (CoreWeave) → pulls directly from `ghcr.io`

### Cost

- Multi-region → same-continent region egress is **free** per
  [AR pricing](https://cloud.google.com/artifact-registry/pricing).
- GHCR → AR cache miss incurs internet egress, but only on the first pull per image/tag.

## Infrastructure Setup

### Create AR remote repos (one-time)

```bash
# US multi-region
gcloud artifacts repositories create ghcr-mirror \
  --project=hai-gcp-models \
  --repository-format=docker \
  --location=us \
  --mode=remote-repository \
  --remote-docker-repo=https://ghcr.io \
  --description="Remote proxy for ghcr.io (US multi-region)"

# Europe multi-region
gcloud artifacts repositories create ghcr-mirror \
  --project=hai-gcp-models \
  --repository-format=docker \
  --location=europe \
  --mode=remote-repository \
  --remote-docker-repo=https://ghcr.io \
  --description="Remote proxy for ghcr.io (Europe multi-region)"
```

### Cleanup policies

```json
[
  {
    "name": "delete-older-than-30d",
    "action": {"type": "Delete"},
    "condition": {
      "tagState": "any",
      "olderThan": "2592000s"
    }
  },
  {
    "name": "keep-latest",
    "action": {"type": "Keep"},
    "mostRecentVersions": {
      "keepCount": 16
    }
  }
]
```

```bash
gcloud artifacts repositories set-cleanup-policies ghcr-mirror \
  --project=hai-gcp-models --location=us \
  --policy=/tmp/cleanup-policy.json --no-dry-run

gcloud artifacts repositories set-cleanup-policies ghcr-mirror \
  --project=hai-gcp-models --location=europe \
  --policy=/tmp/cleanup-policy.json --no-dry-run
```

### Verify

```bash
# List remote repos
gcloud artifacts repositories list --project=hai-gcp-models --filter="mode=REMOTE_REPOSITORY"

# Test pull-through
docker pull us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:latest
```

## Code

- **Rewrite logic**: `lib/iris/src/iris/cluster/platform/bootstrap.py`
  - `zone_to_multi_region()`: maps GCP zone → continent (`us`, `europe`, `asia`)
  - `rewrite_ghcr_to_ar_remote()`: rewrites `ghcr.io/...` → `{continent}-docker.pkg.dev/.../ghcr-mirror/...`
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
