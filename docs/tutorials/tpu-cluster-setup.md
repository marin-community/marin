# Setting Up Your Own Ray TPU Cluster on GCP

This guide is for running Marin Ray clusters in **your own GCP project**.
Generated configs in this repo default to Google Artifact Registry paths used by Marin's internal project.
For your own project, you can switch `docker.image` to the public GHCR image.

## Prerequisites

- Completed [Installation](installation.md)
- `gcloud` CLI installed and authenticated
- `ray` CLI installed
- A GCP project with TPU quota in your target region/zone
- A GCS bucket for checkpoints/artifacts (see [Prepare a Storage Bucket](storage-bucket.md))

## 1. Pick a Starter Cluster Config

Copy one of the generated cluster configs and customize it:

```bash
cp infra/marin-us-central2.yaml infra/my-ray-us-central2.yaml
```

Edit at least these fields in `infra/my-ray-us-central2.yaml`:

- `cluster_name`
- `provider.project_id`
- `provider.region`
- `provider.availability_zone`
- `docker.image`
- `setup_commands` entries for:
  - `gcloud config set project ...`
  - `gcloud config set compute/region ...`
  - `gcloud config set compute/zone ...`
- `BUCKET` / `MARIN_PREFIX` env values in `docker.worker_run_options` and `docker.head_run_options`

## 2. Use the Public GHCR Cluster Image (Recommended for external projects)

Recommended image format:

```yaml
docker:
  image: "ghcr.io/marin-community/marin-cluster:<tag>"
```

Published tags include:

- `latest`
- `YYYYMMDD` (UTC build date)
- `<short-commit-hash>`

These are published by the scheduled Docker workflow in
`.github/workflows/docker-images.yaml`.

## 3. Secrets: Required vs Optional

### Required (default config)

- `RAY_AUTH_TOKEN` (if your config has Ray token auth enabled)

The generated configs enable token auth by default. Create the secret in your project:

```bash
ray get-auth-token --generate

gcloud secrets create RAY_AUTH_TOKEN --replication-policy=automatic || true
gcloud secrets versions add RAY_AUTH_TOKEN --data-file="$HOME/.ray/auth_token"
```

If this secret is missing while token auth is enabled, cluster setup fails during `setup_commands`.

### Optional

- `HF_TOKEN`
- `OPENAI_API_KEY`
- `RAY_CLUSTER_PUBLIC_KEY`

Cluster setup now checks whether these secrets exist and skips them if missing, so they are optional.

If you want **no Ray token auth**, remove the token-related entries from your config:

- `RAY_AUTH_MODE=token` and `RAY_AUTH_TOKEN_PATH=...` in Docker run options
- setup commands that write `/home/ray/.ray/auth_token`

## 4. Launch and Operate the Cluster

```bash
# Start
uv run scripts/ray/cluster.py --config infra/my-ray-us-central2.yaml start-cluster

# Status
uv run scripts/ray/cluster.py --config infra/my-ray-us-central2.yaml get-status

# Open dashboard tunnel
uv run scripts/ray/cluster.py --config infra/my-ray-us-central2.yaml open-dashboard

# Stop
uv run scripts/ray/cluster.py --config infra/my-ray-us-central2.yaml stop-cluster
```

You can also use raw Ray commands:

```bash
ray up -y infra/my-ray-us-central2.yaml
ray down -y infra/my-ray-us-central2.yaml
ray attach infra/my-ray-us-central2.yaml
```

## 5. Submit a Test Job

```bash
uv run scripts/ray/cluster.py \
  --config infra/my-ray-us-central2.yaml \
  submit-job "python experiments/tutorials/hello_world.py"
```

Or via Ray directly (after `ray dashboard ...` and exporting `RAY_ADDRESS`):

```bash
export RAY_ADDRESS=http://localhost:8265
ray job submit --working-dir . -- python experiments/tutorials/hello_world.py
```

## Troubleshooting

1. Workers do not join:
   - Check autoscaler logs:
   ```bash
   ray exec infra/my-ray-us-central2.yaml "tail -n 200 /tmp/ray/session_latest/logs/monitor*"
   ```
2. Docker pull fails:
   - Verify `docker.image` exists in the registry you configured and the tag is valid.
3. Auth failures:
   - If token auth is enabled, confirm `RAY_AUTH_TOKEN` exists in the same GCP project as `provider.project_id`.
4. Region mismatch / high costs:
   - Keep cluster zone and bucket region aligned.
