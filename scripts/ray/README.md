# Ray Cluster Management

This directory contains scripts for comprehensive Ray cluster management on Google Cloud Platform (GCP) with TPU support.

## Overview

The `cluster.py` script is the main entry point for managing Ray clusters. It provides commands to:

- Start, stop, restart, and monitor Ray clusters
- Manage cluster configurations and workers
- Submit and track Ray jobs with backup/restore capability
- SSH into cluster nodes
- Clean up resources and temporary files
- Monitor cluster health with optional wandb integration

## Shared Keys with `--marin-config`

You can provide a project-level `.marin.yaml` to supply shared environment keys (for example `HF_TOKEN`, `WANDB_API_KEY`).
Pass the file via `--marin-config` and the CLI will load those keys into the environment for the session (existing env vars are not overridden).

Example:

```bash
# Load shared keys, then operate on a specific cluster config
uv run scripts/ray/cluster.py \
  --marin-config .marin.yaml \
  --config infra/marin-us-east5-a.yaml \
  list-jobs

# Or specify by cluster name resolution
uv run scripts/ray/cluster.py \
  --marin-config .marin.yaml \
  --cluster us-east5-a \
  get-status
```

## Command Reference

### Specifying Clusters

Most commands expect a cluster to operate against. You can specify the cluster
to target using either the config file `infra/...yaml` or the cluster name. Cluster names can be inconsistent so this mechanism is best effort.


```bash
# Using config file path
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml <command>
uv run scripts/ray/cluster.py --cluster us-central1 <command>
```

#### `list-configs`
List all available cluster configurations:

```bash
uv run scripts/ray/cluster.py list-configs
```

#### `update-configs`
Update all cluster configuration files from templates.

```bash
uv run scripts/ray/cluster.py update-configs
```

- Regenerates cluster configs from Jinja2 templates
- Updates all files in the infra directory

#### `start-cluster`
Start the specified cluster.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml start-cluster
```

- Checks if cluster is already running to prevent conflicts
- Warns if cluster head is already detected

#### `stop-cluster`
Stop the cluster and terminate all nodes.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml stop-cluster
```

#### `restart-cluster`
Restart a cluster.

This attempts to restart any currently running jobs on the new cluster.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml restart-cluster
```

- Backs up all running jobs
- Stops the cluster
- Starts the cluster with fresh configuration
- Restores backed-up jobs


#### `add-worker`

Add a manual TPU worker to cluster.
This is usually done to take advantage of reserved quota which is not handled by autoscaling.
Manual workers are _not_ currently destroyed as part of `down`, but will automatically
connect to a new cluster at startup.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml add-worker v4-8 [--capacity preemptible] [--name custom-name]
```

- **Capacity types**: `reserved`, `preemptible`, `best_effort`

#### `init-worker`
Initialize Ray on a manual TPU worker.

This is typically used for debugging or to update the configuration of an existing worker.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml init-worker --name worker-name
```

- Connects existing TPU to the Ray cluster
- Configures Ray worker processes

#### `ssh-tpu`
SSH to TPU node by IP address.

```bash
uv run scripts/ray/cluster.py ssh-tpu 10.128.0.42 [--project PROJECT] [--zone ZONE] [-- extra-ssh-args]
```

#### `ssh-head`
SSH to cluster head node.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml ssh-head [-- extra-args]
```

#### `list-jobs`
List all Ray jobs with status and details.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml list-jobs
```

- Returns JSON array of job information
- Shows job ID, status, submission time, and entrypoint

#### `submit-job`
Submit a new Ray job to the cluster.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml submit-job "python train.py" \
  [--working-dir /path/to/code] \
  [--runtime-env '{"pip": ["torch", "transformers"]}']
```

- **entrypoint**: Command to execute
- **working-dir**: Directory containing job code
- **runtime-env**: JSON string with environment specification

#### `clean-tpu-processes`
Kill straggling TPU processes.

```bash
uv run scripts/ray/cluster.py clean-tpu-processes [--tpu-type v4-8]
```

- Submits cleanup job for specified TPU type

#### `open-dashboard`
Open Ray dashboard with port forwarding.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml open-dashboard
```

#### `auth`
Open a single cluster dashboard, copy the Ray auth token to clipboard, and open a browser.

```bash
uv run scripts/ray/cluster.py --cluster us-central2 auth
```

When the browser shows the token prompt, paste the token (already in your clipboard) and click **Submit**.
If you are not prompted, you already have a `ray-authentication-token` cookie for this host; if the token is rotated
you may need to clear that cookie or use an incognito window.

If you want to override the Secret Manager secret:

```bash
uv run scripts/ray/cluster.py --cluster us-central2 auth --secret RAY_AUTH_TOKEN
```

You can also install the token locally with:

```bash
make get_ray_auth_token
```

#### `auth-env`
Emit shell exports for Ray token authentication (ssh-agent style).

Recommended usage (exports `RAY_AUTH_MODE=token` and `RAY_AUTH_TOKEN_PATH=...`):

```bash
eval "$(uv run scripts/ray/cluster.py --cluster us-central2 auth-env)"
```

To export the token value directly (less secure):

```bash
eval "$(uv run scripts/ray/cluster.py --cluster us-central2 auth-env --inline-token)"
```

#### `monitor-cluster`
Monitor cluster health with optional wandb logging.

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml monitor-cluster [--wandb]
```

- Displays cluster health summary
- With `--wandb`: logs metrics to Weights & Biases
- Monitors TPU utilization, memory, and job status

#### `show-logs`

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml show-logs [--tail 100]
```

- Shows Ray monitor logs from head node
- Default: last 100 lines with follow mode
- Use Ctrl+C to stop tailing

## Usage Examples

### Basic Cluster Lifecycle

```bash
# Start cluster
uv run scripts/ray/cluster.py --cluster us-central1 start-cluster

# Check status
uv run scripts/ray/cluster.py --cluster us-central1 get-status

# Submit a job
uv run scripts/ray/cluster.py --cluster us-central1 submit-job "python my_script.py"

# Monitor in dashboard
uv run scripts/ray/cluster.py --cluster us-central1 open-dashboard

# Stop cluster
uv run scripts/ray/cluster.py --cluster us-central1 stop-cluster
```
