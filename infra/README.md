# Autoscaling Cluster for Marin Data Processing
In Marin, we use GCP TPUs (provided by TRC) to do all of our work, including non-training tasks.
We have several clusters for Marin, each with a different TPU type:

- `marin-us-central2` (default): v4's
- `marin-us-west4`: v5e's
- `marin-eu-west4`: v5e's
- `marin-us-east5` (v6e's)
- `marin-us-east1` (v6e's)
- `marin-us-central1` (v6e's)
- `marin-big-run` (v4, reserved for the hero run, whatever it may be. Do not use for anything else.)



## Our Cluster

At a high-level, this directory provides all setup scripts and configuration files for standing up a Ray Cluster and
interacting with it to do lightweight monitoring and reconfiguration. The architecture of our cluster is as follows:

- **Head Node**: A *persistent* (on-demand) [`n2-standard-8` GCP VM](https://cloud.google.com/compute/docs/general-purpose-machines) with 8 CPUs and 32 GB of
  RAM, and a 200 GB disk.
- **Worker Nodes**: An autoscaling number of **preemptible** TPU v4-8 or v5e VMs; a minimum of 4 VMs will be kept alive
 at all times, with a maximum of 1024 VMs alive at once (we can increase this number).

In the v4 cluster, we use v4-8's as our worker nodes. In the v5e clusters, we use v5e-1's as our worker nodes.

The head node is responsible for coordinating the cluster, while the worker nodes are responsible for executing the
actual tasks. In general, we try to avoid running any actual computation on the head node, as it is a shared resource.

## Ray

[Ray](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) provides the underlying
cluster infrastructure for Marin. We use Ray for:
- **Cluster management**: Autoscaling, node provisioning, job scheduling
- **Training**: Distributed model training via Levanter
- **Inference**: GPU/TPU actor pools for model serving

### Ray token authentication

Marin clusters use Ray token authentication (Ray >= 2.52). Ray APIs (dashboard, jobs, status) require a shared token.

- Cluster-side: the token is fetched from GCP Secret Manager into `/home/ray/.ray/auth_token` during `setup_commands`
  (runs inside the Ray container) and `RAY_AUTH_MODE=token` is set via Docker run options.
- Client-side (your laptop): `scripts/ray/cluster.py` will automatically fetch/cache the token into `~/.ray/` when you
  connect to a specific cluster config. You can also install tokens explicitly (see below).

For the staging cluster (`marin-us-central2-staging`):

```bash
uv run scripts/ray/cluster.py --cluster us-central2-staging auth
```

If you need the staging token locally for CLI/SDK usage:

```bash
make get_ray_auth_token_staging
```

Note: the Ray dashboard stores the token in a `ray-authentication-token` cookie scoped to the host (e.g. `localhost`).
If you switch between clusters with different tokens (staging vs non-staging), you may need to clear that cookie or use
an incognito window (or open via `127.0.0.1` vs `localhost`).

For non-staging clusters, install the default token locally:

```bash
make get_ray_auth_token
```

If you prefer ssh-agent style exports (sets `RAY_AUTH_MODE=token` and `RAY_AUTH_TOKEN_PATH=...` for the current shell):

```bash
eval "$(uv run scripts/ray/cluster.py --cluster us-central2 auth-env)"
```

#### Known issue: dashboard Logs tab

As of Ray 2.52.x, the Ray dashboard **Logs** tab is currently unreliable with token authentication enabled (you may see
`UNAUTHENTICATED` / “Invalid or missing authentication token” errors when fetching logs). This appears to be an upstream
Ray issue (tracking: [ray-project/ray#59614](https://github.com/ray-project/ray/issues/59614)).

Workarounds:
- Prefer `ray_run.py` output (driver logs stream in the terminal).
- SSH to the head node and `tail -f` files under `/tmp/ray/session_latest/logs/`.

#### One-time: create the production token secret

Non-staging clusters read their token from the `RAY_AUTH_TOKEN` Secret Manager secret.

If `make get_ray_auth_token` fails because the secret doesn’t exist yet, create it once:

```bash
export RAY_AUTH_MODE=token
ray get-auth-token --generate  # writes ~/.ray/auth_token (don’t commit this)
make init_ray_auth_token_secret
```

For **data processing** (downloads, transforms, deduplication), we use Zephyr instead of raw Ray.

**Useful Documentation**:
- [Ray Cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html): Cluster architecture and key concepts
- [Ray on GCP](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html): GCP-specific deployment

## Preemptibility

It is important to understand that almost all of our compute is **preemptible**.  This means that the VMs can be shut down at
any time by Google, and we will lose all data on them. Preemptibility imposes a lot of constraints on how we design:

- When possible, setup should be fast.
- Jobs should be written into small checkpointable units, that can be rescheduled if they fail.
- Jobs should be idempotent and should be able to be restarted from the last checkpoint and not get confused if any partial mess is left behind.
- Checkpoint often, use GCS for anything durable.
- If you need absolutely need something to not crash, ask to schedule it on the head node. Do not do anything heavy on the head node.

## Data Processing with Zephyr

For data processing jobs (downloads, transforms, deduplication, etc.), we use **Zephyr**, a lightweight Dataset
abstraction that handles parallelism and fault tolerance automatically.

### Quick Example

```python
from zephyr import Dataset, flow_backend

def process_file(input_path: str, output_path: str) -> None:
    # Your processing logic here - no @ray.remote needed
    ...

def main():
    backend = flow_backend()  # Backend configured via CLI flags
    pipeline = (
        Dataset.from_list(input_files)
        .filter(lambda f: not output_exists(f))
        .map(lambda f: process_file(f["input"], f["output"]))
    )
    list(backend.execute(pipeline))
```

Run with:
```bash
uv run zephyr --backend=ray --max-parallelism=200 --memory=2GB script.py
```

### Documentation

- **Quick start**: See `lib/zephyr/README.md`
- **Design & API**: See `lib/zephyr/docs/design.md`
- **Migration patterns**: See `.agents/docs/zephyr-migration.md` for patterns like bounded parallel map, flat_map for file processing, and nested parallelism

### Design Principles

Jobs should still follow these principles for preemptible compute:
- **Idempotent**: Can be restarted without side effects (use `skip_existing=True` in writers)
- **Checkpointable**: Write to GCS frequently, use small atomic units of work
- **Streaming**: Avoid materializing entire datasets in memory

---

# Maintaining a Ray Cluster

## Setup

Install [gcloud](https://cloud.google.com/sdk/docs/install). On MacOS, you can download the CLI with `brew install gcloud-cli`.

You will also need to authenticate with GCP and set the default project.

```bash
gcloud auth login  # follow instructions
gcloud auth application-default login  # follow instructions
gcloud config set project hai-gcp-models
```


## Cluster Management

Each cluster config is in a separate file in the `infra` directory. These files are automatically generated by the
`scripts/ray/cluster.py` script, which reads the `infra/marin-cluster-template.yaml` file. **Do not edit the
`cluster.yaml` files directly**. Instead, edit the template file and run the script to update the cluster configs.
For short term testing, it's fine to edit the `cluster.yaml` directly, but remember to update the template file and
regenerate the configs before merging. Check in the generated configs to the repo.

### Cluster management tool

For most operations, you can use the cluster management tool at `scripts/ray/cluster.py`. You can find the documentation
in [scripts/ray/README.md]. Some sample commands:

```
uv run ./scripts/ray/cluster.py --config=infra/marin-us-central2.yaml {start-cluster,stop-cluster,restart-cluster}
uv run ./scripts/ray/cluster.py --config=infra/marin-us-central2.yaml {add-worker}
uv run ./scripts/ray/cluster.py --config=infra/marin-us-central2.yaml dashboard
```

### Ray Commands

You can also use the Ray commands to directly manipulate clusters.


```bash
export CLUSTER=us-central2

# Launch the Cluster -- will automatically provision the head node, and start configuring the minimum number of workers
ray up -y infra/marin-$CLUSTER.yaml

# Kill the Cluster (takes a while to gracefully terminate nodes)
ray down -y infra/marin-$CLUSTER.yaml

# Monitor the Ray Autoscaler Logs (in case there are problems spinning up workers)
#   =>> Note that `exec` lets you run arbitrary commands on the head node!
ray exec infra/marin-$CLUSTER.yaml "tail -n 100 -f /tmp/ray/session_latest/logs/monitor*"

# SSH into the Head Node
ray attach infra/marin-$CLUSTER.yaml
```

By default, each cluster is provisioned with a persistent `n2-standard-8` VM acting as
the head node. At any given time, there should be a minimum of 4 TPU `v4-8` VMs acting as workers, with an autoscaling
limit of 1024 VMs (so a maximum of 8 * 1024 = 8,192 total v4 cores or 4,096 v4 chips).

Each TPU `v4-8` VM actually has a surprising amount of CPUs and RAM (~240 CPUs, 200GB+ of RAM). However, Ray doesn't
actually do anything under the hood to ensure that a job is actually using the number of logical resources specified
(e.g., a job that on paper requests 1 CPU can use arbitrary cores/RAM). To mitigate this on the scheduler side,
the config file configures each worker with only 120 visible CPUs.

### Restarting the Cluster
There is currently an error on the Ray autoscaler side with spot-TPU instances, where the Ray autoscaler is not able
to detect when spot-TPU instances are dead and as a result, we may be left in a state with just the head node and
no more spot-TPU worker instances starting up. When this state occurs, please message in the #infra Discord
that you are going to restart the cluster, and then run `uv run scripts/ray/cluster.py --config <config> restart-cluster`.

Notes:
* Please check whether there are any running jobs from other users before restarting so that you do not kill all their
jobs without getting permission first.
* You can specify `--preserve-jobs=0` when restarting the cluster if you want to skip backing up running jobs and start
with a completely clean slate (the default value is `--preserve-jobs=1`, which backs up jobs and resubmits them after the restart).
Example: `uv run ./scripts/ray/cluster.py --config=infra/marin-us-central2.yaml restart-cluster --preserve-jobs=0`
* See the instructions below if there are any reserved workers on the cluster, though in many cases the command above is all you need.

### Adding manual workers

Ray cannot automatically schedule TPUs using our reserved capacity. These must be added to the cluster manually.

```bash
export CLUSTER=us-east5-b-vllm
uv run scripts/ray/cluster.py --config infra/marin-us-east5-b-vllm.yaml add-worker v6e-8 --capacity reserved
```

Remember to:
1. Message in the #marin Discord channel before restarting
2. Wait for the cluster to fully initialize before running jobs
3. Be patient with the first job after restart as it may take ~10 minutes for workers to spin up

### Reconfiguring the Cluster

To reconfigure the cluster, you should generally use the `scripts/ray/cluster.py` script and the template
file `infra/marin-cluster-template.yaml` and not modify the `cluster.yaml` directly. This script will update all the
cluster configs in the `infra` directory with your changes.

In general, for additive operations like increasing the `max_workers` for autoscaling, you can just call `ray up`
against the already-running cluster. For larger changes, like changing the machine type of the workers, you should bring
the cluster down (`ray down`) and then bring it back up (`ray up`).

If you need to change something else about the cluster, e.g. if you're changing any of the initialization/setup
commands, it's best to bring the entire cluster down (`ray down`), *then edit the `marin-cluster-template.yaml`*, and
then bring the cluster back up (`ray up`); note that this will kill all VMs, including the head node.

#### Docker Image

If you need to make substantive changes to the machine software, you should change the Docker file at
`docker/marin/Dockerfile.cluster`. Then run `make cluster_docker` to rebuild the Docker image and push it to the
Google Artifact Registry. (Note that by default this will update the dockers for all clusters; if you only want to update it for one cluster, you can modify `CLUSTER_REPOS` variable in the Makefile). This will create a new image and a new tag, of the form
`"us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:<TAG>"`. Tags can include the latest commit hash and the
date, for example:

```makefile
CLUSTER_REPOS = us-central2 europe-west4 us-west4
TAG_VERSIONS = latest $(shell git rev-parse --short HEAD) $(shell date -u +"%Y%m%d")
```

The `cluster_docker` command will handle creating artifact repositories if they don't exist, building the Docker image,
tagging it, and pushing it to all relevant regions and versions. If you run into a permissions error (e.g., 403) when pushing the Docker image, you may need to authenticate to the repo:

``` diff
gcloud auth configure-docker us-central2-docker.pkg.dev
gcloud auth configure-docker europe-west4-docker.pkg.dev
gcloud auth configure-docker us-west4-docker.pkg.dev
```

After building the Docker image and pushing it to the relevant regions and versions, you need to update the
Ray configuration files to point to the latest version. `make cluster_docker` should update the `LATEST` tag
in `src/main/cluster/config.py` for you, but you can check just in case.

Run `uv run scripts/ray/cluster.py update-configs` to regenerate the cluster configs. This will update each cluster
config in the `infra` directory with the corresponding new Docker image tag.

After that, you can restart each cluster with `ray down` and `ray up`.

**If you use a cluster, please use the corresponding bucket, as data transfer costs between regions are high**

If you need to change something else about the cluster, e.g. if you're changing any of the initialization/setup
commands, it's best to bring the entire cluster down (`ray down`), *then edit the `cluster.yaml`*, and then bring the
cluster back up (`ray up`); note that this will kill all VMs, including the head node (nothing lasts forever).

**Word of Warning**: Ray looks at the actual `cluster_name` and various worker names/configs to "identify" existing/new
clusters. To prevent orphaned states, do not change the names of the clusters without first bringing
the cluster down!


#### Environment Variables

We are currently using Google Secret Manager to store the environment variables that are needed to run the cluster.
You can edit those secrets by going to the Google Cloud Console and navigating to the Secret Manager. Once you add
a new version, you can cause the changes to propagate by killing the workers or restarting the cluster.


### Adding TPU Nodes Manually to the Cluster

Ray only supports on demand and preemptible TPUs. For reserved nodes, we need to add them manually to the cluster.

The unified cluster manager provides this functionality:

```bash
# Add a reserved TPU worker (functionality consolidated from manual_ray_worker_launch.py)
uv run scripts/ray/cluster.py --config infra/marin-us-central2.yaml add-worker v4-128 --capacity reserved
```

**Note**: This functionality is currently being integrated. Contact the team for assistance with adding reserved workers if needed.

## Artifact Registry Cleanup Policy Management

To keep our Docker artifact registries tidy, we provide a script and Makefile target to automatically configure a cleanup policy for all our standard GCP regions. This policy deletes images older than 30 days from the registry,
except we keep the most recent 16 tags.

### Script: `infra/configure_gcp_registry.py`
- This script sets a cleanup policy on a GCP Artifact Registry repository to delete images older than 30 days.
- Usage:
  ```bash
  python infra/configure_gcp_registry.py <repository-name> --region=<region> [--project=<gcp-project>]
  ```
  - `repository-name`: Name of the Artifact Registry repository (default is usually `marin`).
  - `--region`: GCP region (e.g., `us-central2`).
  - `--project`: (Optional) GCP project ID. If omitted, uses the current gcloud project.

### Makefile Target: `configure_gcp_registry_all`
- To apply the 30-day cleanup policy to all standard regions (as defined in the `CLUSTER_REPOS` variable in the Makefile), run:
  ```bash
  make configure_gcp_registry_all
  ```
- This will call the script for each region, setting the policy for the `marin` repository in each.
- To use a different repository name, edit the `default_registry_name` variable in the Makefile.
- To specify a project, you can modify the Makefile target or call the script directly with `--project`.

**When to use:**
- After creating new Artifact Registry repositories in new regions.
- Periodically, to ensure all regions have the correct cleanup policy applied.
- After onboarding a new GCP project or changing repository names.

```
