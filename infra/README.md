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



## Ray

[Ray](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) provides the underlying
cluster infrastructure for Marin. We use Ray for:
- **Cluster management**: Autoscaling, node provisioning, job scheduling
- **Training**: Distributed model training via Levanter
- **Inference**: GPU/TPU actor pools for model serving

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

### Documentation

- **Quick start**: See `lib/zephyr/README.md`
- **Design & API**: See `lib/zephyr/docs/design.md`
- **Migration patterns**: See `.agents/docs/zephyr-migration.md` for patterns like bounded parallel map, flat_map for file processing, and nested parallelism

### Design Principles

Jobs should still follow these principles for preemptible compute:
- **Idempotent**: Can be restarted without side effects (use `skip_existing=True` in writers)
- **Checkpointable**: Write to GCS frequently, use small atomic units of work
- **Streaming**: Avoid materializing entire datasets in memory

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
