# Marin infrastructure

This page routes infrastructure work to the repository's operational and
deployment documentation. New team members should start with
[Internal onboarding](../docs/dev-guide/guidelines-internal.md). Most experiment
development does not require the deployment procedures on this page.

## Start here

| Task | Documentation |
| --- | --- |
| Submit, inspect, or debug a job | [Iris operations](../lib/iris/OPS.md) |
| Understand Iris scheduling and configuration | [Iris README](../lib/iris/README.md) |
| Build or debug a data pipeline | [Zephyr README](../lib/zephyr/README.md) and [operations guide](../lib/zephyr/OPS.md) |
| Add or change Pulumi-managed infrastructure | [Pulumi project guide](pulumi.md) |
| Deploy Marin services | [Deployment guide](deploy/README.md) |
| Build or operate a Marina app | [Marina README](marina/README.md) |

## Shared compute

Marin compute runs on GCP TPUs and CoreWeave GPUs. The files in
`lib/iris/config/` are the source of truth for named clusters, regions, and
accelerator pools. Use Iris to list the clusters available in the current
checkout; do not copy a static fleet list into other documentation.

## Cluster infrastructure

Marin clusters run on [Iris](../lib/iris/README.md) for orchestration (job/task
scheduling, node provisioning), fray for distributed execution (Iris-backed),
and [zephyr](../lib/zephyr/README.md) for data pipelines.

## Worker loss and durable state

Google TPU workers are preemptible. CoreWeave GPU nodes are non-preemptible,
but node and runtime failures still occur. Code for both platforms as though a
worker can disappear at any time. Its local disk is not durable.

- Keep worker startup and task environment setup fast.
- Split work into checkpointable units that Iris can reschedule.
- Make each unit idempotent so retrying it after partial output is safe.
- Write checkpoints and other durable artifacts to the cluster-provided
  `MARIN_PREFIX`, not to a worker's local disk.

The [Datakit sampling pipeline](../experiments/datakit/cluster/domain/v0/sample.py)
shows the data-processing pattern: each shard has an independent output and the
writer uses `skip_existing=True`, so a restart preserves completed shards. For
training, see [Grug checkpoints and resume](../experiments/grug/README.md#checkpoints-and-resume).

Use [Iris operations](../lib/iris/OPS.md) for current scheduling and recovery
procedures. Do not start, stop, or restart a shared cluster without explicit
approval.

## Data Processing with Zephyr

For data processing jobs (downloads, transforms, deduplication, etc.), we use **Zephyr**, a lightweight Dataset
abstraction that handles parallelism and fault tolerance automatically.

### Quick Example

```python
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

def process_file(input_path: str, output_path: str) -> None:
    # Your processing logic here - no manual worker orchestration needed
    ...

def main():
    ctx = ZephyrContext(max_workers=100)
    pipeline = (
        Dataset.from_list(input_files)
        .filter(lambda task: not output_exists(task["output"]))
        .map(lambda task: process_file(task["input"], task["output"]))
    )
    ctx.execute(pipeline)
```

### Documentation

- **Quick start**: See `lib/zephyr/README.md`
- **Operational notes**: See `lib/zephyr/OPS.md`
- **Archived migration patterns**: See `.agents/docs/zephyr-migration.md` for historical examples of replacing legacy distributed loops with Zephyr patterns

### Design Principles

Jobs should still follow these principles for preemptible compute:
- **Idempotent**: Can be restarted without side effects (use `skip_existing=True` in writers)
- **Checkpointable**: Write to GCS frequently, use small atomic units of work
- **Streaming**: Avoid materializing entire datasets in memory

## Artifact Registry Cleanup Policy Management

To keep our Docker artifact registries tidy, we provide a script and Makefile target to automatically configure a cleanup policy for all our standard GCP regions. This policy deletes images older than 30 days from the registry,
except we keep the most recent 16 tags.

The canonical region list is sourced from `lib/iris/config/marin.yaml`, the
same source used by the regional data-bucket Pulumi component. Scripts read
that map so they stay aligned with the runtime fleet.

### Script: `infra/configure_gcp_registry.py`
- This script sets a cleanup policy on a GCP Artifact Registry repository to delete images older than 30 days (keeping the 16 most recent tags).
- A registry repo's "location" is the GCP region.
- Usage:
  ```bash
  uv run infra/configure_gcp_registry.py <repository-name> --region=<region> [--project=<gcp-project>]
  uv run infra/configure_gcp_registry.py <repository-name> --all-regions [--project=<gcp-project>]
  uv run infra/configure_gcp_registry.py <repository-name> --all-regions --dry-run
  ```
  - `repository-name`: Name of the Artifact Registry repository (usually `marin`).
  - `--region`: GCP region (e.g., `us-central2`). Mutually exclusive with `--all-regions`; exactly one is required.
  - `--all-regions`: Apply to every region in `lib/iris/config/marin.yaml`.
  - `--dry-run`: Print the gcloud command(s) that would run, per region, without executing.
  - `--project`: (Optional) GCP project ID. If omitted, uses the current gcloud project.

### Makefile Target: `configure_gcp_registry_all`
- To apply the 30-day cleanup policy to the `marin` repository across every canonical region, run:
  ```bash
  make configure_gcp_registry_all
  ```
- This runs `uv run infra/configure_gcp_registry.py marin --all-regions`, iterating over the regions in `lib/iris/config/marin.yaml`.
- To target a single region, a different repository, or a specific project, call the script directly with `--region` / `--project`.

**When to use:**
- After creating new Artifact Registry repositories in new regions.
- Periodically, to ensure all regions have the correct cleanup policy applied.
- After onboarding a new GCP project or changing repository names.
