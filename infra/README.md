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



## Cluster Infrastructure

Marin clusters run on [Iris](../lib/iris/README.md) for orchestration (job/task
scheduling, node provisioning), fray for distributed execution (Iris-backed),
and [zephyr](../lib/zephyr/README.md) for data pipelines.

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
from zephyr import Dataset, ZephyrContext

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

The canonical region list is sourced from `config/marin.yaml`
(us-central1, us-central2, us-east1, us-east5, us-west4, europe-west4) — the same
single source of truth used by `infra/configure_buckets.py`. Scripts read that map
rather than hardcoding regions, so they never drift from the runtime view of the fleet.

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
  - `--all-regions`: Apply to every region in `config/marin.yaml`.
  - `--dry-run`: Print the gcloud command(s) that would run, per region, without executing.
  - `--project`: (Optional) GCP project ID. If omitted, uses the current gcloud project.

### Makefile Target: `configure_gcp_registry_all`
- To apply the 30-day cleanup policy to the `marin` repository across every canonical region, run:
  ```bash
  make configure_gcp_registry_all
  ```
- This runs `uv run infra/configure_gcp_registry.py marin --all-regions`, iterating over the regions in `config/marin.yaml`.
- To target a single region, a different repository, or a specific project, call the script directly with `--region` / `--project`.

**When to use:**
- After creating new Artifact Registry repositories in new regions.
- Periodically, to ensure all regions have the correct cleanup policy applied.
- After onboarding a new GCP project or changing repository names.
