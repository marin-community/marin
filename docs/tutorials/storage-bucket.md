# Preparing a Storage Bucket for Marin and Levanter

Many Marin and Levanter workflows expect a durable object store for checkpoints, dataset shards, logs, and executor outputs.
This tutorial walks through setting up a Google Cloud Storage (GCS) bucket that you can reference via `MARIN_PREFIX` or `trainer.checkpointer.base_path`.

## When You Need This

- Running local GPU or TPU experiments that write checkpoints to `gs://...` paths.
- Launching TPU jobs on the shared
  [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
  cluster. CoreWeave GPU jobs use the object storage described in [Training on
  Cloud GPUs](cloud-gpu.md).
- Hosting tokenized datasets or compilation caches that multiple jobs should reuse.

If you only run experiments locally with `local_store/` you can skip this, but migrating to GCS early prevents churn later.

## Step 1: Choose a Region and Name

Pick a region that matches your compute (e.g., `us-central2` for v4/v5e TPUs or `us-west4` for west-coast GPUs). Using the same region keeps egress costs low and improves throughput. Bucket names are global, so choose something descriptive like `gs://marin-<team>-us-central2`.

For the storage class, decide between:

- **Standard**: Lowest latency and predictable performance; slightly higher cost but ideal if training jobs read/write checkpoints frequently.
- **Autoclass**: Google automatically moves objects to colder tiers if they sit idle, which can cut storage costs but occasionally delays reads when objects are thawed. Use this if you mostly archive checkpoints and don't mind rare rehydration pauses.

Marin will attempt to prevent cross-region egress by raising an error in training jobs that write to a different region than the compute, but it's best to avoid that situation entirely.

!!! warning
    Avoid multi-region buckets (e.g., `us` or `us-west`) because they incur higher costs and have more complex performance characteristics. Single-region buckets are cheaper and more predictable for Marin workloads.

## Step 2: Create the Bucket

```bash
PROJECT_ID=your-gcp-project
BUCKET=gs://marin-yourteam-us-central2
REGION=us-central2

# Create the bucket with uniform access and no public exposure.
gcloud storage buckets create "$BUCKET" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --uniform-bucket-level-access \
  --default-storage-class=STANDARD  # add --enable-autoclass to enable automated tiering when you can tolerate slower cold reads

# Grant yourself (or a service account) Storage Admin if needed.
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="user:you@example.com" \
  --role="roles/storage.objectAdmin"
```

Uniform bucket-level access ensures IAM policies apply consistently; keep the bucket private unless you intentionally publish checkpoints.

## Step 3: Disable Soft Delete


!!! warning
   Disabling soft delete is critical to avoid runaway storage costs. Marin creates many large, short-lived files that should be deleted immediately.
   Of course, disabling soft delete means you cannot recover deleted files, so consider implementing lifecycle rules or replication for backups if needed.

GCS enables soft delete by default on new buckets. That feature retains deleted objects for at least seven days, which quickly explodes storage usage for Marin/Levanter workloads because training jobs constantly create and remove multi-gigabyte checkpoints and compilation caches. Disable soft delete immediately after creating the bucket:

```bash
# Permanently disable soft delete for this bucket.
gcloud storage buckets update "$BUCKET" --clear-soft-delete

# Optional: verify that the policy is cleared.
gcloud storage buckets describe "$BUCKET" \
  --format="value(soft_delete_policy)"
```

Clearing the policy ensures that once a training job deletes temporary files they disappear immediately, preventing runaway storage bills. You can still enable backups via lifecycle rules or replication if you need recovery.

## Step 4: TTL Scratch Prefix on Main Buckets (`marin-{region}/tmp/ttl=Nd/`)

For intermediate checkpoints and other short-lived data, Marin reserves a `tmp/` prefix on each `marin-{region}` bucket with lifecycle rules that delete objects based on a `tmp/ttl=Nd/` path prefix — for example, objects under `gs://marin-us-central2/tmp/ttl=3d/my-job/` are deleted three days after they are written.

Supported TTLs: 1, 2, 3, 4, 5, 6, 7, 14, and 30 days. The canonical list lives in `config/marin.yaml` (`data.temp.ttl_days`); call `marin_temp_bucket(ttl_days=N, prefix=...)` to build a path.

The shared `marin-*` buckets on GCS, CoreWeave, and R2 are declared by `DataBuckets` in
[`infra/pulumi`](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md).
Their names and regions come from `config/marin.yaml` and `config/coreweave.yaml`; Pulumi owns
the complete lifecycle policy on every backend and GCS soft-delete disablement.
To add a region, add its GCS bucket entry to the reviewed `data.region_buckets` map and preview
the `marin` stack:

```bash
uv sync --package marin-iac --extra deploy
export PULUMI_PYTHON_CMD="$PWD/.venv/bin/python"
export COREWEAVE_API_TOKEN=...
export CLOUDFLARE_API_TOKEN=...  # account token with Workers R2 Storage Write
pulumi -C infra/pulumi stack select marin
pulumi -C infra/pulumi preview
```

If the bucket already exists, adopt it with the Program-first workflow in
[Adopting live resources](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md#adopting-live-resources) before running
`pulumi up`. A new bucket is created by the normal update.

### Custom lifecycle rules for your own buckets

For non-scratch buckets, you can still set up lifecycle rules manually. For example, delete files under a prefix after seven days:

```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 7, "matchesPrefix": ["tmp/"]}
    }
  ]
}
```

Save this as `lifecycle.json` and apply it:

```bash
gcloud storage buckets update "$BUCKET" --lifecycle-file=lifecycle.json
```

Adjust prefixes to match how your experiments organize outputs.

## Step 5: Wire It Into Marin / Levanter

Set the bucket as your default prefix whenever you run tutorials:

```bash
export MARIN_PREFIX=$BUCKET
export WANDB_PROJECT=marin
export WANDB_ENTITY=your-entity
```

For Levanter configs, point the checkpointer to the same bucket:

```yaml
trainer:
  checkpointer:
    base_path: "$BUCKET/your-run"
```

For `iris job run`, put personal launch values in the gitignored `.marin.yaml`
at the checkout root:

```yaml
env:
  MARIN_PREFIX: "gs://your-bucket"
  WANDB_PROJECT: marin
  WANDB_ENTITY: your-entity
```

The Iris CLI loads this `env` section when run from that directory. Direct SDK
submissions and commands run from another directory do not load it. A GCS
`MARIN_PREFIX` in this file also overrides the CoreWeave storage default. Omit
it from checkouts that submit CoreWeave jobs, or set it only on the GCP job that
needs it.

## Ongoing Hygiene Checklist

- Re-run `gcloud storage buckets describe` monthly to confirm soft delete stays disabled.
- Use `gcloud storage ls --buckets --soft-deleted` to ensure no surprise buckets exist in soft-delete state.
- Monitor storage costs in Cloud Monitoring or set up alerts when the bucket exceeds an expected size.

With this setup you have a clean, low-overhead bucket tailor-made for Marin and Levanter experiments without the surprise bills that soft delete can cause.
