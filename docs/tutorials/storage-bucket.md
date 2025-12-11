# Preparing a Storage Bucket for Marin and Levanter

Many Marin and Levanter workflows expect a durable object store for checkpoints, dataset shards, logs, and executor outputs.
This tutorial walks through setting up a Google Cloud Storage (GCS) bucket that you can reference via `MARIN_PREFIX` or `trainer.checkpointer.base_path`.

## When You Need This

- Running local GPU or TPU experiments that write checkpoints to `gs://...` paths.
- Launching TPU jobs with `scripts/tpu/launch.py` or Ray clusters, where every worker streams artifacts to a shared prefix.
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

## Step 4: Configure Lifecycle Rules (Optional but Helpful)

Large experimentation buckets benefit from automatic cleanup. For example, delete files under `tmp/` after seven days and move old checkpoints to colder storage:

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
gsutil lifecycle set lifecycle.json "$BUCKET"
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

Commit these defaults in `.levanter.yaml` or `.envrc` so every launch script uses the same location.

## Ongoing Hygiene Checklist

- Re-run `gcloud storage buckets describe` monthly to confirm soft delete stays disabled.
- Use `gcloud storage ls --buckets --soft-deleted` to ensure no surprise buckets exist in soft-delete state.
- Monitor storage costs in Cloud Monitoring or set up alerts when the bucket exceeds an expected size.

With this setup you have a clean, low-overhead bucket tailor-made for Marin and Levanter experiments without the surprise bills that soft delete can cause.
