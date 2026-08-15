# Training on Cloud GPUs

Marin runs shared H100 and GB200 workloads on CoreWeave through Iris. Use
`scripts/iris/dev_gpu.py` for short interactive tests. Submit a full Iris job
after the code works on the reserved node.

For a GPU you control directly, see [Setting up a Local GPU
Environment](local-gpu.md). For the training script itself, see [Training an
LM](train-an-lm.md).

## Choose a cluster

| Cluster | Accelerator |
| --- | --- |
| `cw-rno2a` | H100, 8 per node |
| `cw-us-east-02a` | H100, 8 per node |
| `cw-us-east-08a` | GB200, 4 per compute tray |

The `marin` cluster federates a complete job to one of these peers. Use
`--target-cluster` to select the peer. The [federation
reference](https://github.com/marin-community/marin/blob/main/lib/iris/docs/federation.md)
describes the parent/child job boundary.

When choosing between the H100 clusters, inspect current availability:

```bash
uv run iris --cluster=<cluster> rpc controller list-backends
```

## Submit a job

Run from the checkout root so Iris loads job-specific environment variables
from the gitignored `.marin.yaml`:

```bash
uv run iris --cluster=marin job run \
  --target-cluster cw-rno2a \
  --cpu=1 --memory=2G --extra=cpu \
  -- python -m experiments.tutorials.train_tiny_model \
    --device h100x8 --dataset wikitext --version dev --run
```

The outer process and its child jobs stay on the target cluster. The tutorial
script defines H100 device entries. Add a GB200 entry to that script, or use an
experiment that already requests GB200, before targeting `cw-us-east-08a`.

## Request GPU resources

GPU tasks must allow any region because the CoreWeave peers do not advertise a
GCP region:

```python
from fray.types import ANY_REGION, ResourceConfig

h100 = ResourceConfig.with_gpu(
    "H100", count=8, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION]
)
gb200 = ResourceConfig.with_gpu(
    "GB200", count=4, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION]
)
```

Use one whole node or tray when practical. Raise the task replica count for a
multi-node job.

## Keep data on CoreWeave

CoreWeave task pods receive `MARIN_PREFIX` and object-storage credentials from
the cluster. Use that prefix for inputs, outputs, and caches. Use a
lifecycle-managed path for disposable data:

```python
from rigging.filesystem.cluster_config import marin_temp_bucket

scratch = marin_temp_bucket(ttl_days=1, prefix="my-experiment")
```

Do not read or copy data from GCS while running on CoreWeave without explicit
user approval. The transfer can incur egress charges.

The checkout's `.marin.yaml` supplies job-specific values such as W&B or
Hugging Face credentials. CoreWeave object-storage access comes from the
cluster-managed `iris-task-env` Kubernetes Secret.

## Watch the job

```bash
uv run iris --cluster=marin job logs -f /<user>/<job-name>
uv run iris --cluster=marin job summary /<user>/<job-name>
```

Logs are relayed from the CoreWeave peer. `job summary` reads the mirrored job
state and can be more current than a busy log stream.

After the `dev` run finishes, check
`<MARIN_PREFIX>/users/<username>/checkpoints/tiny-wikitext-h100x8/dev/tracker_metrics.jsonl`.
Its `summary["throughput/device_kind"]` value should name the requested accelerator.
