# Training on Cloud GPUs

Marin's GPU capacity is a fleet of H100 nodes in CoreWeave, reached through the
[Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) `marin` cluster.
You submit to `marin`, as you would for a TPU job, and Iris *federates* the job to a
CoreWeave cluster that has the GPUs. This guide covers what you must set differently for a
GPU run compared to a TPU one.

For running on a GPU you already own, see [Setting up a Local GPU
Environment](local-gpu.md). For the anatomy of a training script, see [Training an
LM](train-an-lm.md).

## The clusters

| Cluster | Hosts | Accelerators |
| --- | --- | --- |
| `marin` | GCP | TPU v4/v5e/v5p/v6e, CPU |
| `cw-rno2a` | CoreWeave, Reno | H100 (8 per node) |
| `cw-us-east-02a` | CoreWeave, US East | H100 (8 per node) |

`marin` has no GPUs of its own. It is configured with the two CoreWeave clusters as
federation *peers*: they report the shapes they can host (`device-type=gpu`,
`device-variant=h100`), and `marin` hands whole jobs to them. Peers have no user-facing
endpoint — you never submit to CoreWeave directly, and `--cluster` always stays `marin`.

## Submitting a GPU job

```bash
uv run iris --cluster=marin job run \
  --target-cluster cw-rno2a \
  --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.tutorials.train_tiny_model --device h100x8 --dataset wikitext
```

Three things differ from a TPU submission. Each is explained below.

### 1. Pin the whole job with `--target-cluster`

`--target-cluster` federates the *entire* job — the coordinator process and every sub-job
it dispatches — to the named peer.

You need it because **only a whole root job is ever federated**. A peer runs a handed-off
job under the same job id, so it can only accept a job whose parent it also has. The
training sub-job that `StepRunner` dispatches from inside your script is a *child* job, and
a child never crosses the federation boundary: it runs on whatever cluster runs its parent.
Submit the coordinator to `marin` without `--target-cluster` and its H100 sub-job is refused
— `marin` has no GPUs, and the sub-job cannot federate on its own.

The cost is that the coordinator process also occupies a CoreWeave CPU node, and a job
pinned to CoreWeave cannot dispatch a TPU sub-job. A single experiment therefore targets
GPUs or TPUs, not both.

### 2. Keep storage on CoreWeave object storage

CoreWeave task pods carry S3 credentials for CoreWeave AI Object Storage and **no GCP
credentials at all**. A `gs://` prefix that works for a TPU run is unreadable from a GPU job.

You normally do nothing here: a CoreWeave cluster already sets `MARIN_PREFIX` to
`s3://marin-us-east-02a/marin`, the one bucket both CoreWeave clusters share, and pods read
it through the in-cluster LOTA cache with no endpoint or credentials of your own. Override
it only to write somewhere else under that bucket:

```bash
-e MARIN_PREFIX "s3://marin-us-east-02a/scratch/my-experiment"
```

The rule that bites is that *every* artifact the run reads or writes must live under an
`s3://` prefix, tokenized caches included. A cache a TPU run built into `gs://` is not
reachable from a GPU run, which rebuilds it under `s3://` instead. Passing a `gs://`
`MARIN_PREFIX` to a CoreWeave job fails on the first read.

### 3. Let the GPU sub-job leave the coordinator's region

A sub-job inherits the region of the worker that submitted it, which normally keeps it near
its data. CoreWeave's peers advertise no region, so an inherited GCP region excludes every
host that has an H100. GPU resources must opt out with `regions=[ANY_REGION]`:

```python
from fray.types import ANY_REGION, ResourceConfig

ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION])
```

Without it the sub-job fails at submit with `no scaling group provides device gpu:h100`,
listing only `marin`'s local TPU and CPU groups. The `h100x1` and `h100x8` entries in
[`experiments/tutorials/train_tiny_model.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model.py)
already set it.

## Choosing a GPU shape

A GPU request is a variant and a count. Unlike a TPU slice, whose VM is an atomic
scheduling unit, GPUs pack: a 1-GPU task and a 7-GPU task can share one 8-GPU node.

```python
ResourceConfig.with_gpu("H100", count=1, cpu=8, disk="128G", ram="64G", regions=[ANY_REGION])
ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION])
```

`count` is GPUs per task, up to the 8 on a node. JAX sees them as local devices in one
process, so a single-node job needs no gang scheduling. Ask for more than 8 by raising
`replicas`; multi-node gangs are admitted together over InfiniBand.

Keep `cpu` and `ram` within one node's share (128 vCPU and 2 TiB across 8 GPUs).

## Watching a run

```bash
uv run iris --cluster=marin job logs -f /<user>/<job-name>
uv run iris --cluster=marin job summary /<user>/<job-name>
```

A federated job's logs are relayed from the peer back to `marin`, so `iris job logs` shows
them without your ever connecting to CoreWeave. The relay is asynchronous and lags behind a
log-heavy job; `job summary` reads job and task state, which is mirrored from the peer
independently of logs, so it is the reliable answer to "is it still running".

## Verifying the run used the GPUs

`train_lm` mirrors its metrics next to the run's output, under `${MARIN_PREFIX}/<name>/<version>/`.
`tracker_metrics.jsonl` carries the device facts straight from JAX, and
`checkpoints/eval_metrics.jsonl` the losses:

| Field | Meaning |
| --- | --- |
| `throughput/device_kind` | `NVIDIA H100 80GB HBM3` |
| `throughput/theoretical_flops_per_device` | 9.895e14 for one H100 |
| `throughput/theoretical_flops` | that times the number of GPUs JAX saw |
| `throughput/total_tokens` | `batch_size × seq_len × num_train_steps` |

Pass an explicit `run_id` to `train_lm`. A run that omits it takes the last segment of its
output path — the *version* — as its W&B run id, so every `version="dev"` run in the project
reports into one W&B run and its mirrored summary is another run's metrics.
