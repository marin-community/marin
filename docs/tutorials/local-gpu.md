# Setting up a Local GPU Environment

This guide will walk you through the steps to set up a local GPU environment for Marin.
By "local", we mean a machine that you run jobs on directly, as opposed to dispatching them to a shared cluster via [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).
To dispatch a GPU job to Marin's shared H100 fleet instead, see [Training on Cloud GPUs](cloud-gpu.md).

## Prerequisites

Make sure you've followed the [installation guide](installation.md) to do the basic installation.

In addition to the prerequisites from the basic installation, we have one GPU-specific system dependency:

- NVIDIA driver 580 or newer

We assume you are running Ubuntu 24.04.

## NVIDIA driver and runtime

Install an NVIDIA driver that supports CUDA 13. Verify that the driver is at least 580 and that
`nvidia-smi` reports CUDA 13.x:

```bash
nvidia-smi
```

Marin uses [JAX](https://docs.jax.dev/en/latest/index.html) as a core library. The `gpu`
extra installs the CUDA 13 JAX runtime, including CUDA, cuDNN, and NCCL Python wheels:

```bash
uv sync --extra=gpu
```

If you install a local CUDA toolkit for custom kernels, use CUDA 13 and keep older CUDA libraries
out of `LD_LIBRARY_PATH` so they do not override the JAX wheel libraries.

See [JAX's installation guide](https://docs.jax.dev/en/latest/installation.html) for more options.

!!! tip
If you are using a DGX Spark or similar machine with unified memory, you may need to dramatically reduce the memory that XLA preallocates for itself. You can do this by setting the `XLA_PYTHON_CLIENT_MEM_FRACTION` variable, to something like 0.5:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

    You can also set this in your `.bashrc` or `.zshrc` file.
    ```bash
    echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5' >> ~/.bashrc
    ```

    For broader JAX/Levanter memory tuning (sharding, checkpointing, offloading), see [Making Things Fit in HBM](../references/hbm-optimization.md).

## Running an Experiment

Now you can run an experiment.
The unified tutorial script [`experiments/tutorials/train_tiny_model.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model.py)
accepts a `--device` flag that selects the accelerator:

```bash
export MARIN_PREFIX=local_store
export WANDB_ENTITY=...
uv run python experiments/tutorials/train_tiny_model.py --device h100x8 --dataset wikitext
```

`MARIN_PREFIX` sets the root directory for all outputs; it can be a local path or anything
fsspec supports, such as `s3://` or `gs://`.

The same script runs on CPU, GPU, and TPU — only `--device` and `--dataset` change. The GPU
device entry in `train_tiny_model.py` configures resources and batch size:

```python
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.experiment.train import train_lm

# "h100x8" entry in DEVICES (resources, batch_size)
resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G")
batch_size = 256
```

Whereas `--device cpu` uses `ResourceConfig.with_cpu()` and a batch size of 4, `--device h100x8`
uses eight H100s with a larger batch. Adding a new device is one entry in the `DEVICES` dict —
no separate file needed.

To scale up, submit the same script to Marin's shared H100 fleet — see [Training on Cloud
GPUs](cloud-gpu.md), which covers the storage prefix, region, and cluster-pinning a GPU job needs.
