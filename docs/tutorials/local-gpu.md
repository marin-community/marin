# Setting up a Local GPU Environment

This guide will walk you through the steps to set up a local GPU environment for Marin.
By "local", we mean a machine that you run jobs on directly, as opposed to dispatching them to a shared cluster via [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).
Similar steps will let you run Marin on a cloud GPU environment under Iris (the Marin team runs production GPU workloads on CoreWeave), but we defer that to a future guide.

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
Let's start by running the tiny model training script (GPU version) [`experiments/tutorials/train_tiny_model_gpu.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_gpu.py):

```bash
export MARIN_PREFIX=local_store
export WANDB_ENTITY=...
uv run python experiments/tutorials/train_tiny_model_gpu.py --prefix local_store
```

The `prefix` is the directory where the output will be saved. It can be a local directory or anything fsspec supports,
such as `s3://` or `gs://`.

Let's take a look at the script.
Whereas the [CPU version](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_cpu.py)
requests `resources=ResourceConfig.with_cpu()`,
the [GPU version](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_gpu.py)
requests `resources=ResourceConfig.with_gpu(...)`:

```python
from fray.cluster import ResourceConfig

nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G"),
    train_batch_size=256,
    num_train_steps=100,
    learning_rate=6e-4,
    weight_decay=0.1,
)
```

To scale up, submit to Marin's shared [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) cluster
via `uv run iris --cluster=marin job run ...` (see `lib/iris/OPS.md` for the CLI reference).
