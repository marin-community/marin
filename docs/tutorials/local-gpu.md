# Setting up a Local GPU Environment

Marin was mainly developed on Google Cloud Platform (GCP) using TPUs.
However, it is possible to run Marin on GPUs as well.

This guide will walk you through the steps to set up a local GPU environment for Marin.
Similar steps will let you run Marin on a cloud GPU environment using Ray's autoscaler,
but we defer that to a future guide.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- CUDA Toolkit (version 12.1 or higher)
- cuDNN (version 9.1 or higher)

## Installation Steps

1. Clone the Marin repository:
   ```bash
   git clone https://github.com/stanford-crfm/marin.git
   cd marin
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   or with conda:
   ```bash
   conda create --name marin python=3.10 pip
   conda activate marin
   ```
3. Install the required packages:
   ```bash
   pip install -e . jax[cuda12]
   ```
   (Marin uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as a core library. See [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more options.)
4. Set WANDB and HF tokens:
   ```bash
   wandb login
   huggingface-cli login
   ```
   For this gude, you'll need access to [Meta's Llama 3.1 8B model](https://huggingface.co/meta-llama/Llama-3.1-8B).

## Running an Experiment

Now you can run an experiment. For example, to run the tiny model training script [`experiments/tutorial/exp1076_train_tiny_model.py`](https://github.com/stanford-crfm/marin/blob/main/experiments/tutorial/exp1076_train_tiny_model.py):

```bash
python  experiments/tutorial/exp1076_train_tiny_model.py --prefix /path/to/output
```

The `prefix` is the directory where the output will be saved. It can be a local directory or anything fsspec supports,
such as `s3://` or `gs://`.

That's it! You should now be able to run Marin on your local GPU environment.

## Explaining the Script

Let's take a look at the script. In general, it works much the same as the TPU version. The only difference is
that we request GPU resources from Ray:

```python
nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=GpuConfig(gpu_count=1),
    train_batch_size=128,
    num_train_steps=1000,
    learning_rate=6e-4,
    weight_decay=0.1,
)
```

To scale up, you can use Ray's autoscaler to launch a cluster of GPU nodes. We defer that to a future guide,
but you can see Ray's [autoscaler documentation](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/) for more information.
