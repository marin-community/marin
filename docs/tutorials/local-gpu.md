# Setting up a Local GPU Environment

This guide will walk you through the steps to set up a local GPU environment for Marin.
By "local", we mean a machine that you run jobs on directly, as opposed to using Ray's autoscaler to launch a cluster of GPU nodes.
Similar steps will let you run Marin on a cloud GPU environment using Ray's autoscaler, but we defer that to a future guide.

## Prerequisites

Make sure you've followed the [installation guide](installation.md) to do the basic installation.

In addition to the prerequisites from the basic installation, we have GPU-specific dependencies:

- CUDA Toolkit (version 12.1 or higher)
- cuDNN (version 9.1 or higher)

We assume you are running Ubuntu 24.04.

## CUDA installation

Install CUDA 12.9.0:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.run
```

Install cuDNN 9.9.0 (Instructions from [NVIDIA's cuDNN download page](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)):

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.10.0/local_installers/cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
sudo apt-get -y install nvidia-cuda-toolkit
```

Verify your setup by checking the CUDA version:

```bash
nvcc --version
```

Marin uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as a core library.
Install JAX with the CUDA 12.9.0 backend:

```bash
pip3 install -e . jax[cuda12]
```

See [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more options.

## Running an Experiment

Now you can run an experiment.
Let's start by running the tiny model training script (GPU version) [`experiments/tutorials/train_tiny_model_gpu.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_gpu.py):

```bash
python3 experiments/tutorials/train_tiny_model_gpu.py --prefix local_store
```

The `prefix` is the directory where the output will be saved. It can be a local directory or anything fsspec supports,
such as `s3://` or `gs://`.

Let's take a look at the script.
Whereas the [CPU version](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_cpu.py)
requests `resources=CpuOnlyConfig(num_cpus=1)`,
the [GPU version](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model_gpu.py)
requests `resources=GpuConfig(gpu_count=1)`:

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