# Setting up a Local GPU Environment

Marin was mainly developed on Google Cloud Platform (GCP) using TPUs.
However, it is possible to run Marin on GPUs as well.

This guide will walk you through the steps to set up a local GPU environment for Marin.
Similar steps will let you run Marin on a cloud GPU environment using Ray's autoscaler,
but we defer that to a future guide.

## Prerequisites

This part walks you through having the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- CUDA Toolkit (version 12.1 or higher)
- cuDNN (version 9.1 or higher)

## Basic system setup

### 1. System setup
First, refresh the system's package index:

```bash
sudo apt update
```

Install pip if you are working on a brand new system:

```bash
python3 -m ensurepip
```

### 2. Install Git

```bash
sudo apt install git
```

### 3. CUDA installation
Install CUDA 12.9.0:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.run
```

### 4. cuDNN installation
Install cuDNN 9.9.0 (Instructions from https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local):

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.10.0/local_installers/cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```

### 5. Verification
After installation, you can verify your setup by checking the CUDA version:

```bash
nvcc --version
```

## Installation Steps

1. Clone the Marin repository:
   ```bash
   git clone https://github.com/stanford-crfm/marin.git
   cd marin
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
   or with conda:
   ```bash
   conda create --name marin python=3.10 pip
   conda activate marin
   ```

3. Install the required packages:
   ```bash
   pip3 install -e . jax[cuda12]
   ```
   
   (Marin uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as a core library. See [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more options.)

4. Set WANDB and HF tokens:
   ```bash
   wandb login
   huggingface-cli login
   ```
   
   For this guide, you'll need access to [Meta's Llama 3.1 8B model](https://huggingface.co/meta-llama/Llama-3.1-8B).

## Running an Experiment

Now you can run an experiment. For example, to run the tiny model training script [`experiments/tutorial/exp1076_train_tiny_model.py`](https://github.com/stanford-crfm/marin/blob/main/experiments/tutorial/exp1076_train_tiny_model.py):

```bash
python3 experiments/tutorial/exp1076_train_tiny_model.py --prefix /path/to/output
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
