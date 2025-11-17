# Getting Started with Marin

In this tutorial, you will install Marin on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- uv (Python package manager)
- Git
- On macOS, install additional build tools for SentencePiece:
    ```bash
    brew install cmake pkg-config coreutils
    ```
- A [Weights & Biases](https://wandb.ai) account for experiment tracking (optional but recommended)

This document focuses on basic setup and usage of Marin.
If you're on a GPU, see [Local GPU Setup](local-gpu.md) for a GPU-specific walkthrough for getting started.
If you want to set up a TPU cluster, see [TPU Setup](tpu-cluster-setup.md).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marin-community/marin.git
   cd marin
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and dependencies.

    Use `uv sync` to install dependencies and the local Marin package (editable) in one step:
   ```bash
   # Resolve and install dependencies + local package (editable)
   uv sync --all-packages
   ```

4. Setup [Weights and Biases (WandB)](https://wandb.ai) so you can monitor your runs:
   ```bash
   wandb login
   ```

5. Setup the Hugging Face CLI so you can use gated models/tokenizers (such as [Meta's Llama 3.1 8B model](https://huggingface.co/meta-llama/Llama-3.1-8B)):
   ```bash
   huggingface-cli login
   ```

## Hardware-specific Setup

Marin runs on multiple types of hardware (CPU, GPU, TPU).

!!! info "Install `marin` for different accelerators"

    Marin requires different JAX installations depending on your hardware accelerator. These installation options are defined in our `pyproject.toml` file and will install the appropriate JAX version for your hardware.

    === "CPU"
        ```bash
        # Install CPU-specific dependencies (local package included)
        uv sync --all-packages --extra=cpu
        ```

    === "GPU"
         If you are working on GPUs you'll need to set up your system first by installing the appropriate CUDA version. In Marin, we default to 12.9.0:
         ```bash
         wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
         sudo sh cuda_12.9.0_575.51.03_linux.run
         ```
         Now we'll need to install cuDNN, instructions from [NVIDIA docs](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local), via following:
         ```bash
         wget https://developer.download.nvidia.com/compute/cudnn/9.10.0/local_installers/cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
         sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.0_1.0-1_amd64.deb
         sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.0/cudnn-*-keyring.gpg /usr/share/keyrings/
         sudo apt-get update
         sudo apt-get -y install cudnn
         sudo apt-get -y install cudnn-cuda-12
         ```
         Once system is setup you can verify it via:
         ```bash
         nvcc --version
         ```
         Finally install Python deps for GPU setup:

         ```bash
         # Install GPU-specific dependencies (local package included)
         uv sync --all-packages --extra=cuda12
         ```

    === "TPU"

        ```bash
        # Install TPU-specific dependencies
        uv sync --all-packages --extra=tpu
        ```

## Trying it Out

To check that your installation worked, you can go to the [First Experiment](first-experiment.md) tutorial, where
you train a tiny language model on TinyStories on your CPU.  For a sneak preview, simply run:

```bash
wandb offline  # Disable WandB logging
python experiments/tutorials/train_tiny_model_cpu.py --prefix local_store
```

This will:

1. Download and tokenize the TinyStories dataset to `local_store/`
2. Train a tiny language model
3. Save the model checkpoint to `local_store/`

## Next Steps

Now that you have Marin set up and running, you can either continue with the
next hands-on tutorial or read more about how Marin is designed for building
language models.

1. Follow our [First Experiment](first-experiment.md) tutorial to run a training experiment
2. Read our [Language Modeling Pipeline](../explanations/lm-pipeline.md) to understand Marin's approach to language models
