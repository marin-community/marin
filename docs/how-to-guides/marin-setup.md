# Marin Setup Guide

This guide will help you set up your system and install the necessary dependencies for Marin.

This guide assumes that an user is setting up his machine from scratch. Please feel free to skip to [Marin Installation](#marin-installation) if your system has been set up.

Tested on a new g6dn.2xlarge instance running on Ubuntu 24.04.2 LTS.

## System Requirements
- Ubuntu 24.04 (lower version is OK)
- NVIDIA GPU with CUDA support (for running jobs locally)
- Administrator privileges (sudo access)

## Basic system setup

### 1. System Update
First, ensure your system is up to date:
```bash
sudo apt update && sudo apt upgrade
```

### 2. CUDA related installation (for running GPU jobs)
Install CUDA 12.9.0
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.run
```

Install cuDNN 9.9.0
```bash
# Download cuDNN package
wget https://developer.download.nvidia.com/compute/cudnn/9.9.0/local_installers/cudnn-local-repo-ubuntu2404-9.9.0_1.0-1_amd64.deb

# Install the package
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.9.0_1.0-1_amd64.deb

# Copy the keyring file
sudo cp /var/cudnn-local-repo-ubuntu2404-9.9.0/cudnn-*-keyring.gpg /usr/share/keyrings/

# Update package list
sudo apt-get update

# Install cuDNN
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```

### 3. Verification
After installation, you can verify your setup by checking the CUDA version:
```bash
nvcc --version
```

## Marin Installation
Now that your system is set up with CUDA and cuDNN, you can install Marin:

1. Clone the repository:
```bash
git clone https://github.com/stanford-crfm/marin.git
cd marin
```

2. Create and activate a Python virtual environment. Replace `.venv` with your desired venv name.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Marin and its dependencies:
```bash
pip3 install -e . jax[cuda12]
```

This will:
- Install Marin in development mode (`-e` flag)
- Install JAX with CUDA 12 support
- Install all required dependencies specified in the project's `setup.py`

## Next Steps
After installation, you can:
1. Train a model on your local machine. See [Local GPU Training Guide](local-gpu.md)
2. Submit a speedrun. See [Submitting Speedrun Guide](submitting-speedrun.md)
3. Explore data filtering. See [Data Filtering Guide](filtering-data.md)

For additional help, please refer to the [Marin GitHub repository](https://github.com/stanford-crfm/marin) or open an issue there.