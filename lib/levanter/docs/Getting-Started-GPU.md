# Getting Started on GPU


!!! note
    We only test on Ampere GPUs (e.g., A100s or 30xx series). If it works with JAX, it should work, though. We have done limited testing on H100 GPUs, but we do not have regular access to them.

!!! tip "Deterministic Training"
    If you want fully deterministic results when training on GPU, set `XLA_FLAGS="--xla_gpu_deterministic_ops=true"` in your environment before launching Levanter.

We have two installation options for Levanter:

1. [Using `uv` Virtual Environments](#using-uv-virtual-environments): This is the simplest way if you don't have root access to your machine (and don't have rootless docker installed).
2. [Using a Docker Container](#using-a-docker-container): This is the best way to achieve the fastest training speeds, because the Docker container has [TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html), and Levanter uses TransformerEngine's FusedAttention implementation to accelerate training.

## Using `uv` Virtual Environments

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
uv run --extra gpu <command>
```

### WandB Login

By default, Levanter logs training runs to Weights and Biases. You can sign up for a free WandB account at https://wandb.ai/site.

You can obtain an API token from [Weights and Biases](https://wandb.ai/authorize) and use it to log into your WandB account on the command line as follows:

To use WandB, you can log in to your WandB account on the command line as follows:

```bash
uv run wandb login ${YOUR TOKEN HERE}
```

For more information on getting set up with Weights and Biases, visit https://wandb.ai/site.

If you do not want to use WandB, you can disable it by running:

```bash
uv run wandb offline
```

#### Using a Different Tracker

You can also use TensorBoard for logging. See [Trackers and Metrics](./reference/Trackers.md) for more information.

## Using a Docker Container

To take advantage of the fastest training speeds Levanter has to offer, we recommend using the Docker container image
that is part of NVIDIA's [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox). The image is continuously updated with the latest versions of JAX, CUDA, TransformerEngine, and Levanter.
Training speeds are accelerated by [TransformerEngine's](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) [FusedAttention](https://arxiv.org/abs/2205.14135) implementation, which requires a TransformerEngine installation in your environment. Luckily, the offical image has Levanter and TransformerEngine installed for us.

### Ensure You Have Docker Installed
To check if you have Docker installed, run

```bash
sudo docker --version
```

If it is not installed, you can follow the [installation instructions on their website](https://docs.docker.com/engine/install/).

You'll also need to have the `nvidia-container-toolkit` installed. You can follow the [installation instructions on their website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Download the Container Image

Technically optional, since the first time you run the container it will be downloaded, but you can download the inage ahead of time with the following command:

```bash
sudo docker pull ghcr.io/nvidia/jax:levanter
```

### Running the Docker Container For Levanter Users

If you just want to use Leventer out of the box to train models, these are the Docker setup steps you should follow.

If you're interested in actively developing Levanter while using a Docker container, see the [Developing in a GPU Docker Container](./dev/GPU-Docker-Dev.md) guide.

#### Running an Interactive Docker Shell
To run a docker container interactively, you can use the following command:

```bash
sudo docker run -it --gpus=all --shm-size=16g ghcr.io/nvidia/jax:levanter
```

Then, you can run training commands from within your Docker container as follows:

```bash
python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/llama_small_fast.yaml
```

#### Running a Job in a Docker Container
You can also run a job in a Docker container with the following command:

```bash
sudo docker run \
    --gpus=all \
    --shm-size=16g \
    -i ghcr.io/nvidia/jax:levanter \
    python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/llama_small_fast.yaml
```

For more information on how to train models in Levanter, see our [User Guide](Getting-Started-Training.md).

### Mounting a Local Fork of Levanter Inside the Docker Container

If you are planning to add to or extend Levanter for your own use case, follow these Docker setup steps.

First, clone the Levanter repository:


```bash
git clone https://github.com/stanford-crfm/levanter.git
```

Then run an interactive Docker container with your Levanter directory mounted as a volume. For example, if your Levanter
repo is located at `/nlp/src/username/levanter`, then run the command below to make that directory accessible to the Docker container.

```bash
sudo docker run -it --gpus=all -v /nlp/src/username/levanter:/levanter --shm-size=16g ghcr.io/nvidia/jax:levanter
```

Once your container starts, the Levanter repo you cloned will be available at `/levanter`.
You should `cd` into the `levanter` directory and run the install command for Levanter from that directory.

```bash
cd /levanter
pip install -e .
```

Now, you should be able to run training jobs in this container using the version of Levanter from your mounted directory:

```bash
python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml
```


### Things to Watch Out For When Using Docker + Levanter

1. To use the Levanter datasets available on Google cloud within a Docker container, you need to install gcloud and login inside the docker container. See Google Cloud Setup instructions at the top of [Getting Started on TPU VMs](./Getting-Started-TPU-VM.md).

2. If you are using a Docker container on the Stanford NLP cluster, you need to check which GPUs have been allocated to you within your slurm job. Run `nvidia-smi` before you start your docker container and note the `Bus-Id` for each GPU. Then, after starting your docker container, run `nvidia-smi` again to discover the indices of the GPUs you've been allocated within the full node. The GPU index is listed to the left of the GPU name in the left most column. Run `export CUDA_VISIBLE_DEVICES=[YOUR GPU INDICES]` so the container will only use your allocated GPUs and not all the GPUs on the node. For example, if you are using GPUs `[2, 3, 4, 5]` you would run `export CUDA_VISIBLE_DEVICES=2,3,4,5`.

## Running a Job

For more details on how to configure training runs, please see the [Getting Started Training](Getting-Started-Training.md) guide.
Here are some examples of running a job.

### Running a job locally

```bash
python -m levanter.main.train_lm --config config/llama_small_fast
```

### Running a job on Slurm

#### Single Node: 1 Process per Node

Here's a simple example of running a job on a single node. This example assumes you have cloned the Levanter repository
and are in the root directory of the repository.

```bash
srun --account=nlp --cpus-per-task=128 --gpus-per-node=8 --job-name=levanter-multi-1 --mem=1000G  --open-mode=append --partition=sphinx --time=14-0 infra/run-slurm.sh python src/levanter/main/train_lm.py --config_path config/llama_small_fast.yaml
```

#### Single Node: One Process Per GPU

This example uses `sbatch` to submit a job to Slurm. This example assumes you have cloned and installed the Levanter repository.

Nvidia recommends using this method (rather than one process per node) for best performance.

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=levanter-test
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=levanter_%j.log
#SBATCH --mem=16G

## On the Stanford NLP cluster you might need this:
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')

## Activate your virtual environment
source levanter/bin/activate

srun python -m levanter.main.train_lm --config config/llama_small_fast --trainer.per_device_parallelism -1
```

Then, submit the job with sbatch:

```bash
sbatch my-job.sh
```

### Multi-Node GPU Training
For multi-gpu training, you need to additionally have [nvidia-fabricmanager](https://docs.nvidia.com/datacenter/tesla/pdf/fabric-manager-user-guide.pdf) installed on each of your nodes.

```
sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
```

#### Multi-Node Docker Environment Setup
If you are using a docker container to train your model, your docker run command should look similar to this

```
sudo docker run -it --network=host -v ~/src/levanter/cache:/cache -v /home/user/levanter:/levanter --gpus=all --shm-size=16g  ghcr.io/nvidia/jax:levanter
```
The main difference between the command here and the one found in the [GPU Docker Development Guide](dev/GPU-Docker-Dev.md) is the `--network=host` argument. This tells the docker container to use the host machine's network instead of the default docker `bridge` network. Using `host` is the easiest way to do multi-node networking with docker and should be sufficient for your training purposes. Please see docker's [host](https://docs.docker.com/network/network-tutorial-host/) and [bridge](https://docs.docker.com/network/network-tutorial-standalone/) network documentation for more information.

#### Multi-Node Training Command
We use [JAX Distributed](https://jax.readthedocs.io/en/latest/multi_process.html) to help manage multi-node training in Levanter. On each node you can run a command like the following to kick off a training job:

```bash
NCCL_DEBUG=INFO python src/levanter/main/train_lm.py \
  --config_path config/gpt2_7b.yaml \
  --trainer.ray.auto_start_cluster false \
  --trainer.per_device_parallelism -1 \
  --trainer.distributed.num_processes 4 \
  --trainer.distributed.local_device_ids "[0,1,2,3,4,5,6,7]" \
  --trainer.distributed.coordinator_address 12.345.678.91:2403 \
  --trainer.distributed.process_id 0
```
This will start a 4 node job where each node has 8 GPUs.

- `--trainer.distributed.num_processes` - sets the number of nodes used in this training run
- `--trainer.distributed.local_device_ids` - sets the ids of the local GPUs to use on this specific node
- `--trainer.distributed.coordinator_address` - is the IP address and port number of the node that will be leading the training run. All other nodes should have network access to the port and IP address set by this argument. The same IP address and port number should be used for this argument in every node's run command.
- `--trainer.distributed.process_id` - The process ID of the current node. If the node is coordinator for the training run (its IP address was the one specified at `--trainer.distributed.coordinator_address`), its process ID needs to be set to zero. All other nodes in the train run should have a unique integer ID between [1, `num_processes` - 1].

When the above command is run on the coordinator node, it will block until all other processes connect to it. All the other nodes will connect to the coordinator node before they can begin training. All other training run arguments have the same meaning as with single node runs. We recommend thinking about increasing your `--trainer.train_batch_size` value when you scale from single node to multi-node training, as this is the global batch size for your training job and you've now increased your compute capacity.

#### Launching a Multi-Node Slurm Job
Here is an updated Slurm script example where we've added `#SBATCH --nodes=2`.
***NOTE: This script hasn't been tested yet.***

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=levanter-test
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=levanter_%j.log
#SBATCH --mem=16G
#SBATCH --nodes=2

# On the Stanford NLP cluster, you might need this:
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')

CONTAINER_PATH="ghcr.io/nvidia/jax:levanter"
TRAINING_COMMAND="python -m levanter.main.train_lm --config_path config/gpt2_7b.yaml --trainer.ray.auto_start_cluster false --trainer.per_device_parallelism -1"

srun docker run --gpus=all --shm-size=16g --rm $CONTAINER_PATH $TRAINING_COMMAND
```
If you're Slurm (and using Pyxis), you won't need to do provide the distributed arguments described in the previous section. JAX/Levanter will infer them for you.

### Switching Between GPU and TPU
In Levanter, you can switch between using TPUs and GPUs in the middle of a training run. See our tutorial on [Switching Hardware Mid-Training Run](Hardware-Agnostic-Training.md) to learn more.

## FP8 Training

On H100 and newer GPUs, you can train with FP8 precision. To do this, you just need to add the following to your config:

```yaml
trainer:
  # ...
  quantization:
    fp8: true
```

For details on how it works, see the [Haliax FP8 docs](https://haliax.readthedocs.io/en/latest/fp8/) and
Transformer Engine's [FP8 docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

## Miscellaneous Problems

For solutions to common problems, please see the [FAQ](faq.md).

###  CUDA: `XLA requires ptxas version 11.8 or higher`

See FAQ entry, but some variant of this should work:

```bash
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')
```

The issue is that the system-installed CUDA is being used instead of the CUDA installed by JAX.
