# Getting Started on TPU VMs

This guide will walk you through the steps to get started with Levanter on TPU VMs.

## Overview

An important thing to know about TPU VMs is that they are not a single machine (for more than a vX-8). Instead, they
are a collection of workers that are all connected to the same TPU pod. Each worker manages a set of 8 TPUs.
This means that you can't just run a single process on a TPU VM instance, you need to run a distributed process,
and you can't just set up one machine, but a whole cluster. We have some scripts to help with this.

Our approach is to use Docker to package up the code and run it on the workers. TPU VMs already have Docker installed,
so we just need to build the image and run it. We use a combination of `gcloud` and `docker` to do this, and it's
mostly wrapped up in a script called `launch.py`. For handling preemptible compute and other failures, we have a
new script called `launch_on_ray.py` that uses Ray to automatically spin up TPUs, run jobs, and restart them if they fail.

We also have a legacy script called `spin-up-vm.sh` that can be used to create a TPU VM instance without any of the Docker stuff.

### Preemptible TPUs

Since much of our compute is preemptible, we have to account for the fact that TPU VMs can be preempted at any time.
Levanter is designed to be robust to this, but we still have to actually restart the job when it happens.
We refer to this as "babysitting" the job. We have two options for "babysitting" training jobs.

1. `launch_on_ray.py` is a new, experimental script that uses Ray to manage the job and restart it if it fails.
   This script is still in development, but it seems to basically work.
2. `launch.py` has a `--retries` flag that will automatically restart the job if it fails.  To use this,
 `launch.py` must be running in foreground mode and must maintain a connection to the TPU VM instance.

## Installation

### Install Levanter

First, you need to clone the Levanter repository and install the dependencies. You can do this with the following commands:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### Docker

Docker is a tool that allows you to package up code and run it in a container. You should install Docker
on your local machine. Here are some instructions for [installing Docker](https://docs.docker.com/engine/install/)
if you don't already have it. If you're not planning on using `launch.py` or `launch_on_ray.py`, you don't need Docker.

### Google Cloud setup

First you need gcloud installed and configured. You can find instructions for that [here](https://cloud.google.com/sdk/docs/quickstarts)
or if you're a conda person you can just run `conda install -c conda-forge google-cloud-sdk`.

Second, you need to follow some steps to enable Cloud TPU VM. You can follow Google's directions [here](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
but the gist of it is you need to enable the TPU API and the Compute Engine API. You can do this by running:

```bash
gcloud auth login  # if you haven't already
gcloud auth application-default login  # on your local machine
gcloud components install alpha
gcloud services enable tpu.googleapis.com
gcloud config set account your-email-account
gcloud config set project your-project
```

You can follow more steps there to get used to things like creating instances and such, but we'll only discuss the
most important details here.

Google recommends not running those first two commands on a VM, and instead using tokens and service accounts. You can
find more information about that [here](https://cloud.google.com/docs/authentication/production#auth-cloud-implicit-python).
Honestly, if you're working outside of a corp environment and not dealing with private data, I don't bother...

You may also need to create an SSH key and add it to your Google Cloud account. Consider using
[gcloud's guide on ssh keys](https://cloud.google.com/compute/docs/connect/add-ssh-keys#metadata) (or OS Login if you do that)
to set up ssh keys and [using `ssh-agent`](https://kb.iu.edu/d/aeww) to make executing the SSH commands easier.

## Using `launch.py`

### Configuration

You will need a [Docker installation](https://docs.docker.com/engine/install/)
on your development machine to build and run images on TPUs.

First create a configuration file for future launches in your Levanter directory:

```bash
cat > .levanter.yaml <<EOF
env:
    WANDB_API_KEY:
    WANDB_ENTITY:
    WANDB_PROJECT:
    HF_TOKEN:
    TPU_STDERR_LOG_LEVEL: 2
    TPU_MIN_LOG_LEVEL: 2
    LIBTPU_INIT_ARGS: <extra args to libtpu>  # Optional

# Optional: specific environment variables for TPUs based on the TPU type
accel_env:
  v6e:
     # If you're lucky enough to have a v6e, you can set the following, which is pretty important for performance
     LIBTPU_INIT_ARGS: "--xla_tpu_scoped_vmem_limit_kib=98304"

docker_repository: levanter  # default
zone: us-west4-a  # if not set, will use your default zone
tpu: test-spin-up-32  # name of the TPU
capacity_type: "preemptible"
subnetwork: "default"  # default
EOF
```

If you want to customize the docker image that is created and uploaded to Artifact Registry, you can add config `image_name: "YOUR-DOCKER-NAME"`.

#### (Optional) Using GitHub Container Registry

Note that you can also Configure docker to push to GHCR by setting

```yaml
docker_registry: ghcr
github_user: <YOUR USERNAME>
github_token: <YOUR TOKEN>
```

By default, the TPU instance won't be able to access the Docker image, so you may need to make it public. To do
so, navigate to the GitHub profile or organization that owns the Docker image (e.g. https://github.com/orgs/stanford-crfm/packages),
click on the package, and then click on the "Make public" button. GitHub will display a scary warning about how
this will make the package public, but that's what you want.

To get a GitHub token, see [this guide on creating access tokens](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)
and [the GitHub Container Registry docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry).

### Launch a GPT-2 Small in the background

Now run `launch.py`. This will package your current directory into a Docker image and run it on your workers. Everything after the `--` is run on each worker.

```bash
python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

The command you run should be run as though it's being run on the TPU VM, from the root of the Levanter repo. Everything
in your current directory not covered by `.dockerignore` will be copied to the TPU VM. (This can lead to surprises
if you have large files in your directory that you don't want to copy over.)

### Launch a GPT-2 Small in interactive mode

To run in the foreground, use `--foreground` with the `launch.py` script. You should use tmux or something for long-running jobs for this version.

```bash
python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

### Running your own config

If you want to run your own config, we suggest you start from one of the existing configs. Just copy it to
a new file:

`cp config/gpt2_small.yaml config/my_config.yaml`

If you're using `launch.py`, the config will be automatically uploaded as part of your Docker image, so you
can just reference the local config path in your command line:

```bash
python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/my_config.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

Afterward, you can use the config directly from the TPU VM instance, e.g.:

```bash
    python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/
```

With this configuration (unless `trainer.load_checkpoint` is false), Levanter will automatically
try to load the latest checkpoint if it exists.

Tokenizers and configuration files are loaded via `fsspec` which supports remote
filesystems, so you can also copy your tokenizer or config file to GCS and use
a `gs://` path to access it.

### Using an external directory or file

In case that you want to reference some external directory/file outside of the levanter repo, you can do it by adding the external directory/file to the docker image so that it becomes accessible in TPU instances. You can specify the path you want to add as extra buildl context by `--extra_context` with the `launch.py` script. Then, you should be able to use the external files in arguments in `train_lm.py` etc.

```bash
python infra/launch.py --extra_context <external path> -- python src/levanter/main/train_lm.py --config_path <external path> --trainer.checkpointer.base_path gs://<somewhere>'
```

### Babysitting script for preemptible TPUs

If you are using a preemptible TPU VM, you probably want to use the "babysitting" version of the script to keep an eye on
the VM. This is because preemptible instances can be preempted and will always be killed every 24 hours.
You can run `launch.py` with the `--retries` and `--foreground` parameter to accomplish this.
If `--retries` is greater than 1, `launch.py` will automatically attempt to re-create the VM and re-run the command if it fails. (`--foreground` is necessary to keep the script from returning immediately.)

```bash
    python infra/launch.py --retries=100 --foreground --tpu_name=my_tpu -- python src/levanter/main/train_lm.py --config_path config/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/
```

That `--` is important! It separates the spin up args from the running args.
Also you should always use `--foregrouund` with `babysit-tpu-vm`, as the
background mode will always return immediately.


## Using the Ray Autoscaler

We use Ray's autoscaler to manage the TPU VM instances. This is a more robust way to manage the instances, as it will
automatically restart them if they fail. It also allows you to easily scale up the number of instances if you need more
compute.

### Configuration

Since Levanter already uses Ray, you don't need to install anything new. You just need to set up your configuration file.
We have a template configuration file in `infra/cluster/job-cluster.yaml`. You can modify this file to suit your needs.
In particular, you should set the GCP project, zone, and which TPU slice types you want to use. The default configuration
enables v4 slices of various sizes.

**Note that the default configuration uses an n2-standard-2 instance as the head node. This costs about $70/month.**
This is considerably smaller than [Ray's guidance for the head node](https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html#configuring-the-head-node).
If you need to save money, you can also look into committing to a year of usage to save money, or potentially you could
use a non-preemptible TPU VM instance as the head node if you have non-preemptible TRC TPUs.

### Launching the Cluster

To launch the cluster, you can run the following command:

```bash
ray up infra/cluster/job-cluster.yaml
```

This will create the head node and the minimum number of workers. You can then submit jobs to the cluster. First,
you should establish a connection to the Ray dashboard:

```bash
ray dashboard infra/cluster/job-cluster.yaml
```

Then, **in a separate terminal**, you can submit a job to the cluster. To replicate the previous example, you can run:

```bash
export RAY_ADDRESS=http://localhost:8265  # tell ray where the cluster is
python infra/launch_on_ray.py --tpu_type v4-32 --foreground -- python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

Even without `--foreground`, the job will be restarted if it fails. The `--tpu_type` flag is required, and should be
one of the TPU types you enabled in the cluster configuration.

This command will print various options for monitoring the job. You can use the Ray dashboard to monitor the job, or you can
stop the job with:

```bash
ray job stop <job_id>
```

If `--foreground` is present, the script will tail the logs of the job.

### Monitoring the Cluster

If you've launched the cluster, you can look at the Ray dashboard to see the status of the cluster by
navigating to `http://localhost:8265` in your browser. You can also monitor the autoscaler logs with the following command:

```bash
ray exec infra/cluster/job-cluster.yaml "tail -n 100 -f /tmp/ray/session_latest/logs/monitor*"
```

## Common Issues

### Can't find TPUs

If you get the warning `No GPU/TPU found, falling back to CPU.` then something else might be using the TPU, like a zombie python
process. You can kill all python processes with `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'pkill python'`

If that fails, you can try rebooting the TPU VM instance. This is harder than it sounds, because you can't just reboot the instance
from the console or with a special command. You have to do it with ssh:

```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'sudo reboot'
```

**and then you have to ctrl-c this after about 10 seconds**. Otherwise, gcloud will think the command failed and will
try again, and get stuck in a loop forever. (You can ctrl-c it at any point after 10 seconds.)

### Docker-related Issues

If you get a `permission denied` error with your `docker.sock`,
you should be able to fix it by running the following script in the launching machine and then restarting your shell:

```shell
sudo usermod -aG docker $USER
```

If you get an error like `denied: Unauthenticated request. Unauthenticated requests do not have permission "artifactregistry.repositories.uploadArtifacts"`,
You should run the following authentication script:

```shell
# change based on your GCP zone
gcloud auth configure-docker ${GCP_ZONE}.pkg.dev
# for example:
gcloud auth configure-docker us-central2-docker.pkg.dev
```

#### Too big of a Docker image

If you're concerned your Docker images are taking too long to push, especially after a first push, you can try to
reduce the size of the image. One way to do this is to add more entries to the `.dockerignore` file in the root of the
Levanter repo. This file is used by Docker to determine what files to ignore when building the image.

To see what files are likely taking up the most space, you can run the following command:

```bash
ncdu -X .dockerignore
```

This will show you a list of files and directories in the repo, sorted by size, excluding files that are in the `.dockerignore` file.
(There are slight differences between how `ncdu` and Docker interpret the `.dockerignore` file, so this isn't perfect, but it's usually pretty close.)

## Creating a TPU VM Instance



### Automatic Setup

!!! warning
    This approach is deprecated and will be removed in the future. Please use `launch.py` or `launch_on_ray.py` instead.

You can use `infra/spin-up-vm.sh` to create a TPU VM instance. In addition to creating the instance, it will set up
the venv on each worker, and it will clone the repo to `~/levanter/`.

```bash
bash infra/spin-up-vm.sh <name> -z <zone> -t <type> -n <subnetwork> [--preemptible] [--use-alpha]
```

Defaults are:
- `zone`: `us-east1-d`
- `type`: `v3-32`
- `subnetwork`: `default` (set to custom VPC subnet, useful for tpuv4 configs)
- `preemptible`: `false`
- `use-alpha`: `false` (mark `true` for tpuv4s in alpha zones like `us-central2`)

**Notes**:

* This uploads setup scripts via scp. If the ssh-key that you used for Google Cloud requires passphrase or your ssh key
path is not `~/.ssh/google_compute_engine`, you will need to modify the script.
* The command will spam you with a lot of output, sorry.
* If you use a preemptible instance, you probably want to use the ["babysitting" script](#babysitting-script) to
the VM. That's explained down below in the [Running Levanter GPT-2](#running-levanter-gpt-2) section.


## Useful commands

### SSHing into one TPU VM worker

`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=0`

### Running a command on all workers (in parallel)
`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=all --command="echo hello"`

### SCPing a file to all workers
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=all`

### SCPing a file to one worker
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=0`

### SCPing a file from one worker
`gcloud compute tpus tpu-vm scp $name:path/to/file my_file --zone us-east1-d --worker=0`
