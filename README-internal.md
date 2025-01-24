# Marin (internal)

This documentation is for internal developers.

## Setup

Behind the scenes, we run an instance of Marin on Google Cloud Platform (GCP).
Make sure to [install `gcloud`](https://cloud.google.com/sdk/docs/quickstarts) and then run:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set account <your-email-account>
gcloud config set project hai-gcp-models

# [Verification] Should show a [core] entry with `account = <your-email-account` and `project = hai-gcp-models`
gcloud config list

# [Verification] Should not throw a permission error
gcloud storage ls gs://marin-us-central2
```

If you don't have permissions for `hai-gcp-models` or you run into permissions
issues, contact David Hall or Sidd Karamcheti for help!

## Ray Cluster + Job Submission

Once authenticated for GCP, all other work happens through our
[Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). The entire cluster configuration template is stored in [`infra/marin-cluster-template.yaml`](./infra/marin-cluster-template.yaml). **Ray uses this file as the single-source of
truth for all cluster operations** -- you can think of this file as an alternative to managing your own SSH keys,
remembering the IP address of the cluster head node, what port the dashboard is running on, etc. For more info about this template and connecting to specific clusters, see [`infra/README.md`](./infra/README.md).

There are two steps necessary for 1) establishing a connection to the cluster and 2) submitting/monitoring jobs on the
cluster. **You will need at least two terminal processes running for the following steps** (make sure to activate your
`marin` Python environment as well):

```bash
# [Terminal 1] Establish a Connection to the Ray Dashboard (launches an ssh connection w/ port-forwarding)
#   =>> Assumes `marin` Python environment is active, and you're running scripts from the repository root directory
ray dashboard infra/marin-cluster.yaml

# [Browser] Navigate to `http://localhost:8265` (or whatever URL is output by the above command)
#   =>> You should see the Cluster Overview Page (with a list of recent jobs, node status, resource status)
```

In addition to linking you to the cluster dashboard, the above command will establish a (persistent) SSH connection to
our cluster's head node (if you're familiar with the NLP SLURM workflow, think of this as a connection to `sc`). Keep
this terminal open!

To submit jobs, we use the
[Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#submitting-a-job).
This requires that your Python script is formatted in a certain way, calling some boilerplate Ray functions prior to
launching tasks -- see [`tests/test_quickstart.py`](./tests/test_quickstart.py) for a minimal example. To launch:

```
# [Terminal 2] Submit a Ray Job (specified via a Python script)
#   =>> Assumes `marin` Python environment is active, current working directory == repository root == "."
#   =>> Will output a Job ID like `raysubmit_pAJM8vKfHPhiyHBa`
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python tests/test_quickstart.py

# Get Job Status (given Job ID = raysubmit_pAJM8vKfHPhiyHBa)
ray job status --address http://127.0.0.1:8265 raysubmit_pAJM8vKfHPhiyHBa

# Get Job Logs (Console Out)
ray job logs --address http://127.0.0.1:8265 raysubmit_pAJM8vKfHPhiyHBa

# Kill / Stop Job (if necessary / error / bug)
ray job stop --address http://127.0.0.1:8265 raysubmit_pAJM8vKfHPhiyHBa
```

To avoid passing `--address ...` you can set the environment variable `export RAY_ADDRESS="http://127.0.0.1:8265"`

**Quality of Life**: If you like `tmux` and `conda` (with environment name `marin`), feel free to run
[`infra/marin-tmux.sh`](./infra/marin-tmux.sh) that automates launching the dashboard for you. Make sure to read the
script before running!

## Programming Guidelines

### Using Ray

When using Ray, use the `ray.remote` decorator for any function that you want to run distributed.
`ray.remote` executes the function in a separate process, possibly on a separate machine.

```python
@ray.remote
def my_function():
    ...

result_future = my_function.remote()

# Get the result of the function
result = ray.get(result_future)
```

Ray is a resource-aware job scheduler, so you can specify the resources that a job requires:

```python
@ray.remote(num_cpus=4)
def my_cpu_job():
    ...
```

Please see the [Ray documentation](https://docs.ray.io/en/latest/index.html) for more information, though
see the next section for some important notes about using TPUs.

### Using TPUs on Ray

You can use our workers like normal CPU instances, just using the `ray.remote` decorator. However, if you want to use
the TPUs, you need to tell Ray that you need them. To use TPUs on our cluster, you should use the following:

```python
@ray.remote(num_cpus=8, resources={"TPU": 4, "TPU-v4-8-head": 1})
def my_tpu_job():
    ...
```

Always use the `TPU-v4-8-head` resource when requesting TPUs unless you specifically want a multi-node slice. This will
ensure you don't accidentally grab part of a multi-node slice, which will lead to weird errors.

Also, despite it saying `"TPU": 4`, you're actually getting all the TPUs. TPU v4 slices have 4 boards with 2 cores each,
so you're getting 8 cores total, but they present as 4 TPUs.

**IMPORTANT**: Ray and `libtpu` don't always get along. If you are using TPUs, you should either fork a process that
uses the TPUs or force remove the libtpu lockfile when your task finishes. The latter is very hacky, but it works.
We offer a utility decorator to do this for you:

```python
from marin.utils import remove_tpu_lockfile_on_exit

@ray.remote(num_cpus=8, resources={"TPU": 4, "TPU-v4-8-head": 1})
@remove_tpu_lockfile_on_exit
def my_tpu_job():
    ...
```

## Evaluation

By default, we run evaluations on TPU using `lm-evaluation-harness` via Levanter. See
[`experiments/evals/README_eval.md`](./experiments/evals/README_eval.md) for more details.
