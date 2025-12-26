# Marin (internal)

This documentation is for internal developers.

## Prerequisites

- Please read the general guidelines in [guidelines.md](../explanations/guidelines.md)
- Complete the environment setup in [installation.md](../tutorials/installation.md)

## Setup

Behind the scenes, we run an instance of Marin on Google Cloud Platform (GCP).

Ensure that someone (e.g. @dlwh) adds you to the `hai-gcp-models` project on GCP
as a `Marin Dev`. Make sure to [install `gcloud`](https://cloud.google.com/sdk/docs/quickstarts) and then run:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set account <your-email-account>
gcloud config set project hai-gcp-models

# [Verification] Should show a [core] entry with `account = <your-email-account` and `project = hai-gcp-models`
gcloud config list

# [Verification] Should not throw a permission error
gcloud storage ls gs://marin-us-central2

# Ensure you have the cluster ssh key
make dev_setup
```

If you don't have permissions for `hai-gcp-models` or you run into permissions
issues, contact David Hall for help!

## Ray Cluster + Job Submission

Once authenticated for GCP, all other work happens through our
[Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). The entire cluster configuration template is stored in [marin-cluster-template.yaml](https://github.com/marin-community/marin/blob/main/infra/marin-cluster-template.yaml). **Ray uses this file as the single-source of
truth for all cluster operations** -- you can think of this file as an alternative to managing your own SSH keys,
remembering the IP address of the cluster head node, what port the dashboard is running on, etc. For more info about this template and connecting to specific clusters, see [README.md](https://github.com/marin-community/marin/blob/main/infra/README.md).

There are two steps necessary for 1) establishing a connection to the cluster and 2) submitting/monitoring jobs on the
cluster. **You will need at least two terminal processes running for the following steps** (make sure to activate your
`marin` Python environment as well):

```bash
# [Terminal 1] Establish Ray dashboard connections (port-forwarding)
uv run scripts/ray/cluster.py dashboard

# [Browser] Navigate to the URL printed above (typically http://localhost:9999) for the overall dashboard.
# Clicking a cluster opens its overview page with jobs/nodes/resources.
```

In addition to linking you to the cluster dashboard, the above command will establish a (persistent) SSH connection to
our cluster's head node (if you're familiar with the NLP SLURM workflow, think of this as a connection to `sc`). Keep
this terminal open!

To submit jobs, we use the
[Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#submitting-a-job).
This requires that your Python script is formatted in a certain way, calling some boilerplate Ray functions prior to
launching tasks -- see [test_integration_test.py](https://github.com/marin-community/marin/blob/main/tests/test_integration_test.py) for a minimal example. To launch:

```bash
# [Terminal 2] Submit a Ray Job (specified via a Python script)
#   =>> Will output a Job ID like `raysubmit_pAJM8vKfHPhiyHBa`
uv run python marin/run/ray_run.py --no_wait --env_vars WANDB_API_KEY=${WANDB_API_KEY} -- python experiments/hello_world.py

# Get Job Status
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml list-jobs

# Kill / Stop Job (if necessary / error / bug)
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml stop-job raysubmit_pAJM8vKfHPhiyHBa
```
