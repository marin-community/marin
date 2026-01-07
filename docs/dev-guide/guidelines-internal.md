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

### Ray token authentication

Marin clusters use Ray token authentication (Ray >= 2.53). You will typically interact with clusters through SSH
port-forwarding on `localhost`, but you still need the token.

This is the easiest flow:

1. Install the default token locally (or re-run `make dev_setup`):
    ```bash
    make get_ray_auth_token
    ```
    If this fails because the Secret Manager secret doesn’t exist yet, someone needs to create it once (see
    `infra/README.md`).
2. Authenticate to the cluster dashboard:
    ```bash
    # Starts SSH port-forwarding, copies the token to clipboard, and opens the dashboard.
    uv run scripts/ray/cluster.py --cluster us-central2 auth
    ```

#### Using Ray CLI/SDK with token auth

If you are using `uv run scripts/ray/cluster.py ...`, you generally don’t need to set any Ray auth env vars manually.

However, if you want to use Ray’s CLI/SDK directly (e.g. `ray job submit ...`), you will need to set the appropriate
environment variables for Ray token authentication.

To set them for your current shell (sets `RAY_AUTH_MODE=token` and `RAY_AUTH_TOKEN_PATH=...`), ensure you have a Ray auth
token locally at `~/.ray/auth_token` (or set `RAY_AUTH_TOKEN_PATH`) and then:

```bash
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="${RAY_AUTH_TOKEN_PATH:-$HOME/.ray/auth_token}"
```

You can add these lines to your shell profile (e.g. `~/.bashrc` or `~/.zshrc`) if you want them to be set
automatically in future terminal sessions.

### Connecting to the Ray Cluster Dashboard and Submitting Jobs

There are two steps necessary for 1) establishing a connection to the cluster and 2) submitting/monitoring jobs on the
cluster. **You will need at least two terminal processes running for the following steps** (make sure to activate your
`marin` Python environment as well):

```bash
# [Terminal 1] Establish Ray dashboard connections (port-forwarding)
uv run scripts/ray/cluster.py dashboard

# [Browser] Navigate to the URL printed above for the overall dashboard.
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
uv run lib/marin/src/marin/run/ray_run.py --cluster infra/marin-us-central1.yaml --no_wait -e WANDB_API_KEY ${WANDB_API_KEY} -- python experiments/tutorials/hello_world.py

# Get Job Status
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml list-jobs

# Get Job Logs for a specific job
# Ensure that the dashboard for the correct cluster is running (run this in another terminal)
# > uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml dashboard
ray job logs --address "http://127.0.0.1:8265" <JOB_ID>

# Kill / Stop Job (if necessary / error / bug)
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml stop-job raysubmit_pAJM8vKfHPhiyHBa
```
