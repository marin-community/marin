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

make dev_setup
```

If you don't have permissions for `hai-gcp-models` or you run into permissions
issues, contact David Hall for help!

## Iris Cluster + Job Submission

Once authenticated for GCP, all other work happens through our
[Iris cluster](https://github.com/marin-community/marin/blob/main/lib/iris/README.md). Cluster config templates live
under [`lib/iris/examples/`](https://github.com/marin-community/marin/tree/main/lib/iris/examples):

- `marin.yaml` — production TPU cluster on GCP
- `marin-dev.yaml` — dev TPU cluster on GCP (smaller scale caps)
- `coreweave.yaml` — GPU cluster on CoreWeave

**Iris uses these configs as the single source of truth for cluster operations** -- controller URL, SSH tunneling, and
auth are all derived from the config file you pass with `--config`. You don't need to manage SSH keys, head-node IPs,
or dashboard ports yourself.

For day-to-day debugging and operating live clusters, see
[`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).

### Connecting to the Dashboard and Submitting Jobs

There are two steps necessary for 1) establishing a connection to the cluster and 2) submitting/monitoring jobs on the
cluster. **You will need at least two terminal processes running for the following steps** (make sure to activate your
`marin` Python environment as well):

```bash
# [Terminal 1] Open SSH tunnel, print the dashboard URL, and block.
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard

# [Browser] Navigate to the URL printed above.
# Clicking a job opens its per-task/log view.
```

The tunnel lives as long as this process does — keep the terminal open. (If you're familiar with the NLP SLURM
workflow, think of this as your connection to `sc`.)

To submit jobs, use `iris job run`. Job IDs are canonical paths of the form `/<user>/<job-name>-<timestamp>`:

```bash
# [Terminal 2] Submit a job and return immediately.
#   =>> Will print a line like `Job submitted: /<user>/hello_world-20260420-120000`
uv run iris --config lib/iris/examples/marin.yaml job run \
    --no-wait --extra marin:tpu --tpu v5litepod-16 \
    -- python experiments/tutorials/hello_world.py

# List jobs (filter by --state, --user, --prefix, etc).
uv run iris --config lib/iris/examples/marin.yaml job list

# Follow logs (--include-children streams per-task output as well).
uv run iris --config lib/iris/examples/marin.yaml job logs --include-children /<user>/<job-name>

# Kill / Stop Job (if necessary / error / bug) -- kills the job and all its child tasks.
uv run iris --config lib/iris/examples/marin.yaml job stop /<user>/<job-name>
```

Notes:

- `--extra marin:tpu` installs the Marin TPU deps into the task container; use `--extra marin:cpu` for CPU-only
  entrypoints and `--extra gpu` (with `--gpu H100x8`) on CoreWeave.
- Request TPU hardware with `--tpu v5litepod-16` (or similar). `--reserve` only holds capacity for scheduling; it does
  not attach accelerator devices to the task container.
- `WANDB_API_KEY`, `HF_TOKEN`, `HF_DATASETS_TRUST_REMOTE_CODE`, and `TOKENIZERS_PARALLELISM` are auto-injected from
  your shell. Pass other env vars with `-e KEY VALUE` (two positional args — quote `$VALUE` if it may be unset).
- See `iris job run --help` and [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
  for the full flag list (`--memory`, `--job-name`, priority bands, etc.).

## Forking External Packages

See [Forking Policy](forking-policy.md) for our requirements on maintaining
forked dependencies under `marin-community/`.

## Precommit

`./infra/pre-commit.py --all-files --fix`
