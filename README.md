# Marin

> "*I am not afraid of storms, for I am learning how to sail my ship."*<br/>
> – Louisa May Alcott

## Getting Started

To get set up, create a new virtual environment (or `conda` environment) with the appropriate Python version (3.10),
then run the following:

```bash
git clone https://github.com/stanford-crfm/marin
cd marin
pip install -e ".[dev]"
```

This will install all the core dependencies and build `marin` as a Python package. Installing the `[dev]` requirements
will additionally install test, linting, and debugging dependencies (e.g., `pytest`).


## Ray + Google Cloud Platform Quickstart

In order to run data processing and curation workloads, we use [Ray](https://docs.ray.io/), a nifty Python library for
configuring and launching distributed applications. We use Ray on top of Google Cloud Platform to 1) automatically 
provision and maintain a cluster of virtual machines (VMs), and 2) to launch individual "jobs" (units of work) on our
cluster. For more detailed information [see the `infra` README](./infra/README.md).

#### Google Cloud (`gcloud`) Setup

The **most important prerequisite** is making sure that your development environment (e.g., laptop) is set up for 
Google Cloud Platform (GCP) for project `hai-gcp-models`. Make sure to 
[install `gcloud`](https://cloud.google.com/sdk/docs/quickstarts), then: 

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set account <your-email-account>
gcloud config set project hai-gcp-models

# [Verification] Should show a [core] entry with `account = <your-email-account` and `project = hai-gcp-models`
gcloud config list

# [Verification] Should not throw a permission error
gcloud storage ls gs://marin-data
```

If you don't have permissions for `hai-gcp-models` or you run into permissions issues, contact David Hall or 
Sidd Karamcheti for help!

#### Ray Cluster + Job Submission

Once authenticated for GCP, all other work happens through our 
[Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). The entire cluster configuration
is stored in [`infra/marin-cluster.yaml`](./infra/marin-cluster.yaml). **Ray uses this file as the single-source of 
truth for all cluster operations** -- you can think of this file as an alternative to managing your own SSH keys, 
remembering the IP address of the cluster head node, what port the dashboard is running on, etc. 

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
launching tasks -- see [`tests/test_ray_cluster.py`](./tests/test_ray_cluster.py) for a minimal example. To launch:

```
# [Terminal 2] Submit a Ray Job (specified via a Python script)
#   =>> Assumes `marin` Python environment is active, current working directory == repository root == "."
#   =>> Will output a Job ID like `raysubmit_pAJM8vKfHPhiyHBa`
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python tests/test_ray_cluster.py

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

---

## Organization

### Scripts

Scripts go in `scripts/$domain/`. Once there's a script for actually creating the domain, let's make a README.md in 
that directory that explains how to use the script.

### Source

Markdown conversion goes in `marin/markdown/`.

Library-y source code goes in `marin/$domain/`.


# Domains

## Web

### Working on Markdown conversion

My (@dlwh) workflow looks like this:

* `export PYTHONPATH=.:$PYTHONPATH`, or use an IDE that does this for you.
* find a web page that I'm concerned about
* run `python3 scripts/web/process_url.py <url>` 
* look at the outputs in `output/`. In particular compare `outputs/name.readability.html` to `outputs/name.md` to see what the conversion looks like.
* If you need to, make a gist of the md at https://gist.github.com/ and look at how GitHub renders it. This is the gold standard for what we're aiming for.

#### Adding a test case

We use snapshot testing in addition to unit tests. To add a test case, do the following:

* Add an html file to `tests/snapshots/inputs/` that you want to test.
* Add the expected markdown output to `tests/snapshots/expected/` with the same name as the input file.
* Commit these files.

Pro-tip: You can copy the markdown from `process_url.py`'s output to the expected file and edit it as needed.

If it's reasonable, try to add a unit test as well. This will help ensure that the conversion is correct.

#### Running tests

To run the tests, run `pytest` in the root directory. This will run the unit tests and snapshot tests.

#### Updating snapshots

If you've made a change that you think is correct, you can update the snapshots by copying `tests/snapshots/outputs/` to `tests/snapshots/expected/`. This will overwrite the expected output with the new output. You should review these changes before committing them.


## Wikipedia

TODO

## ArXiv

TODO

## Code

TODO

## Instruction Data

XXX

## Books

XXX

## StackExchange

XXX

## Reddit/Forums

XXX

## CC News?

XXX

## Semantic Scholar??

XXX

## PubMed

TODO

## Law

TODO
