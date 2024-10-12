# Ray - Autoscaling Cluster for Marin Data Processing

[Ray](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) provides a simple interface
for programmatically spinning up compute and scheduling/running jobs at scale. For Marin, we structure our data
processing jobs to be Ray compatible, so we can quickly parallelize across different nodes.

At a high-level, this directory provides all setup scripts and configuration files for standing up a Ray Cluster and
interacting with it to do lightweight monitoring and reconfiguration. The architecture of our cluster is as follows:

+ **Head Node**: A *persistent* (on-demand)
  [`n2-standard-8` GCP VM](https://cloud.google.com/compute/docs/general-purpose-machines) with 8 CPUs and 32 GB of
  RAM, and a 200 GB disk.
+ **Worker Nodes**: An autoscaling number of TPU v4-8 VMs; a minimum of 4 VMs will be kept alive at all times, with a
  maximum of 1024 VMs alive at once (we can increase this number).

**Useful Documentation** -- Time-permitting, consult the official Ray documentation to learn more about how Ray works,
with helpful examples for working with Clusters and Jobs:
- [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html): Provides a walkthrough of Ray "tasks" (atomic
  unit of work to be distributed), as well as `@ray.remote` syntax. For concrete code, try
  [this guide with examples](https://docs.ray.io/en/latest/ray-core/tasks.html).
- [Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html): Ray Jobs
  quickstart, with additional details about "special case" jobs (specifying CPUs, adding new Python packages).
- [Ray Cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html): Nitty-gritty of Ray Clusters.
  After reading overview, see
  [this guide for deploying on GCP](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html)
  and [this repository for setting up with TPUs](https://github.com/tensorflow/tpu/tree/master/tools/ray_tpu/src/serve).


## Structuring Code & Running Jobs

The way we're using Ray for data curation / processing is really as a distributed task queue -- we define an individual
Python script for each "job" (e.g., running a toxicity classifier on Reddit), breaking the computation into "tasks" that
are parallelized across the entire cluster.

A task is a "pure" function that operates over some atomic unit of the overall job (e.g., running the toxicity
classifier on a single Reddit thread). A good rule of thumb is to break a job down into individual tasks that can
complete within 1-10 minutes.

The following provides a simple sketch of what a job script might look like:

```python
from marin.toxicity import toxicity_classifier
from marin.utils import fsspec_exists

import fsspec
import ray


def get_sentinel_path(gcs_file_path: str) -> str:
    return f"{gcs_file_path}.COMPLETE"


def finalize_sentinel(sentinel_gcs_path: str, content: str = "Task Completed") -> None:
    with fsspec.open(sentinel_gcs_path, "w") as f:
        f.write(content)


# Task Function --> this actually will be parallelized across cluster workers
@ray.remote
def classify_reddit_toxicity(gcs_reddit_input: str, gcs_toxicity_output: str) -> bool:
    """Read from input, perform toxicity classification, write to output -- return success/failure."""

    # [Short-Circuit] If "sentinel" file exists, task has been successfully completed
    sentinel_gcs_path = get_sentinel_path(gcs_toxicity_output)
    if fsspec_exists(sentinel_gcs_path):
        return True

    # Read and validate `gcs_reddit_input`
    ...

    # Run toxicity classifier and get scores --> write to `gcs_toxicity_output` as you go!
    with fsspec.open(gcs_toxicity_output, "w", compression="infer") as out:
        ...

    # Finalize Sentinel
    finalize_sentinel(sentinel_gcs_path)

    return True


# Main Function (runs in a single processing on cluster head node) --> responsible for "dispatching" tasks
def run_toxicity_classifier_reddit() -> None:
    """Iterate over Reddit threads and run toxicity classifier."""

    # Load List of GCS Input Paths (e.g., one file for K threads)
    gcs_reddit_inputs: List[str] = [...]

    # Create Corresponding List of GCS Output Paths (to store toxicity scores)
    gcs_toxicity_outputs: List[str] = ["<...>" for input_path in gcs_reddit_inputs]

    # Initialize Connection to Ray Cluster
    ray.init()

    # Invoke / Dispatch Tasks (call .remote() --> return *promises* -- a list of references to task output)
    success_refs = []
    for i in range(len(gcs_reddit_inputs)):
        success_refs.append(classify_reddit_toxicity.remote(gcs_reddit_inputs[i], gcs_toxicity_outputs[i]))

    # Resolve / Verify Task Successes (call .get() on individual references)
    #   =>> TODO (siddk) :: There's actually a cleaner API for this that doesn't block on task ordering... fix!
    task_successes = {input_path: False for input_path in gcs_reddit_inputs}
    for i, input_path in enumerate(gcs_reddit_inputs):
        successful = ray.get(success_refs[i])  # Blocks until ready
        task_successes[input_path] = successful

    # Cleanup -- Write Successes / Failures to GCS
    ...


if __name__ == "__main__":
    run_toxicity_classifier_reddit()
```

Note the "short-circuiting" structure of the `@ray.remote` decorated function `classify_reddit_toxicity`; in general,
writing your job scripts and tasks so that they are
[*idempotent*](https://stackoverflow.com/questions/1077412/what-is-an-idempotent-operation) (can be invoked multiple times, always returning the same output) is a good idea, and gracefully
handles cases where individual tasks crash or a VM worker gets preempted in the middle of running a task.

---

# Maintaining a Ray Cluster

The [`cluster.yaml`](./marin-cluster.yaml) file has all the necessary configuration details for the cluster. Some useful
commands:

```bash
# Launch the Cluster -- will automatically provision the head node, and start configuring the minimum number of workers
ray up -y infra/marin-cluster.yaml

# Kill the Cluster (takes a while to gracefully terminate nodes)
ray down -y infra/marin-cluster.yaml

# Monitor the Ray Autoscaler Logs (in case there are problems spinning up workers)
#   =>> Note that `exec` lets you run arbitrary commands on the head node!
ray exec infra/marin-cluster.yaml "tail -n 100 -f /tmp/ray/session_latest/logs/monitor*"

# SSH into the Head Node
ray attach infra/marin-cluster.yaml
```

By default, the cluster is set up in availability zone `us-central-2b` with a persistent `n2-standard-8` VM acting as
the head node. At any given time, there should be a minimum of 4 TPU `v4-8` VMs acting as workers, with an autoscaling
limit of 1024 VMs (so a maximum of 8 * 1024 = 8,192 total v4 cores or 4,096 v4 chips).

Each TPU `v4-8` VM actually has a surprising amount of CPUs and RAM (~240 CPUs, 200GB+ of RAM). However, Ray doesn't
actually do anything under the hood to ensure that a job is actually using the number of logical resources specified
(e.g., a job that on paper requests 1 CPU can use arbitrary cores/RAM). To mitigate this on the scheduler side, the
`cluster.yaml` file configures each worker with only 120 visible CPUs.

### Restarting the Cluster
There is currently an error on the Ray autoscaler side with spot-TPU instances, where the Ray autoscaler is not able
to detect when spot-TPU instances are dead and as a result, we may be left in a state with just the head node and
no more spot-TPU worker instances starting up. When this state occurs, please message in the #marin-infra slack
that you are going to restart the cluster (call `ray down infra/marin-cluster.yaml` and then `ray up infra/marin-cluster.yaml`).

Please note that after restarting the cluster and queueing the first job, it will likely stall because it takes a while for the head node to actually spin up the worker sometimes (~10min).


### Reconfiguring the Cluster

In general, for additive operations like increasing the `max_workers` for autoscaling, you can just call `ray up`
against the already-running cluster.

If you need to make substantive changes to the machine software, you should change the Docker file at
`docker/marin/Dockerfile.cluster`. Then run `make cluster_docker` to rebuild the Docker image and push it to the
Google Artifact Registry. This will create a new image and a new tag, of the form
`"us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:20241011"`. Replace the date with the current date
and edit the `cluster.yaml` file to point to the new image:

```yaml
...
docker:
  image: "us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:20241011"
```

After that, you can restart the cluster with `ray down` and `ray up`.

If you need to change something else about the cluster, e.g. if you're changing any of the initialization/setup
commands, it's best to bring the entire cluster down (`ray down`), *then edit the `cluster.yaml`*, and then bring the
cluster back up (`ray up`); note that this will kill all VMs, including the head node (nothing lasts forever).

**Word of Warning**: Ray looks at the actual `cluster_name` and various worker names/configs to "identify" existing/new
clusters. To prevent orphaned states, do not change the names of anything in the `cluster.yaml` without first bringing
the cluster down!


#### Environment Variables

We are currently using Google Secret Manager to store the environment variables that are needed to run the cluster.
You can edit those secrets by going to the Google Cloud Console and navigating to the Secret Manager. Once you add
a new version, you can cause the changes to propagate by killing the workers or restarting the cluster.
