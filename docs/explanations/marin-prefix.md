# Understanding `MARIN_PREFIX`

Marin uses a designated storage location, referred to as a "prefix," to save all outputs from experiments. This includes tokenized data, model checkpoints, experiment logs, and other artifacts. You specify this location with the `MARIN_PREFIX` environment variable.

## What is the Prefix Used For?

The prefix defines the root directory where Marin will store:
-   **Tokenized Datasets:** Processed datasets ready for training.
-   **Model Checkpoints:** Saved states of your models during and after training.
-   **Experiment Configurations and Logs:** JSON files detailing experiment setups, along with other logs.
-   **Evaluation Results:** Outputs from evaluation harnesses.

Every artifact lands at `{prefix}/{name}/{version}`, so the prefix is the only part of an
artifact's address that depends on where you are running.

## Specifying the Prefix

Set `MARIN_PREFIX` in the environment the experiment runs in. There is no command-line
override; scripts read the variable at run time.

```bash
export MARIN_PREFIX="/path/to/your/output_directory"
python experiments/your_experiment.py --version dev --run
```

For a one-off run, set it inline:

```bash
MARIN_PREFIX=local_store python experiments/your_experiment.py --version dev --run
```

On a cluster, the task environment supplies `MARIN_PREFIX` — Iris sets it for GCP and
CoreWeave tasks — so a job inherits the prefix for the region it lands in.

## Acceptable Storage Backends and Paths

Marin leverages the `fsspec` library, allowing you to use various storage backends. The path you provide should be a URI understandable by `fsspec`. Common examples include:

*   **Local Filesystem:**
    *   `export MARIN_PREFIX=/path/to/local/directory`
    *   `export MARIN_PREFIX=./relative/path/to/output` (relative to where you run the script)

*   **Amazon S3 / CoreWeave AI Object Storage:**
    *   `export MARIN_PREFIX=s3://your-s3-bucket/path/to/output`
    (Requires appropriate credentials and the `s3fs` library installed: `uv pip install s3fs`)

*   **Google Cloud Storage (GCS):**
    *   `export MARIN_PREFIX=gs://your-gcs-bucket/path/to/output`
    (Requires appropriate GCP credentials and `gcsfs` library installed: `uv pip install gcsfs`)

## Important Considerations for Distributed Environments

When running Marin in a distributed setup (e.g., across multiple nodes via Iris), it is **critical** that the specified prefix path:

*   **Is accessible by all worker nodes:** Each machine involved in the experiment must have the necessary permissions and network access to read from and write to this location.
*   **Points to the same shared storage location for all workers:** Using a local path like `/tmp/marin_output` on each machine will result in data being scattered and inaccessible, not a unified output. You must use a shared filesystem (like NFS) or a cloud storage solution (S3, GCS) for distributed runs.

Choosing a suitable shared storage solution is crucial for the successful execution and reproducibility of your experiments in a distributed setting.
