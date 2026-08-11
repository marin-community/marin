# Understanding `MARIN_PREFIX` and `--prefix`

Marin uses a designated storage location, referred to as a "prefix," to save all outputs from experiments. This includes tokenized data, model checkpoints, experiment logs, and other artifacts. You can specify this location using either the `MARIN_PREFIX` environment variable or the `--prefix` command-line argument when running experiment scripts.

## What is the Prefix Used For?

The prefix defines the root directory where Marin will store:
-   **Tokenized Datasets:** Processed datasets ready for training.
-   **Model Checkpoints:** Saved states of your models during and after training.
-   **Experiment Configurations and Logs:** JSON files detailing experiment setups, along with other logs.
-   **Evaluation Results:** Outputs from evaluation harnesses.

## Specifying the Prefix

You have two ways to specify this output location:

1.  **`--prefix` Command-Line Argument:**
    You can directly provide the path when running an experiment script:
    ```bash
    python experiments/your_experiment.py --prefix /path/to/your/output_directory
    ```
    Or for a cloud storage bucket:
    ```bash
    python experiments/your_experiment.py --prefix s3://your-bucket-name/path/to/output
    ```

2.  **`MARIN_PREFIX` Environment Variable:**
    Alternatively, you can set the `MARIN_PREFIX` environment variable in your shell. Marin will automatically use this path if `--prefix` is not provided.
    ```bash
    export MARIN_PREFIX="/path/to/your/output_directory"
    # Now you can run your script without --prefix
    python experiments/your_experiment.py
    ```

    This method is convenient if you consistently use the same output location for multiple experiments.

**Precedence:** If both the `MARIN_PREFIX` environment variable is set and the `--prefix` command-line argument is provided, the `--prefix` argument will always take precedence.

## Acceptable Storage Backends and Paths

Marin leverages the `fsspec` library, allowing you to use various storage backends. The path you provide should be a URI understandable by `fsspec`. Common examples include:

*   **Local Filesystem:**
    *   `--prefix /path/to/local/directory`
    *   `export MARIN_PREFIX=/path/to/local/directory`
    *   `--prefix ./relative/path/to/output` (relative to where you run the script)
    *   `export MARIN_PREFIX=./relative/path/to/output`

*   **Amazon S3:**
    *   `--prefix s3://your-s3-bucket/path/to/output`
    *   `export MARIN_PREFIX=s3://your-s3-bucket/path/to/output`
    (Requires appropriate AWS credentials and `s3fs` library installed: `uv pip install s3fs`)

*   **Google Cloud Storage (GCS):**
    *   `--prefix gs://your-gcs-bucket/path/to/output`
    *   `export MARIN_PREFIX=gs://your-gcs-bucket/path/to/output`
    (Requires appropriate GCP credentials and `gcsfs` library installed: `uv pip install gcsfs`)

## Important Considerations for Distributed Environments

When running Marin in a distributed setup (e.g., across multiple nodes via Iris), it is **critical** that the specified prefix path (whether via `MARIN_PREFIX` or `--prefix`):

*   **Is accessible by all worker nodes:** Each machine involved in the experiment must have the necessary permissions and network access to read from and write to this location.
*   **Points to the same shared storage location for all workers:** Using a local path like `/tmp/marin_output` on each machine will result in data being scattered and inaccessible, not a unified output. You must use a shared filesystem (like NFS) or a cloud storage solution (S3, GCS) for distributed runs.

Choosing a suitable shared storage solution is crucial for the successful execution and reproducibility of your experiments in a distributed setting.

## GCS region checks for Iris jobs

An Iris worker derives a region-local `MARIN_PREFIX` when the job does not set
one. This lets an unpinned coordinator run wherever Iris has capacity without
moving its default storage to another GCP region.

An explicit GCS `MARIN_PREFIX`, supplied with `iris job run -e` or the `env`
section of `.marin.yaml`, requires `--region` or `--zone`. Before submission,
Iris reads the bucket location and rejects the job unless every requested region
matches it. This check happens before the root job or any child job is submitted.
For child submissions, Iris checks the effective environment and placement after
including values inherited from the parent job.

```bash
uv run iris --cluster=marin job run \
  --region us-central1 \
  -e MARIN_PREFIX gs://marin-us-central1/my-run \
  -- python experiments/my_pipeline.py
```

Use `--allow-cross-region` only for an intentional transfer. It sets
`MARIN_I_WILL_PAY_FOR_ALL_FEES=1` in the job, logs the prefix and placement, and
emits an `iris.job.cross_region_override` telemetry event. Passing the same
environment variable with `-e` is equivalent. Local and S3 prefixes are not
part of this GCS launch check.
