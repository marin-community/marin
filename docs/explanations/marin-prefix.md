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

## Per-User Namespacing for Dev Runs

Within a prefix, *mutable* (`dev`) training checkpoints are automatically isolated per user. A mutable checkpoint addresses identity on `{name}/dev` alone, so without isolation two people iterating on the same experiment would write to the same path and clobber each other (and a resumption checkpointer could resume from whichever run last touched it).

To prevent this, Marin prefixes a mutable training step's name with `users/{username}/`, so each author gets their own scratch namespace under the shared prefix:

```
<MARIN_PREFIX>/checkpoints/users/<username>/<experiment-name>/dev/...
```

The username is resolved from your OS login by `rigging.provenance.username_segment()` — the same identity that stamps provenance `built_by`. An email-like login keeps its full local name (`russell.power@host` → `russell-power`) so distinct users stay distinct, and resolution raises rather than silently falling back to a shared `users/unknown/` path.

Fixed (calendar-versioned) checkpoints and **all** datasets keep their shared, un-namespaced names, so published runs stay citable and the expensive multi-TB tokenized caches still cache-hit across users. Only mutable `dev`/`<label>-dev` steps are moved. See `marin.experiment.namespacing.user_namespaced_name` for the exact policy.

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
