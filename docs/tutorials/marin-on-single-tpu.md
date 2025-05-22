# Running Marin on a Single TPU Node (e.g., v4-8 using Levanter)

Welcome to this guide on setting up and running Marin experiments on a single Google Cloud TPU node (like a v4-8)! This tutorial is the TPU equivalent of our "Setting up a Local GPU Environment" guide. Here, we'll focus on leveraging [Levanter's `launch.py`](https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/#using-launchpy) script to configure the TPU environment and execute your Marin tasks. 

This guide will walk you through the necessary prerequisites, configuration, and common commands to get you started.

# Prerequisites

## Levanter Installation

Marin uses Levanter for its core functionality, especially for TPU interaction via `launch.py`.

1.  **Clone Levanter (if not already present):**
    If you haven't cloned Levanter into the `submodules/` directory yet, you can do so using:
    ```bash
    # From the root of your Marin project directory
    mkdir -p submodules
    git clone https://github.com/stanford-crfm/levanter.git submodules/levanter
    ```

2.  **Install Levanter:**
    After ensuring Levanter is cloned into `submodules/levanter`, install it:
    ```bash
    # From the root of your Marin project directory
    cd submodules/levanter
    pip install -e .
    cd ../.. # Return to Marin project root
    ```

## Docker Installation

Docker is required for using `launch.py` to manage TPU resources. `launch.py` will build a Docker image containing your Levanter code and deploy it to the TPU.

*   **Install Docker:** Follow the official Docker installation guide for your operating system: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

## Weights & Biases (WandB) Account and Credentials

Marin uses Weights & Biases (WandB) for logging experiment metrics, results, and general run information. You'll need a WandB account to track your experiments.

*   **Create a WandB Account:** If you don't have one, sign up at [https://wandb.ai/](https://wandb.ai/).
*   **Obtain Credentials:** Once you have an account, you'll need three pieces of information:
    1.  **API Key:** Found in your WandB account settings.
    2.  **Entity:** This is typically your WandB username or the name of the organization/team you're working under.
    3.  **Project Name:** You can decide on a project name (e.g., "marin-tpu-experiments") where your runs will be logged. This project will be created in WandB if it doesn't exist.

These values (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`) will be configured in the `.levanter.yaml` file later in this guide.

## Hugging Face Token (Optional but Recommended)

A Hugging Face token allows you to download models and datasets that might be gated or private. Even for public resources, using a token can prevent rate-limiting issues.

*   **Get your Token:** You can find or create a Hugging Face token in your account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A "read" token is usually sufficient.

This token (`HF_TOKEN`) will also be configured in the `.levanter.yaml` file.

## Google Cloud SDK Installation and Configuration

The Google Cloud SDK (`gcloud`) is necessary for interacting with Google Cloud Platform services, including TPUs.

*   **Install Google Cloud SDK:** Follow the quickstart guide: [https://cloud.google.com/sdk/docs/quickstarts](https://cloud.google.com/sdk/docs/quickstarts)

After installation, configure `gcloud` with the following commands. These are typically run on your local development machine, not the TPU VM itself.

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install alpha  # May be needed for the latest TPU types
gcloud services enable tpu.googleapis.com # Enable the TPU API
gcloud config set account YOUR_EMAIL_ACCOUNT # e.g., user@example.com
gcloud config set project YOUR_PROJECT_ID   # Your GCP Project ID
```

*   **SSH Key Setup for GCP:** You will also need to set up SSH keys to connect to your GCP resources. Follow Google's guide for adding SSH keys: [https://cloud.google.com/compute/docs/connect/add-ssh-keys#metadata](https://cloud.google.com/compute/docs/connect/add-ssh-keys#metadata).

# Levanter Configuration (`.levanter.yaml`)

Levanter's `launch.py` script (located at `submodules/levanter/infra/launch.py`) requires a configuration file named `.levanter.yaml` to function. This file **must be placed in root of your Marin repo**. When you run `python submodules/levanter/infra/launch.py ...` from your Marin project root, `launch.py` will look for this YAML file in the current working directory.

This tutorial focuses on the environment variables essential for Marin. For a comprehensive guide to all `.levanter.yaml` options (like `zone`, `tpu` name, `capacity_type`, Docker settings, etc.), please refer to [Levanter's official documentation](https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/).

Here's a minimal example of `.levanter.yaml` highlighting Marin-relevant settings:

```yaml
# .levanter.yaml (place in the root of your Levanter clone, typically submodules/levanter/)
# Refer to Levanter's official docs for all options: https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/

env:
    # Marin-specific: Path for experiment outputs
    MARIN_PREFIX: gs://YOUR_GCS_BUCKET_NAME/marin_outputs # IMPORTANT: Set this to your GCS bucket and desired output path.
    # For WandB logging (replace with your actual credentials)
    WANDB_API_KEY: YOUR_WANDB_API_KEY
    WANDB_ENTITY: YOUR_WANDB_ENTITY    # Your W&B username or organization
    WANDB_PROJECT: YOUR_WANDB_PROJECT_NAME # Your W&B project name

    # For Hugging Face model/tokenizer downloads (replace with your token if needed)
    HF_TOKEN: YOUR_HUGGINGFACE_TOKEN

    # Optional: Levanter/TPU specific performance arguments. 
    # These are advanced settings, often tuned for specific models or TPU hardware.
    # For recommended defaults used in some Marin setups, you can inspect marin/run/vars.py.
    # launch.py may not strictly require these if Levanter's defaults are sufficient for your use case.
    # LIBTPU_INIT_ARGS: >-
    #   --xla_tpu_scoped_vmem_limit_kib=81920
    #   ... (and other args from marin/run/vars.py if needed)

# Levanter-specific: TPU configuration (required by launch.py)
# These should be set according to your GCP project and TPU details.
# See Levanter docs for more details: https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/
# You can instead pass these as command line args to launch.py
zone: YOUR_ZONE # e.g., us-central2-b
tpu: YOUR_TPU_NAME  # e.g., my-v4-8-tpu

# Other Levanter configurations (refer to Levanter docs):
# capacity_type: "on-demand" # or "reserved", "preemptible"
# docker_repository: ... # etc.
# image_name: ...
# subnetwork: ...
```

### Key Marin-Relevant Configuration Fields:

*   **`env`**: Defines environment variables for the TPU execution environment.
    *   `MARIN_PREFIX`: **Essential for Marin.** Specifies the base GCS path for all Marin experiment outputs (checkpoints, logs, tokenized data). **Replace `gs://YOUR_GCS_BUCKET_NAME/marin_outputs` with your actual GCS bucket and desired path.**
    *   `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`: Your Weights & Biases credentials. Necessary if you intend to use W&B logging with Marin.
    *   `HF_TOKEN`: Your Hugging Face token. Required if your Marin experiments need to download private models or tokenizers from Hugging Face Hub.
    *   `LIBTPU_INIT_ARGS`: (Commented out by default in this minimal example) These are advanced XLA flags for fine-tuning TPU performance and stability. While Levanter might have its own defaults, Marin sometimes uses a specific set of these arguments for larger-scale runs. If you encounter issues or are scaling up, you might refer to `marin/run/vars.py` for a list of recommended arguments. For basic tutorial steps, they are often not strictly necessary.


**For all other settings in `.levanter.yaml`** (like `capacity_type`, `docker_repository`, `image_name`, `subnetwork`, other environment variables like `TPU_STDERR_LOG_LEVEL`, etc.), please consult the [Levanter documentation](https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/). This Marin tutorial assumes you have a working Levanter setup as per their guidelines, focusing only on the variables Marin directly depends on or commonly uses.

Ensure you create this `.levanter.yaml` file in `submodules/levanter/.levanter.yaml` and populate it with your specific details before proceeding.

# Using `launch.py` for Single TPU Nodes

The `launch.py` script, located in `submodules/levanter/infra/launch.py` (relative to your Marin project root), is the primary tool for deploying and running your Marin code on TPUs. It handles packaging your Marin project (including the Levanter submodule) into a Docker image, deploying it to the TPU, and executing your specified command.

## Basic Usage

The fundamental command structure for `launch.py` (run from the root of your Marin project directory) is:

```bash
python submodules/levanter/infra/launch.py -- YOUR_COMMAND_TO_RUN_ON_TPU
```

Key things to note:

*   **Path to `launch.py`:** The script is now at `submodules/levanter/infra/launch.py`.
*   **`--` Separator:** It separates the arguments for `launch.py` itself from the command that will be executed on the TPU VM.
*   **Command Execution Context:** `YOUR_COMMAND_TO_RUN_ON_TPU` is executed from the root of your Marin project directory *on the TPU VM*. `launch.py` sets up the Docker container so that the working directory is the root of your project.
*   **Docker Packaging:** `launch.py` packages the current state of your Marin project directory (including the Levanter submodule, respecting any patterns in your `.dockerignore` files at both Marin root and Levanter submodule root) into a Docker image.

## Running a Marin Experiment Script

With your `.levanter.yaml` configured for your TPU environment, you can now use `launch.py` to execute your Marin experiment scripts on the TPU. The Marin script itself, like the [`train_tiny_model_tpu.py` example](https://github.com/marin-community/marin/tree/main/experiments/tutorials/train_tiny_model_tpu.py) defines the model, data, and training parameters, including the `TpuPodConfig` which specifies the TPU resources Marin expects.

**You will need to update the TpuPodConfig to use the kind of TPU you are using.**


Here's how to run the `train_tiny_model_tpu.py` script (assuming your Marin project root is the current directory):

```bash
# Ensure you .levanter.yaml is configured with your TPU details 
# (YOUR_TPU_NAME, YOUR_ZONE) and, most importantly, your MARIN_PREFIX 
# (e.g., MARIN_PREFIX: gs://YOUR_GCS_BUCKET_NAME/marin_outputs).

python submodules/levanter/infra/launch.py -- \
  python experiments/tutorials/train_tiny_model_tpu.py
  
  # or
  
  python submodules/levanter/infra/launch.py --zone us-central2-b --tpu_type v4-8 --tpu_name marin_tutorial --preemptible -- python experiments/tutorials/train_tiny_model_tpu.py
```

**Explanation:**

*   `python submodules/levanter/infra/launch.py --`: Invokes Levanter's launch script.
*   `python experiments/tutorials/train_tiny_model_tpu.py`: The Marin script to execute on the TPU.
*   **Output Path Management with `MARIN_PREFIX`**:
    *   The `MARIN_PREFIX` environment variable, which you defined in `submodules/levanter/.levanter.yaml` (e.g., `gs://YOUR_GCS_BUCKET_NAME/marin_outputs`), is automatically picked up by Marin's execution logic within the `train_tiny_model_tpu.py` script (and other Marin experiment scripts).
    *   Marin uses this `MARIN_PREFIX` as the base directory in Google Cloud Storage for all outputs related to your experiments.
    *   The script itself (e.g., `train_tiny_model_tpu.py` which defines the `nano_wikitext_model` step) will then create a specific subdirectory structure under this `MARIN_PREFIX`. For example, checkpoints for the `nano_wikitext_model` might be saved to something like `gs://YOUR_GCS_BUCKET_NAME/marin_outputs/checkpoints/llama-nano-wikitext-VERSION_HASH/`. The exact naming and structure (like including `checkpoints/` and the step name with a version hash) are handled by Marin's internal step and executor logic.
    *   You don't need to pass a `--prefix` argument directly in the `launch.py` command for basic output location if `MARIN_PREFIX` is correctly set in `.levanter.yaml`. Your Marin script will use the `name` argument from its `Step` definitions (e.g., `name="llama-nano-wikitext"` in `train_tiny_model_tpu.py`) to create unique output paths under `MARIN_PREFIX`.

The `train_tiny_model_tpu.py` script uses `TpuPodConfig` to declare its TPU resource needs (e.g., `tpu_type="v4-8"`). This should align with the TPU details (`tpu` and `zone`) in your `submodules/levanter/.levanter.yaml`. The "v4-8" in the script is a common default.

This example replaces older examples that used Levanter-specific configurations like `config/gpt2_small.yaml`. In Marin, the experiment's model, data, and training details are typically defined within the Python script itself or via configuration objects loaded by the script.

## Important `launch.py` Flags for Single Node Usage

While `launch.py` has many options (run `python infra/launch.py --help` for a full list), a few are particularly useful when running Marin scripts on a single TPU node:

*   **`--foreground`**:
    This flag runs the launch script in the foreground. Instead of detaching and running in the background, it will stream the logs from the TPU directly to your terminal. This is very useful for interactive development and debugging.

*   **`--retries=N`**:
    This option allows `launch.py` to automatically attempt to restart your job up to `N` times if it fails. This can be helpful for preemptible TPUs or if transient network issues occur.

**Example combining these flags with the Marin script:**

```bash
# Ensure submodules/levanter/.levanter.yaml is configured with MARIN_PREFIX and your TPU details.
python submodules/levanter/infra/launch.py --foreground --retries=3 -- \
  python experiments/tutorials/train_tiny_model_tpu.py
```

## A Note on Docker Context and Code Availability

When you run `python submodules/levanter/infra/launch.py ...` from your Marin project root:

*   `launch.py` (from the Levanter submodule) uses the **Marin project root** as the primary Docker build context.
*   All files in your Marin project, including the `submodules/levanter/` directory, are included in the Docker image.
*   A `.dockerignore` file in your Marin project root is respected for the primary context. Levanter also has its own `.dockerignore` (e.g., `submodules/levanter/.dockerignore`) for the submodule directory.

This setup is standard for Marin development.

# Interacting with your TPU Node (Optional)

While `launch.py` handles the deployment and execution of your Marin scripts, you might occasionally need to interact directly with the TPU for debugging.

The most common command is to SSH into a TPU worker:
```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE --worker=0
```
Replace `YOUR_TPU_NAME` and `YOUR_ZONE` with your specific TPU details. This connects you to worker 0 of your TPU slice.

For a more comprehensive list of `gcloud` commands for TPU interaction (like copying files with `scp`, or running commands on all workers), please refer to the Levanter official documentation: [Useful Commands for TPUs](https://levanter.readthedocs.io/en/latest/Getting-Started-TPU-VM/#useful-commands).

# Troubleshooting / Common Issues

*   **"No GPU/TPU found, falling back to CPU."**:
    *   This often means another process is using the TPU. Try stopping existing Python processes on the TPU workers:
        ```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE --worker=all --command="sudo pkill -f python" # Kills python processes
        ```
        (Using `-f` to match the full command line, and `sudo` if processes were started by root via Docker).
    *   As a last resort, you can reboot the TPU VM. **Warning**: This command will hang as the SSH connection drops. Manually terminate it (Ctrl+C) after 10-15 seconds.
        ```bash
        gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE --worker=all --command="sudo reboot" # Reboots the TPU
        ```
        Wait a few minutes for the TPU to reboot before trying again.

*   **Docker Permission Denied (e.g., `permission denied while trying to connect to the Docker daemon socket`)**:
    *   This error occurs on the machine where you are running `launch.py` (your local machine or development server). Add your user to the `docker` group:
        ```bash
        sudo usermod -aG docker $USER
        ```
    *   You'll need to restart your shell session or log out and log back in for this change to take effect.

*   **Docker GCR/Artifact Registry Authentication Issues (e.g., `denied: Unauthenticated request`)**:
    *   If using Google Container Registry (GCR) or Artifact Registry for Docker images (specified in `submodules/levanter/.levanter.yaml`) and encountering authentication errors when `launch.py` tries to pull/push images, configure Docker to use `gcloud` as a credential helper on your local machine:
        ```bash
        # For Artifact Registry (replace YOUR_REGION, e.g., us-central1)
        gcloud auth configure-docker YOUR_REGION-docker.pkg.dev

        # For GCR (less common now, but if used)
        # gcloud auth configure-docker gcr.io
        ```

This concludes the tutorial on running Marin experiments on a single TPU node using Levanter. For more advanced topics, refer to other Marin and Levanter documentation. Happy training!

# Next Steps

Congratulations on successfully setting up and running a Marin experiment on a single TPU node! Here are a few suggestions for what you can explore next:

## Next Steps

Congratulations! You have trained your first model in Marin.  Choose your next adventure:

- Train a real [1B or 8B parameter language model](train-an-lm.md) using Marin.
- Learn about the [Executor framework](../explanations/executor.md).
- Read more about the full [language modeling pipeline](../explanations/lm-pipeline.md), including data processing.
