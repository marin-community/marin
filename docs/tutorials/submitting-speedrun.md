# Submitting to the Speedrun Leaderboard

There are just three easy steps to submit a speedrun experiment to Marin:

![Marin speedrun 3 steps](../images/marin-speedrun-3-steps.png){width=75%}

## tl;dr (using GPUs with CUDA 12)

If you are at a terminal with GPUs, the following should work for you.

_Prerequisites you may already fulfill:_

<details><summary>Install uv</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

**1. Fork & Setup Marin**

[Click here](https://github.com/marin-community/marin/fork) to fork Marin, then clone your fork. Run the Marin setup script:

```bash
curl -LsSf https://raw.githubusercontent.com/marin-community/marin/refs/heads/main/scripts/speedrun/onboarding_setup.sh | bash
```

You can skip manually creating the fork if you have GitHub CLI installed and authenticated.

<details><summary>Manual Setup Steps (alternative)</summary>

Inside your fork of Marin:

```bash
uv venv --python 3.11
. .venv/bin/activate
uv sync --all-packages --extra=cuda12
```

Add your HuggingFace and WandB tokens if you haven't already:

```bash
export WANDB_API_KEY='...' # or `uvx wandb login`
export HF_TOKEN='...' # or `uvx hf auth login`

```

Create a subdirectory under `experiments/speedrun` and copy a starter file there (see details below).

</details>

**2. Develop & Test Submission**

You can now work on your speedrun submission! You can check your code and your estimated compute cost using a dry run

```bash
python -m experiments.speedrun.my_submission.main --dry_run true --prefix local_store
```

then fire off training on your hardware.

**3. Open PR & Merge**

When you are ready, open a PR and contribute to Marin. We ask that you

- give a brief explanation of your approach (model architecture, training strategy, optimizations)

- include the output of `print_run_info()` in the PR description (obtainable via a dry run), and `speedrun_results.json` files.

- leave "Allow edits by maintainers" on so we can help work on your code and scale up your ideas on TPU clusters.

Once the PR is merged, your run will appear on the [public leaderboard](https://marin.community/speedrun/).

## The Complete Introduction to Marin Speedruns

Marin Speedrun, inspired by the [nanogpt Speedrun](https://github.com/KellerJordan/modded-nanogpt), is a benchmark
aimed at improving the compute efficiency of language model training, by
incentivizing researchers to develop new model architectures, optimizers, and
training strategies. The idea is to allow participants to experiment at a smaller scale, where compute is less of a bottleneck, before scaling things up.

See [the overview of Marin Speedrun](../explanations/speedrun.md) for a more detailed explanation of speedrun. This tutorial assumes you are familiar with the idea of a speedrun, and will show you how to submit a speedrun to the leaderboard.

Submitting to the Speedrun Leaderboard consists of the following steps:

1. Define a configuration for training a model (or additionally define a model architecture).
2. Train it on your hardware; both GPUs and TPUs are supported. It is possible to create a good speedrun on just one/few GPUs.
3. Submit the configuration and the generated results file to the Marin GitHub repository via a pull request.
4. Watch it appear on the leaderboard once merged!

Here is a [hello world submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py),
and here is the [current leaderboard](https://marin.community/speedrun). You can explore the code for different runs [here](https://github.com/marin-community/marin/tree/main/experiments/speedrun). We will now use the hello-world example to show you how to submit a speedrun to the leaderboard.

## Prerequisites

Before you get started, you will need the following:

- Basic [installation](installation.md)
- Set up your [local GPU](local-gpu.md)
- You need to make sure that your Ray GPU cluster is started.

## Creating Your Speedrun Submission

1.  Create a new directory for your run:

    ```bash
    mkdir -p experiments/speedrun/my_submission
    ```

2.  Create your training script in this directory. You can start by copying the "[hackable transformer](https://github.com/marin-community/marin/tree/main/experiments/hackable_transformer_starter_template.py)" starter file, where a generic transformer language model is implemented for you to make changes easily, in addition to all other necessary code for you to make a submission. To see examples of other speedruns and configurations, check out the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun). You can also [add new optimizers](https://github.com/marin-community/marin/blob/main/docs/tutorials/add-optimizer.md), change learning rate schedules, play with hyperparameters, etc.

3.  Before running your speedrun, you may find it helpful to call [speedrun_config.print_run_info()](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/speedrun/speedrun.py#L76) to see the estimated training HW FLOPs, model FLOPs, model size, and your speedrun configs displayed. This will help you sanity-check your run, and also can be helpful in estimating the resources needed for your run.

4.  Set relevant environment variables needed for the run:

    ```
    export WANDB_API_KEY=...
    export HF_TOKEN=...
    export MARIN_PREFIX=...
    export WANDB_ENTITY=...
    export WANDB_PROJECT=...
    ```

    - `MARIN_PREFIX` tells Marin where to put artifacts generated during the execution of any steps.
      For examples, training checkpoints usually will be written to `${MARIN_PREFIX}/checkpoints/`.
      You can set this to an fsspec-recognizable path eg. a GCS bucket, or a directory on your machine. (For a detailed explanation of `MARIN_PREFIX` and other ways to specify the output path, see [Understanding `MARIN_PREFIX` and `--prefix`](../explanations/marin-prefix.md).)

    - `WANDB_ENTITY` and `WANDB_PROJECT` are used to specify the Weights and Biases team/entity and project names. You may use your own organization and project identifiers, but we ask that the run link corresponding to your speedrun is publicly accessible when submitting to the leaderboard.

5.  Train your model (example):
    ```bash
    python -m experiments.speedrun.my_submission.main --force_run_failed true --prefix local_store
    ```
    If you have setup a remote Ray cluster, use
    ```bash
    python marin/run/ray_run.py -- python -m experiments.speedrun.my_submission.main --force_run_failed true --prefix local_store
    ```

## Submitting Your Run

We walked through a hello-world example above, but if you do want to submit your run, follow these steps:

1. Add the resulting `speedrun_results.json` file to your run directory:

   ```bash
   cp ${MARIN_PREFIX}/checkpoints/speedrun/speedrun_results.json experiments/speedrun/<your_run_name>/
   ```

2. Create a pull request including:

   - Your run directory (training script and results file)
   - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated, and your run will appear on the [public leaderboard](https://marin.community/speedrun/).

!!! note "Speedrun Time Benchmarks"
To give you a sense of training times on GPU, a 50M parameter model trained on 4× A100 GPUs (80GB) takes about 33 minutes to complete 7600 steps, processing approximately 1 billion tokens. This should help you estimate the resources needed for your own speedrun experiments.

## Helpful Resources

- To see examples of other speedruns, check out the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun).
- Check out the [leaderboard](https://marin.community/speedrun).
- Find out more about speedrun itself in the [Speedrun explanation](../explanations/speedrun.md).
