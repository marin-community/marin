# Submitting to the Speedrun Leaderboard

## tl;dr (using GPUs w/ CUDA 12)

Prerequisites you may already fulfill:

<details><summary>Install uv</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

<details><summary>Fork marin (or have GitHub CLI)</summary>

Fork Marin at <a href="https://github.com/marin-community/marin/fork">https://github.com/marin-community/marin/fork</a>, or let the script do it for you automatically if you have installed and logged in to the GitHub CLI.

</details>

Run the following to setup

```bash
git clone https://github.com/marin-community/marin
cd marin
uv venv --python 3.11
. .venv/bin/activate
uv sync --all-packages --extra=cuda12
```

Add your HuggingFace and WandB tokens if you haven't already:

```bash
export WANDB_API_KEY='...' # or `uvx wandb login`
export HF_TOKEN='...' # or `uvx hf auth login`

```

Start training the hackable transformer:

```bash
python -m experiments.hackable_transformer_starter_template \
    --force_run_failed true --prefix local_store

# prefix specifies the output directory of all artifacts
```

When you are done, open a PR and contribute to Marin! We ask that you include the output of `print_run_info()` in the PR description (obtainable via a dryrun with `--dry_run true`), and any `speedrun_results.json` files generated.

Leave "Allow edits by maintainers" on so we can work on your code and provide compute to scale up your ideas.

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

Here is an [hello world submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py),
and here is the [current leaderboard](https://marin.community/speedrun). You can explore the code for different runs [here](https://github.com/marin-community/marin/tree/main/experiments/speedrun). We will now use the hello-world example to show you how to submit a speedrun to the leaderboard.

## Prerequisites

Before you get started, you will need the following:

- Basic [installation](installation.md)
- Set up your [local GPU](local-gpu.md)
- You need to make sure that your Ray GPU cluster is started.

## Creating Your Speedrun Submission

1.  Create a new directory for your run:

    ```bash
    mkdir -p experiments/speedrun/YOUR_RUN_NAME
    ```

2.  Create your training script in this directory- you'll need to specify congfigs related to author/affiliation, model configuration, and training configuration (as part of which you should also specify what kind of hardware you're using via `GpuConfig` or `TpuPodConfig`).

Marin typically uses [Levanter](https://github.com/marin-community/levanter) for model training. In the example above, we use the `llama_nano` configuration for the Llama model implemented in Levanter. Feel free to go through each of the configuration classes' definitions to get an idea of the design space here. The training configuration can either be a `SimpleTrainConfig` or `TrainLmOnPodConfig`. Also, note that in this example, you only have one Python file, but if your submission is more complex, you can split it into multiple files. To see examples of other speedruns and configurations, check out the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun).

If you are interested in modifying model architectures, you can refer to the "[hackable transformer](https://github.com/marin-community/marin/tree/main/experiments/speedrun/hackable_transformer_starter/hackable_transformer_attn_sink.py)" starter file, where a generic transformer language model is implemented for you to make changes easily (without needing to merge a Levanter pull request), in addition to all other necessary code for you to make a submission.

You can also [add new optimizers](https://github.com/marin-community/marin/blob/main/docs/tutorials/add-optimizer.md), change learning rate schedules, play with hyperparameters, etc.

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

5.  Train your model:
    ```bash
    python marin/run/ray_run.py -- python experiments/speedrun/llama_nano_tpu_speedrun/llama_nano_tpu_speedrun.py
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
