# Submitting to the Marin Speedrun

The Marin Speedrun, inspired by the [nanogpt Speedrun](https://github.com/KellerJordan/modded-nanogpt), is a benchmark aimed at improving the compute efficiency of language model training. This tutorial assumes you are familiar with the core premise of the Marin Speedrun—if not, check out [the overview of Marin Speedrun](../explanations/speedrun.md) for a more detailed explanation. Let's walk through how to submit your first speedrun to the leaderboard.

![Marin speedrun 3 steps](../images/marin-speedrun-3-steps.png){width=75%}

## Quickstart (GPU environment with CUDA 12)

**1. Fork & Setup Marin**

[Click here](https://github.com/marin-community/marin/fork) to fork Marin, then clone your fork. Run the Marin setup script:

```bash
curl -LsSf https://raw.githubusercontent.com/marin-community/marin/refs/heads/main/scripts/speedrun/onboarding_setup.sh | bash
```

You can skip manually creating the fork if you have GitHub CLI installed and authenticated, which the script will use.

<details><summary>Manual Setup Steps (alternative)</summary>

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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

Create a subdirectory under <code>experiments/speedrun</code> and copy a starter file there (see <a href="#faq">FAQ</a> below).

</details>

**2. Develop & Test Submission**

You can now work on your speedrun submission! You can check your code and your estimated compute cost using a dry run:

```bash
python -m experiments.speedrun.my_submission.main --dry_run true --prefix local_store
```

then fire off training on your hardware.

**3. Open PR & Merge**

When you are ready, open a PR and contribute to Marin. We ask that you:

- Give a brief explanation of your approach (model architecture, training strategy, optimizations)
- Include the output of `print_run_info()` in the PR description (obtainable via a dry run), and `speedrun_results.json` files
- Leave "Allow edits by maintainers" on so we can help work on your code and scale up your ideas on TPU clusters

Once the PR is merged, your run will appear on the [public leaderboard](https://marin.community/speedrun/).

## FAQ

### How do I create my speedrun submission?

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/my_submission
   ```

2. Create your training script in this directory. You can start by copying the "[hackable transformer](https://github.com/marin-community/marin/tree/main/experiments/hackable_transformer_starter_template.py)" starter file, where a generic transformer language model is implemented for you to make changes easily. To see examples of other speedruns and configurations, check out the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun). You can also [add new optimizers](https://github.com/marin-community/marin/blob/main/docs/tutorials/add-optimizer.md), change learning rate schedules, play with hyperparameters, etc.

3. Before running your speedrun, you may find it helpful to call [speedrun_config.print_run_info()](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/speedrun/speedrun.py#L76) to see the estimated training HW FLOPs, model FLOPs, model size, and your speedrun configs displayed.

### What environment variables do I need to set?

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
export MARIN_PREFIX=...
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

- `MARIN_PREFIX` tells Marin where to put artifacts generated during the execution of any steps. For example, training checkpoints usually will be written to `${MARIN_PREFIX}/checkpoints/`. You can set this to an fsspec-recognizable path (e.g., a GCS bucket) or a directory on your machine. See [Understanding `MARIN_PREFIX` and `--prefix`](../explanations/marin-prefix.md) for details.

- `WANDB_ENTITY` and `WANDB_PROJECT` specify the Weights and Biases team/entity and project names. You may use your own organization and project identifiers, but we ask that the run link corresponding to your speedrun is publicly accessible when submitting to the leaderboard.

### How do I run training?

Train your model locally:

```bash
python -m experiments.speedrun.my_submission.main --force_run_failed true --prefix local_store
```

If you have a remote Ray cluster set up:

```bash
python marin/run/ray_run.py -- python -m experiments.speedrun.my_submission.main --force_run_failed true --prefix local_store
```

### How do I submit my results?

1. Add the resulting `speedrun_results.json` file to your run directory:
   ```bash
   cp ${MARIN_PREFIX}/checkpoints/speedrun/speedrun_results.json experiments/speedrun/<your_run_name>/
   ```

2. Create a pull request including:

   - Your run directory (training script and results file)
   - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated and your run will appear on the [public leaderboard](https://marin.community/speedrun/).

### How long does training take?

A 50M parameter model trained on 4× A100 GPUs (80GB) takes about 33 minutes to complete 7600 steps, processing approximately 1 billion tokens. This should help you estimate the resources needed for your own speedrun experiments.

## Helpful Resources

- [Hello world submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py) – a minimal example to get started
- [Speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun) – see other submissions for inspiration
- [Leaderboard](https://marin.community/speedrun) – view current results
- [Speedrun explanation](../explanations/speedrun.md) – understand how metrics are calculated
