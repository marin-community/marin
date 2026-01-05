# Submitting to the Marin Speedrun

The Marin speedrun, inspired by the [nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt),
is a benchmark aimed at improving the compute efficiency of language model training.
If you haven't already done so,
check out the [conceptual explanation of the Marin speedrun](../explanations/speedrun.md).
Let's walk through how to submit a speedrun to the leaderboard.

## Prerequisites

Follow the [installation guide](installation.md) to set up Marin.

We assume you have a GPU with CUDA 12 installed.

## Develop your submission

1. Create a new directory for your submission, where `<submission-name>` is the name of your submission (e.g., `muonh_qwen3_scaling`):

```bash
mkdir -p experiments/speedrun/<submission-name>
```

2. Create your training script in this directory.
You can start by copying the "[hackable transformer](https://github.com/marin-community/marin/tree/main/experiments/hackable_transformer_starter_template.py)" starter file,
where a generic Transformer language model is implemented for you to make changes easily. To see examples of other speedruns and configurations, check out the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun). You can also [add new optimizers](https://github.com/marin-community/marin/blob/main/docs/tutorials/add-optimizer.md), change learning rate schedules, play with hyperparameters, etc.

```bash
cp experiments/hackable_transformer_starter_template.py experiments/speedrun/<submission-name>/main.py
```

3. Think of clever ideas of how to improve the compute efficiency of language
model training (could be model architecture, optimizer, training strategy,
etc.).

4. Run the training script with a dry run to see the estimated training HW
FLOPs, model FLOPs, model size, and your speedrun configs displayed (invoking
[speedrun_config.print_run_info()](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/speedrun/speedrun.py#L76)).

```bash
uv run experiments/speedrun/<submission-name>/main.py --dry_run true
```

5. When you are ready, fire off the training run:

```bash
uv run experiments/speedrun/<submission-name>/main.py --force_run_failed true
```

This should generate a `speedrun_results.json` file in `${MARIN_PREFIX}/checkpoints/speedrun/`.

A 50M parameter model trained on 4× A100 GPUs (80GB) takes about 33 minutes to
complete 7600 steps, processing approximately 1 billion tokens. This should help
you estimate the resources needed for your own speedrun experiments.

## Submit

When you are satisfied with your work, move on to submitting your results to the leaderboard.

1. Add the resulting `speedrun_results.json` file to your submission directory:
   ```bash
   cp ${MARIN_PREFIX}/checkpoints/speedrun/speedrun_results.json experiments/speedrun/<submission-name>/
   ```

2. Give a brief explanation of your approach (model architecture, training strategy, optimizations) in your submission script `experiments/speedrun/<submission-name>/main.py`.

3. Set up your repository so you can [submit a PR](submitting-pr.md).

4. Include the output of `print_run_info()` and `speedrun_results.json` in the PR description.

5. Leave "Allow edits by maintainers" on so the Marin team can help work on your code.

6. The Marin team will review your PR and leave comments if needed.  Once you have addressed any comments,
the Marin team will merge your PR and your submission will appear on the [public leaderboard](https://marin.community/speedrun/).

Good luck!

## Helpful Resources

- [Hello world submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py) – a minimal example to get started
- [Speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun) – see other speedrun submissions for inspiration
- [Leaderboard](https://marin.community/speedrun) – view current results
- [Speedrun explanation](../explanations/speedrun.md) – understand how metrics are calculated
