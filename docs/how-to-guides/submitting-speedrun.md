# Submitting a Speedrun to the Leaderboard

This guide will walk you through the process of creating and submitting your own speedrun entry to the Marin Speedrun leaderboard.

## Overview

Marin Speedrun is a community-driven model training contest and framework where participants can train language models and submit them to Marin. The goal is to further the mission of collaborative model development through a leaderboard that tracks different models and their performance.

The development of an LLM involves several factors, such as the model architecture, scale, number of tokens and hyperparameters. While it would be ideal to train an test LLMs at a larger scale, this is quite expensive and requires lots of compute and time.

An alternative approach is to experiment with different recipes at smaller scale, and use the findings to extrapolate to a larger scale. The goal here is to enable such prototyping and experimentation of ideas to try out new ideas at a managaeable scale before scaling models up.

## Tracks

We maintain two tracks:
- **FineWeb-Edu track**: Uses the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by default. You can experiment with any model architecture, optimizers, learning rate schedules, or hyperparameters.

## Prerequisites

1. Follow the setup instructions in [Marin's main README](../../README.md)
2. Set up Weights and Biases (W&B) for tracking your runs
3. Ensure you have access to the required hardware (GPUs or TPUs)

## Creating Your Speedrun

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/<your_run_name>
   ```

2. Create your training script in this directory; here's a reference speedrun file `llama_75m_fineweb_edu.py`:

   ```python
   """
    Sample speedrun with an 75M LLaMA model.
    """

    import logging

    from experiments.exp72_baselines import fineweb_edu_tokenized
    from experiments.llama import llama_75m
    from experiments.simple_train_config import SimpleTrainConfig
    from marin.execution.executor import executor_main
    from marin.resources import TpuPodConfig
    from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

    logger = logging.getLogger("ray")

    speedrun_config = SpeedrunConfig(
        model_config=llama_75m,
        train_config=SimpleTrainConfig(
            TpuPodConfig(tpu_type="v4-128"),
            train_batch_size=512,
            num_train_steps=3000,
            learning_rate=3e-3,
            weight_decay=0.1,
            steps_per_eval=1000,
        ),
        tokenized_dataset=fineweb_edu_tokenized,
        hardware_config=HardwareConfig(
            device_type="v4-128",
            num_devices=64,
            device_flops=275e12,
        ),
    )

    if __name__ == "__main__":
        executor_main(steps=default_speedrun("75M_llama_fineweb_edu", speedrun_config))

3. Set relevant environment variables needed for the run:

    ```
    export WANDB_API_KEY=...
    export MARIN_PREFIX=...
    export HF_TOKEN=...
    ```

    `MARIN_PREFIX` tells Marin where to put artifacts generated during the execution of any steps. For examples, training checkpoints usually will be written to `${MARIN_PREFIX}/checkpoints/`. You can set this to
    either a directory on your machine, or an fsspec-recognizable path eg. a GCS bucket.

    A more detailed description of these can be found in [docs/<TODO>](../../docs/<TODO>).

3. Train your model:
   ```bash
   python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY} -- python experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py
   ```

## Submitting Your Run

1. Add your `speedrun_results.json` file to your run directory:
   ```bash
   cp ${MARIN_PREFIX}/speedrun_results.json experiments/speedrun/llama_75m_fineweb_edu/
   ```

2. Create a pull request with:
   - Your run directory (training script and results file)
   - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated, and your run will appear on the public leaderboard at https://crfm.stanford.edu/marin/speedrun/.


## Evaluation and metrics

We track the following key metrics:
- **C4-EN BPB (Bits per Byte)**: Compression performance on the C4 dataset
- **FLOPs Used**: Total compute used during training
