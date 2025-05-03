# Submitting a Speedrun to the Leaderboard

This guide will walk you through the process of creating and submitting your own speedrun entry to the Marin Speedrun leaderboard.

## Overview

Marin Speedrun is a community-driven model training contest and framework where participants can train language models and submit them to Marin. The goal is to further the mission of collaborative model development through a leaderboard that tracks different models and their performance.

## Tracks

We maintain two tracks:
- **FineWeb-Edu track**: Uses the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by default. You can experiment with any model architecture, optimizers, learning rate schedules, or hyperparameters.
- **Varying datasets**: You can experiment with different datasets. These runs will appear in a separate track on the leaderboard.

## Prerequisites

1. Follow the setup instructions in [Marin's main README](../../README.md)
2. Set up Weights and Biases (W&B) for tracking your runs
3. Ensure you have access to the required hardware (GPUs or TPUs)

## Creating Your Speedrun

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/<your_run_name>
   ```

2. Create your training script `<your_run_name>.py` in this directory. Here's a template:

   ```python
   from experiments.exp72_baselines import fineweb_edu_tokenized
   from experiments.simple_train_config import SimpleTrainConfig
   from marin.execution.executor import executor_main
   from marin.resources import TpuPodConfig  # or GpuConfig for GPU runs
   from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

   # Define your model configuration
   model_config = your_model_config

   speedrun_config = SpeedrunConfig(
       model_config=model_config,
       train_config=SimpleTrainConfig(
           TpuPodConfig(tpu_type="v4-128"),  # or GpuConfig(gpu_count=1) for GPU
           train_batch_size=512,
           num_train_steps=3000,
           learning_rate=3e-3,
           weight_decay=0.1,
           steps_per_eval=1000,
       ),
       tokenized_dataset=fineweb_edu_tokenized,
       hardware_config=HardwareConfig(
           device_type="v4-128",  # or "gpu" for GPU runs
           num_devices=64,
           device_flops=275e12,
       ),
   )

   if __name__ == "__main__":
       executor_main(steps=default_speedrun("<your_run_name>", speedrun_config))
   ```

3. Train your model:
   ```bash
   python experiments/speedrun/<your_run_name>/<your_run_name>.py
   ```

## Submitting Your Run

1. Add your `speedrun_results.json` file to your run directory:
   ```bash
   cp speedrun_results.json experiments/speedrun/<your_run_name>/
   ```

2. Create a pull request with:
   - Your run directory (training script and results file)
   - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated, and your run will appear on the public leaderboard at https://crfm.stanford.edu/marin/speedrun/.


## Evaluation and metrics

We track the following key metrics:
- **C4-EN BPB (Bits per Byte)**: Compression performance on the C4 dataset
- **FLOPs Used**: Total compute used during training
