# Submitting to the Speedrun Leaderboard

Marin Speedrun, inspired by the [nanogpt Speedrun](https://github.com/KellerJordan/modded-nanogpt), is a benchmark
aimed at improving the compute efficiency of language model training, by
incentivizing researchers to develop new model architectures, optimizers, and
training strategies.

We fix the dataset to be the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
and allow participants to specify their own FLOPs budget.
This allows us examine the tradeoff between compute and model quality.

Submitting to the Speedrun Leaderboard consists of the following steps:

1. Define a configuration for training a model.
2. Train it on your hardware; both GPUs and TPUs are supported.
3. Submit the configuration and the generated results file to the Marin GitHub repository via a pull request.
4. Watch it appear on the leaderboard!

Here is an [example submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py),
and here is the [current leaderboard](https://marin.community/speedrun).

## Prerequisites

Before you get started, you will need the following:

- Basic [installation](installation.md)
- Set up your [local GPU](local-gpu.md)
- You need to make sure that your Ray GPU cluster is started.

## Framework

You are given the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
but you can define any training scheme you'd like.
Each speedrun submission will generate the following key metrics:

- **C4-EN BPB (Bits per Byte)** (quality): Bits-per-byte on the validation portion of the `c4-en` (English portion of the C4) dataset; lower values are better.
- **FLOPs Used** (compute): Floating point operations used to train the model; lower values are better.

We also define the **Pareto frontier**, the set of submissions that are not
outperformed in both metrics by any other submission. In other words, a
submission is Pareto-optimal if no other model achieves better BPB and uses
fewer FLOPs.

It is important that the Speedrun leaderboard span multiple compute scales, since the best strategy might differ across scales,
and having multiple scales allows us to fit scaling laws and extrapolate out to much larger scales.

## Creating Your Speedrun Submission

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/YOUR_RUN_NAME
   ```

2. Create your training script in this directory. See [`llama_75m_fineweb_edu.py`](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py) for a reference:
   ```python
   """
   Sample speedrun with a 75M model that uses the Llama architecture.
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
   ```
   In this example, you only have one Python file, but if your submission is more complex, you can split it into multiple files.

3. Set relevant environment variables needed for the run:

    ```
    export WANDB_API_KEY=...
    export HF_TOKEN=...
    export MARIN_PREFIX=...
    ```

    `MARIN_PREFIX` tells Marin where to put artifacts generated during the execution of any steps.
    For examples, training checkpoints usually will be written to `${MARIN_PREFIX}/checkpoints/`.
    You can set this to an fsspec-recognizable path eg. a GCS bucket, or a directory on your machine.

4. Train your model:
   ```bash
   python marin/run/ray_run.py -- python experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py
   ```

## Submitting Your Run

1. Add the resulting `speedrun_results.json` file to your run directory:
   ```bash
   cp ${MARIN_PREFIX}/speedrun_results.json experiments/speedrun/${YOUR_RUN_NAME}/
   ```

2. Create a pull request including:
    - Your run directory (training script and results file)
    - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated, and your run will appear on the [public leaderboard](https://marin.community/speedrun/).