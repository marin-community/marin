# Submitting to the Speedrun Leaderboard


Marin Speedrun is a community-driven model training framework and contest that promotes collaborative language model development. By focusing on small-scale prototyping, it allows researchers to iterate quickly, test new ideas efficiently, and contribute to a shared leaderboard.

Submitting a speedrun consists of the following steps:
- defining a configuration for training a model
- training it on your hardware; both GPUs and TPUs are supported
- submitting the configuration and the generated results file to Marin via a pull request, and having it appear on the leaderboard!

This guide will walk you through the process of creating and submitting your own speedrun to the Marin Speedrun leaderboard. For an example, see [llama_75m_fineweb_edu.py](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py)- we will also use this as a reference as we go through this guide.

## Motivation for Marin Speedrun
Training large language models (LLMs) is compute-intensive, expensive, and time-consuming. To address this, Marin Speedrun provides a lightweight framework for rapid prototyping. Modeled after the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), it enables researchers to explore new ideas using smaller-scale models before investing in large-scale training. The goal is to make it easier to iterate quickly on model architectures, training strategies, and optimization techniques—and to identify promising approaches by comparing performance and efficiency against other submissions.

## Framework

We use the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by default. You can experiment with any model architecture, optimizers, learning rate schedules, or hyperparameters.
[TODO (Nikil): The tokenized, shuffled dataset is also available on HF [link] to make speedrun more accessible]. We track the following key metrics:

- **C4-EN BPB (Bits per Byte)**: Bits-per-byte on the `c4-en` (English portion of the C4) dataset; lower values are better.
- **FLOPs Used**: Floating point operations used in training the model.


While tracking performance on `c4-en` helps us understand model performance/capability, we opt to track a Pareto frontier to highlight trade-offs between model performance and scale. The Pareto frontier represents the set of models that are not outperformed in both metrics by any other submission. In other words, a model is Pareto-optimal if no other model achieves better BPB and uses fewer FLOPs.

This approach encourages creative and efficient experimentation. For example:

- A large model with very low BPB but needing a lot of compute to train may be Pareto-optimal.
- A smaller or more efficient model with worse BPB but dramatically lower compute may also make the frontier.

The Pareto frontier should provide a clearer picture of what's achievable with different model scales and approaches, both in terms of model performance/capability, and the amount of compute it requires.


## Prerequisites

The rest of this guide assumes you have already:
1. Cloned the codebase and set up the environment and env vars for Marin by following [the setup instructions](../tutorials/getting-started.md). If you are using a local GPU, follow [local-gpu.md](../tutorials/local-gpu.md) for GPU-specific setup instructions.
2. Set up [Weights and Biases](https://wandb.ai) for tracking your runs. You can sign up for a free account [here](https://wandb.ai). You will need a W&B API key to track metrics and submit your run.

## Creating Your Speedrun Submission

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/<your_run_name>
   ```

2. Create your training script in this directory; here's a reference speedrun file `llama_75m_fineweb_edu.py`:

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

    The folder and `.py` file names can be different, but should be descriptive of the configuration you use eg. something like `llama_75m_fineweb_edu`. Note that in principle, you could organize your directory experiments/speedrun/<your_run_name> however you like (multiple python files, etc) if you'd like to; in this guide, we'll go with one file to keep things simple.

3. Train your model:
   ```bash
   python marin/run/ray_run.py -- python experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py
   ```

## Submitting Your Run

1. Add your `speedrun_results.json` file to your run directory:
   ```bash
   cp ${MARIN_PREFIX}/speedrun_results.json experiments/speedrun/llama_75m_fineweb_edu/
   ```

2. Create a pull request including:
   - Your run directory (training script and results file)
   - A brief explanation of your approach (model architecture, training strategy, optimizations)

3. Once reviewed and merged, the leaderboard gets updated, and your run will appear on the public leaderboard at https://marin.community/speedrun/.


TODO (Nikil): add some details about what happens afterwards (pic of leaderboard, scaling up to larger models, etc.)
