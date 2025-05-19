# Submitting to the Speedrun Leaderboard

Marin Speedrun, inspired by the [nanogpt Speedrun](https://github.com/KellerJordan/modded-nanogpt), is a benchmark
aimed at improving the compute efficiency of language model training, by
incentivizing researchers to develop new model architectures, optimizers, and
training strategies. The idea is to allow participants to experiment at a smaller scale, where compute is less of a bottleneck, before scaling things up. See [docs](../explanations/speedrun.md) for more details on Marin Speedrun. This tutorial will show you how to submit a speedrun to the leaderboard.

Submitting to the Speedrun Leaderboard consists of the following steps:

1. Define a configuration for training a model.
2. Train it on your hardware; both GPUs and TPUs are supported. It is possible to create a good speedrun on just one/few GPUs.
3. Submit the configuration and the generated results file to the Marin GitHub repository via a pull request.
4. Watch it appear on the leaderboard!

Here is an [hello world submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py),
and here is the [current leaderboard](https://marin.community/speedrun). You can explore the code for different runs [here](https://github.com/marin-community/marin/tree/main/experiments/speedrun). We will now use the hello-world example to show you how to submit a speedrun to the leaderboard.

## Prerequisites

Before you get started, you will need the following:

- Basic [installation](installation.md)
- Set up your [local GPU](local-gpu.md)
- You need to make sure that your Ray GPU cluster is started.

!!! danger "GPU Setup"
    If you are working on GPU, make sure you've correctly installed and set up the environment by following the [local GPU setup guide](local-gpu.md). This ensures your environment is properly configured for training with GPU.

## Framework

You are given the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) with a fixed seed, but you can define any training scheme you'd like. Since the full dataset it more than 9TB, we have also provided a pre-tokenized, shuffled subset of the dataset consisting of 10B tokens [here](https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B).
(The speedrun code will download this for you when you run the speedrun, unless you override it to use the full dataset!)

Each speedrun submission will generate the following key metrics:

- **C4-EN BPB (Bits per Byte)** (quality): Bits-per-byte on the validation portion of the `c4-en` (English portion of the C4) dataset; lower values are better.
- **FLOPs Used** (compute): Floating point operations used to train the model; lower values are better. See [here](../explanations/speedrun-flops-accounting.md) for more details on how we calculate FLOPs.

We also define the **Pareto frontier**, the set of submissions that are not outperformed in both metrics by any other submission. In other words, a submission is Pareto-optimal if no other model achieves better BPB and uses fewer FLOPs.

It is important that the Speedrun leaderboard span multiple compute scales, since the best strategy might differ across scales, and having multiple scales allows us to fit scaling laws and extrapolate out to much larger scales.

## Creating Your Speedrun Submission

1. Create a new directory for your run:
   ```bash
   mkdir -p experiments/speedrun/YOUR_RUN_NAME
   ```

2. Create your training script in this directory. See [`llama_75m_fineweb_edu.py`](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu/llama_75m_fineweb_edu.py) for a reference:

    === "GPU"
        ```python
        import logging

        from experiments.llama import llama_nano
        from experiments.simple_train_config import SimpleTrainConfig
        from marin.execution.executor import executor_main
        from marin.resources import GpuConfig
        from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

        logger = logging.getLogger("ray")

        speedrun_config = SpeedrunConfig(
            author=Author(
                name="your name",
                affiliation="your affiliation",
                url="your url",
            ),
            description="your description",
            model_config=llama_nano,
            train_config=SimpleTrainConfig(
                GpuConfig(gpu_count=1, accelerator_type="A100"),
                train_batch_size=32,
                num_train_steps=100,
                learning_rate=3e-3,
                weight_decay=0.1,
                steps_per_eval=500,
            ),
            # tokenized_dataset need not be set here; by default it will be set to a FineWeb-Edu 10B token dataset
            # that we have pre-tokenized and shuffled, available at https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B
            # tokenized_dataset=fineweb_edu_subcache_10B,
        )

        # Shows your speedrun configuration, model FLOPs, model size and (training) hardware FLOPs- you
        # can use this before actually kicking off a run to validate your setup
        speedrun_config.print_run_info()

        if __name__ == "__main__":
            executor_main(steps=default_speedrun("llama_nano_gpu_speedrun", speedrun_config))
        ```
    === "TPU"
        ```python
        import logging

        from experiments.llama import llama_nano
        from experiments.simple_train_config import SimpleTrainConfig
        from marin.execution.executor import executor_main
        from marin.resources import TpuPodConfig
        from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

        logger = logging.getLogger("ray")

        speedrun_config = SpeedrunConfig(
            author=Author(
                name="your name",
                affiliation="your affiliation",
                url="your url",
            ),
            description="Nano model based on Llama architecture.",
            model_config=llama_nano,
            train_config=SimpleTrainConfig(
                TpuPodConfig(tpu_type="v4-128"),
                train_batch_size=512,
                num_train_steps=3000,
                learning_rate=3e-3,
                weight_decay=0.1,
                steps_per_eval=1000,
            ),
            # tokenized_dataset need not be set here; by default it will be set to a FineWeb-Edu 10B token dataset
            # that we have pre-tokenized and shuffled, available at https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B
            # tokenized_dataset=fineweb_edu_subcache_10B,
        )

        speedrun_config.print_run_info()

        if __name__ == "__main__":
            executor_main(steps=default_speedrun("llama_nano_tpu_speedrun", speedrun_config))
        ```

    In this example, you only have one Python file, but if your submission is more complex, you can split it into multiple files.

3. Before running your speedrun, you may find it helpful to call [speedrun_config.print_run_info()](https://github.com/marin-community/marin/blob/main/marin/speedrun/speedrun.py#L76) to see the estimated training HW FLOPs, model FLOPs, model size, and your speedrun configs displayed. This will help you sanity-check your run, and also can be helpful in estimating the resources needed for your run.

4. Set relevant environment variables needed for the run:

    ```
    export WANDB_API_KEY=...
    export HF_TOKEN=...
    export MARIN_PREFIX=...
    ```

    `MARIN_PREFIX` tells Marin where to put artifacts generated during the execution of any steps.
    For examples, training checkpoints usually will be written to `${MARIN_PREFIX}/checkpoints/`.
    You can set this to an fsspec-recognizable path eg. a GCS bucket, or a directory on your machine.

5. Train your model:
   ```bash
   python marin/run/ray_run.py -- python experiments/speedrun/llama_nano_tpu_speedrun/llama_nano_tpu_speedrun.py
   ```

## Submitting Your Run

This is just a hello-world example, but if you do want to submit your run, follow these steps:

1. Add the resulting `speedrun_results.json` file to your run directory:
   ```bash
   cp ${MARIN_PREFIX}/speedrun_results.json experiments/speedrun/<your_run_name>/
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
