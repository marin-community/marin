# First Experiment: Train a Tiny Model on TinyStories

In this tutorial, you will run your first Marin experiment: training a tiny
language model on the TinyStories dataset.
Don't get too excited, for this model will not be good;
our goal is just to run something.

In this tutorial, we will cover the following:

1. Creating an experiment script
2. Creating the steps to train a tiny model on CPU or GPU
3. Running the experiment

We assume you've already gone through the [installation](installation.md) tutorial
to install Marin and set up your environment.

## Creating an Experiment Script

Experiments scripts live in the `experiments` directory. For instance, the script for this tutorial is in
`experiments/tutorials/train_tiny_model_cpu.py` or `experiments/tutorials/train_tiny_model_gpu.py` if you want to test on a GPU machine.

By convention, experiments are usually named `exp{GITHUB_ISSUE_NUMBER}_{DESCRIPTOR}.py`, where `GITHUB_ISSUE_NUMBER` is the issue number for your experiment and `DESCRIPTOR` is a brief description.
But, to follow along with this tutorial, you can name it whatever you want.

To train a model, we need to:

1. Choose a dataset
2. Tokenize the dataset
3. Define training configuration
4. Train the model

Let's do that step by step.

### 1. Choose a Dataset

For this tutorial, we will use the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
This dataset is a collection of synthetic short stories, each about 100 tokens long.

We'll just reference it by its Hugging Face name:

```python
tinystories_hf_id = "roneneldan/TinyStories"
```

### 2. Tokenize the Dataset

To tokenize the dataset, we use the `default_tokenize` function from `experiments.defaults`.
This function takes a dataset and a tokenizer and returns an `ExecutorStep` that produces a tokenized dataset.
The tokenized dataset is a directory containing one file per shard of the dataset.

```python
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,  # path to write tokenized files (tokenized/ will be prepended)
    dataset=tinystories_hf_id,  # HF dataset id
    tokenizer=llama3_tokenizer,
)
```

An [`ExecutorStep`](../explanations/executor.md#steps) is the basic unit of work in Marin.
A step defines a piece of work to be done, such as tokenizing a dataset or training a model.
Each step has a name, a function to execute, and a configuration object.
The function takes the configuration object and produces some output.
The output of a step can be used as the input to another step.

By convention, we name steps based on what they produce. The `tinystories_tokenized` step produces a tokenized dataset,
and you can use that step anywhere a path to a tokenized dataset is expected.

### 3. Define Training Configuration

For this tutorial, we will use the `SimpleTrainConfig` class from `experiments.simple_train_config`.
This class defines basic training configuration that is sufficient for most experiments.

!!! info "Training Configuration for Different Accelerators"
    You need to provide the appropriate resource configuration based on your hardware setup. Marin supports different accelerator types through various [resource configurations](../references/resource-config.md). The `CpuOnlyConfig` is one such resource configuration that requests a certain number of CPUs. Other resource configurations include `GpuConfig` for requesting GPUs and `TpuPodConfig` for requesting TPUs.

    === "CPU"
        ```python
        nano_train_config = SimpleTrainConfig(
            # Here we define the hardware resources we need.
            resources=CpuOnlyConfig(num_cpus=1),
            train_batch_size=4,
            num_train_steps=100,
            # set hyperparameters
            learning_rate=6e-4,
            weight_decay=0.1,
            # keep eval quick for tutorial
            max_eval_batches=4,
        )
        ```

    === "GPU"
        ```python
        nano_train_config = SimpleTrainConfig(
            # Here we define the hardware resources we need.
            resources=GpuConfig(gpu_count=1),
            train_batch_size=4,
            num_train_steps=100,
            # set hyperparameters
            learning_rate=6e-4,
            weight_decay=0.1,
            # keep eval quick for tutorial
            max_eval_batches=4,
        )
        ```

    === "TPU"
        ```python
        nano_train_config = SimpleTrainConfig(
            # Here we define the hardware resources we need.
            resources=TpuPodConfig(tpu_type="v4-8"),
            train_batch_size=4,
            num_train_steps=100,
            # set hyperparameters
            learning_rate=6e-4,
            weight_decay=0.1,
            # keep eval quick for tutorial
            max_eval_batches=4,
        )
        ```

The `CpuOnlyConfig` is a [resource configuration](../references/resource-config.md) that requests a certain number of CPUs;
`GpuConfig` requests GPUs; and `TpuPodConfig` for requests TPUs.

### 4. Train the Model

To train the model, we use the `default_train` function from `experiments.defaults`.
This function takes a tokenized dataset, a model configuration, and a training configuration and returns (a step that trains) the model.

```python
nano_tinystories_model = default_train(
    name="marin-tutorial-nano-tinystories",
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "tinystories", "tutorial"],
    eval_harness_tasks=[],
    use_default_validation=False,
)
```

Let's break this down:

- `name`: The name of the training run. This is used to form the output path for the `ExecutorStep`.
- `tokenized`: The tokenized dataset to train on. Here, we're passing the `tinystories_tokenized` step we defined earlier.
- `model_config`: The model configuration to use. This is a `LmConfig` from [Levanter](https://github.com/stanford-crfm/levanter).
   We use the `llama_nano` configuration from `experiments.llama`.
- `train_config`: The training configuration to use, which we just defined.
- `tags`: Tags to add to the WandB tracker. This is useful for filtering runs in the WandB UI.
- `eval_harness_tasks`: A list of evaluation harness tasks to run during training. We pass an empty list here because we don't want to run any evals on this tiny model.
- `use_default_validation`: Whether to use the default validation sets. We pass `False` here because we just want to evaluate on TinyStories' validation set.

### Invoking the Executor

Finally, we need to invoke the executor to run our experiment.
We do this by calling `executor_main` from `marin.execution.executor`.
We put this in a `if __name__ == "__main__":`:

```python
if __name__ == "__main__":
    executor_main(
        steps=[
            nano_tinystories_model,
        ]
    )
```

This invokes the executor with our training step.
The executor will automatically run any dependencies of the training step, such as tokenization,
so we don't need to explicitly list them.

## Running the Experiment

To run the experiment locally, we can run the script directly:

=== "CPU"
    ```bash
    python experiments/tutorials/train_tiny_model_cpu.py --prefix local_store
    ```

=== "GPU"
    ```bash
    python experiments/tutorials/train_tiny_model_gpu.py --prefix local_store
    ```

The `--prefix` argument specifies the output directory for the experiment. It can be a local directory or anything [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) supports,
such as `s3://` or `gs://`.

This will take a few minutes to run. (Marin isn't optimized for low latency and Ray can take a while to schedule tasks initially...)

At the end, you should see a message like this:

```
2025-05-07 23:00:24,099	INFO executor.py:1036 -- Executor run took 263.37s
2025-05-07 23:00:24,099	INFO executor.py:1037 -- Experiment info written to local_store/experiments/train_tiny_model_cpu-0b39f4.json
2025-05-07 23:00:24,099	INFO executor.py:1038 -- View the experiment at https://localhost:5000/experiment?path=gs%3A%2F%2Fmarin-us-central2%2Fexperiments%2Ftrain_tiny_model_cpu-0b39f4.json
```

The JSON file contains a record of the experiment, including the steps and dependencies.
(Note that this link won't work unless you [start the data browser](data-browser.md).)

If you were to rerun this experiment, the executor would detect that the training step has already been run and skip it.

### Inspecting the Output

You can look at the artifacts logged to the output directory. Typically, for training jobs, most
interesting logging goes to WandB, which you likely disabled.

```
$ ls local_store/*
local_store/checkpoints:
marin-nano-tinystories-b4157e

local_store/experiments:
train_tiny_model_cpu-0b39f4.json

local_store/tokenized:
roneneldan
```

The `experiments` directory contains the experiment JSON file (which tracks all the steps and their dependencies).
The `checkpoints` directory contains the trained model.
The `tokenized` directory contains the tokenized dataset.

## Next Steps

Congratulations! You have trained your first model in Marin.  Choose your next adventure:

- Train a real [1B or 8B parameter language model](train-an-lm.md) using Marin.
- Read about Marin's key concepts and principles in [Concepts](../explanations/concepts.md).
- Learn about the [Executor framework](../explanations/executor.md).
- Read more about the full [language modeling pipeline](../explanations/lm-pipeline.md), including data processing.

## Troubleshooting

### The job failed, how do I rerun it?

By default, the Marin executor won't rerun steps that failed.
To force it to rerun a step, you can use the `--force_run_failed true` flag.

=== "CPU"
    ```bash
    python experiments/tutorials/train_tiny_model_cpu.py --prefix local_store --force_run_failed true
    ```

=== "GPU"
    ```bash
    python experiments/tutorials/train_tiny_model_gpu.py --prefix local_store --force_run_failed true
    ```

### I want to rerun the step after it succeeded, how do I do that?

By default, the Marin executor won't rerun steps that succeeded either.
The easiest way to do this is to remove the output directory for the step.
For instance, if the step is named `marin-nano-tinystories-b4157e`, you can remove the output directory with:

```bash
rm -rf local_store/marin-nano-tinystories-b4157e
```

Note, however, that WandB does not like reusing the same run ID, so you may need to change the `name` argument to `default_train` to a new value.
(If you're using WandB in offline mode, this isn't an issue.)

### I don't want to use WandB, how do I disable it?

`wandb offline` will disable WandB in your current directory.  Run `wandb online` to re-enable it.