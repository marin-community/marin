# Marin

<a href="https://marin.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation" src="https://readthedocs.org/projects/marin/badge/?version=latest">
</a>
<a href="">
    <img alt="License" src="https://img.shields.io/github/license/marin-community/marin?color=blue" />
</a>

<!--marin-intro-start-->

> "*I am not afraid of storms, for I am learning how to sail my ship.*"<br/>
> – Louisa May Alcott

[Marin](https://marin.community) is an open-source framework for the research and development of [foundation models](https://en.wikipedia.org/wiki/Foundation_model).

A key feature of Marin is **reproducibility**: every step, from raw data to the final model are recorded, not just the end result.
This includes failed experiments, so the entire research process is transparent.

Marin's primary use case is training language model like Llama, DeepSeek, Qwen, etc.
Notably, this includes data curation, transformation, filtering, tokenization, training, and evaluation.

We used Marin to train the first open-source 8B parameter model to outperform Llama 3.1 8B.
You can see the [training script](https://github.com/marin-community/marin/blob/main/experiments/tootsie/exp600_tootsie.py)
or read the [retrospective](docs/reports/marin-8b-retro.md).

<!--marin-intro-end-->

The documentation for Marin is available on [ReadTheDocs](https://marin.readthedocs.io/en/latest/) or in the [`docs/`](docs/) folder.

<!--marin-first-steps-start-->

To get started with Marin:

- [Install](docs/tutorials/installation.md) Marin.
- Train a [tiny language model](docs/tutorials/first-experiment.md) using Marin.
- See how to run a much larger [DCLM 1B/1x](docs/tutorials/train-an-lm.md) experiment using Marin.
- See a [summary of the experiments](docs/reports/index.md) we've run.
- Participate in the Marin [Speedrun competition](docs/tutorials/submitting-speedrun.md) to try to find the most efficient way to train a language model.
- Try out the [Marin Datashop](docs/tutorials/datashop.md) to contribute and create data for your use case.
- Join the [Marin Discord](https://discord.gg/J9CTk7pqcM) to chat with the community.

<!--marin-first-steps-end-->

## Example

Marin experiments are defined as a set of steps that can depend on each other and are executed in a topological order,
like a Makefile.

As a brief example of how you can use Marin, here is a complete script for training a tiny model on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
You can check out the [full script](https://github.com/marin-community/marin/blob/main/experiments/tutorial/train_tiny_model_cpu.py) for more details.

<!--marin-example-start-->

```python
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import CpuOnlyConfig

# 1. Choose a dataset
tinystories_hf_id = "roneneldan/TinyStories"

# 2. Tokenize the dataset
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,  # path to write tokenized files (tokenized/ will be prepended)
    dataset=tinystories_hf_id,  # HF dataset id
    tokenizer=llama3_tokenizer,
)

# 3. Define training configuration
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

# 4. Train the model
nano_tinystories_model = default_train(
    name="marin-nano-tinystories",
    # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    # wandb tags
    tags=["llama", "nano", "tinystories", "tutorial"],
    # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
    # during training, but there's no point in running evals on such a tiny model
    eval_harness_tasks=[],
    # to keep tutorial fast, skip default validation sets
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[
        nano_tinystories_model,
    ])
```

Here, we create two [steps](docs/explanation/executor.md#steps), one for tokenizing the dataset and one for training the model.
The training step depends on the tokenized dataset step, so it will be executed after the tokenization step is completed.

<!--marin-example-end-->

With slight modifications, you can extend this to train a [larger model on a larger dataset](docs/tutorials/train-an-lm.md),
a [mixture of datasets](docs/tutorials/train-an-lm.md#mixture-of-sources), even scaling to very large TPU pods
(or multislice TPU, and, soon, multi-node GPUs!).

## Agent-Friendly Recipes

- New: See `docs/recipes/add_dataset.md` for a step-by-step guide to adding new datasets, designed for both humans and coding agents.
