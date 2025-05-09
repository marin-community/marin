# Marin

<a href="https://github.com/marin-community/marin/actions?query=branch%3Amain++">
    <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/marin-community/marin/run_tests.yaml?branch=main">
</a>
<a href="https://marin.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation" src="https://readthedocs.org/projects/marin/badge/?version=latest">
</a>
<a href="">
<img alt="License" src="https://img.shields.io/github/license/marin-community/marin?color=blue" />
</a>

<!--marin-intro-start-->

> "*I am not afraid of storms, for I am learning how to sail my ship.*"<br/>
> – Louisa May Alcott

Marin is a modular, open-source framework for the research and development of
foundation models.  A key property of Marin is reproducibility: All the steps
raw data to the final model are recorded for posterity (not just the final
model).  Moreover, all experiments (whether successful
or not) are also recorded, so the entire research process (not just the
development of the final model) is transparent.

The core part of Marin is minimal, consisting of basically an
[executor framework](docs/explanation/executor.md), which manages the execution of a set of
arbitrary steps. Marin also has a [data browser](https://crfm.stanford.edu/marin/data_browser/) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.

Marin's primary use case is to build a language model like Qwen 3,
which involves data curation, transformation, filtering, tokenization,
training, and evaluation (see [overview](docs/lm/overview.md)).
Note that for now, all this specific code resides in this repository.

<!--marin-intro-end-->

## Documentation

The documentation for Marin is available on [ReadTheDocs](https://marin.readthedocs.io/en/latest/) or in the [`docs/`](docs/) folder.
TODO XXX temporarily here: https://marin-mkdocs.readthedocs.io/en/latest/

<!--marin-first-steps-start-->

To learn more about Marin and how it works, you can:

- Train a [tiny language model](docs/how-to-guides/train-an-lm.md) using Marin.
- Read about Marin's key concepts and principles in [Concepts](docs/explanation/concepts.md)
- Follow a [Executor Hello World tutorial](docs/tutorials/executor-101) to get a feel for how Marin works.
- Learn about the [Executor framework](docs/explanation/executor.md): how to manage Python libraries, run big parallel jobs using Ray, how versioning works, etc.
- Read about [Experiments](docs/explanation/experiments.md): how we use the executor framework to run machine learning experiments.

<!--marin-first-steps-end-->

## Getting Started

Full documentation on setup is available on [ReadTheDocs](https://marin.readthedocs.io/en/latest/tutorials/getting-started/)
TODO XXX temporarily here: https://marin-mkdocs.readthedocs.io/en/latest/tutorials/getting-started/

### Installation

```bash
git clone https://github.com/marin-community/marin
cd marin
```

Create a new virtual environment (or `conda` environment)
with the appropriate Python version (3.11) and run `pip install`. For example, with `virtualenv`:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.11 marin-venv
source marin-venv/bin/activate
pip install -e .
```

To contribute to Marin, please see the [Contributing Guide](CONTRIBUTING.md).

## Example

Marin experiments are defined as a set of steps that can depend on each other and are executed in a topological order,
almost like a Makefile.

As a brief example of how you can use Marin, here is a complete script for training a tiny model on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
You can check out the [full script](https://github.com/marin-community/marin/blob/main/experiments/tutorial/exp1191_train_tiny_model_cpu.py) for more details.

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
        ]
    )
```

Here, we create two [steps](docs/explanation/executor.md#steps), one for tokenizing the dataset and one for training the model.
The training step depends on the tokenized dataset step, so it will be executed after the tokenization step is completed.

<!--marin-example-end-->

With slight modifications, you can extend this to train a [larger model on a larger dataset](docs/how-to-guides/train-an-lm.md),
a [mixture of datasets](docs/how-to-guides/train-an-lm.md#mixture-of-sources), even scaling to very large TPU pods
(or multislice TPU, and, soon, multi-node GPUs!).


## Get Involved

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

- We also have a list of [good first issues](https://github.com/marin-community/marin/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- You can find us on [Discord](https://discord.gg/J9CTk7pqcM).
