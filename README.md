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
You can see the [training script](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)
or read the [retrospective](docs/reports/marin-8b-retro.md).

<!--marin-intro-end-->

The documentation for Marin is available on [ReadTheDocs](https://marin.readthedocs.io/en/latest/) or in the [`docs/`](docs/) folder.

<!--marin-first-steps-start-->

To get started with Marin:

- [Install](docs/tutorials/installation.md) Marin.
- Train a [tiny language model](docs/tutorials/first-experiment.md) using Marin.
- See how to run a much larger [DCLM 1B/1x](docs/tutorials/train-an-lm.md) experiment using Marin.
- See a [summary of the experiments](docs/reports/index.md) we've run.
- Join the [Marin Discord](https://discord.gg/J9CTk7pqcM) to chat with the community.

<!--marin-first-steps-end-->

## Example

Marin experiments are defined as a set of steps that can depend on each other and are executed in a topological order,
like a Makefile.

As a brief example of how you can use Marin, here is a complete script for training a tiny model on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
You can check out the [full script](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model.py) for more details.

<!--marin-example-start-->

```python
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# 1. Tokenize the dataset as a lazy handle — nothing downloads yet.
tinystories_tokenized = tokenized(
    name="tokenized/tinystories",
    source="roneneldan/TinyStories",
    tokenizer=marin_tokenizer,
    sample_count=1000,  # cap at 1 000 samples per shard to keep the tutorial fast
)

# 2. Train the model — depends on the tokenized dataset above.
nano_tinystories_model = train_lm(
    name="checkpoints/marin-nano-tinystories",
    version="v1",
    model=llama_nano,
    optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
    # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
    datasets={tinystories_tokenized: 1.0},
    batch_size=4,
    seq_len=2048,
    num_train_steps=100,
    z_loss_weight=None,
    evals=None,  # no point evaluating such a tiny model
    resources=ResourceConfig.with_cpu(),
)

if __name__ == "__main__":
    StepRunner().run([lower(nano_tinystories_model)])
```

Here, we create two steps, one for tokenizing the dataset and one for training the model.
The training step depends on the tokenized dataset step, so it will be executed after the tokenization step is completed.

<!--marin-example-end-->

With slight modifications, you can extend this to train a [larger model on a larger dataset](docs/tutorials/train-an-lm.md),
a [mixture of datasets](docs/tutorials/train-an-lm.md#mixture-of-sources), even scaling to very large TPU pods
(or multislice TPU, and, soon, multi-node GPUs!).

## Agent Skills

- See `.agents/skills/` (also `.claude/skills/`) for loadable agent skills. For example, `.agents/skills/add-dataset/` has a step-by-step guide to adding new datasets.
