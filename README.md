# Marin

> "*I am not afraid of storms, for I am learning how to sail my ship."*<br/>
> – Louisa May Alcott

Marin is a modular, open-source framework for the research and development of
foundation models.  A key property of Marin is reproducibility: All the steps
raw data to the final model are recorded for posterity (not just the final
model).  Moreover, all [experiments](docs/explanation/experiments.md) (whether successful
or not) are also recorded, so the entire research process (not just the
development of the final model) is transparent.

The core part of Marin is minimal, consisting of basically an
[executor framework](docs/explanation/executor.md), which manages the execution of a set of
arbitrary steps.

Marin's primary use case is to build a language model like Qwen 3,
which involves data curation, transformation, filtering, tokenization,
training, and evaluation (see [overview](docs/lm/overview.md)).
Note that for now, all this specific code resides in this repository.

## Setup

To install Marin, create a new virtual environment (or `conda` environment)
with the appropriate Python version (3.11), and then run the following:

```bash
git clone https://github.com/stanford-crfm/marin
cd marin
```

To run the tests you need to install Node:
```bash
# For macOS
brew install node
# For Linux (not tested)
sudo apt-get update
sudo apt-get install -y nodejs npm
# For Windows (not tested)
choco install nodejs
```

Then you can install all the core dependencies and build `marin` as a Python
package with `make init`.
Installing the `[dev]` requirements will additionally install test,
linting, and debugging dependencies (e.g., `pytest`).

## Linting & Tests

1. We have linters set up to ensure code quality. You can run them with `make lint`
2. To run the tests, run `make test`

## Hello world example

Let's run your first [hello world experiment](experiments/hello_world.py),
which has two steps:

1. Generate some numbers.
2. Compute some statistics of the numbers.

To run this example (and output), simply type:

```bash
python experiments/hello_world.py --prefix var
```

This command should create the following assets:

1. `var/experiments/hello_world-7063e5.json`: stores a record of all the steps in this experiment.
2. `var/hello_world/data-d50b06`: the output of step 1 (generate some numbers, stored in `numbers.json`).
3. `var/hello_world/stats-b5daf3`: the output of step 2 (compute some statistics, stored in `stats.json`).

Note that if you run the same command again, it will detect that both steps
have been already run and return automatically.

## Data browser

Marin comes with a [data browser](data_browser/README.md) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, run:

```bash
cd data_browser
python server.py --config conf/local.conf
```

Once the server is started, go to
[http://localhost:5000](http://localhost:5000) and navigate around to the
experiment JSON file to get a nicer view of the experiment (the URL is also
printed out when you run the experiment).

See the [data browser README](data_browser/README.md) for more details.

## What's next?

To learn more about the core infrastructure:
- [Concepts](docs/explanation/concepts.md): key concepts and principles that underpin the Marin project.
- [Executor framework](docs/explanation/executor.md): how to manage Python libraries, run big parallel jobs using Ray, how versioning works, etc.
- [Experiments](docs/explanation/experiments.md): how we use the executor framework to run machine learning experiments.

Or you can jump directly to learn about our [language modeling efforts](docs/lm/overview.md).

For getting started and installation please see our [documentation](docs/index.md) or visit our [readthedocs](https://marin.readthedocs.io/en/latest/)

