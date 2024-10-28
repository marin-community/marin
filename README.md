# Marin

> "*I am not afraid of storms, for I am learning how to sail my ship."*<br/>
> – Louisa May Alcott

Marin is an open-source framework for building foundation models in a
reproducible and transparent way.  All the code, data, and experiments for all
stages of the pipeline (e.g., data curation, transformation, filtering,
tokenization, training, evaluation) are accessible on the platform with full
provenance.

Marin leverages several tools:
- For training, it uses [levanter](https://github.com/stanford-crfm/levanter),
  a Jax-based framework that's legible, scalable, and reproducible.
- For scheduling distributed jobs over a cluster for data processing and
  training, we use [Ray](https://docs.ray.io/).
- We use the same data formats as [Dolma](https://github.com/allenai/dolma).

## Setup

To get set up, create a new virtual environment (or `conda` environment) with
the appropriate Python version (3.10), then run the following:

```bash
git clone https://github.com/stanford-crfm/marin
cd marin
pip install -e ".[dev]"
```

This will install all the core dependencies and build `marin` as a Python
package. Installing the `[dev]` requirements will additionally install test,
linting, and debugging dependencies (e.g., `pytest`).

## Quickstart

To get started, you can run a toy example which starts with raw HTML, processes
it, trains a quality classifer, filters the data, and performs deduplication.
TODO: add training and evaluation

```bash
python experiments/quickstart.py
```

## Running tests

To run the tests, run `pytest` in the root directory. This will run the unit
tests and snapshot tests.

### Snapshot tests

For HTML-to-text conversion, we have snapshot unit tests.  To add a test case,
do the following:

* Add an html file to `tests/snapshots/inputs/` that you want to test.
* Add the expected markdown output to `tests/snapshots/expected/` with the same
  name as the input file.
* Commit these files.

Pro-tip: You can copy the markdown from `process_url.py`'s output to the
expected file and edit it as needed.

If it's reasonable, try to add a unit test as well. This will help ensure that
the conversion is correct.  If you've made a change that you think is correct,
you can update the snapshots by copying `tests/snapshots/outputs/` to
`tests/snapshots/expected/`. This will overwrite the expected output with the
new output. You should review these changes before committing them.
