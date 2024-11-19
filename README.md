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

## Dev Notes

To run the tests locally, first run `export RAY_ADDRESS=` to run Ray in local mode.

NOTE: The first time you run tests locally, you will need install `pandiff` to run the snapshot tests. This is a one-time install that can be done using the following commands:

```bash
brew install node
conda install -c conda-forge pandoc
npm install -g pandiff
```

1. We have linters set up to ensure code quality. You can run them with:
   ```bash
   pre-commit run --all-files
   ```
2. To run the tests, run `PYTHONPATH=tests:. pytest tests --durations=0 -n 4 --tb=no -v`
3. When submitting to cluster, we recommend using run script `marin/run/run.py`. See `marin/run/README.md`
   for more details.


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


## Submitting jobs to Ray cluster

The Ray cluster only has the core ray and vllm packages installed.
We are still keeping vllm as it requires significant time to build.
If you run an experiment directly on ray, expect missing dependencies.
To resolve this, you can now install dependencies in a fine-grained way using [ray_run.py](../marin/run/ray_run.py) or
using pip_dependency_groups argument in `ExecutorStep`

Dependency Management:

1. Use [ray_run.py](../marin/run/ray_run.py) to handle dependencies across an entire experiment.
Just add any extra packages you need with `--pip_deps`. Core dependencies (levanter, draccus, fspsec, etc.)
are automatically installed from [pyproject.toml](pyproject.toml).
2. For step-specific dependencies, use `pip_dependency_groups` in `ExecutorStep`.
This takes a list where each item is either (1) A key from `project.optional-dependencies` dictionary in pyproject.toml
or (2) A specific pip package. Check out [quickstart.py](experiments/quickstart.py) for an example.

Example usage:

If you earlier used the command

```bash
ray job submit --working-dir . -- python experiments/check_pip_packages_env_variables.py
```

Now you can run the same command with

```bash
python marin/run/ray_run.py --env_vars HF_TOKEN hf_abcd --pip_deps trafilatura,dolma
-- python experiments/check_pip_packages_env_variables.py
```

The new command will:
1. Install all the core dependencies in pyproject.toml
2. Install `trafilatura` and `dolma` as additional dependencies

Example for using `pip_dependency_groups` in `ExecutorStep`:

```python
number_of_restarts = ExecutorStep(
    name=...,
    fn=...,
    config=...,
    pip_dependency_groups=["quality_dedup_consolidate", "google-cloud-logging"],
)
```

This will install the dependencies specified in the `quality_dedup_consolidate` groups
and also pip install `google-cloud-logging`.
