# Contributing to Marin

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Set up the Git hook that runs `infra/pre-commit.py`

```bash
git clone https://github.com/marin-community/marin.git
cd marin
uv venv --python 3.11
source .venv/bin/activate
uv sync --package marin --group dev
make setup_pre_commit
```

Alternatively, you can install all the core dependencies and build `marin` as a Python
package with `make init`.

### Linting

The Git hook configured above runs `./infra/pre-commit.py` before each commit so that the repo-standard lint/format checks pass.
You can also run them manually with `./infra/pre-commit.py --all-files --fix` or via `make lint`.

### Testing

For most changes, start with targeted fast tests:

```bash
uv run pytest -m 'not slow' <relevant test paths>
```

Use `make test` when you need the full default test suite.

*Note* that to run the unit tests, you must not have set `RAY_ADDRESS`. You can unset it with `unset RAY_ADDRESS` or `export RAY_ADDRESS=""`.

### Opening a pull request

Before opening a pull request:

1. Run `./infra/pre-commit.py --all-files --fix`.
2. Run `uv run pytest -m 'not slow'` for the files or packages you changed.
3. Make sure the PR body references an issue with `Fixes #NNNN` or `Part of #NNNN`.

For the end-to-end branch and fork workflow, follow the PR steps in [Submitting to the Marin Speedrun](../tutorials/submitting-speedrun.md#submit). The same GitHub push and PR process applies to non-speedrun changes.

## Guidelines

Please see the [guidelines](../explanations/guidelines.md) for principles and practices for Marin.


# Data browser

Marin comes with a [data browser](https://github.com/marin-community/marin/tree/main/data_browser) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, the recommended development loop is:

```bash
cd data_browser
uv sync
npm install
uv run python run-dev.py --config conf/local.conf
```

If you only need the backend API, run:

```bash
cd data_browser
DEV=true uv run python server.py --config conf/local.conf
```

For more details, see the [data browser tutorial](../tutorials/data-browser.md).
