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
cat <<'EOF' > .git/hooks/pre-commit
#!/bin/sh
set -e
cd "$(git rev-parse --show-toplevel)"
uv run python infra/pre-commit.py --fix
EOF
chmod +x .git/hooks/pre-commit
```

Alternatively, you can install all the core dependencies and build `marin` as a Python
package with `make init`.

### Linting

The Git hook configured above runs `uv run python infra/pre-commit.py` before each commit so that the repo-standard lint/format checks pass.
You can also run them manually with `uv run python infra/pre-commit.py --all-files` or via `make lint`.

### Testing

You can run the tests with `make test`.

*Note* that to run the unit tests, you must not have set `RAY_ADDRESS`. You can unset it with `unset RAY_ADDRESS` or `export RAY_ADDRESS=""`.

## Guidelines

Please see the [guidelines](../explanations/guidelines.md) for principles and practices for Marin.


# Data browser

Marin comes with a [data browser](https://github.com/marin-community/marin/tree/main/data_browser) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, run:

```bash
cd data_browser
python server.py --config conf/local.conf
```

For more information, see the [data browser README](https://github.com/marin-community/marin/blob/main/data_browser/README.md).
