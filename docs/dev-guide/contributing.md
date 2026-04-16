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
3. If your change adds, removes, renames, or rewires docs pages or docs-owned links, run `uv run python infra/check_docs_source_links.py`.
4. If your change is docs-heavy, run `uv run mkdocs build --strict`.
5. If your change adds or rewrites substantial prose, do a final prose-only review using `./.agents/skills/writing-style/SKILL.md`. Remove generic significance framing, stock AI-writing templates, and polished filler that does not add information.
6. Keep the PR description concise and plain text because it becomes the squash-merge commit message. The `.github/PULL_REQUEST_TEMPLATE.md` file shows the expected style.
7. Make sure the PR body references an issue with `Fixes #NNNN` or `Part of #NNNN`.
8. After pushing, verify the relevant GitHub CI checks pass before considering the PR ready for review.

## Guidelines

Please see the [guidelines](../explanations/guidelines.md) for principles and practices for Marin.


# Data browser

The data browser lives in its own repository: [marin-community/data_browser](https://github.com/marin-community/data_browser). See its README for setup and development instructions.
