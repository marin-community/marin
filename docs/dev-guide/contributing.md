# Contributing to Marin

We welcome contributions that add missing features or fix serious issues. If you
are unsure whether a change is wanted, open an issue or ask in the
[Marin Discord](https://discord.gg/J9CTk7pqcM) before sending a PR. We are
unlikely to merge typo fixes, stylistic rewrites, or speculative refactors that
are not tied to an issue.

## AI-generated contributions

We use coding agents ourselves and accept contributions made with them. We do
not accept drive-by PRs that an agent produced in one shot.

The rule of thumb: if your agent can do it in one shot, so can ours, so please
do not burden us with the PR. A throwaway agent-generated change costs us more to
review than it cost you to file. If you do contribute agent-assisted work, hold
it to the same bar as the rest of this guide: confirm the change is needed, that
it is correct, and that you can explain why it matters.

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Set up the Git hook that runs `infra/pre-commit.py`

```bash
git clone https://github.com/marin-community/marin.git
cd marin
uv venv --python 3.12
source .venv/bin/activate
uv sync --package marin-core --group dev
make setup_pre_commit
```

Alternatively, you can install all the core dependencies and build the `marin-core`
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

### Opening a pull request

Before opening a pull request:

1. Run `./infra/pre-commit.py --all-files --fix`.
2. Run `uv run pytest -m 'not slow'` for the files or packages you changed.
3. If your change adds, removes, renames, or rewires docs pages or docs-owned links, run `uv run python infra/check_docs_source_links.py`.
4. If your change is docs-heavy, run `uv run mkdocs build --strict`.
5. If your change adds or rewrites substantial prose, do a final prose-only review using `./.agents/skills/writing-style/SKILL.md`. Remove generic significance framing, stock AI-writing templates, and polished filler that does not add information.
6. Follow `./.agents/skills/writing-style/pull-requests.md` for the PR title and body. The body becomes the squash-merge commit message: lead with what changes, then why, and retain the evidence and caveats a future reader needs.
7. If the work came from an issue, end the PR body with `Fixes #NNNN` or `Part of #NNNN`.
8. After pushing, verify the relevant GitHub CI checks pass before considering the PR ready for review.

## Guidelines

Please see the [guidelines](../explanations/guidelines.md) for principles and practices for Marin.


# Data browser

The data browser lives in its own repository: [marin-community/data_browser](https://github.com/marin-community/data_browser). See its README for setup and development instructions.
