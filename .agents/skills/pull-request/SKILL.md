---
name: pull-request
description: Authoring and reviewing pull requests in Marin. Use when creating a PR, writing a PR description, running pre-PR checks, or reviewing a PR.
---

# Recipe: Pull Requests

Guidelines for authoring and reviewing pull requests in Marin. Applies to both
human and agent authors.

## PR Descriptions

The PR description doubles as the squash-merge commit message. Write it
accordingly: no markdown formatting beyond plain text paragraphs and bullets,
concise, and self-contained.

**Required elements:**

1. **Title** -- short imperative sentence, optionally prefixed with a scope tag
   in brackets (e.g. `[RL] Fix loss normalization`).
2. **Body** -- one paragraph stating what changed and why. Follow with a few
   bullets for specific changes if needed. Link the issue with `Fixes #NNNN`.

**Style rules:**

- No markdown headers, tables, or images in the body.
- No filler ("This PR...", "I noticed..."). State the change directly.
- Keep it under ~150 words. A reviewer should absorb it in 30 seconds.
- Include example output or metrics when the change is substantial.

### Example

```
Title: [RL] Fix loss: use global token normalization instead of per-example

Body:
Fixes a regression in the DAPO loss computation by switching from per-example
normalization (/ n_i) back to global token normalization (/ N). Per-example
normalization gives shorter responses disproportionately more gradient weight,
which hurts math reasoning tasks where correct answers often require detailed,
longer derivations.

- Switch normalization in `compute_dapo_loss()` from per-example to global.
- Add regression test in `tests/test_dapo_loss.py`.

Fixes #1234
```

## Linking Issues

Every PR should reference a GitHub issue. Use `Fixes #NNNN` in the description
body so GitHub auto-closes the issue on merge. For partial work, use
`Part of #NNNN` instead.

Prefer more specific issues when possible, falling back to general issues like tracking epics if needed.

If no issue exists yet, create one with `gh issue create` before opening the PR.

## Testing

Before marking a PR ready for review:

- Run `./infra/pre-commit.py --all-files --fix` and fix any issues.
- Do not substitute `uv run pre-commit ...`; use `./infra/pre-commit.py` directly so checks run in all environments.
- Run `uv run pytest -m 'not slow'` for the relevant test directories.
- When a PR adds, deletes, renames, or rewires docs pages or docs-owned links, run `uv run python infra/check_docs_source_links.py` before pushing so stale GitHub source links fail locally instead of in CI.
- For docs-heavy changes, also run `uv run mkdocs build --strict`.
- Ensure all CI checks pass after pushing. Monitor with
  `gh pr view <number> --json statusCheckRollup`.
- If CI fails, investigate and push fixes. Do not consider the PR complete
  until relevant checks pass.

### What to test

- Write tests _before_ or alongside your fix, not after.
- Prefer behavioral/integration tests over mocks.
- Do not test obvious behavior (a type exists, a constant has a value).
- Do not reimplement the system under test inside your assertions. Tests should
  verify externally-observable outputs and side effects, not mirror the
  production logic with `assert` statements.
- Focus on tests that would catch real regressions: boundary conditions, error
  paths, and integration points between modules.
- Use existing test files when appropriate.

## Self-Review

Before requesting review, go through your own diff:

- Add high-level comments explaining non-obvious changes (e.g. "moved from
  file A to file B", "this block handles the edge case where...").
- Verify the diff matches the PR description. If it doesn't, update one or
  the other.

## Specifications for Large PRs

Any PR that adds or modifies more than ~500 lines of code must include a
specification. Smaller PRs may include one if the change is architecturally
significant or touches many modules.

Prefer to put the specification in one of (in preferred order):

- The associated issue for this work
- A GitHub PR comment (first comment after the description).
- `docs/design/<topic>.md` or `.agents/projects/<topic>.md` -- when it is important for the spec to be referenced and used in future work.

A specification must contain:

1. **Problem** -- what is broken or missing, with file/line references.
2. **Approach** -- concrete plan: which modules change, what gets added/removed.
3. **Key code** -- short snippets (10-30 lines) for non-obvious logic.
4. **Tests** -- what is tested, how, and why that set of tests is sufficient.

Optional: non-goals, future work, alternatives considered. Follow the format
in `.agents/skills/design-doc/` for longer standalone specifications.

A specification passes the **reproduction test**: could a different engineer,
given only the spec and access to the repo, produce substantially the same PR?

## Review Process

### Human review: specification-first

Human reviewers focus on the specification, not the diff:

1. Is the problem statement accurate?
2. Is the approach sound?
3. Is the scope right?
4. Are the listed files complete?

### Agent review: specification compliance

Automated review validates the implementation matches the specification:

1. Every change in the diff should be traceable to the spec.
2. The implementation follows the described approach.
3. Described test scenarios exist and test what they claim. Tests should
   exercise meaningful behavior, not merely restate production logic.
4. No bugs, logic errors, resource leaks, or race conditions.

The standard PR review prompt includes specification validation when a spec is present.

## Agent-Specific Rules

- Add the `agent-generated` label when creating a PR.
- Never credit yourself in commits or PR descriptions.
- When the PR addresses an issue, include `Fixes #NNNN` in the body.

## See Also

- `.agents/skills/github-pr-review/` -- PR review prompt
- `.agents/skills/design-doc/` -- design document format
- `.agents/skills/fix-issue/` -- end-to-end issue fix workflow
- `AGENTS.md` -- coding guidelines
