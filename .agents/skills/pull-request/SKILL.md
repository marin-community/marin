---
name: pull-request
description: Authoring pull requests in Marin. Use when creating or updating a PR, and whenever changing a branch that is already associated with a PR.
---

# Skill: Author a Pull Request

This skill defines the exact output format for pull requests. Follow it
literally when creating or updating a PR. Also apply it whenever you make
changes to a branch that already has an associated PR.

When a branch is already associated with a PR, check the current PR title and
description after your code changes. Keep them aligned with the actual scope of
the branch, and update them if they are no longer correct or sufficient.

## PR Description Format

The PR description becomes the squash-merge commit message. Write it as
plain text — no markdown.

**Title:** Short imperative sentence. Optional scope tag in brackets.

**Body:** 1-3 sentences stating what changed and why. End with issue link if one exists.

**Hard rules — violations will be rejected:**

- No markdown: no headers (`##`), no bullet lists (`-`/`*`), no tables, no images, no `[text](url)` links.
- No checkboxes (`- [ ]`, `- [x]`).
- No section headers like `## Summary`, `## Test plan`, `## Changes`.
- No filler phrases: "This PR...", "I noticed...", "Summary of changes:".
- No emoji.
- Under ~80 words total.

### Example (follow this exactly)

```
Title: [RL] Fix loss: use global token normalization instead of per-example

Body:
Switch DAPO loss from per-example normalization (/ n_i) to global token
normalization (/ N). Per-example normalization over-weights short responses,
hurting math reasoning where correct answers need longer derivations.
Adds regression test.

Fixes #1234
```

## Issue Linking

If the work originated from a GitHub issue, reference it:

- `Fixes #NNNN` — auto-closes the issue on merge.
- `Part of #NNNN` — for partial work.

Do not create an issue solely to satisfy this rule. When there is no
pre-existing issue (e.g., user-directed work in a conversation), omit the
issue link.

## Pre-Push Checklist

Run these before pushing. Do not skip any step.

1. `./infra/pre-commit.py --all-files --fix` — resolve all issues. Do not substitute `uv run pre-commit ...`.
2. `uv run pytest -m 'not slow'` — relevant test directories.
3. If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`
4. If docs-heavy: `uv run mkdocs build --strict`

After pushing, monitor CI: `gh pr view <number> --json statusCheckRollup`.
Fix failures before considering the PR complete.

## Specifications (>500 LOC)

PRs over ~500 lines must include a specification. Put it in (preferred order):

1. The associated GitHub issue
2. First PR comment after the description
3. `docs/design/<topic>.md` or `.agents/projects/<topic>.md`

A specification contains:

1. **Problem** — what is broken or missing, with file/line references.
2. **Approach** — which modules change, what gets added/removed.
3. **Key code** — 10-30 line snippets for non-obvious logic.
4. **Tests** — what is tested, how, and why sufficient.

## Creating the PR

Unless the user says otherwise, and when permissions allow, push directly to a
branch on the main repository and open the PR from that branch. Do not default
to pushing to a fork. Use a fork only when direct push to the main repository
is not available or the user explicitly asks for it.

Use `gh pr create` with these flags:

```bash
gh pr create \
  --title "<title>" \
  --body "<plain text body>" \
  --label agent-generated
```

- Add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.

## See Also

- `.agents/skills/github-pr-review/` — PR review skill (separate concern)
- `.agents/skills/fix-issue/` — end-to-end issue fix workflow
- `AGENTS.md` — coding guidelines
