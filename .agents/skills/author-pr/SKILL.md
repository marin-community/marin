---
name: author-pr
description: Author or update a Marin PR in the required plain-text format. Use when creating or updating a PR.
---

# Skill: Author a Pull Request

The exact output format for pull requests. Follow it literally when creating or
updating a PR, including any time you change a branch that already has one — keep
the PR title and description aligned with the branch's actual scope.

## PR Description Format

The PR description becomes the squash-merge commit message. Write it as plain text.

**Title:** Short imperative sentence. Optional scope tag in brackets.
**Body:** 1-3 sentences stating what changed and why. End with an issue link if one exists.

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

If the work originated from a GitHub issue, reference it: `Fixes #NNNN`
(auto-closes on merge) or `Part of #NNNN` (partial work). Do not create an issue
just to satisfy this rule — omit the link when there is no pre-existing issue.

## Pre-Push Checklist

Run these before pushing; do not skip any step.

1. `./infra/pre-commit.py --changed --fix` — resolve all issues. Do not substitute `uv run pre-commit ...`.
2. `uv run pytest -m 'not slow'` — relevant test directories.
3. If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`
4. If docs-heavy: `uv run mkdocs build --strict`

After pushing, monitor CI with `gh pr view <number> --json statusCheckRollup`;
fix failures before considering the PR complete.

When opening a new PR, if an automation feature is present (e.g.
`codex_app.automation_update`), use it to schedule follow-up checks for
automated reviews and review comments. The automation
should inspect the PR, address comments automatically only when the fix is
unobjectionable and certainly correct, run the relevant checks, and push the
follow-up commit. Defer anything that needs human judgment, design choice, or
tradeoff evaluation back to the user instead of changing it automatically.

## Specifications (>500 LOC)

PRs over ~500 lines must include a specification, placed (preferred order) in the
associated GitHub issue, the first PR comment, or `docs/design/<topic>.md` /
`.agents/projects/<topic>.md`. It contains:

1. **Problem** — what is broken or missing, with file/line references.
2. **Approach** — which modules change, what gets added/removed.
3. **Key code** — 10-30 line snippets for non-obvious logic.
4. **Tests** — what is tested, how, and why sufficient.

## Creating the PR

Unless the user says otherwise and permissions allow, push directly to a branch
on the main repository and open the PR from that branch — use a fork only when
direct push is unavailable or the user asks for it.

```bash
gh pr create \
  --title "<title>" \
  --body "<plain text body>" \
  --label agent-generated
```

- Add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.
- If present, schedule the follow-up automation described above after the PR exists.

## See Also

- `.agents/skills/review-pr/` — PR review skill (separate concern)
- `.agents/skills/fix-issue/` — end-to-end issue fix workflow
- `AGENTS.md` — coding guidelines
