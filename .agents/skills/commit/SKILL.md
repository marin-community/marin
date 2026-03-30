---
name: commit
description: Run lint/format, create an informative commit describing recent work, and push to the current remote branch. Use when checkpointing progress on a feature branch.
---

# Skill: Commit

Lint, commit, and push the current work-in-progress to the remote branch.

## Steps

### 1. Lint and format

```bash
./infra/pre-commit.py --all-files --fix
```

If pre-commit reports errors that `--fix` cannot resolve automatically, fix
them manually before proceeding. Do not skip or weaken checks.

### 2. Stage changes

Stage all modified and new files that are part of the current work. Review
`git status` and `git diff` before staging.

- Stage specific files — avoid `git add -A` or `git add .`.
- Never stage secrets (`.env`, credentials, tokens).
- If unrelated changes are present, ask the user before including them.

### 3. Write the commit message

Examine the diff of staged changes and recent git log to understand what was
done. Write a commit message following these rules:

- **Subject line**: short imperative sentence (<72 chars). Optional `[scope]`
  prefix matching the area of work (e.g. `[zephyr]`, `[iris]`, `[docs]`).
- **Body** (optional, separated by blank line): 1-3 sentences explaining *why*
  the change was made, not *what* changed (the diff shows that). Include
  context a reviewer would need.
- No emoji, no markdown, no bullet lists in the subject.
- Do not credit yourself in the commit message.

### 4. Commit

Create the commit. If the commit fails due to a pre-commit hook, fix the
issue and create a **new** commit (do not amend).

### 5. Push

Push to the remote tracking branch. If no upstream is set, push with `-u`:

```bash
git push -u origin HEAD
```

If the push fails (e.g. diverged history), stop and ask the user — do not
force-push.

## Rules

- Never amend a commit unless the user explicitly asks.
- Never force-push.
- Never skip hooks (`--no-verify`).
- If there are no changes to commit, say so and stop.
