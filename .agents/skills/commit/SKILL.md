---
name: commit
description: Lint, commit, and push work-in-progress to the branch.
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

Examine the staged diff and recent git log, then write a message:

- **Subject**: imperative sentence (<72 chars). Optional `[scope]` prefix
  (e.g. `[zephyr]`, `[iris]`, `[docs]`).
- **Body** (optional, blank-line separated): 1-3 sentences on *why*, not
  *what*. Include context a reviewer needs.
- No emoji, no markdown, no bullets in the subject.
- Do not credit yourself.

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
