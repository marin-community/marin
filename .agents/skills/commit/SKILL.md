---
name: commit
description: Lint, run the pre-PR checks, commit, push, and author or update the branch's pull request in the required plain-text format. Use when committing, pushing, or creating/updating a PR.
---

# Skill: Commit & PR

Lint, run the pre-PR checks, commit, push the current work, and — when the
branch is ready for review — open or update its pull request. The checks run
**before** the push, so "ready for review" is never something you discover after
the branch is already public.

For a quick work-in-progress checkpoint, run steps 1, 4, 5, 6 and stop. Run the
full sequence (including the review pass and the PR step) before you open or
update a PR.

## 1. Lint and format

```bash
./infra/pre-commit.py --changed-files --fix   # diff-scoped; use --all-files for a full sweep
```

`./infra/pre-commit.py` is the required entry point — never `uv run pre-commit`,
never `--no-verify`. If `--fix` cannot resolve something, fix it by hand. Do not
skip or weaken checks.

## 2. Lint-catalog review (before every PR)

```bash
./infra/pre-commit.py --review --agent-command='<your headless CLI>'
```

Always run this before opening a PR. Pass your own headless invocation
(`--agent-command='claude -p'` for Claude Code, `'codex exec'` for Codex;
defaults to `claude -p`). Then fix or respond to all findings, reporting your
actions to the user. You may search the `infra/lint` files for more information
about each code. Treat lints as guidelines, follow them if they make the code
_better_, but your goal is always to achieve high-quality code, not blind
adherence.

## 3. Tests and docs checks (when relevant)

- `uv run pytest -m 'not slow'` over the test directories your change touches.
- If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`.
- If the change is docs-heavy: `uv run mkdocs build --strict`.

## 4. Stage changes

Review `git status` and `git diff`, then stage the specific files that are part
of this work.

- Stage specific files — avoid `git add -A` / `git add .`.
- Never stage secrets (`.env`, credentials, tokens).
- If unrelated changes are present, ask the user before including them.

## 5. Commit

- **Subject**: imperative sentence (<72 chars), optional `[scope]` prefix
  (`[iris]`, `[zephyr]`, `[docs]`, …).
- **Body** (optional, blank-line separated): what, and why, keep it short & readable
  Include context a reviewer needs.
- No emoji, no markdown, no bullets in the subject. Do not credit yourself.

Create the commit. If a pre-commit hook fails, fix the issue and make a **new**
commit — never amend (unless the user asks) and never force-push.

Never credit yourself or add a tag line in commits "Created by Claude", "Created by Codex", etc.

## 6. Push

Push to the remote tracking branch (`git push -u origin HEAD` if no upstream is
set). If the push is rejected (diverged history), stop and ask the user — do not
force-push.

## 7. Open or update the PR

Do this once the branch is ready for review. The PR description becomes the
squash-merge commit message — write it as plain text.

**Title:** short imperative sentence; optional `[scope]` tag.
**Body:** what changed and why. End with an issue link if one exists.

Keep the title and body aligned with the branch's actual scope — including when
you change a branch that already has a PR.

**Hard rules — violations are rejected:**

- No "Validation" / "Testing" / "written by …" filler. The body is *what & why*,
  not *how I tested it* or *who wrote it*.
- No checkboxes (`- [ ]`, `- [x]`), no emoji.
- No filler openers ("This PR…", "I noticed…", "Summary of changes:").
- Under ~500 words.
- Don't credit yourself, no "Created by Claude" filler.
- Only use Markdown & sections if this is a large change that requires structured description

Example:

```
Title: [RL] Fix loss: use global token normalization instead of per-example

Body:
Switch DAPO loss from per-example normalization (/ n_i) to global token
normalization (/ N). Per-example normalization over-weights short responses,
hurting math reasoning where correct answers need longer derivations.

Fixes #1234
```

**Issue linking.** If the work came from a GitHub issue, add `Fixes #NNNN`
(auto-closes on merge) or `Part of #NNNN` (partial work). Do not invent an issue
just to satisfy this — omit the link when none exists.

**Specifications (>500 LOC).** Large PRs must carry a spec.
This may be in the linked issue, or part of the PR description.
If a design doc exists, may also link to it instead.
Specs cover: **Problem** (what is broken/missing, with file/line refs), **Approach**
(which modules change, what is added/removed), and **Key code** (10–30 line
snippets for non-obvious logic).

**Create it.** Unless the user says otherwise and permissions allow, push to a
branch on the main repository and open the PR from it (use a fork only when
direct push is unavailable or the user asks):

```bash
gh pr create --title "<title>" --body "<plain text body>" --label agent-generated
```

- Always add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.
- You must monitor PR status after creation if you have the ability (via a monitor skill or automation update and the `gh` CLI).
- Address CI failures, apply fixes, and repush. Continue monitoring.
- Respond to both human and agent comments: address obvious comments directly and resolve
  them. If you are unsure, prepare a report with your analysis of the remaining comments and
  your proposed action and present it to the local user.
- Continue monitoring until instructed otherwise, or the PR is merged.

## Rules

- `./infra/pre-commit.py` is the only lint entry point.
- Never amend a commit unless the user explicitly asks.
- If there are no changes to commit, say so and stop.

## See Also

- `.agents/skills/review-pr/` — multi-agent PR correctness review (separate concern).
- `.agents/skills/fix-issue/` — end-to-end issue-fix workflow.
- `AGENTS.md` — coding guidelines.
