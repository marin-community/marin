---
name: lint-review
description: In CI, run the infra/lint catalog review over a PR.
allowed-tools: Bash(./infra/pre-commit.py:*), Bash(gh pr comment:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh api:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git merge-base:*), Bash(git rev-parse:*), Bash(git status:*), mcp__github_inline_comment__create_inline_comment
---

# Lint-catalog review on a PR

Run ./infra/pre-commit.py --review over the PR branch diff and report every
surviving finding. You are read-only except for posting comments: never edit,
stage, commit, push, or run a state-changing git/gh command.

The review's lanes and composer are authoritative. Copy each finding verbatim,
one comment per finding; do not drop, merge, reword, soften, re-judge, or invent
findings.

## Workflow

With --comment, first check for the marker
<!-- marin-lint-review --> in both issue comments and inline comments:

~~~bash
gh pr view <PR> --json comments
gh api repos/{owner}/{repo}/pulls/<PR>/comments --paginate
~~~

If present, stop to avoid a duplicate pass. Otherwise, from the repository root
run:

~~~bash
./infra/pre-commit.py --review
~~~

Raw lane output and combined findings are under
/tmp/marin-linter/<branch>/<timestamp>-<uniq>/ (the command prints the path).
Each finding line is:

~~~text
<path>:<line>: ml-<code> (<confidence>) <message>
~~~

Without --comment, print findings and stop.

Distinguish a clean run from a failed run. On exit 0 with no findings (including
the message Lint review: no findings.), print that status and post nothing. On a
non-zero run, all lanes failing, missing agent, or unresolved merge base, report
that the review could not run and why; post nothing.

With --comment, post one inline comment per finding using
mcp__github_inline_comment__create_inline_comment. Use the finding path and
line, with exactly:

~~~text
`ml-<code>` · confidence <confidence>

<verbatim message>

<!-- marin-lint-review -->
~~~

If an inline post fails because its line is outside the PR diff, continue with
the remaining findings. Then post one gh pr comment containing every such
unanchorable finding verbatim, each with the marker. Do not post a fallback when
all findings were anchored.

Format that fallback as:

~~~text
Lint review:

These infra/lint findings anchor on lines not in the PR diff, so they could not
be attached inline.

- <path>:<line>: ml-<code> (<confidence>) <message>
~~~

Use gh for GitHub; the review covers the branch diff against the merge base with
origin/main, including committed and uncommitted work.
