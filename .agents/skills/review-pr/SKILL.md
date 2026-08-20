---
name: review-pr
description: Multi-agent correctness review of a pull request.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(uv run infra/codehealth/log_stats.py:*), mcp__github_inline_comment__create_inline_comment
---

Provide a high-signal code review for the given pull request.

## Review

First use a haiku agent to stop if the PR is closed, draft, clearly does not
need review, or already has a Claude comment (check gh pr view <PR> --comments)
unless a maintainer explicitly requested re-review. Claude-generated PRs still
require review.

Use a haiku agent to return paths (not contents) for root and applicable
CLAUDE.md/AGENTS.md files. Use an opus agent to summarize the PR and check its
title/body against writing-style/pull-requests.md: title <=72 characters,
imperative, no conventional prefix; body should lead with behavior and
motivation and omit diff narration, testing sections, boilerplate headings,
checklists, emoji/attribution/session URLs, inventories, and advocacy language.
Do not flag a terse plain body merely for lacking Markdown.

Launch four parallel agents: two opus agents audit applicable CLAUDE.md/AGENTS.md
rules (and root TESTING.md plus module testing docs for changed tests), and two
opus agents scan the diff for definite bugs or security issues. Give each the PR
title and description. In experiments/grug, do not flag intentional duplication
when behavior/contracts are correct.

Flag only high-confidence findings: code that cannot compile/parse, definite
wrong results, or a directly quotable scoped instruction violation. Do not flag
style, coverage, speculative input/state issues, or subjective improvements.
Validate each candidate from the bug/compliance agents with a parallel opus
agent; keep only validated findings.

## Stats and output

Before any early stop, best-effort emit one stats event from the repo root. Do
not retry or surface logging failures:

~~~bash
cat <<'EOF' | uv run infra/codehealth/log_stats.py
{
  "tool": "review-pr",
  "invocation": {
    "trigger": "local",
    "agent_cli": "claude",
    "pr_number": <PR>,
    "agent_exit_code": 0,
    "timed_out": false
  },
  "findings": [
    ["<file>", <line>, "<category>", 1.0, "<first 200 chars of issue description>"]
  ]
}
EOF
~~~

Category is bug or claude-md-adherence; use an empty findings array when clean.

Print each validated issue briefly. If none, print exactly:
No issues found. Checked for bugs and CLAUDE.md compliance.
Report PR-description problems separately. Without --comment, stop without
posting.

With --comment, post one top-level gh pr comment for description problems
(prefixed with 🤖), whether or not code issues exist. If there are no code issues,
post the same no-issues summary and stop. Otherwise post one inline comment per
validated issue with confirmed: true. Include a brief issue description and a
committable suggestion only when it completely fixes a small self-contained
issue; never duplicate a unique issue.

Do not flag pre-existing issues, correct behavior, linter-detectable issues,
general quality concerns, speculative security concerns, or explicitly silenced
rules. Use gh rather than web fetch. For changed tests, flag only concrete
testing-policy violations.
Cite each issue in its inline comment, including a scoped AGENTS rule when
applicable. Inline code links require a full SHA, repository name, #, and at
least one context line before and after the commented line.
