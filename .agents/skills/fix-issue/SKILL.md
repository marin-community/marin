---
name: fix-issue
description: End-to-end workflow to fix a GitHub issue in marin-community/marin.
---

# Fix GitHub Issue

Fix the issue specified by the user. Read AGENTS.md first. If the task cannot
be completed, leave a GitHub comment with the last status and blocker.

Keep issue comments terse: state new facts, use annotated links, and omit
preambles, repeated issue text, and implementation narration. Research the
issue and code, then post one comment with # Research (root cause plus at most
five relevant links) and # Proposed Fix (smallest fix and call chain when
useful).

Before coding, check for duplicate work:

~~~bash
gh pr list --state open --search "<issue-id or keyword>"
gh issue list --state open --search "<keyword>"
~~~

If overlapping work exists, do not open a parallel implementation PR; comment
and hand off or limit the change to non-overlapping follow-up work.

## Implement and verify

Use branch agent/{YYYYMMDD}-fix-{issue-id}. Write a behavior-focused test before
the fix when a regression test is appropriate; follow root TESTING.md and the
nearest module testing guide for mocks, markers, and boundaries. Run the
default affected tests without overriding their marker expression, then:

~~~bash
./infra/pre-commit.py --all-files --fix
~~~

Resolve all reported issues before upload. Slow, integration, live-cluster,
Docker, and manual tests belong to their dedicated jobs unless explicitly
required.

## Publish and monitor

When checks pass, use .agents/skills/commit/SKILL.md exactly for commit, push,
PR text, and issue linking. Add a GitHub issue comment summarizing the fix.
After opening the PR, use that skill's wait_for.py loop through an exit
condition; do not start a separate polling loop. Investigate failures, push
fixes, and re-arm the wait as it specifies. Unit tests and lint must pass;
build-docs should pass when documentation changed.
