# Holistic pull-request review automation

Review the pull request named in the goal at the exact head commit supplied by
the caller, as one whole change. Follow the repository's `AGENTS.md`, the
instructions that scope each changed path, and `TESTING.md` when tests change.
When the goal supplies a workflow revision, load the review policy with
`git show <workflow-sha>:.agents/skills/review-pr/SKILL.md` and read its
holistic reference and linked review instructions from that same commit.
Fetch the commit from origin if needed. Skip the Skill tool for this
invocation because it resolves the copy in the PR worktree.

The session is read-only except for creating or replacing the original
holistic overview comment and, on re-review, optionally creating one new
top-level comment containing only new or materially changed actionable findings.
Do not edit files, commit, push, change labels, or open pull requests. Run
`review-pr` in holistic mode with `--comment`, publish according to its Posting
contract, then append a concise typed `result` to the session's Loom channel and stop.
Record stale or already-reviewed heads in the result without posting.
