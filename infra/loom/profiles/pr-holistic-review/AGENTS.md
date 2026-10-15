# Holistic pull-request review automation

Review the pull request named in the goal at the exact head commit supplied by
the caller, as one whole change. Follow the repository's `AGENTS.md`, the
instructions that scope each changed path, `TESTING.md` when tests change, and
the `review-pr-holistic` skill. Read beyond the diff: base classes, callers,
sibling modules, skipped-marker tests, and open pull requests on the same
paths.

The session is read-only except for one GitHub comment. Do not edit files,
commit, push, change labels, or open pull requests. Run
`review-pr-holistic --comment`, post the packet according to its contract,
then append a concise typed `result` to the session's Loom channel and stop.
When the skill reports a stale head or a small delta since the last reviewed
head, post nothing and record that in the result. The profile archives idle
sessions after Loom's 15-minute safety window.
