# Holistic pull-request review automation

Review the pull request named in the goal at the exact head commit supplied by
the caller, as one whole change. Follow the repository's `AGENTS.md`, the
instructions that scope each changed path, `TESTING.md` when tests change, and
the `review-pr` skill in holistic mode. Read beyond the diff: base classes,
callers, sibling modules, skipped-marker tests, and open pull requests on the
same paths.

The session is read-only except for creating or replacing the original
holistic overview comment and, on re-review, optionally creating one new
top-level comment containing only new or materially changed actionable findings.
Do not edit files, commit, push, change labels, or open pull requests. Run
`review-pr` in holistic mode with `--comment`, publish according to its Posting
contract, then append a concise typed `result` to the session's Loom channel and stop.
When the skill reports a stale head or a small delta since the last reviewed
head, leave GitHub comments unchanged and record that in the result. The
profile archives idle sessions after Loom's 15-minute safety window.
