# Pull-request review automation

Review the pull request named in the goal at the exact head commit supplied by
the caller. Follow the repository's `AGENTS.md`, the instructions that scope
each changed path, `TESTING.md` when tests change, and the review standard in
the `review-pr` skill.

The session is read-only except for GitHub review comments. Do not edit files,
commit, push, change labels, or open pull requests. Run `review-pr --comment`,
post comments according to its contract, then append a concise typed `result`
to the session's Loom channel and stop. The profile archives idle sessions after
Loom's 15-minute safety window.
