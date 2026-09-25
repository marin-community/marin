# Pull-request lint automation

Review the pull request named in the goal at the head commit it names. Read the
repository's `AGENTS.md` and `.agents/skills/lint-review/SKILL.md`. Check out
that commit, fetch `origin/main`, and run the skill with `--comment` so every
catalog finding is reported on the PR. Do not edit files, commit, push, or open
a PR.

After the review command succeeds and its findings are reported, add the
`agentic-lint` label to the PR, even when there were no findings. The label
records that the PR has had a lint pass and stops later events from launching
another review. If the review failed to run, leave the label off and report the
failure. Append a concise typed `result` to the session's Loom channel.
