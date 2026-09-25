# Pull-request lint automation

Review the pull request named in the goal at its exact head commit. Read the
repository's `AGENTS.md` and `.agents/skills/lint-review/SKILL.md`. Check out
the head commit, fetch `origin/main`, and run the skill with `--comment` so every
catalog finding is reported. Do not edit files, commit, push, or open a PR.

After the review command succeeds and all findings are reported, fetch the PR's
current head SHA. Add the `agentic-lint` label only if it still equals the SHA
in the goal. If the head changed, record that the review is stale and stop; the
new `synchronize` event launches another review. If the review failed to run,
leave the label absent and report the failure. Append a concise typed `result`
to the session's Loom channel.
