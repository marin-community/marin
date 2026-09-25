# Pull-request lint automation

Review the pull request named in the goal at its exact head commit. Read the
repository's `AGENTS.md` and `.agents/skills/lint-review/SKILL.md`. Check out
the head commit, fetch `origin/main`, and run the skill with `--comment` so every
catalog finding is reported. Do not edit files, commit, push, or open a PR.

After the review command succeeds and all findings are reported, fetch the PR's
current head SHA. If it differs from the goal, record that the review is stale
and stop. Otherwise post one `🤖` PR comment containing the exact marker
`<!-- marin-agentic-lint:<head SHA> -->` and a concise completion statement.
Remove and then add the `agentic-lint` label to retrigger the policy check,
even if the label was already present from a prior review. The policy check
requires both this marker from the Loom GitHub App and the label. If the review
failed to run, post no marker and report the failure. Append a concise typed
`result` to the session's Loom channel.
