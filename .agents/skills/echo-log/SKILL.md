---
name: echo-log
description: Read and append Echo's shared agent work log. Use at the start of non-trivial shared work to see what other agents are doing, and at a milestone (result, decision, blocker, or handoff) to record one distilled entry.
---

# Echo work log

The work log is Echo's append-only team logbook. Each entry records one
milestone of an agent's work: a result, a decision, a blocker, or a handoff.
The Echo dashboard shows recent entries at `/echo/conversation`.

## Read

At the start of non-trivial shared work, check whether another agent already
works on it:

```bash
uv run infra/marina/apps/echo/cli.py work-log recent --days 7
uv run infra/marina/apps/echo/cli.py work-log recent --project <slug>
uv run infra/marina/apps/echo/cli.py work-log show <id>
```

`recent` prints one summary line per entry, newest first. `show` prints one
entry with its body.

## Write

Add one entry at each milestone:

```bash
uv run infra/marina/apps/echo/cli.py work-log add --project <slug> --title "<result or decision>" --body - <<'EOF'
<two or three sentences with links to the PR, issue, run, or report>
EOF
```

- Use one stable lowercase kebab-case `--project` slug for each thread of work,
  across sessions and agents. Run `recent` to find an existing slug before you
  make a new one.
- Write the title as the result or decision. Keep the body short and link the
  evidence. Do not paste transcripts, raw logs, or secrets.
- Echo records the signed-in caller as the author.

## Access

The CLI uses the shared Marin IAP login (`iris login`) or ambient
service-account credentials. See
[Echo setup and access](../../../infra/marina/apps/echo/README.md).

If Echo refuses a write because the caller has no credentials or no IAP access,
`add` prints one warning and exits 0. In a Claude Code or Codex session, later
`add` calls in that session do nothing. Do not retry the write, change
credentials, or debug access for the work log. Continue the task.
