# Monitoring Dispatcher

The monitoring dispatcher manages **monitoring collections** — bindings of
operator prompts, logbooks, branches, issues, and run pointers. It processes
them one tick at a time, launching agent sessions (Claude Code or Codex) to
diagnose failures, advance research threads, and post updates.

## Quick Start

```bash
# Register a collection (no runs yet)
uv run scripts/dispatch.py register \
  --name isoflop-sweep \
  --prompt "Monitor the isoflop sweep. If OOM, reduce batch size." \
  --logbook .agents/logbooks/isoflop-sweep.md \
  --branch research/isoflop-sweep \
  --issue 3560

# Add a Ray run
uv run scripts/dispatch.py add-run isoflop-sweep \
  --track ray \
  --job-id raysubmit_abc123 \
  --cluster us-central2 \
  --experiment experiments/isoflop_sweep.py

# Run a manual tick
uv run scripts/dispatch.py tick --collection isoflop-sweep --event-kind manual

# List all collections
uv run scripts/dispatch.py list

# Show full details
uv run scripts/dispatch.py show isoflop-sweep
```

## Cron Setup

Add a crontab entry to poll every 10 minutes:

```cron
*/10 * * * * cd /path/to/marin && uv run scripts/dispatch.py tick --event-kind scheduled_poll >> /var/log/dispatch.log 2>&1
```

This processes all non-paused collections. To target a specific collection:

```cron
*/10 * * * * cd /path/to/marin && uv run scripts/dispatch.py tick --collection isoflop-sweep >> /var/log/dispatch.log 2>&1
```

## Collection Schema

Collections are stored as YAML in `.agents/collections/<name>.yaml`:

```yaml
name: isoflop-sweep
prompt: |
  Monitor the isoflop sweep training run.
  If it fails with OOM, reduce batch size.
logbook: .agents/logbooks/isoflop-sweep.md
branch: research/isoflop-sweep
issue: 3560
created_at: "2026-03-22T10:00:00Z"
paused: false
runs:
  - track: ray
    ray:
      job_id: raysubmit_abc123
      cluster: us-central2
      experiment: experiments/isoflop_sweep.py
```

Runtime state is tracked in `<name>.state.yaml` (not version-controlled).

## Agent Selection

Use `--agent` to choose the agent backend:

```bash
uv run scripts/dispatch.py tick --agent claude-code   # default
uv run scripts/dispatch.py tick --agent codex
```

## Managing Collections

```bash
# Pause a collection (skip during scheduled polls)
uv run scripts/dispatch.py update isoflop-sweep --paused true

# Resume
uv run scripts/dispatch.py update isoflop-sweep --paused false

# Update the prompt
uv run scripts/dispatch.py update isoflop-sweep --prompt "New instructions..."

# Remove a run by index
uv run scripts/dispatch.py remove-run isoflop-sweep --index 0

# Delete a collection entirely
uv run scripts/dispatch.py delete isoflop-sweep
```

## How Dispatch Works

Each `tick` invocation:

1. Loads the collection config and run state
2. Queries current status of each run (Ray or Iris)
3. Decides whether to dispatch an agent (status change, periodic health check, or manual trigger)
4. Creates a git worktree on the collection's branch
5. Launches the agent with the collection prompt and run context
6. Appends the agent's logbook entry and commits/pushes
7. Posts issue comments on meaningful progress
8. On 3+ consecutive failures, posts an escalation comment and stops dispatching

## Troubleshooting

**Agent produces no output:** Check that `claude` or `codex` CLI is installed and authenticated.

**Push failures:** The dispatcher retries with `git pull --rebase` up to 3 times. If rebase conflicts occur on non-logbook files, the push is aborted and logged as a failure.

**Escalation posted:** After 3 consecutive agent failures on a run, the dispatcher posts an escalation comment and stops dispatching for that run. Fix the issue manually, then reset by updating the state file or re-registering.
