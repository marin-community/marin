---
name: dispatch-monitor
description: Cron-driven monitoring dispatcher for long-running research threads. Use when setting up or managing event-driven monitoring collections instead of in-turn babysit loops.
---

# Skill: Dispatch Monitor

## When to Use

Use this skill instead of `babysit-job` when:

- The monitoring needs to survive across sessions (cron-driven)
- Multiple runs need coordinated monitoring under one research thread
- You want automated logbook updates and issue comments

Use `babysit-job` when you need interactive, in-session monitoring.

## Quick Reference

```bash
# Register a collection
uv run scripts/dispatch.py register \
  --name <NAME> --prompt "<INSTRUCTIONS>" \
  --logbook .agents/logbooks/<name>.md \
  --branch research/<name> --issue <NUM>

# Add runs
uv run scripts/dispatch.py add-run <NAME> \
  --track ray --job-id <ID> --cluster <CLUSTER> --experiment <SCRIPT>

# Manual tick
uv run scripts/dispatch.py tick --collection <NAME> --event-kind manual

# Check status
uv run scripts/dispatch.py show <NAME>
```

## Workflow

1. **Register** a collection with operator prompt, logbook path, branch, and issue
2. **Add runs** (Ray or Iris) to the collection
3. **Set up cron** or run manual ticks
4. **Monitor** via `list` and `show` commands
5. **Pause/resume** collections as needed

## See Also

- [Dispatch tutorial](../../../docs/tutorials/dispatch.md)
- [babysit-job](../babysit-job/SKILL.md) — interactive in-session monitoring
- [agent-research](../agent-research/SKILL.md) — research thread management
