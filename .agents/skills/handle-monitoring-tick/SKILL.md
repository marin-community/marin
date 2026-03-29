---
name: handle-monitoring-tick
description: Default instructions for a monitoring agent dispatched by the dispatcher to check on a research run. Covers status checks, diagnosis, corrective action, and reporting.
---

# Skill: Handle Monitoring Tick

You have been dispatched by the Marin monitoring system to check on a research run.

## What to do

1. **Check status** — query the run's current state via Ray dashboard, Iris CLI, or logs.
2. **Check metrics** — if W&B is configured, look at loss curves and training throughput.
3. **Diagnose failures** — read recent logs for errors (OOM, NaN loss, node failures, preemptions).
4. **Take corrective action** if possible — resubmit a failed job, reduce batch size, swap out a bad node.
5. **Write a logbook entry** — summarize what you found and any actions taken.
6. **Post an issue comment** — only on meaningful events (status change, failure, milestone reached).
7. **Escalate** — if the problem needs human intervention, mark it for escalation.

## What NOT to do

- Do not restart or bounce Ray/Iris clusters.
- Do not move large amounts of data across GCS regions.
- Do not make speculative changes to training configs without evidence of a problem.
- Do not post noisy issue comments when nothing has changed.

## Output Format

Your output is parsed by the dispatcher. Use these markers exactly:

- Wrap your logbook entry between `<<<LOGBOOK_ENTRY>>>` and `<<<END_LOGBOOK_ENTRY>>>`.
- If you have a meaningful update for the GitHub issue, wrap it between `<<<ISSUE_COMMENT>>>` and `<<<END_ISSUE_COMMENT>>>`.
- If the situation requires human escalation, include `<<<ESCALATE>>>` on its own line.
