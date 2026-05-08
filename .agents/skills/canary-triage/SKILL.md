---
name: canary-triage
description: Triage a failed canary ferry run. Gather diagnostics, identify root cause, file a GitHub issue, and write a Slack summary. Used by CI on scheduled canary failures.
---

# Skill: Canary Triage

Triage a failed canary ferry run. Diagnose root cause, file a GitHub issue,
write a Slack summary. Diagnosis and reporting only — no code changes, no PRs.

## Inputs (environment variables)

| Variable | Description |
|---|---|
| `CANARY_LANE` | `gpu` (CoreWeave) or `tpu` (GCP) |
| `CANARY_JOB_ID` | Iris job ID |
| `CANARY_RUN_ID` | W&B run ID |
| `IRIS_CONFIG` | Path to Iris cluster config |
| `IRIS_NAMESPACE` | Kubernetes namespace (CW only) |
| `WANDB_ENTITY` | W&B entity |
| `WANDB_PROJECT` | W&B project |
| `GHA_RUN_URL` | Full URL to the GitHub Actions run |

## Steps

### 1. Gather diagnostics

The cluster is still live. Collect signal now — it will be torn down after you.

- Iris job state via `.venv/bin/iris --config=$IRIS_CONFIG job list --json`
- **GPU lane:** you have kubectl at `~/.kube/coreweave-iris`, namespace `$IRIS_NAMESPACE` (defaults to `iris-ci` — the canary shares this namespace with PR CI).
  Get pod status, controller logs, task pod logs, warning events, pod describe.
  **Filter by `iris.job_id=<CANARY_JOB_ID with '/' replaced by '.'>`** so you only see this canary's pods, not co-tenant CI pods. Example: `kubectl -n iris-ci get pods -l iris.job_id=runner.iris-run-job-abc123`.
- **TPU lane:** use `iris process logs` and `iris job list`.
- Re-run `scripts/canary/validate_canary_metrics.py` if you need the validation output.

### 2. Identify root cause

Classify into one of: **infra/scheduling**, **training crash**, **metric regression**,
**controller bug**, **data/storage**.

Use hypothesis-driven diagnosis: state hypothesis, gather evidence, narrow.
Attempt to reproduce the issue locally and minimally.
Triple check that you're narrowing down on the same issue as the one that actually broke the canary.

### 3. File a GitHub issue

Follow the `file-issue` skill. Use the bug-report template.

- **Title:** `[canary-{lane}] {short failure description}`
- **Labels:** `bug`, `agent-generated`, `canary`
- **Body must include** a "Canary run context" section with: lane, job ID,
  GHA run URL, W&B run URL, date.
- Support your claims using supporting data (e.g. runtime logs)
- Keep the issue concise and maximally readable for humans.
- Use GFM to make the details (e.g. log traces, code to reproduce issue) optional and declutter the issue.
- Use `--body-file` with a temp file (see `file-issue` skill for the pattern).

### 4. Write `slack_message.md`

Write to the repo root. The workflow reads this file and sends it to Slack.
Always write this file, even if issue creation failed.

Format — keep to 4 lines max:

```
:red_circle: *{GPU|TPU} Canary failed* — {one-line summary}
*Root cause:* {category} — {1 sentence}
*Issue:* {github issue URL}
*GHA run:* {GHA_RUN_URL}
```

If root cause is unclear, say so: `root cause unclear` with your best-guess signals.

