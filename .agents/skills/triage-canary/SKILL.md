---
name: triage-canary
description: Triage a failed canary ferry run only when invoked by CI with the required CANARY context.
---

# Skill: Triage Canary

Triage a failed canary ferry run. Diagnose root cause and file a GitHub issue.
The workflow sends the immediate Slack failure notice. Diagnosis and reporting
only — no code changes, no PRs.

## CI invocation context

The Loom launch includes these non-secret values in its goal. Use those values
directly instead of inspecting the process environment.

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

The Actions runner may have exited. Start with the diagnostics artifact linked
from `GHA_RUN_URL` and durable Iris and Finelog logs. Use live cluster commands
only when the relevant cluster is still running.

- Iris job state via `iris --config=$IRIS_CONFIG job list` when accessible.
- **GPU lane:** you have kubectl at `~/.kube/coreweave-iris`, namespace `$IRIS_NAMESPACE` (defaults to `iris-ci` — the canary shares this namespace with PR CI).
  Get pod status, controller logs, task pod logs, warning events, pod describe.
  **Filter by `iris.job_id=<CANARY_JOB_ID with '/' replaced by '.'>`** so you only see this canary's pods, not co-tenant CI pods. Example: `kubectl -n iris-ci get pods -l iris.job_id=runner.iris-run-job-abc123`.
- **TPU lane:** use `iris process logs` and `iris job list`.
- Re-run `scripts/ci/validate_canary_metrics.py` if you need the validation output.

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

### 4. Publish the result

Append a typed `result` in the Loom channel with the issue URL, diagnosis, and
`GHA_RUN_URL`. If root cause is unclear, state that and the best available signal.
