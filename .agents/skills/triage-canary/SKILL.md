---
name: triage-canary
description: Triage a failed canary ferry run (CI-invoked).
---

# Triage Canary

Diagnose a failed canary, file one GitHub issue, and write the Slack summary.
This workflow is diagnosis/reporting only: no code changes and no PRs.

## Inputs

Required environment variables:

| Variable | Meaning |
|---|---|
| CANARY_LANE | gpu (CoreWeave) or tpu (GCP) |
| CANARY_JOB_ID | Iris job ID |
| CANARY_RUN_ID | W&B run ID |
| IRIS_CONFIG | Iris cluster config path |
| IRIS_NAMESPACE | CoreWeave namespace, default iris-ci |
| WANDB_ENTITY | W&B entity |
| WANDB_PROJECT | W&B project |
| GHA_RUN_URL | GitHub Actions run URL |

## Diagnose

Collect diagnostics while the cluster is live:

- Iris state: .venv/bin/iris --config=$IRIS_CONFIG job list.
- GPU: use kubectl with ~/.kube/coreweave-iris and $IRIS_NAMESPACE for pod
  status, controller/task logs, warning events, and describe. Filter by
  'iris.job_id=<CANARY_JOB_ID with / replaced by .>' so co-tenant PR-CI
  pods are excluded; for example:
  kubectl -n iris-ci get pods -l iris.job_id=runner.iris-run-job-abc123.
- TPU: use iris process logs and iris job list.
- Re-run scripts/ci/validate_canary_metrics.py if its output is needed.

State hypotheses, gather evidence, and narrow to one category:
infra/scheduling, training crash, metric regression, controller bug, or
data/storage. Reproduce minimally when useful and verify the diagnosis matches
the failure that stopped the canary.

## Report

Follow file-issue. Use title [canary-{lane}] {short failure description} and
labels bug, agent-generated, and canary. Include a Canary run context section
with lane, job ID, GHA URL, W&B URL, and date, supported by runtime evidence.
Use a unique temporary --body-file.

Always write repo-root slack_message.md, even if issue creation fails; the
workflow sends this file. Keep it to four lines:

~~~text
:red_circle: *{GPU|TPU} Canary failed* — {one-line summary}
*Root cause:* {category} — {1 sentence}
*Issue:* {github issue URL}
*GHA run:* {GHA_RUN_URL}
~~~

If uncertain, write root cause unclear with the strongest signals.
