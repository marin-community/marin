# RL iteration progress

Maintained source for the private [RL progress dashboard](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/).
The stable UUID is declared in `applet.toml`. Preserve it when releasing updates.

Each run has one row with its attempts in a dropdown, latest first. Failed attempts
are marked. The selected attempt supplies the run's details.
The row shows the gate report's objectives (control → run, change and noise),
guards and invariants, then verdict and effect. Replicate counts identify aggregate
results. Attempts without report details remain descriptive; unknown measurements are explicit.
A run-only link (`#run=<run_id>`) selects its latest attempt. Add
`&attempt=<attempt_id>` to select a particular attempt. W&B links use the recorded
run URL; the Grafana link uses the recorded native execution window. The Model column can filter by
the recorded model. The issue number links to GitHub, and the opt-out is shown in the attempt details.
Linked EvalDash evaluations open the recorded run IDs. The detail view includes per-step family, AAII-category and difficulty means and scored/attempted answer counts.
The attempt details link to a grader report when one was published. It shows
training and evaluation success rates and verifier coverage by task and subset,
with failure counts and grading latency by subset.

Malformed records are skipped individually and named in a visible note.

## Data contract

The app's release migration creates its `rl_runs` table in the application schema.
Each native `(run_id, attempt_id)` identifies one updatable JSON record. Repeat
publication refreshes that attempt's record; null model, issue, EvalDash links, gate and gate-result values
preserve their stored values. Other attempts remain separate.

Native failures and failed training or evaluation invariants set `status: failed` and name `failed_facts`.
Training grader errors are reported in rollout metrics; they do not fail an ungated run's status.
Records carry native identity, source and engine facts, and the launcher's issue.
The planned update count and measured update indices come from the run's effective configuration;
the detail view shows them beside the observed optimizer count. Missing stored values are unknown.
The gate result contains the report's objectives, guards, invariants, verdict and effect; controls and
train-long have no gate verdict. Update counts use stored metrics and checkpoint parents.
Lineage duration sums those native jobs' stored windows. Missing lineage records leave both
measurements unknown; a missing window leaves elapsed time unknown with a timing note.

`POST /api/runs` validates each submitted record with the shared
`finestore.rl.rl_progress.AttemptSummary`, upserts its native key and returns the stored
record. `GET /api/runs` returns stored records and lists malformed historical rows
in `skipped_records`. See the
[Snowball publication rules](../../../../docs/recipes/rl-experiments.md#dashboard-publication)
for record semantics, retries and publishing pinned controls.

## Validate and release

From the repository root:

```bash
node --check infra/marina/apps/rl_progress/dist/app.js
uv run pytest -m integration infra/marina/tests/test_rl_progress.py
uv run marina validate infra/marina/apps/rl_progress
uv run marina release infra/marina/apps/rl_progress --local --json
```

Check cold-loaded run and attempt hash links, refresh after repeat publication,
and the detail dialog's scrolling body, close control, Escape,
and backdrop dismissal. Stop the local stack with Ctrl-C.

The `Marin - Release RL Progress` workflow runs manually from main, with inspection as its default. The `rl-progress` CI job runs the packaged API and `marina journey rl_progress` browser owner.

Deploy a Marina runtime containing `finestore.rl.rl_progress` before releasing the app package.

For production, follow [Marina's release procedure](../../README.md). Inspect the
current private revision, then release with its exact `--base-version`. Merging
source does not publish it. Verify the deployed assets, selected revision and all
stored records after release. Do not change access mode or rewrite historical
results as part of a frontend update.
