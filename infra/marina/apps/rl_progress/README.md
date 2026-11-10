# RL iteration progress

Maintained source for the private [RL progress dashboard](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/).
The stable UUID is declared in `applet.toml`. Preserve it when releasing updates.

Each run has one row with its attempts in a dropdown, latest first. Failed attempts
are marked. The selected attempt supplies the run's details.
The row shows the gate report's climbed metrics (control → run, change and noise),
constraints and invariants, then verdict and label. Replicate counts identify aggregate
results. Attempts without report details remain descriptive; unknown measurements are explicit.
A run-only link (`#run=<run_id>`) selects its latest attempt. Add
`&attempt=<attempt_id>` to select a particular attempt. W&B links use the recorded
run URL; an `issue` field links to the experiment's GitHub issue.

Malformed records are skipped individually and named in a visible note.

## Data contract

The app's release migration creates its `rl_runs` table in the application schema.
Each native `(run_id, attempt_id)` identifies one updatable JSON record. Repeat
publication refreshes that attempt's record; a missing issue preserves its stored
issue link. Other attempts remain separate.

The shared `marin.rl.rl_progress.RunSummary` validates each record. `GET /api/runs` returns stored records and lists malformed rows in `skipped_records`. See the
[Snowball publication rules](../../../../docs/recipes/snowball-rl.md#dashboard-publication)
for record semantics, retries and publishing pinned controls.

## Validate and release

From the repository root:

```bash
node --check infra/marina/apps/rl_progress/dist/app.js
uv run pytest infra/marina/tests/test_rl_progress.py
uv run marina validate infra/marina/apps/rl_progress
uv run marina release infra/marina/apps/rl_progress --local --json
```

Check cold-loaded run and attempt hash links, refresh after repeat publication,
and the detail dialog's scrolling body, close control, Escape,
and backdrop dismissal. Stop the local stack with Ctrl-C.

Deploy a Marina runtime containing `marin.rl.rl_progress` before releasing the app package.

For production, follow [Marina's release procedure](../../README.md). Inspect the
current private revision, then release with its exact `--base-version`. Merging
source does not publish it. Verify the deployed assets, selected revision and all
stored records after release. Do not change access mode or rewrite historical
results as part of a frontend update.
