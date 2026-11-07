# RL iteration progress

Maintained source for the private [RL progress dashboard](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/).
The stable UUID is declared in `applet.toml`. Preserve it when releasing updates.

Each run has one row with its attempts in a dropdown, latest first. Failed attempts
are marked. The selected attempt supplies the run's graph points and details.
Campaign and series come from record fields; names do not define a comparison group.
A run-only link (`#run=<run_id>`) selects its latest attempt. Add
`&attempt=<attempt_id>` to select a particular attempt. W&B links use the recorded
run URL; an `issue` field links to the experiment's GitHub issue.

The frontend leads with reward versus elapsed time. The backend applies each record's
coverage threshold and evaluation identity to comparison eligibility. Incomplete
coverage remains visible, including attempts that meet their recorded threshold.
Malformed records are skipped individually and named in a visible note.

## Data contract

The backend reads the application-owned `rl_runs` table:

```sql
CREATE TABLE rl_runs (
    run_id text NOT NULL,
    attempt_id text NOT NULL,
    record jsonb NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, attempt_id)
);
```

Existing production data needs no migration or reload. A disposable local preview
needs this table and synthetic records loaded through the maintained Marina data
commands against the printed local origin. Never load a partial snapshot over the
production table.

The shared `marin.rl.rl_progress.RunSummary` validates each record. `GET api/runs` adds coverage and
comparison fields and lists malformed rows in `skipped_records`. See the
[Snowball publication rules](../../../../../docs/recipes/snowball-rl.md#dashboard-publication)
for record semantics, retries and publishing pinned controls.

## Validate and release

From the repository root:

```bash
node --check infra/marina/apps/rl_progress/dist/app.js
uv run pytest infra/marina/tests/test_rl_progress.py
uv run marina validate infra/marina/apps/rl_progress
uv run marina release infra/marina/apps/rl_progress --local --json
```

Check campaign filtering, each graph's point links, cold-loaded run hash links,
coverage exclusions, and the detail dialog's scrolling body, close control, Escape,
and backdrop dismissal. Stop the local stack with Ctrl-C.

For production, follow [Marina's release procedure](../../README.md). Inspect the
current private revision, then release with its exact `--base-version`. Merging
source does not publish it. Verify the deployed assets, selected revision and all
stored records after release. Do not change access mode or rewrite historical
results as part of a frontend update.
