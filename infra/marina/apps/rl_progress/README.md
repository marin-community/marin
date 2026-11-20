# RL iteration progress

Maintained source for the private [RL progress dashboard](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/).
The stable UUID is declared in `applet.toml`. Preserve it when releasing updates.

The overview leads with reward versus elapsed time. Unknown evaluation protocols
remain unranked and incomplete coverage stays visible. W&B links identify the
project unless a record supplies an exact run URL.

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

The app creates this table on first release. A disposable local preview can load
synthetic records through the Marina data commands against the printed local origin.

Each record retains its run/attempt identity, label, status, UTC submission/start/
completion timestamps, timing scope, elapsed seconds, optimizer count, source/data/
initializer identities, evaluation identity, baseline/final reward and attempted/
scored counts, artifact URI, and evidence. `GET api/runs` adds coverage and
comparison fields. A runtime evaluation protocol hash is distinct from a declared
source protocol; absent runtime identity remains null. Names do not define a
comparison group. Use `wandb_url` for an exact W&B run link when recorded.

The overview groups runs by the recorded preset. New Quick RL records store
the preset, model ID, eval selection identity, and requested GPU type and count.
The picker also shows the historical Snowball presets. Historical rows with
a recognized series use it as their preset. GPU type and count narrow the
view and distinguish frontiers within it.

## Validate and release

From the repository root:

```bash
node --check infra/marina/apps/rl_progress/dist/app.js
uv run pytest infra/marina/apps/rl_progress/tests
uv run marina validate infra/marina/apps/rl_progress
uv run marina release infra/marina/apps/rl_progress --local --json
```

Check preset and hardware filtering, graph point links, cold-loaded run hash links,
coverage exclusions, and the detail dialog's scrolling body, close control, Escape,
and backdrop dismissal. Stop the local stack with Ctrl-C.

For production, follow [Marina's release procedure](../../README.md). Inspect the
current private revision, then release with its exact `--base-version`. Merging
source does not publish it. Verify the deployed assets, selected revision and all
stored records after release. Do not change access mode or rewrite historical
results as part of a frontend update.

## Import completed Quick RL attempts

The Quick RL coordinator registers each native attempt in the regional CoreWeave
dashboard inbox before starting SkyRL. The scheduled GitHub Actions workflow
`marin-quick-rl-dashboard.yaml` calls the private app's `POST api/sync` every
ten minutes. It uses the existing main-branch Marina deployment identity.
Marina reads the small registrations and reports with its CoreWeave credentials
and looks up coordinator timestamps through its existing VPC connection to Iris.
It imports terminal attempts by their native keys. A repeated import leaves
the existing snapshot intact. A failed import remains eligible on the next tick.

The dashboard prefers the trainer's `elapsed_excluding_eval_seconds` from the
attempt report in CoreWeave. It falls back to coordinator wall time minus both
validation durations, then to unadjusted coordinator wall time. The timing
definition is part of the frontier comparison key. Runs with only unadjusted
wall time or mismatched evaluation identities stay unranked. Historical rows
retain their original timing and evaluation groups.

Run names and chart points open a separate grader page for the selected native
attempt. The page uses the imported grader report, including per-environment
success rates, verifier coverage, failure counts, and latency for training and
evaluation. Failed attempts show any report they collected before termination.
Historical attempts without a report show an empty state.
