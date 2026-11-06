# RL iteration progress

Maintained source for the private [RL progress dashboard](https://applets.marina.oa.dev/a/6c971887-1e22-45b1-9dcb-dd775829df57/).
The stable UUID is declared in `applet.toml`. Preserve it when releasing updates.

This source was recovered from the deployed report contract and retained frontend
assets after its original temporary checkout was lost. It preserves the existing
nine-record report, runtime-campaign filtering, comparison eligibility, and
shareable run details. The frontend leads with reward versus elapsed time;
unknown evaluation protocols remain unranked and incomplete coverage stays visible.
W&B links identify the project unless a record supplies an exact run URL.

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

Each record retains its run/attempt identity, label, status, UTC submission/start/
completion timestamps, timing scope, elapsed seconds, optimizer count, source/data/
initializer identities, evaluation identity, baseline/final reward and attempted/
scored counts, artifact URI, and evidence. `GET api/runs` adds coverage and
comparison fields. A runtime evaluation protocol hash is distinct from a declared
source protocol; absent runtime identity remains null. Names do not define a
comparison group. Use `wandb_url` for an exact W&B run link when recorded.

Snowball records carry their native execution status separately from an optional gate verdict.
They declare a minimum scored fraction (0.99); actual coverage remains visible, and comparison
eligibility uses that threshold. The gain uses the same run's update-0 panel evaluation.
Records without a verdict retain complete-scoring validation. Campaign and series identify the
preset revision and preset, with the exact W&B URL when recorded.

## Validate and release

From the repository root:

```bash
node --check infra/marina/apps/rl_progress/dist/app.js
uv run pytest infra/marina/apps/rl_progress/tests
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
