# echo

The shared context store for Marin's agents: the `context` database on the
`marin-metadata` Cloud SQL instance (`hai-gcp-models:us-central1:marin-metadata`).

- **`chunks`** — the github+discord slice of the [marinmirror](https://marinmirror.exe.xyz)
  corpus: issues, PRs, comments, and Discord messages, ~73k rows, each with a canonical
  URL and a pgvector embedding (`vector(384)`, bge-small) for semantic search. The
  `echo-sync` Cloud Run job polls the marinmirror manifest every 10 minutes and mirrors
  new corpus builds. This is the interim design until marinmirror runs as a service in
  this project.
- **`work_log`** — the agents' shared logbook: one row per distilled milestone (result,
  decision, blocker, handoff), written by agents, read by anyone asking "what is the team
  doing". Append-only for agents.

Agents use these through the `context-search` and `work-log` skills
(`.agents/skills/`), which wrap `scripts/context_search.py` and `scripts/work_log.py`.

Tables are defined in `schema.py` and applied with `migrate.py`; the stack is deployed
with `pulumi up` (see `__main__.py` for the roles and secrets involved).

## Operations

```bash
# run a sync now instead of waiting for the schedule:
gcloud run jobs execute echo-sync --region=us-central1 --project=hai-gcp-models

# recent sync logs:
gcloud logging read 'resource.type="cloud_run_job" resource.labels.job_name="echo-sync"' \
  --project=hai-gcp-models --limit=20 --format='value(textPayload)'
```
