# echo

The shared context store for Marin's agents: the `context` database on the
`marin-metadata` Cloud SQL instance, plus the `echo-api` service and `echo-sync` job in
front of it.

- **`chunks`** — the github+discord slice of the [marinmirror](https://marinmirror.exe.xyz)
  corpus (issues, PRs, comments, Discord messages), each with a canonical URL and a
  pgvector embedding for semantic search, mirrored by the `echo-sync` job.
- **`work_log`** — an append-only logbook agents write to, one row per distilled milestone.

Two ways in: **`echo-api`**, an IAP-gated Cloud Run service exposing an OpenAPI HTTP
interface (`/search`, `/grep`, `/chunks/{id}`, `/work_log`), and **direct SQL** via the
`context-search` / `work-log` skills. Access is Cloud SQL IAM, not passwords: the
`echo@openathena.ai` group and the service accounts are IAM database users.
