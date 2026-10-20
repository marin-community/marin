# Snowball conversation and curriculum applets

These maintained sources update the existing Snowball applets. Each applet keeps
its UUID, database schema, and access mode. Initial source was recovered from the
selected Postgres packages: conversation lab revision 28, public comparison
revision 1, and curriculum revision 2.

| Application | Source directory under `infra/marina/applets/` | UUID |
| --- | --- | --- |
| Conversation lab | `snowball_conversation_lab` | `25f504c6-c6fb-4e14-8401-07dd8e6fbfba` |
| Public 50M-token comparison | `snowball_token_comparison` | `53af6539-ce9d-48c0-8b98-358a82c8b203` |
| Training curriculum | `snowball_curriculum` | `e2bd5c9d-0f93-4ce9-b7af-3c78ca55e5d1` |

## Summaries and browser refresh

Conversation summaries are cached for eight seconds per Marina process and
application revision. The key includes the run and mode (`chat`, `agentic`, or
both). At most 128 summaries are retained. Simultaneous requests share a refresh;
authorization is checked before every cache lookup. Worker assignment and session
revocation therefore take effect on cached reads.

Both conversation frontends use the source and build dependencies under
`snowball_conversation_lab/web/`. Their manifests record that directory as a
shared build input. Each package retains its own backend and access policy:
conversation lab requires login, while the comparison permits public reads.
Browsers poll every four seconds while visible, skip polling while hidden, and
refresh when shown again. An already running refresh can finish after the page
becomes hidden.

## Curriculum search index

The curriculum migration adds a GIN trigram index on `lower(search_text)`, matching
the case-insensitive substring filter used by the count and result queries. The
server must already provide `pg_trgm`; application migrations do not install
extensions. The initial index build runs in the release transaction and blocks
writes to `pool_tasks` until it completes. Release outside a curriculum load.
Broad or shorter-than-three-character searches can still require scans.

## Build and release

Run these commands from the repository root, replacing `APP` with a directory
from the table:

```bash
uv run marina release infra/marina/applets/APP --dry-run --json
uv run marina release-status infra/marina/applets/APP --json
uv run marina applets versions UUID --json
uv run marina release infra/marina/applets/APP --base-version CURRENT_VERSION --json
```

Use the inspected current version for each release. Release from clean current
main after merging. Updating Marina's runtime or merging this source does not
release the applets. Releases preserve the selected private/public mode and do
not reload the stored conversations or curriculum data. A rollback selects an
older package and leaves the compatible search index in place.

See [Building Marina apps](marina.md) for local Postgres previews, validation
scope, and the release lifecycle.
