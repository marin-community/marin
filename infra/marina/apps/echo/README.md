# Echo

Marin's shared agent-context store, served by the Marina kernel under `/echo/`. The API is
mounted at `/echo/api/`; the tables live in the `echo` schema of Marina's database; the
`marina-echo-sync` Cloud Run job keeps the corpus current.

- `chunks` holds issues, pull requests, comments, and Discord messages mirrored from
  [marinmirror](https://marinmirror.exe.xyz). Each row has a canonical URL, a weighted
  PostgreSQL full-text document, and a pgvector embedding.
- `repository_file_chunks` holds the indexed `main` heads of `marin-community/marin`,
  `vllm`, `tpu-inference`, `evalchemy`, `harbor`, and `MarinSkyRL`. A result names both the
  repository and the exact indexed commit and links to that GitHub blob.
- `wiki_entries` holds durable agent-authored notes and incident records: a title, a
  one-sentence `use_when` hint, lowercase kebab-case tags, a body, embeddings, and
  deliberate-reference counters.
- `work_log` is an append-only agent logbook, one row per distilled milestone.
- `search_executions` and `search_execution_results` record every API search and the ranked
  snapshot it returned; `search_feedback` and `search_feedback_grades` record the 0–10
  judgments a caller ties back to those rows.

## CLI

The CLI calls the same IAP-gated service the browser does, so it needs no database access.
Run it from the repository:

```bash
uv run infra/marina/apps/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
uv run infra/marina/apps/echo/cli.py search "ragged_all_to_all" --domain file --domain pr
uv run infra/marina/apps/echo/cli.py search "compare cache implementations" --repository all
uv run infra/marina/apps/echo/cli.py get file:marin-community/marin@main:lib/iris/OPS.md
uv run infra/marina/apps/echo/cli.py grep ragged_all_to_all --source discord
uv run infra/marina/apps/echo/cli.py wiki search "grafana access" --tag ops
uv run infra/marina/apps/echo/cli.py wiki add --file note.md   # OKF markdown
uv run infra/marina/apps/echo/cli.py wiki show <id> > note.md  # edit, then `wiki edit`
uv run infra/marina/apps/echo/cli.py history export > echo-search-history.jsonl
```

Authentication reuses Marin's shared IAP login: run `iris login` once and the CLI mints
Marina tokens from the same cached credential (`~/.config/marin/credentials`). Agents and CI
need no login — ambient service-account credentials mint the token instead. `ECHO_MARINA_URL`
overrides the Marina origin; `ECHO_LOGIN_CLUSTER` selects which cached login to reuse.

`search` returns one ranked result set across five domains: `wiki`, `file`, `discord`, `pr`,
and `issue`. Wiki, files, pull requests, and issues are searched by default; Discord is
opt-in because high-volume conversation is noisier agent context. When files are selected
and `--repository` is omitted, the CLI infers the configured Marin-community repository from
the current Git checkout (a contributor fork with the same repository name maps to its
Marin-community target). Results come from reciprocal-rank fusion of a lexical and a vector
retrieval, then a cross-encoder rerank; a search returns at most 24 results. `grep` is a
case-insensitive literal substring scan over activity, newest first, and needs no model.

Grade what a search returned with `feedback`, passing the `--execution-id` the search
printed and repeating `--grade <result-key>=<0-10>`; a short overall explanation is read
from stdin. Search history is retained indefinitely for search-quality work and exports as
JSONL through `history export`.

## Sync job

`marina-echo-sync` runs `python -m echo.sync.main` from the Marina image every ten minutes.
It refreshes GitHub and Discord activity from marinmirror, then takes one globally
serialized repository turn: a Postgres advisory lock, a durable fair cursor, and one target
per execution, so in steady state the six repositories rotate about once per hour. A lock
loser finishes successfully without consuming a turn. Retries are disabled: an attempt is
one turn.

Each turn checks that repository's head. An unchanged head only advances the check time; a
new head uses GitHub's compare API to delete, fetch, and re-embed changed paths, and falls
back to one repository archive on a first build, a divergent history, or a comparison of at
least 300 files. Builds commit ten files at a time behind a durable watermark, so a partial
build is searchable and a later turn resumes it. The job reuses the `marinmirror-token`
GitHub PAT for authenticated REST calls.

For a cold target that needs longer than the scheduled attempt, run one task with a longer
timeout and a pinned target:

```bash
gcloud run jobs execute marina-echo-sync \
  --project=hai-gcp-models --region=us-central1 --tasks=1 --task-timeout=21600s \
  --update-env-vars=ECHO_REPOSITORY_TARGET=marin-community/vllm --wait
```

The file index accepts source, configuration, and prose files up to 256 KiB and rejects
binary content, lock files, generated or minified files, build and cache directories,
vendored trees, and secret-like names. Working-tree and branch changes are absent until they
reach `main`; use `rg` locally when the current checkout matters.

## Pages and endpoints

The dashboard is a Vue single-page app under `/echo/`: `/` searches, `/wiki` lists recently
updated notes, `/conversation` shows recent work-log entries, `/feedback` shows recent
grades, `/wiki/<id>` opens a note, and `/chunk/<id>` opens an activity chunk. Every data
route is under `/echo/api/`, and `/echo/api/docs` serves the OpenAPI documentation:

`search`, `federated-search`, `search-configuration`, `repository-index`,
`repository-files/{owner}/{repository}@main:{path}`, `grep`, `chunks/{id}`,
`search-results/{id}`, `search-executions`, `wiki/search`, `wiki/{id}`, `wiki`,
`wiki/{id}/references`, `feedback`, `work_log`, `work_log/{id}`, and `health`.

The kernel authenticates every request and binds the caller, so the API attributes wiki
writes, work-log entries, feedback, and search history to that identity; it reads no
authentication headers itself.

Both ONNX models load on the first request that needs them, not at startup: one process
serves every Marina app. The image bakes them in (`FASTEMBED_CACHE_PATH=/app/models`)
because Cloud Run keeps no disk between instances. `INFERENCE_THREADS` is four, which
assumes a search has most of the machine; concurrent searches share those threads.

## Development

```bash
cd infra/marina
uv run marina build --only echo                # compile apps/echo/web into apps/echo/dist
export MARINA_DATABASE_URL=postgresql+pg8000://postgres:...@127.0.0.1:5432/marina
uv run marina migrate --only echo              # create the echo schema
uv run marina dev                              # http://127.0.0.1:8080/echo/
uv run pytest apps/echo                        # unit tests (Docker for the throwaway Postgres)
uv run marina journey echo                     # a browser walk; screenshots in journeys-out/echo
```

The journey seeds its own rows and needs every app's schema present, so run `marina migrate`
(without `--only`) against the journey database first.

## Search quality

`benchmarks/search_queries.jsonl` holds graded relevance judgments for federated search, split
into a `dev` half to tune on and a held-out `test` half. `search_benchmark.py collect` captures
live results and `evaluate` scores them; `benchmarks/README.md` carries the judgment format and
the figures past ranking changes were measured against. Collect serially -- a burst of
concurrent queries scales Cloud Run out, and each new instance loads the ONNX models on its
first search, past the client's timeout.

```bash
uv run infra/marina/apps/echo/search_benchmark.py collect \
  infra/marina/apps/echo/benchmarks/search_queries.jsonl /tmp/echo-search-dev.jsonl \
  --split dev --workers 1
uv run infra/marina/apps/echo/search_benchmark.py evaluate \
  infra/marina/apps/echo/benchmarks/search_queries.jsonl /tmp/echo-search-dev.jsonl --split dev
```

Migrations are `migrations/mNNNN_*.py` modules applied in name order and recorded in
`schema_migrations`; `migrate(engine)` is what `marina migrate` calls. `m0001_init` is the
schema as it stood when Echo moved into Marina and must not be edited — add a new module.

Infrastructure and durable debugging records belong in Echo as wiki entries tagged
`incident`, `debugging`, `ops`, the subsystem, severity, and resolution. Link the URL
printed by `wiki add` or `wiki edit` from the associated PR or issue.
