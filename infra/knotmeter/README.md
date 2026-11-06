# Knotmeter

Knotmeter displays Marin's Snowball and Hero inference measurements using a pinned
[InferenceX-app](https://github.com/SemiAnalysisAI/InferenceX-app) Next.js server and
PostgreSQL. `settings.json` pins the source, Node, Bun and `https://knotmeter.oa.dev`.
The Docker bases are pinned by digest. The application remains GPLv3; `LICENSE.app`,
the upstream pin, `app.patch`, `build.mjs` and `layout.tsx` supply its corresponding source.
`/knotmeter-build.json` identifies those source bytes.

This candidate is for local review. Its Knotmeter cloud names are future configuration.
The earlier InferenceX-named resources and data have not been renamed or replaced.
Cloud Run, IAP, managed TLS and the GitHub cloud-identity path remain unverified.

## Upstream boundary

`app.patch` keeps the upstream schema, ETL, APIs, chart transforms and views.
These are the retained change groups and the consequence of removing each:

| Group | Purpose and consequence of removal |
| --- | --- |
| Snowball/Hero catalog and normalizers | Give the two checkpoints distinct database keys, selectors and AgentX comparison defaults. Without them, the saved bundles cannot appear as the intended models. Unknown architectures and OpenRouter listings stay absent. |
| Knotmeter titles, origin and landing route | Keep local navigation, API examples and exported charts under this deployment's name. The root opens the Snowball agentic BF16 chart; Hero is marked provisional. |
| Runtime and recipe in point metadata | Show the measured vLLM build and recipe beside each point. Removing these loses visible provenance; the API schema and cache version change with the metadata. |
| Exact GitHub attempt and required source lookup | Preserve the original run, attempt, date and Marin commit links. The unpatched importer can read a newer attempt or continue after a failed lookup. |
| Relative phase slicing | Keep the first profiling request when epoch-nanosecond rounding crosses the stage boundary. Removing it drops a real request from the charts. |
| Replay cache metrics | Preserve cache-hit values derived from an already linked trace. The unpatched aggregate upsert overwrites them on repeat import. |
| Collector admission wrapper | Check receipts, source attempts, recipes and topology outside upstream ETL. Removing it admits cancelled, partial or misattributed bundles. |
| Standalone Next.js output | Package the existing server and API for the container. Without it, the runtime image has no standalone server. |

`build.mjs` owns the private build in one step. It retains whole upstream inference,
comparison, history, point-detail, corpus and API routes. It omits public content and
build-time social images, installs the small `layout.tsx`, and disables search indexing.
The layout keeps the upstream query, theme and chunk-recovery providers. Builds need
no database credentials. Updating the pin requires checking those routes and imports.

The overlay leaves dependency manifests and the lockfile unchanged, so frozen installs
precede the overlay and remain cached across catalog edits. It does not edit upstream's
distributable skills, deployment docs, Cypress
configuration or database connection implementation. Postgres.js already accepts
`PGHOST`, `PGUSER`, `PGPASSWORD` and `PGDATABASE`. The private wrapper supplies the
hostless `postgresql:///` URL that lets those native settings take effect.
`INFERENCEX_SOURCE_REPOS`, `@semianalysisai/inferencex-*`, AgentX artifact names,
API headers and original source records retain their upstream protocol identities.

## Local build and data

```sh
docker build -t knotmeter-web infra/knotmeter
docker build --target ingest -t knotmeter-ingest infra/knotmeter
```

Use disposable PostgreSQL 17 with database `knotmeter` and native LOGIN roles
`knotmeter_reader` and `knotmeter_writer`, both
`NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION`. Use `scram-sha-256` for local
socket authentication; `trust` would accept a wrong password. As the administrator,
apply `roles.sql`. As the writer, apply `reader-defaults.sql` before migrations:

```sh
docker run --rm --entrypoint bun \
  --mount "type=bind,src=$SOCKET_DIRECTORY,dst=/run/knotmeter" \
  -e PGHOST=/run/knotmeter -e PGUSER=knotmeter_writer -e PGPASSWORD \
  -e PGDATABASE=knotmeter -e DATABASE_WRITE_URL=postgresql:/// \
  -e DATABASE_DRIVER=postgres -e DATABASE_SSL=false \
  knotmeter-ingest packages/db/src/migrate.ts --yes --no-ssl
```

Start the web image with the same socket mount, `PGUSER=knotmeter_reader`, its
`PGPASSWORD`, `PGDATABASE=knotmeter`, a unique `CACHE_NAMESPACE` and
`INVALIDATE_SECRET`. Publish container port 8080 on loopback. The entrypoint refuses
`DATABASE_WRITE_URL`. Keep `E2E_FIXTURES` unset when reviewing authentic data.

The importer accepts one artifact directory:

```sh
docker run --rm --network host \
  --mount "type=bind,src=$BUNDLE,dst=/artifacts,readonly" \
  --mount "type=bind,src=$SOCKET_DIRECTORY,dst=/run/knotmeter" \
  -e GITHUB_TOKEN -e PGHOST=/run/knotmeter -e PGUSER=knotmeter_writer \
  -e PGPASSWORD -e PGDATABASE=knotmeter -e DATABASE_DRIVER=postgres \
  -e DATABASE_SSL=false -e DASHBOARD_URL -e INVALIDATE_SECRET -e IAP_ID_TOKEN \
  knotmeter-ingest /artifacts
```

The root contains `agentx-source.json`, `bmk_agentic_*` aggregates, sibling
`agentic_*` timelines and optional `changelog-metadata/agentx.json`. The wrapper
checks the exact completed GitHub run/attempt/commit, successful Iris attempt,
recipe, full-hour/393-conversation validation and physical GPU count before invoking
upstream `ingest-ci-run.ts`. Hero requires its complete four-host, sixteen-GB200 gang.
Cancelled sources, partial windows and replaced hosts are excluded. A valid recovered
window can retain a submitting Action whose conclusion is failure or timed out.
The source identity is the original submitting Action, not the collector Action.

`DASHBOARD_URL` and `INVALIDATE_SECRET` request cache refresh after import. Behind
IAP, supply the approved `IAP_ID_TOKEN` in `Proxy-Authorization`; the invalidation
secret uses `Authorization`. Reimport is safe after an import or refresh failure.
The future service remains capped at one instance so refresh reaches its sole cache.

For corpus links, use the writer environment and socket with the upstream CLI:

```sh
docker run --rm --entrypoint bun \
  --mount "type=bind,src=$SOCKET_DIRECTORY,dst=/run/knotmeter" \
  -e PGHOST=/run/knotmeter -e PGUSER=knotmeter_writer -e PGPASSWORD \
  -e PGDATABASE=knotmeter -e DATABASE_WRITE_URL=postgresql:/// \
  -e DATABASE_DRIVER=postgres -e DATABASE_SSL=false \
  knotmeter-ingest packages/db/src/ingest-weka-dataset.ts \
  semianalysisai/cc-traces-weka-062126-256k --variant 256k --no-ssl
```

That CLI reads the Hub rows API. Check the Hub revision before and after import
against the artifact's `dataset_revision` and require 393 conversations. Refresh the
web cache afterward. This supplies corpus structure without adding measurements.

## Acceptance

Use retained authentic bundles from collectors
[40633833787](https://github.com/marin-community/marin/actions/runs/40633833787)
and [40723697736](https://github.com/marin-community/marin/actions/runs/40723697736).
Their source runs are `40625266678/1` and `40705471793/1`, both on October 28, 2026.
Snowball measures 40.14996 output tokens/s/GPU over eight H100s; the older provisional
Hero setup measures 2.23178 over sixteen GB200s. Keep their original runtime and recipe.
No optimized Hero measurement is included. The two points cannot demonstrate a
multi-date trend or multiple loads for one model.

Run the reusable check after import:

```sh
python infra/knotmeter/check.py --url http://127.0.0.1:18096 \
  --bundle "$SNOWBALL_BUNDLE" --bundle "$HERO_BUNDLE"
```

It compares build hashes, source runs/dates, runtime/recipe, physical GPUs, throughput,
latency, profiling counts and every raw request timestamp with the APIs and chart
projection. It uses `gh` for source-attempt metadata; `--source-runs FILE` can reuse
saved API responses keyed by run ID. Check request times within 1024 ns of JSON number
rounding. Warmup remains in raw timelines; profiling summaries exclude it.

Also inspect rendered charts, comparisons, history, point details, request timelines
and corpus links. Repeat both imports and compare persisted IDs, counts, metadata,
metrics and compressed trace hashes. Verify the reader can SELECT, cannot write or
create tables, and fails authentication with a wrong password. Keep fixture tests
separate. Local checks do not establish hosted IAP, TLS or cloud identity behavior.

## Later replacement and runner handoff

No deployment or image push is part of local review. The desired names are:

| Earlier target | Knotmeter target |
| --- | --- |
| `infra/inferencex`, project/stack `marin-inferencex` | `infra/knotmeter`, project/stack `marin-knotmeter` |
| SQL database `inferencex` | `knotmeter` |
| `inferencex_reader`, `inferencex_writer` | `knotmeter_reader`, `knotmeter_writer` |
| `cloudsql-inferencex-reader-password`, `cloudsql-inferencex-writer-password` | `cloudsql-knotmeter-reader-password`, `cloudsql-knotmeter-writer-password` |
| `marin-inferencex-invalidate-secret` | `marin-knotmeter-invalidate-secret` |
| Runtime account, service and registry repository `marin-inferencex` | `marin-knotmeter` |
| Logical component/domain/DNS names `inferencex*` | `knotmeter*` |
| `.github/workflows/inferencex-ingest.yaml` | `.github/workflows/knotmeter-ingest.yaml` |
| `INFERENCEX_INGEST_IMAGE`, `INFERENCEX_INGEST_ENABLED` | `KNOTMETER_INGEST_IMAGE`, `KNOTMETER_INGEST_ENABLED` |
| `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `DATABASE_SOCKET_DIR` | `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PGHOST` |

The shared SQL instance remains `hai-gcp-models:us-central1:marin-metadata`.
The domain remains `knotmeter.oa.dev`. Future image destinations are
`us-central1-docker.pkg.dev/hai-gcp-models/marin-knotmeter/marin-knotmeter@sha256:DIGEST`
and `us-central1-docker.pkg.dev/hai-gcp-models/marin-knotmeter/ingest@sha256:DIGEST`;
these are not claims of published images.

For a later authorized deployment:

1. Inventory current resources, state and stack outputs. Preserve the old database,
   secrets, images and authentic bundles until replacement acceptance. The shared
   SQL stack has had unrelated drift; never apply it wholesale from this checkout.
2. Target only the new `db-knotmeter` and two Knotmeter password-secret shells in
   `marin-cloudsql`. Create the restricted roles, apply both SQL grant files and run
   migrations as writer. Do not replace or reconfigure the shared instance.
3. Create the new application stack's runtime account, repository and invalidation
   secret shell. Populate secret values out of band. A second person must run
   `review-grant` on the focused IAM draft and apply its reviewed startup bindings
   in global stack `marin`. The intended Marina human allowlist is unchanged.
4. Build/push the committed candidate to the future image destinations. Deploy the
   service, then have the reviewer apply the service IAP/invoker bindings. Transfer
   ownership of the existing DNS CNAME to the renamed stack through a reviewed state
   move/import; preserve its value and avoid duplicate DNS resources. Create the
   domain mapping and verify managed TLS.
5. Reimport the two authentic saved bundles and corpus into `knotmeter`; repeat the
   local parity, replay and role checks. Verify authorized browser access, anonymous
   denial on vanity/direct URLs and APIs, and the intended GitHub WIF import/refresh.
   Record the actual serving revision and immutable image digest.
6. Give the runner owner the renamed paths/variables and native PG interface above.
   `GITHUB_TOKEN` still needs contents/actions read. The IAP audience and `github-iris`
   identity remain in `knotmeter-ingest.yaml`. The collector's AgentX artifact contract,
   `AGENTX_STATE_ROOT` and `AGENTX_WEEKLY_ENABLED` remain runner-owned. Leave automatic
   imports and scheduled submission disabled until that owner validates the whole path.

The separate [runner draft](https://github.com/yonromai/marin/blob/cab4b9112a4df1e52b73271ff5649e3411d19839/experiments/evaluation/agentx/README.md)
is not rewritten here. Do not relabel or copy an old cloud image as the renamed
candidate. Retire old resources only in a separately reviewed, authorized cleanup
with a restore reference. For later updates, record the previous serving revision
before changing traffic; application rollback does not roll back SQL migrations.
