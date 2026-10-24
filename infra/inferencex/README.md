# Snowball performance dashboard

This packages the pinned InferenceX-app as a standalone Next.js service at
`https://performance.oa.dev`. Cloud Run's native IAP gate covers the service and its APIs.
The global IAM stack reuses Grafana's viewer grants and adds the existing GitHub Iris
identity for cache refresh. No anonymous invoker is granted. The web process receives
only the `inferencex_reader` password. The import Action receives `inferencex_writer`.

`settings.json` pins the upstream source, Node, Bun and build origin; Docker base images
also use immutable digests. `app.patch` is the complete local overlay. It adds the
Snowball catalog entry, landing defaults, build/recipe detail, internal origin, Unix
socket connections and exact GitHub-attempt ingestion. It leaves the upstream schema,
processing, view transforms and trace importer in use. APIs supply benchmark data at
request time, database-backed OG routes skip static enumeration, and the private sitemap is
empty: building the image needs no database credentials or fixtures.

One phase-boundary fix keeps offsets relative when slicing request charts. Adding an
epoch-nanosecond origin and subtracting it lost precision and dropped the first profiling
request in a real H100 window. The API retained 351 requests while the histogram showed
350; the regression uses that window's origin and offset. No measurement is rewritten.

The importer also preserves cache-hit rates derived from an already linked trace when
an aggregate is imported again. Upstream otherwise overwrote those rates with the
aggregate's server-counter rates while skipping trace processing. Replaying the real
C32 artifact changed 52.68% to 57.38%; a SQL regression covers preserving the linked
trace's rate while still refreshing aggregate-only rows.

The app remains GPLv3; `LICENSE.app` preserves its license. Corresponding source is the
upstream commit in `settings.json` plus the adjacent patch. Keep both with distributed
images. `/marin-build.json` exposes the upstream commit and overlay SHA256. Selected
point details show the measured vLLM build and serving recipe; missing values show `—`.

## Build and local acceptance

```sh
docker build -t inferencex-web infra/inferencex
docker build --target ingest -t inferencex-ingest infra/inferencex
```

Use disposable PostgreSQL 17 with two LOGIN roles created through SQL, both
`NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION`, and an `inferencex` database.
As its administrator, apply `roles.sql`. Run migrations **as the writer** so its default
privileges grant the reader SELECT on later tables. The upstream migration command is:

```sh
docker run --rm --network host --entrypoint bun \
  -e DATABASE_WRITE_URL -e DATABASE_DRIVER=postgres -e DATABASE_SSL=false \
  inferencex-ingest packages/db/src/migrate.ts --yes --no-ssl
```

For Unix sockets also pass `DATABASE_SOCKET_DIR` and mount that directory at the same
path. The URL encodes user/password/database; the explicit socket setting selects the
Postgres.js host. Do not use a URL `host` query parameter as a substitute.

Start the web image with `DATABASE_USER=inferencex_reader`, `DATABASE_PASSWORD`,
`DATABASE_NAME=inferencex`, `DATABASE_SOCKET_DIR`, a unique `CACHE_NAMESPACE` and
`INVALIDATE_SECRET`. Mount the database socket and publish port 8080. The entrypoint
refuses `DATABASE_WRITE_URL`. Leave `E2E_FIXTURES` unset for actual-data acceptance.

Download an `agentx-results-*` artifact from a real collection Action. Its root contains
`agentx-source.json`, upstream `bmk_agentic_*` aggregates and sibling `agentic_*` timelines.
Run the ingest image with that directory mounted at `/artifacts`, `GITHUB_TOKEN` and the
writer connection. The wrapper verifies the exact source GitHub run/attempt/commit,
Iris attempt, recipe and physical GPU count before invoking upstream `ingest-ci-run.ts`.
It rejects local/sentinel source IDs and source-override metadata. No manual SQL metadata
update is part of this path. Retain the source manifest with the original artifact.

Set `DASHBOARD_URL`, `INVALIDATE_SECRET` and, behind IAP, `IAP_ID_TOKEN` to refresh caches
after import. The IAP token travels in `Proxy-Authorization`; the application invalidation
secret travels in `Authorization`. An import or refresh failure is safe to retry.
The web service is limited to one instance so invalidating its in-process/Next cache
reaches the sole live instance. Scaling beyond one instance needs shared cache handling.

Acceptance must compare aggregate/raw counts, latency, throughput and GPU normalization
with `/api/v1/benchmarks`, `/api/v1/benchmarks/history`,
`/api/v1/views/inference`, point detail and request timeline. Repeat the import and check
unchanged benchmark/trace identities and counts. Verify two actual source run IDs in
history, preserving their real timestamps; two runs on one UTC date do not establish a
second calendar date. Check reader SELECT succeeds and INSERT/DDL fail, and a wrong
password is rejected. Throughput and request-token distributions come from the client
profiling records; `total_prompt_tokens` and `total_generation_tokens` preserve vLLM
server counter deltas. Generated tokens can differ from tokens received by the client;
check these totals against `server_metrics.tokens`, not client throughput times duration.
Upstream fixture smoke tests are separate regression evidence.

For request-timeline links into the corpus explorer, bootstrap the dataset once with the
upstream CLI (writer connection and socket mount as above):

```sh
docker run --rm --network host --entrypoint bun \
  -e DATABASE_WRITE_URL -e DATABASE_SOCKET_DIR -e DATABASE_SSL=false \
  inferencex-ingest packages/db/src/ingest-weka-dataset.ts \
  semianalysisai/cc-traces-weka-062126-256k --variant 256k --no-ssl
```

That CLI reads the Hub's current rows API. Verify the dataset's Hub SHA matches the
runner's `dataset_revision` before and after this one-time operation; it must import
393 conversations. The runner independently verifies the immutable corpus bytes.
Refresh `/api/v1/invalidate` afterward. This bootstrap supplies explorer structure;
it does not create or change benchmark measurements.

History uses the original GitHub source run's UTC creation date, following upstream's
convention. Iris submission/start times and raw profiling start/end times retain the
actual queue delay and measurement window. A later collector date is not substituted
for the source date.

## Production activation after human review

No resources are deployed by opening this PR. The IAM change follows
`.agents/skills/add-grant/SKILL.md`: a second person reviews the grant before merge/apply.
The app leaf is intentionally absent from automatic deployment targets. Use Pulumi
directly, following `infra/pulumi.md`. Keep real-identity previews and passwords private.

1. Merge the reviewed dashboard and runner changes. Install the repository's locked
   deploy environment (`uv sync --all-packages --extra deploy`). With the normal operator
   identity, preview/apply `infra/cloudsql`, stack `marin-cloudsql`; this adds the
   `inferencex` database and the two empty password-secret shells to the existing instance.
2. Initialize `infra/inferencex`, stack `marin-inferencex`, with the configured GCP KMS
   provider. For first creation, target only the explicit GCP provider, `inferencex`
   component, its `sa` and `repo` children, and `invalidate-secret`. Read exact URNs from
   the private preview. These are the runtime account, Artifact Registry repository and
   empty invalidation-secret shell; leave the service/image/domain resources for step 5.
3. Review/apply global IAM in `infra/pulumi`, stack `marin`, excluding the two new service
   bindings until the service exists. Their logical names are
   `iap-run-service-us-central1-marin-inferencex-roles-iap-httpsresourceaccessor` and
   `run-service-us-central1-marin-inferencex-roles-run-invoker`. Use their full preview URNs
   with Pulumi's `--exclude`. Review drift; do not replace existing policies manually.
4. Through an authenticated Cloud SQL proxy, create the two restricted LOGIN roles using
   SQL, apply `roles.sql`, and store each generated password in the matching Secret Manager
   secret. Avoid [Cloud SQL's default elevated user membership](https://docs.cloud.google.com/sql/docs/postgres/users).
   Store a separately generated
   invalidation secret in `marin-inferencex-invalidate-secret`. Run upstream migrations as
   the writer. Passwords and secret versions stay outside Pulumi state and Git.
5. Preview/apply the full `marin-inferencex` leaf. It builds and pushes the web image,
   attaches the existing SQL socket, configures IAP and creates a DNS-only CNAME and Cloud
   Run domain mapping. The public build origin must match `custom_domain`; update
   `settings.json` and stack configuration together when changing it. Wait for the managed
   certificate. Then apply the global IAM stack without the two exclusions. A service
   created before those bindings is inaccessible, not public.
6. Build/push the `ingest` target to the same repository. Set repository variable
   `INFERENCEX_INGEST_IMAGE` to its **digest** reference, for example
   `us-central1-docker.pkg.dev/hai-gcp-models/marin-inferencex/ingest@sha256:...`.
   This is a separate image target from the smaller standalone web runtime.
7. Verify an authorized viewer can load the root, which redirects to Snowball Agentic
   Traces/BF16/output tokens per GPU/P90, and an unauthenticated caller cannot read APIs.
   Verify the reader role cannot write and the GitHub identity can use the writer secret
   and IAP-protected invalidation endpoint. Do not infer live IAP/TLS behavior from mocks.
8. Set `AGENTX_STATE_ROOT` as described in the runner runbook, then
   `INFERENCEX_INGEST_ENABLED=true`. Submit an H100 run, collect it, and inspect the import
   Action and API/UI values. Retry its collection/import to confirm idempotence. Only then
   enable `AGENTX_WEEKLY_ENABLED=true`. Verify a current GB200 batch lands within the spare
   allocation before including it in the weekly profile. Do not hard-code an old rack.

The import workflow triggers after each collection and can be retried explicitly:

```sh
gh workflow run inferencex-ingest.yaml -R marin-community/marin \
  -f collector_run_id=COLLECTION_RUN_ID
```

Each job is bounded to 25 minutes, uses an ephemeral checksum-pinned Cloud SQL Auth Proxy,
and closes it on exit. Ingestion uses the original benchmark source run, even though the
artifact belongs to a later collection Action. The upstream importer fetches that exact
attempt's GitHub metadata and uses existing conflict keys for aggregates and timelines.
Schema migration is an operator action, not a per-import operation.

## Rollback and retention

Disable `AGENTX_WEEKLY_ENABLED` to stop new weekly submissions. Cancel active saved plans
before disabling collection. Disable `INFERENCEX_INGEST_ENABLED` to stop database writes.
Keep raw artifacts and collection state so these operations can be retried later.

Before each app deployment, record the current Cloud Run revision and immutable image.
To roll back serving, use `gcloud run services update-traffic marin-inferencex
--project=hai-gcp-models --region=us-central1 --to-revisions=PREVIOUS_REVISION=100`, then
reconcile Pulumi to the reviewed source. This overlay adds no SQL migration. For future
upstream migration changes, take a backup of the existing Cloud SQL instance first and
check backward compatibility; rolling back the app does not roll back a database.

Keep GitHub plan/result artifacts for 90 days. The benchmark collector copies Iris's
seven-day output archives into the configured non-TTL prefix with uploader SHA256,
processed-bundle SHA256, all native attempt metadata and validation receipts. Apply the
organization's retention policy there only after preserving a restore reference. The web
database is a derived view; archived raw results are the reproduction source.

The remaining live acceptance gates are the reviewed merge, IAM/SQL secret bootstrap,
Cloud Run/IAP/vanity-TLS rollout and enabled scheduled workflows. Local production-image
acceptance and temporary-branch GitHub execution do not claim these gates have passed.
