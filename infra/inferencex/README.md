# Marin inference performance dashboard

This leaf packages an internal build of the pinned InferenceX-app as a standalone
Next.js service. Its Pulumi configuration targets `https://performance.oa.dev`
and uses Cloud Run's native IAP gate for the service and its APIs.
The global IAM stack reuses Grafana's viewer grants and adds the existing GitHub Iris
identity for cache refresh. No anonymous invoker is granted. The web process receives
only the `inferencex_reader` password. The import Action receives `inferencex_writer`.

`settings.json` pins the upstream source, Node, Bun and build origin; Docker base images
also use immutable digests. `app.patch` contains the upstream fixes and configuration changes. It adds the
Snowball and Hero catalog entries, landing defaults, build/recipe detail, internal origin, Unix
socket connections and exact GitHub-attempt ingestion. It leaves the upstream schema,
processing, view transforms and trace importer in use. APIs supply benchmark data at
request time. `internal.mjs` retains whole upstream benchmark, comparison, history,
point-detail, corpus-exploration and API routes. It omits public content pages and social
images, and installs `internal-layout.tsx` plus private robots/sitemap entrypoints.
Building the image needs no database credentials or fixtures.
Hero's retained points are provisional. Its calibration and weekly-load selection
remain deferred.

The writer requires its own GitHub source lookup to succeed before database writes.
Retry ingestion when that lookup is unavailable; an import must preserve the original
source date, commit and link.

The internal root omits the public header/footer, decoration, star-count fetch and
site analytics. Updating the pin requires checking route selection and provider
imports in that root. The full app remains the fallback described below.

Request-chart phase slicing keeps nanosecond offsets relative to their captured
origin so the first profiling record remains present. Repeated aggregate imports
preserve cache-hit metrics already derived from a linked trace. Aggregate-only
rows still refresh their metrics from the server counters.

The app remains GPLv3; `LICENSE.app` preserves its license. Corresponding source is the
upstream commit in `settings.json`, the adjacent patch and internal build entrypoints. Keep
these with distributed images. `/marin-build.json` exposes the upstream commit, overlay
SHA256 and internal-profile SHA256. Selected
point details show the measured vLLM build and serving recipe; missing values show `—`.

## Build and local acceptance

```sh
docker build -t inferencex-web infra/inferencex
docker build --target ingest -t inferencex-ingest infra/inferencex
```

Use disposable PostgreSQL 17 with two LOGIN roles created through SQL, both
`NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION`, and an `inferencex` database.
Initialize local/socket authentication with `scram-sha-256`, too; PostgreSQL's
default local `trust` rule would accept an incorrect password on that socket.
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
A recovered valid window remains importable when its short submitting Action ended in
failure or timed out after Iris accepted the job. The exact completed source identity is
still verified and upstream retains its actual conclusion. Cancelled sources and failed
or interrupted native windows remain excluded. A failed collection Action may also
publish valid siblings; import admission checks each point independently.
Calibration accepts any positive integer concurrency: Snowball on eight H100s,
and the pinned Hero step144000 BF16 export on sixteen GB200s across four hosts.
Each point must retain its validated one-hour window and all 393 corpus entries.
Hero additionally requires four distinct nodes, sixteen GPU UUIDs and the exact
successful host-attempt set saved with the measured window. Startup probes and
partial or replaced gangs do not enter the chart.
The pinned processor's `IS_MULTINODE` input selects disaggregated prefill/decode
pools. Hero uses its combined-pool path for the sixteen-GPU denominator, preserves
that processor output as `processor-aggregate.json` beside the raw trace, and
annotates only the final aggregate's physical `is_multinode` marker. The app
retains a combined pool (`disagg=false`) on four hosts. No request metric changes.
It rejects local/sentinel source IDs and source-override metadata. No manual SQL metadata
update is part of this path. Retain the source manifest with the original artifact.

The collector's `changelog-metadata/agentx.json` uses the existing upstream
append-only contract. Each source extends its model/hardware curve while keeping
its actual GitHub date, attempt and commit. On the inference chart, disable
**Optimal Only** (`i_optimal=0`) to inspect non-frontier calibration loads.
The existing run selector and comparison/history views expose individual source
runs and repeats; the latest curve shows the newest measurement for each load
and serving recipe.

Set `DASHBOARD_URL`, `INVALIDATE_SECRET` and, behind IAP, `IAP_ID_TOKEN` to refresh caches
after import. The IAP token travels in `Proxy-Authorization`; the application invalidation
secret travels in `Authorization`. An import or refresh failure is safe to retry.
Obtain the deployment's accepted token through
[IAP programmatic authentication](https://docs.cloud.google.com/iap/docs/authentication-howto).
Production audience and issuer checks remain part of deployment acceptance; the
private local preview does not use IAP.
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

Compare these fields at the pinned processor's precision:

| Value | Delivered artifact | Dashboard API |
| --- | --- | --- |
| Output tokens/s/GPU | `agg.json: request_metrics.throughput.per_gpu.output_tput_tps` | `/api/v1/benchmarks: metrics.output_tput_per_gpu` |
| P90 time to first token, seconds | `agg.json: request_metrics.latency.ttft.p90` | `/api/v1/benchmarks: metrics.p90_ttft` |
| Physical GPUs | `agg.json: num_gpus` | `/api/v1/benchmark-siblings: siblings[].num_decode_gpu` |
| Successful profiling requests | `agentx-source.json: receipt.validation.profiling_requests` | `/api/v1/benchmark-siblings: siblings[].total_requests` |

Per-GPU throughput is client output tokens/s divided by physical GPUs, with the
processor's rounding. Use its `request_metrics.throughput.duration_seconds` interval.
Warmup requests remain in the raw trace and are excluded from profiling counts.
For every raw JSONL record, match `metadata.conversation_id`, `turn_index` and
`benchmark_phase` to the request-timeline API's `cid`, `ti` and `phase`. Adding
`startNs` to each relative `start`/`end` must reproduce `request_start_ns` and
`request_end_ns` within 1024 ns of JSON numeric rounding.

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
8. Set `AGENTX_STATE_ROOT` as described in the
   [runner runbook](../../experiments/evaluation/agentx/README.md), then
   `INFERENCEX_INGEST_ENABLED=true`. Submit an H100 run, collect it, and inspect the import
   Action and API/UI values. Retry its collection/import to confirm idempotence. Only then
   enable `AGENTX_WEEKLY_ENABLED=true` for Snowball's selected three-load grid.
   Hero calibration and weekly-load selection remain deferred.

Automatic imports accept collection completions from this repository's default branch.
An operator can explicitly retry a reviewed collector from another branch:

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

Disable `AGENTX_WEEKLY_ENABLED` to stop new weekly submissions. Cancel active Iris
coordinators before disabling collection, following the
[runner runbook](../../experiments/evaluation/agentx/README.md). Disable `INFERENCEX_INGEST_ENABLED` to stop database writes.
Keep raw artifacts and immutable completed-result descriptors so delivery and ingestion
can be retried later. Coordinator failure can require manual recovery; the runner runbook
contains the exact salvage and fresh-submission commands.

Option A is the original pinned full app plus correctness patches. To rebuild that
fallback, use the preserved original packaging commit in an isolated worktree:

```sh
git worktree add --detach ../inferencex-full-app d58f672d5569e4daadeb0cacdec6d1b7e69698d0
docker build -t inferencex-full-app ../inferencex-full-app/infra/inferencex
```

Both profiles use the same PostgreSQL schema and reader/writer roles.
Continue using the current ingestion image with the full-app web fallback. It contains
the source-lookup and trace-cache correctness fixes.

Before each app deployment, record the current Cloud Run revision and immutable image.
To roll back serving, use `gcloud run services update-traffic marin-inferencex
--project=hai-gcp-models --region=us-central1 --to-revisions=PREVIOUS_REVISION=100`, then
reconcile Pulumi to the reviewed source. This overlay adds no SQL migration. For future
upstream migration changes, take a backup of the existing Cloud SQL instance first and
check backward compatibility; rolling back the app does not roll back a database.

Keep GitHub plan/result artifacts for 90 days. The benchmark collector copies Iris's
seven-day output archives into the non-TTL `AGENTX_STATE_ROOT` prefix with uploader SHA256,
processed-bundle SHA256, all native attempt metadata and validation receipts. Apply the
organization's retention policy there only after preserving a restore reference. The web
database is a derived view; archived raw results are the reproduction source.

The remaining live acceptance gates are the reviewed merge, IAM/SQL secret bootstrap,
Cloud Run/IAP/vanity-TLS rollout and enabled scheduled workflows. Local production-image
acceptance and temporary-branch GitHub execution do not claim these gates have passed.
