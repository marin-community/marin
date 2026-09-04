# Background Research Brief

- Effort / stop rule / date: Medium; stopped when repository, prior-PR, live-resource, and official Cloud Run evidence agreed on the deployment constraints; 2026-09-04.

## Ten-minute EvalDash follow-up

EvalDash's inventory lists only `*/record.json` candidates and revalidates known object metadata on its separate daily schedule. A ten-minute discovery cadence therefore multiplies LIST calls, not full record reads. Google prices regional Standard GCS Class A operations at $0.005 per 1,000 with 5,000 monthly operations free; two one-page GCS prefixes scanned 4,320 times are about $0.04 gross. CoreWeave publishes object storage with no request or egress fees. Cloud Run jobs have a one-minute billing floor and use instance-based rates: at the us-central1 list prices of $0.000018/vCPU-second and $0.000002/GiB-second, a 1-vCPU/1-GiB ten-minute runner is $5.18/month gross when each execution fits in one minute. This supports a dedicated lightweight EvalDash runner rather than running the 4-vCPU Echo job six times per hour.

Sources: [Cloud Run pricing](https://cloud.google.com/run/pricing), [Cloud Storage pricing](https://cloud.google.com/storage/pricing), [CoreWeave pricing](https://coreweave.com/pricing).

## Question

How should Marina move its non-request work into app-declared Cloud Run jobs, run Echo hourly and EvalDash every ten minutes, remove the dedicated migration job, and use request-based Cloud Run billing without serving stale EvalDash data?

## Current Marin Context

The Marina stack hard-codes two Cloud Run jobs. `marina-migrate` runs after each image change and blocks creation of the new service revision. `marina-echo-sync` runs every ten minutes. The service sets `cpu_always_allocated=True` and keeps one instance warm because EvalDash starts reconciliation and catalog-polling loops inside the web process ([Pulumi entry point](https://github.com/marin-community/marin/blob/b51262b432b9edad777c40531a043491ada8434f/infra/marina/__main__.py#L191-L283)).

App discovery is already manifest-driven, but `app.toml` only accepts page metadata and a frontend build command ([manifest parser](https://github.com/marin-community/marin/blob/b51262b432b9edad777c40531a043491ada8434f/infra/marina/src/marina/manifest.py#L17-L65)). The CLI separately discovers all apps for builds and migrations ([CLI](https://github.com/marin-community/marin/blob/b51262b432b9edad777c40531a043491ada8434f/infra/marina/src/marina/cli.py#L42-L76)). This gives jobs a natural declaration and execution boundary without importing application modules into Pulumi.

EvalDash stores a committed catalog in PostgreSQL but serves most reads from an in-memory snapshot. Its web process currently runs both object-store reconciliation and a ten-second generation poll in background tasks ([EvalDash app](https://github.com/marin-community/marin/blob/b51262b432b9edad777c40531a043491ada8434f/infra/marina/apps/evaldash/app.py#L642-L766)). A scheduled reconciler alone would leave an existing web instance stale unless request handling also checks the committed catalog generation.

## Internal Prior Work

The merged [shared Marina service PR](https://github.com/marin-community/marin/pull/8867) introduced the current structure. It deliberately kept one warm, always-allocated service instance for EvalDash ingestion and created separate deploy-migration and Echo-sync jobs. The rollout already copied and verified the production data, so this change only needs to preserve ongoing schema migration and ingestion behavior.

Echo currently advances one of six repository targets per execution. Its ten-minute trigger therefore checks each target about once per hour ([Echo operator notes](https://github.com/marin-community/marin/blob/b51262b432b9edad777c40531a043491ada8434f/infra/marina/apps/echo/README.md#L57-L70)). Changing the runner to hourly without changing Echo's turn model makes per-repository checks about six hours apart. The user requested one sync execution per hour; preserving hourly checks for every repository would require a separate Echo behavior change and longer executions.

Recent live `marina-echo-sync` executions ranged from about one to eighteen minutes, with overlapping ten-minute triggers. The live service has `run.googleapis.com/cpu-throttling=false`, service-level minimum instances of one, four CPUs, and 4 GiB. The live migration job has run twelve times and recent no-op migrations still took roughly two to four minutes including job startup.

## External Prior Art

Google documents request-based billing as the default for sporadic services. CPU is allocated while an instance starts, shuts down, or handles a request. Instance-based billing allocates and charges CPU for the full instance lifetime and is intended for background work. Cloud Run jobs always use instance-based billing for their execution lifetime ([billing settings](https://docs.cloud.google.com/run/docs/configuring/billing-settings), [pricing](https://cloud.google.com/run/pricing)).

Google's supported scheduled-job pattern is a Cloud Scheduler HTTP target that invokes the Cloud Run Jobs `:run` API with OAuth ([scheduled jobs](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule)). `gcloud run jobs execute` also supports per-execution argument overrides, so one scheduled runner definition can execute `marina migrate` during deployment without a second persistent job resource.

## Evidence Map

### Claim: EvalDash is the reason Marina needs CPU outside requests

- Support: The Pulumi comment names EvalDash's ingest loop, and `build_api` starts reconciliation plus catalog polling in mounted-app background tasks.
- Contradictions: Echo model inference is CPU-heavy, but it runs on request. One warm instance affects latency and idle memory cost; it does not require always-allocated CPU.
- Directness to Marin: Direct code and live-service configuration.
- Confidence: High.
- Action: Move production reconciliation to a job and reload catalog generations at the request boundary.

### Claim: The migration resource can be deleted without moving migrations into service startup

- Support: The shared job image already contains `marina migrate`, and Cloud Run execution argument overrides can replace a job's normal runner arguments. `CloudRunServiceArgs.before_deploy` already orders resources between image publication and revision creation.
- Contradictions: Reusing the hourly runner couples deploy readiness to that runner resource and gives the migration execution the runner's larger resource allocation.
- Directness to Marin: Direct code plus installed `gcloud` command contract.
- Confidence: High.
- Action: Execute `marina migrate --reader ...` as an overridden execution of `marina-hourly` before deploying the revision.

### Claim: App manifests can own scheduled work without owning cloud credentials

- Support: Manifests already drive app discovery and reject unknown keys. Pulumi can group their runner declarations while keeping the mapping from logical secret names to Secret Manager resources in the deployment stack.
- Contradictions: Co-scheduled jobs share one process lifetime, resource size, service account, and union of declared secret environment variables.
- Directness to Marin: Direct code; no external abstraction is required.
- Confidence: High.
- Action: Add typed job declarations with runner, cron schedule, argv, and logical secret names.

## Negative / Failed Leads

- Running migrations during every web-container startup would remove the job, but concurrent cold starts would require every app migration chain to be concurrency-safe and would put schema work on the request startup path.
- Creating one Cloud Run Job for every app job gives clean failure and secret isolation, but it does not let light jobs share one image cold start and schedule as requested.
- Keeping EvalDash's catalog polling loop under request-based CPU can resume late and let the first request after an hourly ingest observe the previous generation.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Shared Marina service | PR | https://github.com/marin-community/marin/pull/8867 | Original deployment intent and completed migration | High | Echo result `pr:19094` |
| Marina Pulumi entry point | Marin code | `infra/marina/__main__.py` | Current jobs, billing, schedule, deploy ordering | High | Pinned above |
| Marina manifest and CLI | Marin code | `infra/marina/src/marina/` | Reusable discovery and execution boundary | High | Pinned above |
| EvalDash app | Marin code | `infra/marina/apps/evaldash/app.py` | Reconciliation and in-memory catalog behavior | High | Pinned above |
| Cloud Run billing settings | Official docs | https://docs.cloud.google.com/run/docs/configuring/billing-settings | CPU and billing semantics | High | Current 2026-09-04 |
| Schedule Cloud Run jobs | Official docs | https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule | Scheduler invocation pattern | High | Current 2026-09-04 |
| Live GCP resources | Cloud Run API | `hai-gcp-models/us-central1` | Current flags and execution durations | High | Read-only inspection 2026-09-04 |

## Handoff

- Decision: One hourly Echo command processes all six repository turns.
- Decision: Keep one minimum web instance under request-based billing because the idle cost is acceptable and it avoids Echo model-loading latency.
- Decision: Keep EvalDash's whole-catalog in-memory cache, but validate it against the durable generation at the store query boundary instead of running an ASGI background poll.
