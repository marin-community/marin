# App-declared Marina runners

Marina will use request-based Cloud Run billing. Production work that must progress without an HTTP request will run in app-declared Cloud Run jobs. Echo sync will run hourly, EvalDash ingestion will use a lightweight runner every ten minutes, and the dedicated `marina-migrate` resource will be removed while migrations remain a deploy prerequisite.

## Background

The current Marina service keeps CPU allocated for its full lifetime because EvalDash reconciles object storage and polls catalog generations in the web process. Echo sync and deploy migrations are hard-coded as separate resources in the Marina Pulumi program. The service is now the default deployment for Echo, EvalDash, and TaskTrove, so the remaining background-work exception should become part of the app contract. [Research](research.md) records the live configuration, original PR, code paths, and Cloud Run billing semantics.

## Challenges

EvalDash serves an in-memory snapshot of a PostgreSQL catalog. Moving reconciliation into a scheduled process updates PostgreSQL but does not update snapshots already loaded by web instances. The first request after a job commit must refresh the snapshot before reading it.

Deploy migrations must still finish before a new revision starts. Removing the dedicated migration job cannot turn schema ordering into a manual deploy step or add migrations to every web-container cold start.

Co-scheduled app jobs share one Cloud Run execution. A command timeout or ordinary nonzero exit must not prevent later app jobs from running, and the execution must still fail after all commands finish so Cloud Run records the partial failure. A process-level termination can still stop the runner.

## Costs / Risks

- Echo will check all six repositories in one hourly command. Recent single-turn executions took one to eighteen minutes, so a cold or changed set of repositories can make the combined command long.
- Co-scheduled jobs share the Marina service account and the union of their declared secrets. A runner takes the largest CPU and memory declaration among its jobs; a job that needs isolation declares a distinct runner.
- EvalDash checks the catalog generation at most once per short cache interval. A changed generation reloads the committed snapshot before a query reads it.
- The service keeps one minimum instance to avoid an Echo model-loading cold start. Request-based billing still charges the lower idle rate for that instance's memory and fractional CPU allocation.

## Design

`apps/<name>/app.toml` may contain `[[jobs]]` entries. Each entry declares a job name, a runner name, one unix-cron schedule, an argv array, a command timeout, CPU and memory requirements, and logical secret environment names. Jobs with the same runner must declare the same schedule. The manifest parser rejects duplicate job names, malformed commands, unknown secrets at deployment, and conflicting schedules for one runner.

The `marina run <runner>` CLI discovers manifests, acquires a database advisory lease for the runner, applies app migrations under one shared migration lock, and runs matching commands in app-name and job-name order. A concurrent execution that cannot acquire the runner lease exits successfully without running another copy. The CLI continues after a child timeout, spawn error, or nonzero exit, then exits nonzero with the failed qualified job names. Each child receives only the registered secrets its manifest entry declares.

The Pulumi program groups declarations by runner. Each group becomes `marina-<runner>` plus a Cloud Scheduler trigger using the declared cron expression. A runner receives database environment, Cloud SQL, the maximum declared CPU and memory, and the union of its jobs' logical secrets. Secret Manager IDs remain in Pulumi; app manifests only name the environment variables their commands consume.

Echo declares `sync` on a 4-vCPU, 4-GiB hourly runner with `python -m echo.sync.main` and `MARINMIRROR_TOKEN`. Its normal entry point processes one activity sync followed by all six repository turns. EvalDash declares `ingest` on its own 1-vCPU, 1-GiB runner every ten minutes with `python -m evaldash.ingest` and the two CoreWeave credential variables. The new EvalDash entry point builds its PostgreSQL store, runs one `PostgresIngestor.run_once()`, disposes its engine, and fails after the pass if any configured prefix failed.

The hourly runner's default command is `marina run hourly`, so every scheduled execution migrates the image before it runs app code. During deployment, `CloudRunServiceArgs.before_deploy` executes that same job with migration-only and reader-grant arguments, then waits. A scheduler firing after any runner image changes is safe because every runner takes the shared migration lock. Reader grants stay out of recurring executions, where their table locks could delay app work. The image reference remains the deploy trigger, so unchanged images do not start a deploy migration. The stack requires an hourly runner while it uses that runner as the migration vehicle.

EvalDash keeps background reconciliation only for its database-free local mode. In PostgreSQL mode, the store's read boundary checks the durable catalog generation on a short backoff. It reuses the in-memory snapshot while the generation is unchanged and installs only a newer snapshot under one process lock. A failed check records the catalog error, backs off, and serves the previous committed snapshot. `/status` reads persisted prefix health written by the EvalDash job. `POST /refresh` first loads any already committed generation, then queues the same Cloud Run job and returns `202` with its operation name. Source reconciliation never outlives a web request inside a request-billed service.

The Marina service sets `cpu_always_allocated=False`, which emits `cpu_idle=true` and selects request-based billing. It retains startup CPU boost, one minimum instance, the current maximum of four, and current CPU and memory limits.

## Testing

Manifest tests cover resource aggregation, invalid commands and resource values, and conflicting runner schedules. CLI tests execute real child commands in a temporary directory and verify secret filtering, continue-after-failure behavior, migration-only execution, and final failure status.

EvalDash tests verify that PostgreSQL queries reload a newly committed catalog before returning data, failures back off and clear after recovery, the manual endpoint queues the configured job, and `/status` reads persisted health. Existing reconciliation tests remain the independent behavior coverage.

The production Pulumi preview verifies `cpuIdle=true`, the hourly and EvalDash runners, their cron expressions and capacities, deploy migration execution against the hourly job, and deletion of `marina-migrate` and `marina-echo-sync`.

The ten-minute cadence adds little storage cost. GCS LIST requests are Class A operations; two one-page prefixes produce about 8,640 operations per month, or $0.04 before the 5,000-operation free allowance. CoreWeave publishes no request or egress charge. The one-minute Cloud Run job floor is the meaningful cost: 1 vCPU and 1 GiB costs about $5.18 per month gross at one minute per execution, before the shared free tier. Longer no-op migration and scan time increases that linearly.
