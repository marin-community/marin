# Daytona operations for agentic RL

Use this runbook when Harbor trials fail during sandbox setup, snapshot quota
prevents a launch, or completed trials leave provider capacity occupied. Iris
job operations remain in [Iris OPS](../../lib/iris/OPS.md).

## Difficulty queue schema changes

Difficulty workers validate byte-bound `sources.json` queues against the current
`DifficultySource` schema before contacting a solver. A validation error such as
`declaration.name: extra_forbidden` is a queue metadata failure. Keep healthy
solver pools running while correcting the controller's queue bindings.

Publish a new queue with current served declarations only when each declaration's
population revision matches its retained panel. Preserve the panel, sample, task
hashes, release and converter bindings, and record the old and new queue digests.
Validate every replacement row with `DifficultySource` and read back the published
bytes before updating `DEFAULT_QUEUES`. Keep the old queues immutable. Queue URI
and digest are part of the action identity, so the corrected binding permits a
fresh attempt without deleting previous submissions or writing scored evidence.

Use repeatable `reconcile --exclude-source SOURCE_ID` arguments to quarantine a
source whose independent repair is pending. The launcher forwards these IDs to
the remote controller, which excludes them before planning any solver's work and
rejects unknown IDs. Record the exclusions with the controller handoff; remove
them on a controller relaunch only after the source's canonical release and new
panel binding are verified. An entry in `repair-verifications` prioritizes a
source and does not quarantine it.

## Credentials and account selection

Agentic RL uses the dedicated `DAYTONA_RL_API_KEY`. A generic Daytona key can
authenticate successfully against the wrong account. Do not substitute it for
the RL key. Resolve the credential through the launcher's supported secret
indirection and verify that the coordinator and trial workers receive it.

Validation that needs declarative image builds uses the Daytona **eval** account,
not the RL account. Keep the two credentials separately named and select the
eval credential only in the validation subprocess. For inline Harbor validation,
set the callback's `daytona_api_key_env: DAYTONA_EVAL_API_KEY` and
forward it through both the coordinator and task launcher. Do not globally
replace the RL credential: training containers remain in the RL account.
`DaytonaValidationError: declarative builds are not allowed` is an account/build
permission failure, not a model failure or a zero reward. Inline validation builds
images into snapshots; the Harbor evaluation callback rejects
`image_build: declarative`, which only standalone eval launches may set.

For eval-account validation without named snapshots, explicitly select
`environment: {type: daytona, kwargs: {provisioning: declarative}}` in the
Harbor TrialConfig. This uses `DaytonaDeclarativeMachineFactory` and
`CreateSandboxFromImageParams` to build from the original registry image or
Dockerfile context during sandbox creation. It preserves task resources and
shares the normal command, provider-retry and sandbox-cleanup path. The default
`snapshot` mode still resolves named snapshots. Select the eval credential
separately; provisioning mode does not change accounts. Bound concurrency to the
eval sandbox quota, retain the actual provisioning mode in launch receipts, and
leave other campaigns' snapshots and sandboxes alone. The ten-snapshot RL
suitability census remains a measurement of the source's training configuration.

Shellbox rejects Dockerfile `ADD` instructions before creating a sandbox. This
check follows instruction continuations and `RUN`/`COPY` heredoc boundaries:
an assembly `add` instruction inside a heredoc is passed through unchanged.
Retain the original Dockerfile when diagnosing a preflight rejection. COPY inputs are resolved with the pinned Daytona SDK before a create request is logged. A missing local source raises `MissingDockerCopySource`; verify the original native inventory and record a source build failure rather than generating replacement files. This preflight performs no provider creation or cleanup. If a pinned upstream archive contains an explicit empty directory that Git omitted, import that archive as a new source version and preserve the directory member through the native Harbor path. Bind the archive digest and directory metadata; do not insert placeholder files.

The keys live in Google Secret Manager in project `hai-gcp-models`:
`DAYTONA_RL_API_KEY` for training and `DAYTONA_EVAL_API_KEY` for validation.
Load them into the correspondingly named environment variables before submission.
Do not enable shell tracing or print the environment:

```bash
set +x
export DAYTONA_RL_API_KEY="$(gcloud secrets versions access latest \
  --project=hai-gcp-models --secret=DAYTONA_RL_API_KEY)"
export DAYTONA_EVAL_API_KEY="$(gcloud secrets versions access latest \
  --project=hai-gcp-models --secret=DAYTONA_EVAL_API_KEY)"
: "${DAYTONA_RL_API_KEY:?RL secret resolution failed}"
: "${DAYTONA_EVAL_API_KEY:?Eval secret resolution failed}"
```

The Marin RL coordinator allowlist forwards both names. Configure the SkyRL
task launcher's `terminal_bench_config.harbor` with
`daytona_api_key_env: DAYTONA_RL_API_KEY` and
`eval_daytona_api_key_env: DAYTONA_EVAL_API_KEY`; its Daytona credential
forwarding reads those fields and rejects missing values. In the
`trainer.callbacks` entry with `type: harbor_evaluation`, set
`daytona_api_key_env: DAYTONA_EVAL_API_KEY`. These paths are relative to the
SkyRL training config (`skyrl` in the generated launch config).

With [the explicit coordinator-secret declaration](https://github.com/marin-community/marin/pull/10333),
also set `IrisSkyRLExecution.required_env_vars` to
`("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY")` to reject missing credentials
before coordinator submission. A field
ending in `_env` takes the variable name, not the Secret Manager URI or key value.
At each process boundary, check only `bool(os.environ.get(name))` for these two
names; do not log their values. Account access still requires a successful
provider preflight.
The standalone evaluation launcher has a separate credential-resolution path;
see [evaluation credentials](../../experiments/evaluation/README.md).

For launchers that consume only `DAYTONA_API_KEY`, map the RL key immediately
before invoking the launcher:

```bash
: "${DAYTONA_RL_API_KEY:?Load the dedicated RL key through the configured secret provider}"
export DAYTONA_API_KEY="$DAYTONA_RL_API_KEY"
```

For concurrent Difficulty trials, resolve the DATA key from the Iris secret
bundle and capture the Daytona connection before starting the worker pool.
Pass that host capability to each trial. The pinned SDK's object-storage
constructor temporarily clears process-wide `os.environ`; resolving a connection
inside another trial thread can therefore report missing `api_key` and `target`
even when the job received both. Keep URL and target in ordinary job environment
variables and the DATA key in the named secret bundle.
Launch the inventory subprocess with the current interpreter's absolute path;
the SDK can also temporarily clear `PATH` while another trial starts.
The inventory-admission CLI's `--api-key-env` selects that same named secret;
it reads the mounted file first and then the same environment variable.

Report presence and successful access only; never print key values. Do not copy
private secret files into launch records or source control. The legacy
OpenThoughts-Agent Jupiter launcher also uses `DC_AGENT_SECRET_ENV` to identify
its private secrets file. That is a launcher-specific setting, not an Iris
requirement. See the source Jupiter runbook below for that workflow.

## Difficulty model traffic

The `/tokenize` route counts oversized histories so native context managers can
compact them. Generation requests still enforce the protocol input budget;
token counting alone never dispatches a model request.
The native agent's proactive summary reserve is the protocol context window
minus its input budget, so compaction starts at the input boundary.

Difficulty's declared native judges use a 30-second upstream HTTP timeout and
up to six attempts. A stalled read can retry before the native grader's
300-second client deadline; solver and tier-judge requests retain 600 seconds.
Provider admission waits are recorded separately.
In `judge-requests.tar.gz`, inspect
`*/*/upstream-attempts/*/transport-error.json` and `*/*/provider-waits/*.json`
before changing limits.
Timeout receipts identify the transport error without inventing an upstream
HTTP status. Retries repeat only model requests, not source commands.

## Native Harbor network evidence

For a native shared-machine task with `source-runtime.json`, Daytona records
`network-policy.json` beside the trial result after the sandbox is acquired.
The receipt binds the original runtime digest to the network policy passed to
Shellbox for the shared actor/verifier machine. A conflicting declared policy
fails before creation. Preserve this file in control archives; a historical
archive without it does not prove its declared network policy was applied.

The coordinator-authorized Terminal-Lego historical reconstruction is a separate
`network_policy_reconstructed_native` record. It binds archived task.toml,
TrialResult environment configuration, source runtime and pinned implementation
SHAs. Before using it, compare at least 20 fresh physical policy receipts with
the reconstructed policies and seal every comparison. Any mismatch rejects the
reconstruction. The historical trial/calibration records stay unchanged; never
write a reconstructed policy as a physical `network-policy.json`.

## Sandbox ownership evidence

For a single-trial subprocess started by your campaign, preserve its trial
configuration, job/task ID and complete lifecycle log. The native oracle-filter
launcher passes `--sandbox-lifecycle-log ATTEMPT/sandbox-lifecycle.jsonl` to
`harbor trials start`. This writes a fresh local file and explicitly enables
ownership logging, independent of the process's default log level. The file
contains events from that trial process; do not combine other campaigns' logs.

The `shellbox.backends.daytona.lifecycle` logger emits JSON message payloads with
`kind: daytona_sandbox_lifecycle` and `schema_version: 1`. `create_requested`
records the generated `requested_name`; `created` binds it to the returned
`sandbox_id`. `delete_requested` records the exact ID before deletion, and
`deleted` follows the awaited provider deletion. A failed creation can still
emit deletion events when the provider accepted the request before returning an
error. An `absent` event has no ID: it means only that the generated name was
not found.

Collect IDs only from `created` or `delete_requested` events tied to a preceding
`create_requested` in that owned subprocess. After the trial subprocess and its
normal awaited cleanup finish, use the same account's read-only
`AsyncDaytona.get(id)` on each exact ID. Only `DaytonaNotFoundError` establishes
absence; authorization errors, timeouts and other lookup failures leave it
unverified. Do not infer ownership from status, age or account-wide lists.
Every `create_requested` without a bound ID remains unresolved, including when
there is an `absent` event. An interrupted subprocess or unresolved request keeps
its slots held pending ledger-owner review; an empty ID list cannot release them.

Before returning reserved sandbox concurrency slots to the campaign's capacity
ledger, store an immutable receipt containing the trial/job identity, complete
log URI and SHA-256, exact IDs, check time and each lookup result. Publish the
receipt URI and SHA-256 to the ledger owner. Presence or an unverified lookup
keeps those slots held. These lifecycle records do not establish correct
scoring; run the [grader controls](../../docs/references/task-curation.md#graders-and-controls)
separately.

## Snapshot preflight and reclamation

Shellbox identifies snapshots for all launchers. One snapshot
covers one image source (a registry image, or a Docker build context with its
Dockerfile contents) and one CPU, memory, disk and GPU request. Memory and disk
round up to whole GiB. The name is `shellbox-` plus the first 40 hex characters
of `sha256(image_source_key(source) + json(resources))`, so a changed image
reference, build context or resource request selects a different snapshot. A
moved registry tag keeps its snapshot name. Harbor passes its task
image and resource values to Shellbox and does not name, round or build
snapshots itself.

Use `shellbox-daytona-snapshots` from a checkout instead of ad-hoc SDK scripts.
It reads credentials from an explicit environment variable and never falls back
to the SDK's ambient account selection:

```bash
# RL account used by training.
: "${DAYTONA_RL_API_KEY:?Load the dedicated RL key through the configured secret provider}"
export DAYTONA_TARGET="<the configured target>"

uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
  --api-key-env DAYTONA_RL_API_KEY inventory
```

Pass `--api-key-env DAYTONA_EVAL_API_KEY` to inventory or purge the eval account
used for validation builds. `--target` overrides `DAYTONA_TARGET`. `--api-url`
overrides `DAYTONA_API_URL`.

Read the account's current quota and inventory before submission. Snapshot quota
and live sandbox capacity are separate limits. A successful GPU allocation proves
neither is available.

`shellbox-daytona-usage` reads every page of the sandbox inventory and reports
organization headroom. The preset table below is the source of record for the
limits supplied by the account owner (2026-11-19); it does not call the
organization-usage endpoint, which rejects API keys with 403.

```bash
uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-usage \
  --api-key-env DAYTONA_RL_API_KEY --cpu 2 --memory-gib 8 --disk-gib 8 --json

uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-usage \
  --all --cpu 4 --memory-gib 8 --disk-gib 10 --headroom 0.2 --json
```

`--all` uses the dedicated keys: eval = `DAYTONA_EVAL_API_KEY`,
rl = `DAYTONA_RL_API_KEY`, data = `DAYTONA_DATA_API_KEY`. Missing keys produce
explicit skipped-account entries. A single standard key selects its quota
preset. For another variable, select the limits explicitly, for example
`--api-key-env DAYTONA_API_KEY --limits eval`. Check that this variable holds
the intended account's key. The ambient `DAYTONA_API_KEY` never selects an
account by itself.

The presets are organization totals, across all regions:

| Preset | vCPU | RAM (GB) | Disk (GB) |
| --- | ---: | ---: | ---: |
| eval | 2,500 | 7,500 | 25,000 |
| rl | 6,000 | 12,000 | 20,000 |
| data | 2,000 | 4,000 | 10,000 |

All three presets limit one sandbox to 4 vCPU, 8 GiB RAM and 10 GiB disk.
The report converts decimal GB quotas to GiB to match inventory allocations.
`--memory-gib` and `--disk-gib` take whole GiB. Update the maintained presets
when the account owner changes these limits; the command does not fetch live
quotas. `--target` configures the SDK connection and does not filter inventory.

The report counts every state. Compute is charged for `creating`, `restoring`,
`started`, `starting`, `stopping`, `pending_build`, `building_snapshot`,
`unknown`, `pulling_snapshot`, `forking`, `pausing` and `resuming`.
`resizing` and `snapshotting` also charge compute when their desired state is
`started`. Disk is charged for every compute state plus `stopped`, `paused`,
`archiving`, `resizing` and `snapshotting`, regardless of desired state.
These rules follow the provider's
[compute](https://github.com/daytonaio/daytona/blob/b5a5d9e/apps/api/src/organization/constants/sandbox-states-consuming-compute.constant.ts)
and [disk](https://github.com/daytonaio/daytona/blob/b5a5d9e/apps/api/src/organization/constants/sandbox-states-consuming-disk.constant.ts)
quota state definitions. Unrecognized API states and missing desired states for
conditional compute fail the reading. For each resource, `remaining` is `max(0, quota - used)`,
and `available` is `max(0, quota * (1 - headroom) - used)`. The default
`--headroom 0.2` reserves 20% of the total quota. `max_concurrency.limit` is the
minimum of `floor(available / per_sandbox_request)` across CPU, RAM and disk.
A profile above any per-sandbox maximum gets a zero limit.

JSON includes `organization_id`, `observed_at`, `usage_source`, `quota_source`,
`org_limits`, `headroom_fraction`, `compute_charged_states`,
`conditional_compute_charged_states`, `conditional_compute_desired_state`,
`disk_charged_states`, `resources`, and `sandboxes` (`total`, `started`,
`compute_charged`, `disk_charged`, `by_state`). A supplied profile adds
`max_concurrency` (`profile`, `by_resource`, `exceeds_per_sandbox_maximum`,
`limit`). Resource fields are `quota`, `used`, `remaining` and `available`.
An empty inventory reports zero usage and a null organization ID, without
querying snapshots or volumes. Duplicate IDs, mixed organizations or missing
active allocations fail the reading instead of reporting free capacity.

Inventory is an estimate at read time, not a capacity reservation. Re-read it
before and during submissions, include the entire simultaneous batch, and
back off when it does not fit. Keep lane admission locks and pending-create
reservations: this read-only command does not coordinate concurrent submitters
or enforce creation-rate limits. Snapshot quota is separate.

1. Inventory snapshots in the intended account, including every result page:

   ```bash
   uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
     --api-key-env DAYTONA_RL_API_KEY inventory --json
   ```

   The command paginates through the inventory and reports state, organization
   ownership, resources, last use and whether each snapshot is stale, fresh,
   unclassified or provider-owned (`general`). `general=true` snapshots are
   outside account cleanup scope. Never substitute another account's key to
   overcome a quota limit.
2. Determine the snapshots required by the dataset and pinned backend. Count
   distinct environment contents, including fixtures, rather than assuming one
   snapshot per task or counting Dockerfiles alone. Emit the effective inputs
   from the job config that launches the trials:

   ```bash
   PYTHONPATH=lib/marin/src uv run --isolated --project vendor/harbor --frozen \
     --extra runtime python lib/marin/src/marin/evaluation/harbor/trial_driver.py \
     snapshot-specs experiments/evaluation/configs/harbor/ot-tblite.yaml \
     > /tmp/required-snapshots.json
   ```

   `snapshot-specs` applies the job's `environment.override_*` values and resource
   enforcement policy as a trial does, resolves each dataset task, and emits the
   Shellbox spec JSON. Do not hand-write task-level CPU or memory values: the
   snapshot name follows the effective request. For example, `ot-tblite` launches
   at 2 CPU, 8192 MB memory and 8192 MB storage regardless of task-level values.
   A SkyRL launcher sets the same values under `terminal_bench.harbor` as
   `override_cpus`, `override_memory_mb` and `override_storage_mb`; use those,
   not the task's `task.toml`. Use `--specs -` on the next commands to read stdin.
   When the launch overrides the dataset through a runtime overlay file, pass it
   as a second argument (`snapshot-specs <policy> <overlay>`); otherwise the
   stable policy's own task set is emitted.

   `plan` reports each required name as `missing` or its current state:

   ```bash
   uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
     --api-key-env DAYTONA_RL_API_KEY plan \
     --specs /tmp/required-snapshots.json
   ```

   Creating missing snapshots is the preflight itself. Run `ensure` in the account
   that launches the trials:

   ```bash
   uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
     --api-key-env DAYTONA_RL_API_KEY ensure \
     --specs /tmp/required-snapshots.json
   ```

   `ensure` waits for every snapshot to become active. The RL account rejects
   declarative image builds, so a Dockerfile-only task must be prepared as a
   digest-pinned registry image before preflight. The worked example above does
   not clear this bar as-is: `ot-tblite` resolves only Dockerfile-based tasks
   (every emitted spec names a `dockerfile`, not an image), so running `ensure`
   with that output in the RL account fails with the declarative-build
   rejection. Prebuild those environments, publish them as registry images, and
   repin the task configs to the digest-pinned references before relying on
   `ensure` in the RL account; a dataset whose tasks already use registry
   images needs no extra step. Do not create trial snapshots in the eval
   account. If validation creates declarative snapshots there, purge that
   account with the same commands and `--api-key-env DAYTONA_EVAL_API_KEY`.
3. On every snapshot check, classify a snapshot as stale when its last use was
   more than one hour ago (`now_utc - last_used_utc > 3600 seconds`). Use actual
   last-use evidence, not creation time, snapshot state, or an unfamiliar name.
   Exactly one hour does not exceed the threshold. If last-use evidence is
   missing or ambiguous, report that snapshot as unclassified; do not invent a
   timestamp or silently treat it as fresh. The command uses the provider's
   timezone-aware `last_used_at` value and compares it in UTC.
4. Purge **all stale snapshots** in the authorized account on every check,
   whether or not quota is exhausted. Do not stop after freeing the slots needed
   for the next launch:

   ```bash
   # Preview exact targets first.
   uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
     --api-key-env DAYTONA_RL_API_KEY purge
   # Delete them. Every target is revalidated immediately before deletion.
   uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
     --api-key-env DAYTONA_RL_API_KEY purge --apply
   ```

   `purge` enumerates the account's sandboxes before deleting anything, keeps every
   stale snapshot that a sandbox still references (`kept <name> (referenced by a
   sandbox)`), and aborts without deleting when sandbox references cannot be
   enumerated. It re-reads each remaining target's last use so a snapshot used
   since the inventory is also kept. Provider deletion is asynchronous: `removal
   pending <name>` means the delete was accepted but a follow-up inventory still
   observes it. Do not repeat the deletion merely because removal is asynchronous.
5. Re-inventory after cleanup. Report remaining stale targets and deletion
   failures rather than claiming cleanup succeeded. Recheck quota, create any
   required snapshots that are now missing with `ensure`, and verify their active
   state before launching trials. A reference in a future launch configuration
   alone does not exempt an unused snapshot from the one-hour purge rule.

`--min-idle-seconds` changes the idle threshold for a deliberate one-off check;
the default is 3600 and matches the policy. The same operations are available as
Python entrypoints from `shellbox.backends.daytona.snapshot` (`list_snapshots`,
`classify_snapshot`, `purge_stale_snapshots`, `snapshot_name`) when a launcher
must embed them.

Existing authorization to remove stale snapshots need not be requested again.
The default one-hour rule and full stale-set purge were explicitly supplied by
the operator on 2026-10-07. They apply within the authorized account; they do not
grant access to other accounts or authorize deleting unclassified snapshots.
Report what was removed and whether rebuilding is possible; deletion is not a
recoverable local-cache operation.

Do not assume launcher cleanup enforces this rule. MarinSkyRL campaign revision
`f8d6cf25235c20cb8365069c7561026100231ca7`, in
[`cloud/iris/iris_backend.py`](https://github.com/marin-community/MarinSkyRL/blob/f8d6cf25235c20cb8365069c7561026100231ca7/cloud/iris/iris_backend.py),
uses a two-hour cutoff, selects only names starting with `harbor__`, falls back
to creation time for never-used snapshots, and skips cleanup when the optional
Daytona SDK is unavailable. That implementation does not satisfy the newer
one-hour, all-stale-account-snapshots policy. Run the account-wide check above
with an installed SDK until the selected launcher implements the current rule.

Use the pinned provider SDK through these Shellbox entrypoints for remote
snapshot operations. Harbor's `cache clean` command cleans local Docker images
and the local Harbor cache; it is not a Daytona snapshot-quota remedy.

Quota exhaustion can fail every trial before scoring. Inspect an actual trial
exception and preserve it separately from model errors. After any backend or
snapshot change, run the campaign's end-to-end smoke before scaling collection.

## Sandbox lifecycle and idle cleanup

Shellbox waits for provider destruction before sandbox cleanup returns. The pinned
SDK's `delete(wait=True)` is required; its default only acknowledges the deletion
request. Preserve cleanup failures as infrastructure evidence.

Hard termination can orphan in-flight sandboxes. Reap only sandboxes proven idle
and covered by cleanup authority. Correlate provider IDs/labels with the owning
trial and job; a quiet agent can still be waiting for model inference or tool
execution. Check queued consumers before applying an idle-reap policy.

Cancellation during sandbox creation can leave a live provider sandbox even
when Harbor never receives its handle. In one incident, a sandbox remained
`STARTED` after an `EnvironmentStartTimeoutError`, consuming capacity until
reaping. [Harbor PR #109](https://github.com/marin-community/harbor/pull/109)
added a bounded, instance-label-scoped orphan sweep on
that cancellation path. Check the installed revision rather than assuming that
finalized trials imply successful teardown. See [the incident and fix](https://marina.oa.dev/echo/wiki/320).

Delete only sandboxes that your own run created. Use the exact sandbox IDs from
your own logs. Do not select sandboxes by name pattern, label, state, age or an
account-wide list. Do not convert a local time window to UTC by hand. If you
cannot get the IDs from your own logs, do not delete. Report the candidates.

Sandbox-not-found errors during queued work can indicate an overly aggressive
idle-reap policy. Uniform setup failures point toward image, dependency, account,
or capacity problems; identify the concrete exception before assigning a cause.

## Evidence and progress

Resolve the durable `trace_jobs/` location from the launched configuration or run
metadata. Agentic trials commonly retain their start configuration, final result,
agent/verifier outputs, exceptions, and timing data. Standard dataset-reward RL
has no Harbor trial artifacts; missing `trace_jobs/` is expected in that mode.

Inspect actual rewards, verifier results, stop reasons, and exceptions. All-zero
rewards from the first step can reflect sandbox, data, verifier, or weight-sync
failure. Very short or empty attempts suggest a broken request path or agent
loop. Incoherent output following weight synchronization requires correlation
with reload logs and importance-ratio metrics before concluding learning failed.

When command-output capture fails, Shellbox retains the exception and adds a
`Daytona exec response:` JSON note with the provider's `exit_code`, the first
4,096 diagnostic characters, and `diagnostic_truncated`. Inspect that response
before attributing a missing `/tmp/.shellbox-*/out` file to the task. A capture
failure supplies no task score.

Fresh inference heartbeats, trace counts, and a running controller job can
coexist with a stalled trainer. Require advancing trainer phases/steps and
durable artifacts for training progress. Report dispatched, completed, scored,
correct, retained, and accepted-for-training counts separately.

Measure sandbox demand across all concurrent workers and replicas. Per-worker
`n_concurrent_trials` is not an account-wide concurrency limit. Compare admitted
trial demand with model-serving capacity (`max_num_seqs` times inference-engine
count), then inspect scheduler queues, KV-cache use, and request flow. Before
attributing under-feeding to Daytona, check coordinator CPU/GIL pressure and
staleness throttling.

Separate environment setup, agent setup, agent execution, and verifier timing.
Use bounded-sample medians/tails and state the sampling window. Per-request model
API timing isolates inference within agent execution; the remainder includes
tool work. Startup provisioning bursts do not establish steady-state churn.

Aggregate large trial sets near the running workload and transfer bounded
summaries. Compare timezone-aware timestamps in UTC. Resolved configurations,
logs, and trial metadata can contain capability JWTs inside model ingress URLs;
redact those as well as named API keys before publication.

## Transient provider errors

Read-only sandbox get/status/absence calls use `read_provider_call`: at most five
attempts with exponential backoff capped at 120 seconds and a 60-second request
timeout. HTTP 5xx, 408/429, SDK connection errors and timeouts are retryable.
Snapshot preparation retains its existing caller deadline. Other 4xx errors and
terminal image-build failures propagate.

Sandbox creation and deletion are single-shot. An explicit creation conflict can
resolve the caller's unique name through read-only calls. An ambiguous mutation
must be reconciled from its lifecycle evidence; it is never blindly replayed.

Each retry emits `DAYTONA_PROVIDER_RETRY` with operation, attempt, status and
delay fields. Count these events separately from task outcomes. Provider errors
remain infrastructure evidence and receive no oracle score. Command
execution is not replayed after an ambiguous response.

Harbor selects `ProviderWaitPolicy.JOB` only when the trial has an explicit
`trial_attempt_timeout_sec`. Queue supervision enforces that overall bound on
wall-clock time; standalone trials without it retain phase-bound behavior.
Task-curation calibration and approved retries set a 72-hour overall attempt
bound without changing the native task or environment kwargs.

Under JOB, safe harness controls, artifact reads/writes and resource probes retry
within the overall bound. Controls use a 60-second
per-call timeout; artifact transfers, snapshot operations and sandbox acquisition
retain SDK operation settings, allowing healthy large transfers to complete.
Request duration and retry backoff pause phase and command clocks; source-command
limits remain active during source execution. Normal cleanup also permits these
provider waits. Cancellation cleanup and grace use wall-clock bounds and disable
recovery retries. Retain retry events separately from task verdicts.

Retryable controls include read-only metadata/resource probes, private staging-file
operations and fixed-content artifact transfers.
Source execution remains single-shot after an ambiguous response. An ambiguous
process-group termination falls back to sandbox teardown; it cannot safely be
replayed after PID reuse. Private-file cleanup is idempotent. External cancellation disables recovery retries during cancellation
cleanup and retains its existing cleanup deadline and primary exception.


### Quarantined control rows

`run_quarantined_trials` seals each attempt through its caller before continuing
other rows. A row with ambiguous sandbox ownership has the typed status
`unresolved_cleanup`, exact known IDs, unknown create requests, lifecycle hashes
and its immutable evidence reference. It is not a scored oracle outcome.

At the end of the pass, `resolve_ownership` revalidates the frozen lifecycle
files, gets each exact ID, deletes a present owned sandbox once with `wait=True`,
and checks absence. NotFound clears the ID. Read exhaustion or an ambiguous
cleanup keeps the row quarantined. Only resolved rows are requeued, once; both
attempts remain archived. Unknown create requests are never resolved by account
lists, labels, statuses or time windows.

The caller must declare physical capacity including quarantined sandboxes.
Unresolved IDs and unknown create requests retain slots. When that capacity is
full, unstarted rows remain explicitly deferred for a later cleared pass.

A configured `QuarantineReserveConfig` adds a shared pool through
`QuarantineReservePool` and `RootQuarantineReserve`. Record the typed pool size,
allocation receipt URI and SHA, logical root keys, and one-extra-slot-per-root
limit in campaign receipts. Initialize the CAS object once before dispatch.
The B campaign's granted reserve is six slots shared by its twelve roots.

Reserve claims use the maintained conditional-object backend (local file lock,
GCS generation or S3 ETag). A claim belongs to a logical root and its exact
producer attempt. It never expires or transfers to a replacement producer by
age. Missing state fails closed; it must not recreate apparently free capacity.

A root with one unresolved sandbox can claim one reserve slot to continue other
rows. At two retained sandboxes, or when the shared pool is full, it processes
ownership evidence and waits without creating a third sandbox. Capacity-induced
pass boundaries reconcile exact owned IDs so a full pool cannot deadlock all
roots before their final pass. Each resolved row is requeued once. If that retry
also fails technically, its evidence remains unscored; clearing cleanup cannot
turn it into a completed oracle outcome. Release a reserve claim only when
absence proofs cover all quarantines tracked by that claim.

## Making a dataset Daytona-safe

A dataset is Daytona-safe when it runs on the RL org as configured: its tasks
require at most `DAYTONA_SAFE_SNAPSHOT_LIMIT` (ten) unique environments, and no
task's declared execution requirements are refused by
`shellbox.machine.check_execution_requirements` with the Daytona capability set
(`{shell, filesystem, python, process, network, git_repository}`; no Docker-in-
Docker, no privileged mode, no host mounts). The count and the verdict come from
the tasks' own specs. Never substitute a shared image for an environment a task
specifies: the spec is part of the data.

Only these options are faithful. Each ends with a new dataset version and a
re-measurement; none changes a task's own instruction, files or grader.

1. **Lower onto a few fixed digest-pinned bases.** Keep the archived recipe's
   `RUN` and `COPY` behavior by emitting a TaskCompendium spec whose resources
   carry the copied files and whose `environment_requirements.setup_commands`
   stage them, as `experiments/post_training/task_curation/calibforge_rebuild.py`
   does. Pin every base by digest; a floating tag fails spec validation. A step
   the lowering cannot express (a multi-stage build, a user creation, an
   unsupported interpreter) defers the task instead of approximating it.
2. **Normalize the machine shape.** Identity includes the machine resources, so
   declare one shape (for example 2 CPU / 4096 MB / 10240 MB) for task and
   verifier machines. Differing shapes multiply snapshot names without changing
   the environment.
3. **Prove a no-op before re-declaring the agent image.** A task may declare
   `iris-task` only when
   `experiments/post_training/task_curation/environment_fidelity.py` proves its
   recipe changes nothing: a single `FROM` equal to the digest-pinned agent
   image, no `RUN`/`COPY`/`ADD`/`ENV`/`ARG`/`USER`/`ENTRYPOINT`/`CMD` step, and
   no copied fixtures. That rewrite is a conversion, so it ships as a new dataset
   version whose declaration records the basis. Anything unproven keeps its own
   environment and is built.
4. **Build the archived environment as the task image.** When lowering cannot
   express the recipe and the image is available to the RL org, build it, pin the
   resulting digest, and count each distinct image as one identity.

Re-measure with the production path, not a reimplementation:
`experiments/post_training/task_curation/daytona_identity.py` materializes each
task's environment, builds the `MachineSpec` through Harbor's mapping
(`harbor.environments.daytona.machine_spec`), names it with
`shellbox.backends.daytona.machine.snapshot_identity`, and runs
`daytona_execution_failure` per task. Record the resulting count, the ten-snapshot
threshold and the basis on the source; the verdict
(`finestore.rl.curation.daytona_verdict`) is derived from them. Nothing in this
procedure creates or deletes a snapshot; cleanup stays with the operator
procedure above.

Publish the converted cohort to the `open-athena` Hugging Face org as a **private**
repository (`experiments/post_training/task_curation/calibforge_publish.py`), with a
card that states it is private pending release, and read the upload back through the
same login to compare hashes. Keep the Atlas entry pinned to that repository and
revision.

## Source provenance

Transferred from MarinSkyRL revision
`544d5d6f14116a06bde0209352585903133bd618`:

- [CoreWeave operations](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/coreweave.md):
  provider preflight, idle-only authorized cleanup, trial artifacts, and secrets.
- [Jupiter operations](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/jupiter/README.md):
  dedicated RL credentials and legacy environment-variable mapping.
- [RL diagnostics](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/rl-diagnostics.md):
  agentic failure interpretation, progress, concurrency, and timing semantics.

The snapshot-reclamation checklist incorporates the operator's 2026-10-07
instruction to purge every snapshot unused for more than one hour on every
check. This threshold was supplied separately from the source runbooks. The
cancellation incident is additional operational evidence, not cross-account
cleanup authority.

When reconciliation confirms an exact sandbox ID is absent, exclude that ID
from occupied capacity while retaining the original ownership ledger. Each
unbound create request still consumes one slot. A replacement producer must
carry unresolved quarantine records forward and transfer the stopped producer's
reserve claim without releasing it. Verify the old producer and pod are terminal
before the handoff; never replay an ownership-ambiguous row as a new trial.

## Sealing a stopped difficulty ledger

Stop the controller before releasing its serving pools. Derive leftover Daytona
sandbox IDs from the lane's lifecycle logs and reconcile those exact IDs; retain
unbound create requests as unresolved evidence.

Run the maintained `experiments.post_training.task_curation.difficulty.seal`
entry point with the ledger, destination, provenance JSON and receipt path. It
verifies content-addressed partitions, refuses a changing inventory, preserves
parked repair verifications and REPAIRED rows, and reads every sealed object back.
Its final sequence is the number of unique ledger rows ordered by UTC record
time and attempt ID; the content-addressed ledger has no native sequence counter.
The selected snapshot is recorded in `v1/closed.json`, after which appenders fail
closed. Start any new campaign in its own explicitly versioned ledger namespace.

Uncovered cells have null difficulty and pass rate. Observed partial rates remain
separate; repeated scores on one task do not cover another task. Out-of-range
rewards remain in the raw ledger and make the affected cell unmeasured. Parked
holds remain effective even when old scores exist.

Register the verified manifest URI and hash for the next cut in
`release_stages/difficulty_snapshots.json`. The `atlas-release` graph binds that
receipt into catalog-load identity, verifies it in preflight, and publishes its
release-bound cells and seal reference. Changed source populations never inherit
values from the old seal. Do not publish a seal through a per-release script.
