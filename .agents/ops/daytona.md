# Daytona operations for agentic RL

Use this runbook when Harbor trials fail during sandbox setup, snapshot quota
prevents a launch, or completed trials leave provider capacity occupied. Iris
job operations remain in [Iris OPS](../../lib/iris/OPS.md).

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
permission failure, not a model failure or a zero reward.

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

Report presence and successful access only; never print key values. Do not copy
private secret files into launch records or source control. The legacy
OpenThoughts-Agent Jupiter launcher also uses `DC_AGENT_SECRET_ENV` to identify
its private secrets file. That is a launcher-specific setting, not an Iris
requirement. See the source Jupiter runbook below for that workflow.

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

Snapshot metadata, snapshot creation/listing and named sandbox creation retry
HTTP 5xx, HTTP 408/429 and SDK connection/timeouts. Retry delays are capped at
120 seconds. Retries continue until success or caller cancellation/deadline.
The provider's HTTP 400 "failed to resolve container IP after 3 attempts"
startup state also retries. Other 4xx errors and terminal image-build failures
propagate. Deterministic
snapshot names and caller-supplied sandbox names identify resources whose
creation succeeded despite a lost response.

Each retry emits `DAYTONA_PROVIDER_RETRY` with operation, attempt, status and
delay fields. Count these events separately from task outcomes. Provider errors
remain infrastructure evidence and receive no oracle score. Command
execution is not replayed after an ambiguous response.

Harbor's Daytona adapter selects `ProviderWaitPolicy.JOB`: safe harness controls,
artifact reads/writes and resource probes retry within the enclosing job's
overall timeout. Safe in-sandbox controls, artifact calls and named sandbox deletion
have a 60-second per-call timeout. Snapshot operations and sandbox acquisition
retain their SDK operation settings within the enclosing job bound. Request duration
and retry backoff pause Harbor phase clocks and Shellbox command clocks; the declared
source-command limits remain active during source execution. Run qualification
under a bounded job whose overall clock continues advancing during provider
waits, and retain retry events separately from task verdicts.
Direct Shellbox callers retain phase-bound behavior unless they explicitly select
the job policy.

Retryable controls include read-only metadata/resource probes, private staging-file
operations, fixed-content artifact transfers and named sandbox deletion.
Source execution remains single-shot after an ambiguous response. An ambiguous
process-group termination falls back to sandbox teardown; it cannot safely be
replayed after PID reuse. Named sandbox deletion and private-file cleanup are
idempotent. External cancellation disables recovery retries during cancellation
cleanup and retains its existing cleanup deadline and primary exception.

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
