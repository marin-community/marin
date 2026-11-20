# Daytona operations for agentic RL

Use this runbook when task attempts fail during sandbox setup, snapshot quota
prevents a launch, or completed attempts leave provider capacity occupied. Iris
job operations remain in [Iris OPS](../../lib/iris/OPS.md).

## Credentials and account selection

Agentic RL uses the dedicated `DAYTONA_RL_API_KEY`. A generic Daytona key can
authenticate successfully against the wrong account. Do not substitute it for
the RL key. Resolve the credential through the launcher's supported secret
indirection and verify that the coordinator and rollout workers receive it.

Checkpoint evaluation uses the Daytona **eval** account for declarative image
builds. The evaluation TaskSpecs select the `daytona-eval` backend and training
tasks select `daytona`. Keep both credentials separately named through the
coordinator and task launcher. A `DaytonaValidationError` about declarative
builds identifies an account or build permission failure.

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

The Marin RL coordinator allowlist forwards both names. Under
`environment.task_sessions.daytona`, set `api_key_env: DAYTONA_RL_API_KEY`,
`eval_api_key_env: DAYTONA_EVAL_API_KEY`, and `eval_image_build: declarative`.
The task launcher reads those fields and rejects missing credentials. These
paths are relative to `skyrl` in the generated launch config. Shellbox fixes the
provider endpoint and target for both accounts.

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

## Pi HTTP idle timeout

Pi's HTTP idle timeout is separate from Harbor task and agent deadlines and
provider execution deadlines. For a prepared policy that already selects Pi,
pass the optional `http_idle_timeout` agent kwarg in seconds:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  harbor run --config prepared-policy.yaml \
  --agent-kwarg http_idle_timeout=360
```

Harbor writes Pi's existing `httpIdleTimeoutMs` setting in milliseconds and
preserves other settings, including compaction. Omit the kwarg to keep the
image's setting; the pinned Pi 0.87.0 defaults to five minutes. This override
does not extend Harbor or provider execution deadlines.

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

uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
  --api-key-env DAYTONA_RL_API_KEY inventory
```

Pass `--api-key-env DAYTONA_EVAL_API_KEY` to inventory or purge the eval account
used for validation builds.

Read the account's current quota and inventory before submission. Snapshot quota
and live sandbox capacity are separate limits. A successful GPU allocation proves
neither is available.

`shellbox-daytona-usage` reads every page of the sandbox inventory and reports
organization headroom. It uses the account-owner limits from
`experiments/active/DAYTONA-QUOTAS.md` (2026-11-19); it does not call the
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

The report counts every state and sums allocations for `started`, `starting`
and `creating` sandboxes. For each resource, `remaining` is `max(0, quota - used)`,
and `available` is `max(0, quota * (1 - headroom) - used)`. The default
`--headroom 0.2` reserves 20% of the total quota. `max_concurrency.limit` is the
minimum of `floor(available / per_sandbox_request)` across CPU, RAM and disk.
A profile above any per-sandbox maximum gets a zero limit.

JSON includes `organization_id`, `observed_at`, `usage_source`, `quota_source`,
`org_limits`, `headroom_fraction`, `charged_states`, `resources`, and
`sandboxes` (`total`, `started`, `charged`, `by_state`). A supplied profile adds
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
   For SkyRL, use the selected TaskRolloutSpecs and the deployment's
   `environment.task_sessions.task_spec` bindings to determine the effective
   image and resource requests. Use `--specs -` on the next commands to read stdin.
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

Shellbox creates each sandbox with a 15-minute auto-stop and deletion on stop,
and an open machine renews the sandbox's activity every five minutes. Daytona
does not count a running command as activity. A sandbox stops only when its
owning process stops renewing it, and Daytona then deletes it. Many stopped
`shellbox-` sandboxes that each lived 15 to 20 minutes mean the renewal is
missing. Long commands then fail mid-trial with `failed to resolve container IP
... Is the Sandbox started?`. A provider failure on a sandbox that is no longer
started surfaces as `MachineTerminated`. So does a timed-out command that Shellbox
cannot stop or clean up, because Shellbox then closes the machine. A lost
provider connection, or a 502, 503 or 504 from Daytona's proxy for a started
sandbox, surfaces as `ConnectionError`. A sandbox whose disk is too full to
start a command surfaces as `MachineDiskFull`. Curated evaluations record these
errors as retryable infrastructure.

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

SkyRL writes agentic training attempt receipts under `task-health/` in the run
export. These record setup and verifier outcomes and bounded diagnostics. The
standalone Harbor evaluator and older runs store trial artifacts under `trace_jobs/`.

Inspect actual rewards, verifier results, stop reasons, and exceptions. All-zero
rewards from the first step can reflect sandbox, data, verifier, or weight-sync
failure. Very short or empty attempts suggest a broken request path or agent
loop. Incoherent output following weight synchronization requires correlation
with reload logs and importance-ratio metrics before concluding learning failed.

Fresh inference heartbeats, trace counts, and a running controller job can
coexist with a stalled trainer. Require advancing trainer phases/steps and
durable artifacts for training progress. Report dispatched, completed, scored,
correct, retained, and accepted-for-training counts separately.

Measure sandbox demand across all concurrent workers and replicas. SkyRL divides
`trajectory_runner.max_concurrent_training_machine_tasks` among its rollout
workers. Compare demand with model-serving capacity (`max_num_seqs` times
inference-engine count), then inspect scheduler queues, KV-cache use, and request flow. Before
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
3. **Keep the archive's environment in the converted task.** A converted task
   runs in the environment its archive specifies:
   `environment_fidelity.environment_unused` rejects a spec that replaces the
   archive's recipe with another image, and a replayed recipe must lower onto the
   digest-pinned base with the archive's steps, working directory and variables.
   A conversion that changes the environment is a rewrite, so it ships as a new
   dataset version whose declaration records the basis.
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
