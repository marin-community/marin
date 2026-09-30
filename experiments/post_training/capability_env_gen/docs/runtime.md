# Harbor runtime gate and pilot procedure

The built-in runner is `capability_pipeline/runtime.py`. Synthesis invokes it
inside the exact TaskCompendium `uv` environment with the pinned `harbor` extra.
It explicitly unsets the inherited `VIRTUAL_ENV` and selects an isolated runtime
environment; schema/lowering commands use a separate core environment. Neither
may synchronize the Iris/Marin environment used by the uploader.
It runs every control through `taskcompendium.harbor.runner.run_trial`. Each
declared positive first executes the builder's authored reference with
`ReplayAgent`, proving that the frozen grader accepts its oracle. A separate
GLM-5.3 solver then sees only rendered instructions and public input files.
ShellSim and Docker solvers run through Harbor's interactive `ShellToolAgent`,
so the model observes each command result before choosing its next action. A
solver gets two fresh attempts by default; every attempt is retained, and a miss
stops at adjudication instead of causing the task to be simplified. Fixed
negative and malformed candidates run through `ReplayAgent`. They establish
regression controls but are not represented as an independent adversarial audit.
Every oracle, solver attempt, control, and attack gets a fresh environment.
Summary evidence is accepted only when the raw trial results and full package
tree hashes match.

No-tool tasks work with the pinned Harbor install alone. ShellSim additionally
requires the bridge built from Marin commit
`dc6b501c8604bcd2e3c20c1e9947679845fdfef8` at
`lib/taskcompendium/shellsim-bridge`; set `TASKCOMPENDIUM_SHELLSIM_BRIDGE` to the
resulting executable. The bridge source pins ShellSim commit
`5674a9492c35ffe390a0d23b49c0a340b12beb30`. Docker-bound trials use
`DaytonaHarborEnvironment` with the maintained Daytona 0.200.2 helpers. It
creates a fresh ephemeral sandbox from the immutable image with
`network_block_all=True`, runs Harbor agent and verifier filesystem operations
through that sandbox, deletes it after the trial, and writes a hashed provider
record containing the sandbox ID, image, snapshot, SDK version, and enforced
network setting. Synthesis verifies distinct sandbox IDs across cases. Until a
credentialed trial produces these raw records, a Docker task remains pending.
Credentialed probe 008 caught a real regression: its boundary attack
case-flipped the public token and earned 1 because the probe had inherited
TaskTrove's case-insensitive exact-match default. Probe 009 then exercised the
explicit case-sensitive verifier and retained that case flip as a fixed
regression. Its independent solver earned 1; the malformed control produced the
declared extraction error; three fixed wrong controls and the independent
injection, shortcut, and boundary attacks all earned 0. All eight trials used
distinct `network_block_all` Daytona sandboxes. The runtime evidence SHA-256 is
`62d264d39d45bf3fdbefd609499a042fc146457f7443d305bf0f5e475c578eda`,
the solver transcript SHA-256 is
`1118a7ba0f36373922d51cf5d9089579537e1f6d62ce28f0173b712de06144e3`,
and the passing independent-adversary SHA-256 is
`eb5056bd128d1c4e34ce36624dbcc68b789b1b58b18a889da674747cdccc3965`.
This proves the simple Docker transport and exact-verifier protocol under the
pre-oracle runner. Current admission additionally requires the distinct authored
oracle proof described above. Executable private verifier transport and generated
task semantics retain their own gates.

Executable `ContainerRuntime` verifiers use the trusted
`DaytonaSemanticVerifier` override. It retains TaskCompendium evidence selection,
submission validation, `container_entry.py`, semantic grading, and outcome
mapping while replacing the upstream host `docker run` transport with a separate
fresh Daytona sandbox. That sandbox receives the candidate snapshot and pinned
TaskCompendium/TaskTrove source, blocks network, runs the upstream supervisor,
keeps `/input`, `/result`, and `/tests` root-only, makes every mounted submission
and external directory read-only, removes setuid/setgid elevation paths, applies
the upstream verifier's two-CPU/one-GB resource request, records its sandbox,
snapshot, adapter, and bootstrap hashes in the private result, and is deleted in
`finally`. The snapshot build installs the pinned
supervisor dependencies at the exact versions in the staged TaskCompendium
`uv.lock` on the declared immutable Python image.

Snapshot reuse requires the provider's stored Dockerfile to match the requested
recipe exactly; a matching cache name alone is insufficient. CPU/memory requests
on creation do not prove the limits of an existing cache. Task resource acceptance
requires observed limits, including explicit resizing where the SDK cannot apply
overrides to snapshot-based creation.
Mode-specific tools such as pytest remain the task author's responsibility in
that image. Images without the declared supervisor Python and package installer
fail as infrastructure.

TaskTrove `mode=judge` runs through the pinned native `SemanticVerifier` and
`OpenAIJudgeClient`. Runtime injects only the name `GLM_API_TOKEN`; no credential
is stored in TaskSpec. The task's explicit `JudgeConfig` must match the staged
credential-free `judge-policy.json` provider, model, and normalized `/v1` base
URL. Judge tasks also need private calibration
fixtures with at least 40 model-graded semantic variant groups in each class and
repeated positive/negative controls. Synthesis runs the hash-bound native
calibration immediately after lowering; a missing, failed, or mismatched report
stops at `pending_judge_calibration` before Harbor controls or export.

Tasks that require deterministic gates or measures around a native judge use
`taskcompendium-composite-verifier-v1`. Each step declares pinned private script
checks as gates, weighted criteria, or penalties, plus native judge criterion
weights, must-pass criterion indices, and optional machine-triggered caps on
specific judge sections. Gates run first in fresh network-blocked Daytona
sandboxes; a failed gate returns a graded zero without calling the judge.
Otherwise the pinned native checklist judge sees the declared candidate files,
transcript, original private reference context, and trusted machine results.
The final reward preserves the declared weights, subtracts normalized penalties,
applies section caps, and forces zero on failed gates or critical criteria.
Infrastructure and extraction failures remain ungraded with null reward.

The composite configuration binds the raw TaskSpec, adapter, aggregation policy,
and exact base revision by SHA-256. Lowering stores the valid semantic record as
`composite-specification.json`, replaces the ordinary Harbor
`specification.json` with an explicitly unsupported extension sentinel, and adds
one mandatory `required_extensions` manifest record. Thus an unpatched
TaskCompendium consumer fails decoding instead of silently using native mean
reward. The maintained source overlay makes native `SemanticVerifier` reject
mandatory extensions and changes the Harbor runner's package decode boundary.
That runner loads `composite-specification.json` only after verifying the exact
manifest marker, runner hash, and configured composite-verifier import path.
Only `CompositeSemanticVerifier`, running against that hash-pinned overlay, may
read the preserved specification and grade the package.

For a cluster launch, stage the source from the already-fetched Marin object;
do not fetch a mutable branch on the worker:

```bash
git -C "$MARIN" archive dc6b501c8604bcd2e3c20c1e9947679845fdfef8 \
  lib/taskcompendium | tar -x -C "$STAGE/taskcompendium-source" --strip-components=2
export TASKCOMPENDIUM_SOURCE="$STAGE/taskcompendium-source"
```

Verify every path in `vendor/task_spec/source.lock.json` before submission. The
archive needs the whole `lib/taskcompendium` tree, including `pyproject.toml`, `uv.lock`,
`src/`, `schema/`, `shellsim-bridge/`, and Harbor support files. The worker also
needs `uv`, `cargo` for the trusted bridge build, OMP configured for
`glm-orion/glm-5.3`, `CAPABILITY_OMP_CONFIG`, `PARALLEL_API_KEY`, and the existing
GLM relay credentials. Daytona stages additionally need the five maintained
helpers, Daytona SDK 0.200.2, and `DAYTONA_API_KEY`.

On macOS, build source archives with `COPYFILE_DISABLE=1`, disable xattrs, and
reject every raw `._*`/`.DS_Store` member. Inspect archive members with Python's
`tarfile`, extract to a fresh directory, and run the complete file-set verifier;
checking only the 81 expected member hashes does not detect AppleDouble extras.

Daytona sandbox provisioning retries only structured rate-limit failures. The
adapter makes at most four create attempts, honors `Retry-After` up to 60
seconds, otherwise waits 5, 10, then 20 seconds, and records every attempt.
Exhaustion remains an infrastructure error with null reward. Candidate commands,
verifier execution, model judgment, and semantic grades are never retried by
this policy.

Run the first pilot with one accepted no-tool, ShellSim, and Docker task:

```bash
capability-pipeline synthesize \
  --accepted inputs/source/accepted.json \
  --out results --limit 3 --concurrency 3 \
  --taskcompendium-source "$TASKCOMPENDIUM_SOURCE"
```

Accept the runtime pilot only when all three items have passed the runtime gates
and every `runtime-evidence.json` references readable raw trial and isolation
artifacts. Runtime validation is not publication: admitted construction tasks
also need their exact hash-bound build checklist and an independent post-build
semantic review before export. The
fixed control pass stops at `runtime_controls_passed_pending_adversary`; a
separate independent attack artifact is required before export. A missing
bridge, Daytona provider record, solver failure, verifier infrastructure error,
or absent independent adversarial result is a pending/failing pilot.

After runtime validation, synthesis freezes the complete task, Harbor bundle,
accepted contract, build checklist, and raw evidence into a new immutable
quality-review attempt outside the item tree. A fresh independent OMP context
must cite the frozen file hashes, cover every semantic axis and admitted build
condition, and return `accept`. Runtime success is retained separately as
`runtime_validated`; a repair, rejection, incomplete receipt, or missing evidence
stops at `pending_quality_review`. Only a task with both gates becomes
`quality_accepted` and is copied to the export directory.

Before lowering, synthesis also compares the final public binding with the
admitted solver surface: reasoning maps to no environment and no tools, ShellSim
maps to ShellSim tools, and container maps to Docker tools. A private code or
composite verifier never authorizes extra solver tools. Any change of this
surface requires a new hash-bound admission rather than a builder-authored
fallback.

An independent attack can legitimately earn partial or full credit. In that
case the original report remains `needs_adjudication` and all raw trial,
per-step, private-verifier, and isolation checks still run. A fresh independent
review may resolve only the reward alarm through a frozen, hash-bound sidecar.
The controller revalidates that sidecar against unchanged task and runtime bytes
and includes it in the later semantic-review packet; it never edits the original
reward, report, or runtime attestation. The review is single shot for an
immutable packet: an exploit, uncertain disposition, or incomplete receipt is
reused on resume and enters bounded construction repair. Only changed task or
runtime evidence permits a new adjudication, while every prior attempt remains
retained.
