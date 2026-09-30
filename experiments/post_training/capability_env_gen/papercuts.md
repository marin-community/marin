# Shared infrastructure papercuts

Last updated: 2026-09-22 UTC. Ranked by impact on the capability-generation pilot.
Keep at most **10 entries**; replace lower-impact entries as evidence changes.
Remove resolved entries after verifying their fixes. Current count: **10**.

This list includes useful new features, tools and infrastructure, as well as broken
behavior. Pipeline-local bugs are tracked in [the experiment ledger](docs/experiments.md).
Rank additions by the work they would enable or repeated effort they would remove.
No credentials are included in the repros.

## 1. Remote Daytona tasks still require host Docker for executable grading

**Impact:** Blocks code-verifier trials on Dockerless cluster workers even when the
candidate workspace runs successfully in Daytona.

**Evidence:** At TaskCompendium pin `dc6b501`,
`src/taskcompendium/harbor/container.py::grade_in_container` invokes host
`docker run` and `docker rm`. A `ContainerRuntime` script/pytest verifier reaches
that path independently of the candidate environment backend. Simple and native
judge verifiers are unaffected.

**Useful fix:** Make the private verifier's isolation backend configurable, with
a supported remote Daytona implementation. Preserve its separation from candidate
state, credentials and network policy. A local adapter now passes the real
oracle/solver/control/attack protocol in runtime probe 014; the shared integration
still needs a supported implementation.

## 2. Harbor and the maintained Daytona helpers require different SDK versions

**Impact:** Prevents straightforward reuse of Harbor's Daytona adapter and its
cleanup/retry behavior with the existing OpenAthena environment-generation tools.

**Evidence:** Pinned Harbor's `harbor[daytona]` metadata requires
`daytona>=0.203,<0.204`; the maintained `build_envs/envgen/dt.py` workflow pins
`0.200.2`. This is an observed dependency-contract mismatch, not a demonstrated
claim that every operation on either version is broken. In 0.200.2,
`CreateSandboxFromSnapshotParams` also has no resource field: supplying
`resources=Resources(cpu=2, memory=2)` or direct `cpu`/`memory` extras is silently
ignored by its Pydantic model, while the synchronous create path applies resources
only for image-based creation. The supported resize path requires stopping a
non-ephemeral sandbox before decreasing CPU or memory; the maintained task adapter
creates ephemeral sandboxes, which are deleted on stop. Exact installed-source
hashes and the no-network local model probe are recorded in
`docs/audits/daytona_sdk_resources_001.json` (SHA-256
`5eae132b5f87009bf6ffb88abe4ac2bd8c5e832f057c29e330f219d64ff8dd8c`).

**Useful fix:** Publish one tested Harbor/TaskCompendium/Daytona SDK combination,
with a conformance test covering input staging, execution, download and teardown.
A small public-input bootstrap hook would avoid a second full environment adapter.
Expose resource overrides consistently for both image- and snapshot-based creation,
reject unknown resource arguments instead of dropping them, and report the
provider-observed CPU/memory limits in the sandbox receipt.

The c32 revalidation004 passed its runtime suite but still lacked proof of its
declared 2 CPU/2 GB envelope. Its `nproc` and `/proc/meminfo` readings expose host
capacity, not sandbox limits. A supported cgroup-limit/peak-usage receipt would
prevent that ambiguity; requested snapshot resources alone are insufficient.
The local adapter now records startup/final cgroup samples and freezes explicit
candidate resource requests in evaluation plans. Three diagnostic003 oracle
launches exposed a 2-CPU cgroup quota and 2-GiB memory limit, with retained final
memory peaks. Independent solver load, disk enforcement and the full resource
envelope remain unproved; this workaround does not resolve the shared SDK gap.
See [the diagnostic evidence](docs/audits/c32_evaluation_terminal_003.json).

## 3. Federated Iris log reads fail while job control remains healthy

**Impact:** Makes a live or failed job harder to diagnose; independent durable
logging becomes necessary for every new pipeline.

**Observed repro:** From the Marin checkout,
`uv run --frozen iris --cluster=marin job logs /muchanem/cap-proposal-pilot-001 --max-lines 50`
returned `finelog ... DEADLINE_EXCEEDED: log read exceeded deadline of 10000 ms`.
`job describe` succeeded seconds later. The operational notes also document missing
terminal federated logs; that is a separate failure mode worth covering in tests.

**Useful fix:** Bounded/paginated log reads with partial results, a continuation
cursor and a configurable deadline; retain accessible logs after federation jobs
terminate. Current workaround: stream logs and hash-verified snapshots to S3.

**Related live inspection gap:** Bounded `iris task exec` reads against the
260-builder recovery worker intermittently returned `'NoneType' object has no
attribute 'decode'`. Other reads succeeded. Return structured command exit,
timeout and empty-output states so an observation failure cannot be mistaken for
a stopped worker. The task identifier must include `/0`; the job identifier alone
is rejected.

**Related discovery gap:** The c32 launcher resolved a healthy relay E endpoint
but inspected the killed legacy relay job's route logs, reporting `unknown` and
blocking submission. The local fix binds the resolved address to its owning task
in the authoritative endpoint registry before reading that task's routes. A single
discovery/readiness response containing endpoint owner, advertised model routes and
serving-worker counts would remove this repeated cross-system reconciliation.

## 4. CoreWeave S3 configuration secretly depends on the working directory

**Impact:** A correctly provisioned process fails object-store access when launched
from a different project, even with the right environment and credentials.

**Observed repro:** Calling `configure_coreweave_s3()` through
`uv run --project ~/openathena/marin` from this directory raised
`ValueError: no cluster config declares a 'stores' entry for coreweave`.
The same code worked after changing the actual working directory to the Marin
checkout. `--project` chooses the Python environment; it does not change CWD.

**Useful fix:** Accept an explicit cluster-config/root argument or discover config
relative to the installed package. Include the searched paths in the error.
Current workaround: run the object-store helper with the Marin checkout as CWD.

## 5. Daytona snapshot capacity and shared cleanup need a fleet-level interface

**Impact:** Completed sandboxes can be cleaned up while their image snapshots
still consume the account quota, preventing new task environments from starting.

**Evidence:** Consensus probe010 failed before inference with the preserved
provider message: `Snapshot quota exceeded. Maximum allowed: 40`. Separately,
c17 exhausted four sandbox-creation retries on HTTP429, and the independent c22
fixture replay received a creation rate-limit error. These are distinct provider
limits; neither is a task-scoring failure. Terminal evidence is under
`runs/composite-consensus-probe-010/terminal-pull-0610/`.
The live c32 builder also received `Total CPU limit exceeded. Maximum allowed:
6000` at 06:53:52 and 06:57:00 UTC on September 19, interleaved with 429s. Its
retained `review-pull-0710/.../workspace/dt_calls.jsonl` distinguishes these
responses; snapshot headroom alone does not establish sandbox capacity.

**Initial inventory:** A maintained-client inventory recorded 53 snapshots: 13
provider-general and 40 quota-counted, exactly matching the 40/40 usage response,
plus one live sandbox. The evidence is retained at
`runs/daytona-quota-cleanup-001/cleanup-decision.json`. At that checkpoint no deletion was safe: the
known pipeline snapshots are referenced by live construction/continuation receipts
or retained revalidation evidence, while generic provider snapshots have no
verified creation receipt and no provider-visible reference count. No delete API
call was made at that checkpoint. A later unfiltered inventory returned 3,194
sandbox records with no references to our superseded c17 verifier snapshot.
After checking its retained recipe, ownership and obsolete use, we deleted that
one cache and verified absence; usage fell to 39. Its replacement, built from the
current pinned registry base and verifier bootstrap, is ACTIVE. Health002 then
hit four 429 creation failures. The lifecycle receipts remain under the same run
folder; that cleanup deleted no unrelated or active snapshot.

**Later builder incident:** The terminal c05 repair independently reported deleting
23 shared `harbor__*` caches selected by age, without establishing ownership or
active references. This was an unsupported cleanup, not an approved quota fix;
impact remains unknown. The exact names and evidence are retained in
`docs/audits/c05_terminal_infrastructure_006.json`. Future build/repair instructions
explicitly prohibit this selection rule. Provider-enforced per-task ownership and
deletion permissions would prevent a builder from affecting sibling workloads.

**Useful addition:** Expose current snapshot and aggregate CPU quota/headroom,
provisioning limits, and a typed retry-after/capacity response;
support a shared cache keyed by immutable image/build inputs, creator/run labels,
snapshot-to-sandbox references, and cleanup receipts/refcounts. Production runs
with hundreds of distinct environments also need sufficient quota. Unrelated or
active builder snapshots must remain untouched. This replaces the lower-impact
pool-metric attribution entry, retained in the experiment ledger.

**First-version pilot (2026-09-21 22:46 UTC):** The container builder logged
21 `snapshot.budget_wait` cycles with zero run-owned snapshots and organization
usage 39/40. The maintained helper reserves the final slot (`usage < quota - 1`),
so this delay is a client-side headroom policy, not an observed provider rejection.
Expose that reservation explicitly or allow a single small pilot to use the
remaining slot; otherwise available capacity looks like an hour-long build stall.
The running pilot's frozen helper was not modified.
We subsequently removed only the completed Docker-reset probe's temporary snapshot
after checking its creation evidence and zero listed sandbox references. Exact
absence and organization usage 38/40 were verified at 22:52 UTC; the recipe and
cleanup evidence are retained in
`docs/audits/daytona_reset_snapshot_cleanup_001.json`. No shared cache was removed.

With the user's subsequent authorization to remove other unused rebuildable
caches, two further cleanups removed four old c32 image caches and eight generic
OS/runtime caches. Each deletion followed a fresh complete sandbox-reference
check, and recipes and provider identities were retained. At 00:25 UTC on
September 22, all eight latest deletions were verified absent and quota-counted
inventory was 30/40; current pilot snapshots were preserved. See
`docs/audits/daytona_cache_cleanup_002.json` and
`docs/audits/daytona_cache_cleanup_003.json`. A supported cache lease/refcount and
rebuild-aware garbage collector would remove this repeated manual intervention.

## 6. Integrate a validated production image publisher

**Impact:** Custom Daytona builds produce provider snapshots, but generated
TaskSpec images need real immutable OCI references that other workers can pull.
A successful cached sandbox does not establish portable reconstruction.

**Evidence:** Daytona SDK 0.200.2 exposes a tagged provider reference and build
recipe, not an OCI manifest digest or a registry-resolution credential. The c32
builder used a provider-reference suffix as its candidate image digest and a
Dockerfile hash as its verifier image digest. Both passed string validation;
neither proves an OCI image identity. The c05 builder copied a non-portable
example image ID and its live reconstruction then failed to pull that image.
See [the image contract audit](docs/audits/image_identity_contract_001.md).

**Existing foundation:** `build_envs/envgen/MATERIALISATION.md` and
`probes/push_registry.json` retain three successful authenticated OCI pushes,
manifest digests and registry-backed validation. The registry prototype and
`push_oci.py` already address publication; this is an integration and durability
gap. That report says its registry password remained in a prior session's
scratchpad and a Kubernetes Secret, with durable credential management pending.
The user subsequently authorized registry setup. The existing service is now
verified healthy, a dedicated publisher credential is durable in Secret Manager,
and two CoreWeave probes passed authenticated manifest/config/layer digest
readback with anonymous access rejected. Existing accounts were preserved.
See [the registry runbook](docs/image_registry.md). Both c32 generated images have
now passed content review and authenticated digest-only publication, including
layer/config/manifest readback. Receipts are retained under
`runs/c32-publication-002/results/`. Both images also passed two fresh isolated
cold-boot/reset checks per role, recorded in
`docs/audits/c32_cold_pull_execution_003.json`. TaskSpec pointer migration has now
restored successfully and launched fresh candidate/private sandboxes. After fixing
the controller's omitted workspace replay, the authored oracle received 1.0 in
revalidation-002. That run then stopped on a malformed solver tool call; full task
validation remains unresolved. Publication, cold boots and one oracle pass do not
establish independent solver/control/attack success.

**Measured reconstruction gap:** Both exact published c32 digests import into
Daytona, but a `FROM <digest>` recipe loses the inherited entrypoint and boots
`daytona sleep infinity`. A fresh original-source snapshot boots the reviewed
entrypoint and reaches PostgreSQL readiness. The local reconstruction adapter
must restate launch instructions from digest-verified OCI metadata; a supported
provider image-config import would remove this extra mapping. Evidence and
owned-resource cleanup are in
`docs/audits/c32_cold_pull_provider_recovery_001.json`.

**Useful addition:** Harden and integrate the production publisher with that
registry. Keep its registry-wide credential only in trusted publication workers.
Reject truncated/corrupt compressed layers, preserve image configuration, verify
all uploaded bytes by digest readback, and review candidate/private file boundaries.
Return the canonical
`repository@sha256:<manifest>` reference, retain the exact recipe/context and
manifest bytes, and record a separate mapping to Daytona snapshot IDs. Include
cold-pull validation. The same report's rootfs materialisation prototype avoids
per-task snapshot slots, but retains base-image extra files and requires object
storage egress; those differences need explicit task-specific validation before
adoption under our current network-blocked runtime contract.
The builder CLI pinning request previously ranked here is consolidated into #7;
its original evidence remains in the experiment ledger.

## 7. Expose inference budgets and termination metadata across judge and builder paths

**Impact:** Judge runs depend on serving defaults; the task cannot fully record
or reproduce those inference settings.

**Evidence:** Pinned `judging.py::OpenAIJudgeClient.complete` sends only `model`,
`messages` and `temperature`. It exposes neither `max_tokens` nor
`chat_template_kwargs.reasoning_effort`, unlike the agent path. This is verified
by source inspection; a resulting live judge failure has not yet been established.

**Useful fix:** Add typed budget/reasoning settings to `JudgeModelPolicy`, serialize
them into request and provenance, and test the emitted request against the actual
endpoint. The pilot will retain raw calibration outcomes to distinguish transport
failures from judging errors.

**Measured protocol failures:** The c22 native composite calibration snapshot
`d406a95165c14d81a42a11debbb829e8` retains 91 infrastructure-error trials and
eight graded trials in its aborted 24-way campaign. Nested native verdicts include
55 missing score tokens and 32 inline terminal score tokens. The lower-concurrency
campaign also contains both modes. Raw replies survive, but the native client
discards usage and finish reasons, so these are not evidence of token truncation.
The local newline repair addresses the inline case; bounded protocol retries and
complete request/response provenance are also needed. See
[the artifact audit](docs/audits/c22_native_judge_protocol_002.json).

**Related measured builder need:** Local OMP 18.1.14's model registry advertises a
33K output limit and its CLI help exposes no output-token override; the live
worker's 18.2.6 controls still need separate verification. In c22, nine
zero-exit attempts contained 32 responses stopped at exactly 32,768 output tokens,
21 compactions, and no fixture generator or required handoff. See the
[retained audit](docs/audits/c22_continuation_exhaustion_001.json).
A supported model-budget override and structured termination/usage summary would
make recovery easier. Our controller now reads the transcripts, resumes with
incremental artifact writes, and stops after two attempts without real file
progress. The first resumed c22 attempt completed its 30-file fixture stage;
full task validation remains pending. Process exit zero is not task completion.

**Related tool identity:** Bootstrap downloads OMP from `releases/latest` when
missing. Support a pinned release and expected executable hash, record both with
the effective model limits, and reuse that release for continuations unless an
upgrade is separately recorded. The observed 18.1.14 versus 18.2.6 difference does
not establish that the upgrade caused output-limit failures.

**Resolved controller cap:** The adversary's former 32K/64K ceiling was our
configuration, not a serving limit. Live engine configuration exposes a 262,144
token total context window. The controller now requests 131,072 output tokens,
then the remaining context on length exhaustion, retaining both attempts and
their termination metadata. The first c17 retry exhausted both 131,072 and
258,932 reasoning tokens without an answer. Splitting boundary planning from
artifact generation then succeeded: both phases requested a 131,072-token ceiling
and stopped normally after 2,519 and 977 completion tokens. The resulting invalid
candidate was correctly rejected, and c17 passed its then-current gates. That
historical acceptance was subsequently revoked after a duplicate-JSON exploit;
the serving-budget observation does not establish current task acceptance.
This does not resolve the judge-policy or OMP configuration gaps above. See the
[serving evidence](docs/audits/glm_output_budget_metadata_001.json).

**Solver context recovery:** The solver now also retries an explicit context-length
error with the remaining output budget. A remote replay of the exact failed c32
request reproduced its 32K-reservation rejection, then succeeded with unchanged
messages and tools at 229,499 prompt tokens and 522 completion tokens. Both attempts
and the final snapshot are retained; returned tool calls were not executed. This
resolves that reservation failure, while full-prompt exhaustion and turn-cap recovery
remain separate gaps. See [the request replay](docs/audits/context_budget_probe_001.json).

## 8. The maintained Daytona command helper assumes every image has Bash

**Impact:** Minimal immutable images fail during setup and artifact transfer,
despite being valid Daytona environments.

**Observed repro:** `build_envs/envgen/dt.py::run_in_sandbox` wraps commands in
`bash -c`. In `runs/runtime-probe-005`, the fresh network-blocked sandbox using
official Alpine failed with `timeout: can't execute 'bash': No such file or
directory` before any model request. Alpine provides `/bin/sh` instead.

**Useful fix:** Accept an explicit shell or use portable `/bin/sh` by default,
and test setup, execution and downloads on an Alpine image. Our local adapter
now uses a portable executor; the full live control/attack suite passed in
runtime probe 009. The shared helper still needs the upstream fix.

## 9. Add composition of executable checks and native judges

**Working assumption:** Build tasks using code-and-judge composition now; the user
has authorized extending TaskCompendium to support it. This entry tracks the
shared interface and remaining evidence, rather than a request for permission.

**Value:** Three admitted judge proposals need both task-specific machine checks
and rubric scoring. A supported composition interface would preserve their critical
gates and allow more realistic tasks without redesigning them around a restricted
verifier surface.

**Evidence:** At `dc6b501`, `src/taskcompendium/lowering.py:258` rejects multistep
`ALL_REQUIRED_STEPS`, although that policy exists in TaskSpec. Supported `MEAN`
can award credit despite a failed critical check; `FINAL` discards its reward.
Native `judging.py` supports a reference score or equally weighted binary
checklist, with static judge context and no task-specific executable prepass.

**Useful addition:** Support conjunctive/mixed verification in the target adapter,
including explicit aggregation and private machine-generated judge context.
Add tests where the rubric passes but a mandatory executable check fails, and
where an infrastructure error must remain distinct from a zero reward.
Support ordinal rubric items and explicit multi-judge consensus as part of that
interface. At this pin, thirteen 0-3 items encoded as cumulative binary thresholds
require 78 individual model completions for two score vectors, or 117 when a
third is needed. A typed whole-rubric response could preserve anchor consistency
and reduce request overhead; its grading behavior would need calibration. The
current local extension retains separate completion and grading-attempt counts.

**Plan (authorized 2026-09-18):** Build tasks with their intended composed verifier
and modify TaskCompendium to support it. Preserve critical gates, weights and private
machine-generated context; record and test the extension against the pinned source.
This feature gap does not block proposal construction or require native-only
redesign. Final acceptance still requires the composed verifier to run correctly.
The local extension implements gates, weights, penalties and section caps, with
unit coverage. Live probe 006 passed all four integration cases: correct answer
1.0, failed machine gate 0, failed critical judge criterion 0, and weighted
section-cap/penalty case 0.25. Private machine-generated judge context was used.
Live probe 007 also exercised real two-then-third consensus: agreement used four
criterion completions; a genuine disagreement used six and resolved by median.
Those two-criterion protocol cases are not generated-task calibration.
The c02 continuation exposed a further lowering restriction: `FinalState('.')`
was rejected for judge mode before the composed verifier could be installed.
The pinned local extension now authorizes that combination only with an exact
specification/configuration binding and installs its mandatory extension before
returning the package. Remote FinalState probe 003 completed all three cases
passed: valid work earned 1, a failed executable gate earned 0 without calling
the judge, and a failed critical judge criterion earned 0. It used three distinct
candidate sandboxes and three distinct private verifiers. Iris exited successfully
and its final 64-file snapshot was restored with every member hash verified.
This establishes the synthetic protocol, not generated-task calibration or c02
readmission. See [the probe audit](docs/audits/composite_final_state_remote_probe_launch_003.json) and
[the lowering audit](docs/audits/c02_composite_final_state_lowering_001.json).
The shared upstream feature and generated-task calibration still need completion;
the local integration proof is recorded in [the experiment ledger](docs/experiments.md).

**Related useful tool:** Expose capture and replay of the complete private grading
input, including final workspace, protocol and transcript. The earlier diagnostic
replayed authored commands ten times and checked input-byte equality afterward.
Command-generated timestamps or other legitimate variation can prevent a fixed-input
comparison; capturing once would test verifier stability without requiring the task
itself to produce identical submissions. Preserve this interface for composed
verifiers as well as executable ones.

The local Docker SCRIPT diagnostic now captures once and runs ten private graders,
with explicit logical-directory/mode restoration for object-store snapshots. Local
tests pass, and remote c32 probe003 passed all 70 fixed-input grades with confirmed
cleanup. A shared TaskCompendium API and composed-verifier replay remain outstanding.

## 10. Make Iris job inputs explicit and preserve their exact bytes

**Impact:** A valid job cannot launch when another run's retained artifacts push
the checkout over the bundle limit. Diagnosing and managing that shared size budget
adds work to every launcher.

**Observed repro:** The c17 continuation was rejected with workspace bundles of
26.5 and 26.4 MB against the 25 MB limit. An untracked 9,136 KB source archive in
`marin/capability-pipeline-staging/archives/` was included even though the worker
fetches its source from object storage. Excluding that archive directory and
compressing the staged TaskCompendium source produced a successful 17.5 MB bundle.
See the [submission record](runs/synthesis-c17-continuation-002/provenance/submission-provenance.json).

**Useful addition:** Support a per-job explicit file manifest or isolated bundle
root, with a preflight report of the largest included paths and remaining budget.
Keep durable archives outside the submission inventory by construction. Current
workaround uses bounded staging and a local `.git/info/exclude` entry.

The c32 evaluation diagnostic001 exposed the opposite failure: a shared blanket
ignore for `capability-pipeline-staging/submissions/` kept the checkout small but
silently omitted this job's worker directory. Iris accepted the 15.7 MB bundle;
the task then failed at `cd` before any task code or model call. Its Python client
already supports `extra_bundle_includes`; the CLI needs the same explicit include
surface, and the launcher needs to verify required-file membership before submit.
The current fix adds that narrow include plus a preflight for the selected leaf,
without removing the shared ignore rule.

The c02 continuation later exceeded the same limit with a 64.9 MB workspace
containing a necessary 49.3 MB seed archive. Its local transport now uploads the
unchanged archive under a content-addressed S3 key and verifies the download
before member-level restore checks. Iris's bundle fell to 18.2 MB. First-class
digest-bound large inputs would avoid this extra transport layer.
Continuation003 now verifies that this archive restores remotely and reaches a
GLM repair session while preserving the completed builder-session hashes.

**Additional measured failure:** c17 continuation 005 and c05 image-migration
revalidation 006 both stopped before inference because worker-side input inventories
differed from their manifests. The separately retained source archives verified
locally with complete membership (1,296 c17 seed files and the complete c05 bundle).
This establishes a transport/staging mismatch, not its exact filtering or race
mechanism. A first-class immutable per-job input archive, verified after worker
extraction, would avoid depending on checkout discovery. The launcher is being
updated to use that approach and unique stage directories; failed runs remain
recorded as infrastructure outcomes.
