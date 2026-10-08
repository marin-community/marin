# Task curation library

This package converts source records into TaskSpecs, gathers quality and grading
evidence, and produces final filtering decisions. The
[experiment](../../../../../experiments/post_training/task_curation/README.md)
chooses pinned sources and builds the artifact graph.

A `TaskPolicy` combines normalization, a rubric for the GLM review model and optional checks. A
`DatasetRecipe` binds that policy to source inputs and intended use. Start in
[datasets/](../datasets/) when changing how a family is interpreted: keep its
normalization, rubric and source contract together. Reuse a
family across sources with the same contract; add a family when the contract
differs.

Library factories expose `policy()`. The reusable `run_source_pipeline` procedure
consumes an explicit recipe and writes normalized data, analysis, verification
and reports. Experiments owns `pipeline()` declarations and ArtifactStep wrappers,
including source pins, artifact identities, dependencies, outputs and resources.
TaskCompendium does not import experiments or construct ArtifactSteps.

The `pipeline` package owns acquisition, review, filtering and verification.
Dataset converters live in `taskcompendium.datasets`; source-specific grader
images and declarations live in the experiment's `datasets` package. Those
declarations select the grader and runtime requirements recorded in each
TaskSpec, and TaskCompendium grades by grader kind. Common graders and
source-call transport live in VerifyIT. A `ScriptGrader` runs a supplied command
or an evaluator installed in a pinned image; its verifier resources hold the
command's inputs and any invocation bridge it needs. An evaluator that cannot
run here, or a command without a selected image, becomes a `NoGrader` with the
reason and source contract. No executable placeholder grader is created.

Normalization preserves the boundary between the public problem and hidden
answers or fixtures. Checks and GLM review supply separate evidence. Filtering
keeps or rejects tasks with quality evidence and defers tasks whose review is
unavailable or malformed. Deferred rows remain in the audit, count separately
from task defects, and are excluded from accepted views until reviewed. The audit
retains reasons, confidence and cleanup history.
Optional rewriting creates a candidate that is reviewed again, with its original
evidence retained. Controls run per row unless the source opts into output
sampling. Quality acceptance and grader readiness are separate. Executable export
can rely on individual control results or source-sample evidence; an unsampled
grader has not individually passed controls.

Use [stages.py](stages.py) to follow preparation, sampled quality assessment,
conditional remaining review, filtering, rewriting and merging. Zephyr owns
sharding and retry behavior. GLM completions use an exact-query
cache keyed by the complete request and declared model revision, so unrelated
catalog changes need not repeat inference. Keep artifact construction and
inference-client binding in the experiment.

Audit stages assign each row to a normalization shard using its locator hash,
then deduplicate and pack model requests into batches before distributing review
work. Zephyr's `reshard` moves whole storage chunks, so it cannot spread a small
source stored in one chunk across normalization workers. Manifest rejection
counts use stable reason codes; diagnostics remain in each audit row's
`normalization_detail` field.

Prepared review batches are saved before inference. Normalization, deduplication,
and batch preparation use the full worker resource budget. Deduplication persists
review-input windows before returning filenames, with both `review_batch_size`
and `review_input_bytes` limits (the latter is 64 MiB by default). An oversized
individual audit is preserved alone. Normalized export uses a byte-budgeted
shuffle; moving whole pickle chunks with `reshard` would not bound task payloads.
The separate review
execution uses `review_task_resources`, allowing several model requests to wait
on one worker without applying that smaller budget to shuffle work. Completed
review shards are reused when a source audit resumes.

[source_quality.py](source_quality.py)'s lower-level `assess_source_quality` API
selects 100 eligible unique tasks after conversion, cheap checks and exact
deduplication, or all eligible tasks when the
source is smaller. More than 90% known good judgments skips review of the
remainder; more than 50% known defects rejects the eligible source population.
Both fractions use the entire panel as their denominator. Unresolved responses
count as neither good nor defective. Without decisive evidence, missing or
invalid responses leave the gate `incomplete`; semantic uncertainty or
intermediate quality requires `full_review`. These thresholds are a heuristic.
The canonical `run_source_pipeline` instead draws at most 100 raw records before
conversion and uses that raw panel as the quality denominator.

Unsampled trusted records use `quality_basis="inferred_from_source"`; the gate
never invents per-task reviews. Unresolved sampled tasks remain deferred even
for trusted sources. Sampled judgments are preserved and reused, and
selected tasks are repacked into full inference batches. Output verification
remains independent of quality acceptance.
Preparation retains runtime traces under `checks/<task-id-hash>/attempt-*.json`.

[contract_audit.py](contract_audit.py) accounts for every archive row when no
faithful converter is bound. It retains immutable source locators, archive/file
hashes and decode coverage, emits deferred audit rows with no TaskSpec, and uses
no inference. Archives exceeding inspection limits remain explicit coverage gaps.

Inference uploads have both request-count and serialized UTF-8 byte limits
(64 requests and 4 MiB by default). Larger groups split into separate provider
batches; an oversized single request is deferred with its size diagnostic.
Transient HTTP failures use bounded retries with backoff. Request, submission
and response files retain provider evidence. Review retries default to three
attempts per run. Failed requests leave unavailable review records and do not
count as source defects.

After completed review artifacts are checked, cache lookups match complete
requests, including model, messages, generation options and request IDs, plus the
declared model revision. A completion envelope stores that identity with the raw
response. Missing, invalid or unreadable cache entries issue fresh requests.
Cache write failures leave the review usable. A response lost before caching
can cause repeated inference. An image change can invalidate a review artifact while
leaving this exact-request cache reusable.
FineStore supplies batched cache reads. TaskCompendium validates each stored
request identity and response before reuse.

`RecordedReviewer` can supply separately recorded manual judgments from a
bundle with `schema_version="recorded-review-v1"`, verified against the SHA256
of its complete file. It carries root `provenance` and `records`, each containing
`task_sha256`, `rubric_sha256`, a typed `review`, and record `provenance`.
Each record must match the complete TaskSpec and rubric hashes, including
resources and runtime metadata. Unmatched tasks and rewrite comparisons use the
original reviewer. Matched judgments retain confidence, reference uncertainty,
inspection limits, and reviewer/source provenance in `review_detail` and
`recorded-reviews.json`; they do not create provider completions. Existing quality
thresholds apply, independently of execution
verification. Counters `review/recorded/matched_records` and
`review/recorded/fallback_records` report how records were reviewed.

Zephyr counters report cache hits/misses, cache failures, submitted requests and
bytes, transport failures, review outcomes, and time spent in provider calls.
Cache counters report descriptor lookup count and time, selected shards, payload
read time and bytes returned under `review/cache/`, using FineStore's batched read
diagnostics. Older runs emitted index-refresh counters that the current API no
longer provides; their absence does not establish zero work.
Live worker counters reach telemetry before a shard completes. These are work
observations, including retries; source manifests provide final row counts.
Accumulated provider time includes concurrent calls and is not elapsed stage time.
Preparation also reports check outcomes and time; sampled verification reports
attempts, control outcomes (including skipped goldens), trial outcomes and time.
Verification also records executed and reused attempts globally and under
`verification/suite/<suite.id>/`, with `attempt_seconds` counting executed work.
Review evidence persistence has a separate timer. Finelog exports Zephyr's
aggregated counter snapshots as gauges: take the latest value per full series
(emitting cluster/job, execution ID, metric and attributes) rather than summing
repeated snapshots. Join
the execution IDs in each source's `telemetry.json` to Zephyr stage rows by
execution ID and emitting cluster/job. The sidecar retains separate final counter
dictionaries for every execution, including manifest-count operations, and wall
times for logical phases and the source invocation. `run_source_pipeline` requires
an explicit `canonical_source` keyword; experiments supply the binding key. The
sidecar contains no task payloads.
Do not sum peaks or averages across executions, add phase wall time to its nested
execution times, or confuse concurrent worker durations with campaign wall time.
The sidecar describes the latest invocation; failures retain completed phases and
the exception class. Failed executions have no ID or final counters because the
execution API did not return them; exclude blank IDs from joins. Telemetry status
indicates whether the invocation returned, independently of source quality and
grader readiness. This attribution does not change inference request identities.

GLM sees resource paths, roles, SHA-256 hashes, byte counts and UTF-8 previews.
Up to 256 files share 32,768 preview characters: each text file first receives up
to 100 characters, then earlier previews expand to at most 8,192 characters each.
Binary files retain signatures without text. An aggregate hash covers omitted
files. This describes TaskSpec resources; installed container files require a
separate environment inventory, and previews cannot establish unseen contents.

For grading changes, start with the grader kinds in [models.py](../models.py) and
the packages in [grader.py](../grader.py): a task carries a `VerifyitGrader` with
a standard verifyit mode, or a `ScriptGrader` with verifier resources. Put
generic reusable verification components in [VerifyIT](../../../../../lib/verifyit); keep
dataset-specific behavior in the emitted grader. Runtime requirements must remain
explicit in the TaskSpec.

Source normalizers declare `EnvironmentRequirements.compatible_backends` on
each TaskSpec. Use the existing `shellbox.machine.Backend` values. The agent's
environment and `grader.environment` have separate declarations: a task may
permit ShellSim for shell interaction while its grader needs gVisor. An empty
tuple declares no Shellbox backend; it does not mean every backend is allowed.
Answer-only tasks need no machine declaration.

Use this rubric when authoring a source:

| Contract | Backend declaration |
| --- | --- |
| Built-in shell commands and virtual files preserve the requested behavior | ShellSim may be declared after checking command semantics, paths, quoting, pipes, and required file metadata. |
| Arbitrary Python packages, native binaries, or image-specific dependencies | Use an image-backed backend and pin the required image. ShellSim cannot satisfy a required Docker image. |
| Kernel behavior, subprocesses, networking, or special filesystem behavior | Check the specific Shellbox backend's support. A generic shell capability is insufficient. |
| Grader has different dependencies from the task | Declare compatibility in the grader's environment; preserve grader fixtures across backend choices. |
| Records within a source differ in runtime needs | Emit the correct declaration for each record; do not broaden compatibility to fit the launch. |

Compatibility is an author claim. Verification records evidence for the selected
backend, image and controls; a passing sample on gVisor does not certify ShellSim.
The launch selects an available declared backend. Shellbox still validates its
image source, network policy and host prerequisites, and unsupported choices fail
without fallback. An unavailable compatible runtime is an infrastructure problem,
not proof that the source is defective.

The shared shell environment and `grade_task` enforce the same declarations for
oracle controls and actor episodes. The oracle receives private solution files,
while the actor receives only public files. Both submit the same
declared output paths to the grader. Production executable RL integration remains
separate work: the current TaskSpec Harbor exporter supports direct chat only.

Optional source verification samples up to 100 accepted outputs after filtering
and repair. It uses seeded task-ID hashes across all shards and runs available
controls twice in fresh environments. Missing golden controls are recorded as `skipped`; available
controls still run. The pass fraction is the fraction of tasks with at least one
executed check whose available controls pass on every attempt. Skipped-only tasks
do not count as passes or failures. If all controls are skipped, the source decision is
`skipped` and accepted tasks remain unverified. Unsupported runtimes, infrastructure
errors and an empty sample produce an inconclusive source.
Rejected and inconclusive sources retain their full audit but
publish no accepted rows. Inconclusive checks defer eligible rows; existing review
deferrals and quality rejections keep their dispositions. Failed controls reject
the affected tasks even when other checks are inconclusive.
Known failing sampled tasks are excluded even when the
source's aggregate pass fraction is sufficient. Sampled tasks with skipped
controls remain unverified. Unsampled tasks receive `source_sampled` readiness
only when the passing sample has no skipped controls; otherwise they remain
unverified. Unverified tasks can enter accepted/train/eval views, but cannot
enter executable export. Reports record checked and skipped task counts; these
counts overlap when a task has both executed and skipped checks.

TaskTrove oracle controls run `solution/solve.sh`. Converters
preserve source scripts or create wrappers around supplied solutions; answer
converters may create a script that writes the known reference to the declared
output file. These files are stored in TaskSpec's private `resources.oracle`
group. The check suite runs the script through Shellbox and grades the declared
output files it produces. VerifyIT does not generate a solution during
verification. A missing script skips only the oracle control.

When a source opts into this stage, all its check-suite controls move from audit
and repair to the output sample. Unsampled tasks skip those controls. Changing
sample policy reuses the preceding conversion and review
artifacts. `verification.json` records sample membership, attempts, control
results, source provenance and the final source decision.

One trial executes the entire control suite at one ordinal; the two ordinals
originally run independently in fresh environments. Reruns can reuse complete
trials from retained verification reports without counting an execution twice.
Identity covers the entire task, suite configuration, grading code,
installed dependencies, immutable runtime image and verification policy. Each
trial keeps its original execution ID, checks, rollouts and report provenance.
Infrastructure or unsupported controls cause that entire trial to rerun,
including mixed FAIL/INFRA trials. Superseded trials retain their checks, rollouts
and provenance. Resolved historical infrastructure errors stop blocking readiness;
definite failures survive successful retries. A task is inconsistent when any
current trial passes alongside a current or historical failure. Failures remain
evidence for the source gate; skipped goldens remain skipped.
Prior failures outside the current sample reject only rows with
the same task ID and entire TaskSpec digest; current sample counts remain
independent. Converter-rejected rows without a TaskSpec retain their own rejection.
QEMU reuse requires an immutable worker image pin identifying the packaged
bundle. Online or network-enabled graders always execute fresh because their
remote deployment is not immutable. Code identity conservatively hashes all
TaskCompendium, VerifyIT and Shellbox Python files, so unrelated code changes can
invalidate reuse. Explicit previous reports are read-only inputs, and final
sample membership and the source gate are recomputed.
Identity includes sample size, seed, attempt count and `minimum_pass_fraction`;
changing any invalidates reuse. The driver requires 95% passing checked tasks.
Worker and shard counts do not enter trial identity unless included in suite
parameters; source-artifact admission has its own stricter matching requirements.

See the [task curation reference](../../../../../docs/references/task-curation.md)
for the full stage contracts, review policy, cache identities and output schema.
