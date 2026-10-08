# Task curation library

This package turns the rows of one pinned source into TaskSpecs, reviews and
verifies them, and writes the filtered tasks with a manifest. The
[experiment](../../../../../experiments/post_training/task_curation/README.md)
owns the dataset declarations, the catalog and the ArtifactSteps; TaskCompendium
does not import experiments or construct ArtifactSteps.

## The recipe

`run_source_pipeline` consumes a `SourceRecipe` ([models.py](models.py)):

| Field | Meaning |
| --- | --- |
| `name`, `version` | Source name and converter revision, recorded on every row's `Source`. |
| `source` | `SourceFiles`: the staged dataset, revision, file patterns, format, and optional `select`, `decode` and `read` callables, or `parts` to produce each file's rows, of a known count, in parallel parts. |
| `convert` | `RawRow -> TaskSpec | NormalizedTask | ImportRejection`. The converter fixes the grader. |
| `rubric` | A `ReviewRubric` for model review, or `None` to skip review. |
| `controls` | `Controls(golden, memory_mb)` for grader verification, or `None` to skip it. |
| `intended_use` | `train` or `eval`. |
| `inputs` | Auxiliary staged inputs, passed by name to the source callables. |

Generic conversion techniques live in [convert/](../convert/): answer tasks
graded in process by verifyit, TaskTrove archive decoding, image-installed
source scorers, executable tasks and Nemotron Ultra row decoding. A converter
returns `NormalizedTask` when it records changes to the source (for example a
prompt rewrite), and `ImportRejection` with a stable reason code for rows it
cannot convert.

## Stages

[source_processing.py](source_processing.py) runs these stages; the
[telemetry](#telemetry) section lists the phase and Zephyr execution of each:

1. **Raw sample.** Read the staged files, record every row locator under
   `download/locators/`, and draw at most 100 raw rows. In sample mode, a source
   read in `parts` draws its row indices from the parts' row counts and produces
   only the drawn rows; its ledger has no `raw_input_sha256` for the others.
2. **Normalize and prepare.** Convert the sample and run cheap structural checks
   (`verify_task`): resource layout, answer-format compatibility, and
   reference/wrong-answer checks for in-process numeric, MCQ, exact and action
   graders. The driver deduplicates the checked panel and saves its review
   batches.
3. **Review.** With a rubric, the GLM reviewer judges the sample. These reviews
   are the only model reviews of the source: more than 50% known defects reject
   it, and otherwise it is accepted. The fraction uses the whole panel as the
   denominator; uncertain judgments, unsupported conversions and duplicate rows
   are not defects. Unavailable reviews leave the source `incomplete` only when
   they could still push the defects above 50%; a rerun resumes the same sample.
   Without a rubric, `skip_source_review` keeps every converted row with
   `quality_basis="unreviewed"` and rejects the source only when converter
   defects and failed checks exceed the rejection threshold.
4. **Full expansion.** In full mode an accepted source converts and audits every
   row. Unsampled rows of an accepted source use
   `quality_basis="inferred_from_source"`.
5. **Filter.** Rows are kept, rejected or deferred from quality evidence.
   Deferred rows (unavailable or malformed review) are excluded from `final/`
   and counted separately from defects; the manifest records them as
   `unavailable_reviews`. They make the source `incomplete` only when resolving
   them could change the quality decision.
6. **Verify.** With controls, [source_verification.py](source_verification.py)
   samples kept rows and grades one control submission per task through the
   same grading path rollouts use ([controls.py](controls.py)):
   - `golden(task)`, a known-correct `Reply`, `WorkspaceFiles` or
     `OracleCommand`, which must score 1;
   - otherwise, when the declaration has no `golden` or it returns `None`, an
     empty submission, which shows that the grader runs. A sandbox grader runs
     on it in a fresh machine of its image, with an empty answer file where it
     reads one, a conversation ending in an empty reply, and an empty
     workspace. The control passes when the grader scores it 0 or rejects it.
     A positive reward records a `defect`: an empty reply satisfies the task.
     The row is rejected with `check:empty`, and because the grader ran, the
     trial counts as checked and passed for the source.

   An `OracleCommand` runs with the task's worker and oracle resources
   installed in a fresh machine of the task's agent image, whose tools and
   directories it expects, or of the grader's machine when the task has no
   agent image. Its output files, or the contents of `answer_file`, become the
   submission, which the grader then grades in its own environment. Graders
   that run outside the process get their machines from the campaign's
   `GradingMachines`, which routes each environment by its
   `compatible_backends`: a `local` environment to a bubblewrap sandbox on the worker
   with the Python environment built from its `packages_lock`, and an image
   environment to a sandbox of that image. Every grading machine has network
   access denied. Without controls the stage is skipped and sandbox graders
   stay unverified.

   A source whose converted panel tasks are all graded by a verifyit judge is
   not sampled, with or without controls: no judge control exists yet, so the
   report records `skipped` with reason `judge grader; no control path yet`.
7. **Admit and export.** Each row gets an `admission`:

   | Admission | Meaning |
   | --- | --- |
   | `admitted` | Kept, and the grader runs in process, is a verifyit judge, or passed source verification. |
   | `rejected` | Rejected by conversion, checks, review or verification. |
   | `deferred` | Held back for unavailable review or inconclusive verification. |
   | `no_grader` | The converter found no runnable grader (`NoGrader`). |
   | `unverified` | A sandbox grader, other than a judge, without passing source verification. |

Output layout of one source:

```
download/      row locators of the staged source files
normalize/     every converted row (TaskSpec JSON, rejection reason, normalization changes)
review/        review and filter decisions, report.json, manifest.json
verify/        checks, grader readiness and admission per row, report.json, manifest.json
final/         admitted rows only
manifest.json  status, counts, quality and verification reports, admission summary
telemetry.json Zephyr execution IDs, counters and phase wall times
```

The manifest's `admission` field summarizes the source: `admitted` when any row
is admitted, otherwise `none`.

## Verification trials

Verification samples up to the configured sample size with seeded task-ID hashes
across all shards (the driver samples at most 20 tasks by default) and runs each
task's control in a fresh environment once per attempt (the driver makes one
attempt). An attempt
whose control hits an infrastructure error runs again, up to two more times,
before it is recorded. A task passes when its control passes or records a defect
on every attempt and no earlier trial failed; a task with both a pass and a
definite failure counts as inconsistent. The report's `counts.defective` records
the sampled tasks with a defect. The source passes when the pass fraction meets
`minimum_pass_fraction` (the driver uses 95%). Unsupported runtimes,
infrastructure errors and an empty sample make the source inconclusive, which
defers its eligible rows. Failed and defective controls reject the affected tasks
even when the source passes. Unsampled rows of a passing source get
`source_sampled` readiness.

Reruns reuse complete trials from a previous `verify/report.json`. Trial identity
covers the whole task, the controls code, the machine backend and worker image,
all TaskCompendium, VerifyIT and Shellbox Python files, and the verification
settings. QEMU reuse requires a digest-pinned worker image carrying the bundle;
network-enabled graders always run fresh. Infrastructure errors rerun the whole
trial; definite failures and defects survive successful retries.

## Review execution

Full preparation assigns each row to a normalization shard by locator hash, then
deduplicates and packs model requests into batches before distributing review
work; the driver prepares the panel the same way without a shuffle. Prepared
review batches are saved before inference. Batches respect both
`review_batch_size` and `review_input_bytes` (64 MiB by default); an oversized
audit is preserved alone. Review execution uses `review_task_resources`, so
several model requests can wait on one worker. Completed review shards are
reused when an audit resumes.

Inference uploads are limited to 64 requests and 4 MiB per provider batch; an
oversized single request is deferred with its size diagnostic. Transient HTTP
failures use bounded retries with backoff, and failed requests leave unavailable
review records that do not count as source defects. GLM completions use an
exact-request cache keyed by the complete request and the declared model
revision, read through FineStore, so unrelated catalog changes do not repeat
inference. The quality review reads the cache once for all of a source's sampled
requests.

GLM sees resource paths, roles, SHA-256 hashes, byte counts and UTF-8 previews.
Up to 256 files share 32,768 preview characters: each text file first receives up
to 100 characters, then earlier previews expand to at most 8,192 characters each.
Binary files retain signatures without text, and an aggregate hash covers
omitted files.

## Telemetry

Zephyr counters report review cache hits and misses, submitted requests and
bytes, request failures, review outcomes, check outcomes, control and trial
outcomes, admissions, and time spent in provider calls and grading.
`telemetry.json` keeps the final counter dictionaries of every execution and the
wall time of each phase. Join its execution IDs to Zephyr stage rows by
execution ID and emitting cluster/job; take the latest value per counter series
rather than summing snapshots, and do not add phase wall time to nested
execution times. Failed executions have no ID or final counters.

A source runs these phases in order:

| Phase | Work |
| --- | --- |
| `raw_sample` | One execution scans the staged files, writes the locators and draws the panel. A parted source in sample mode draws the panel by index on the driver, and the execution produces only the panel's rows. |
| `panel_normalize` | One execution decodes, converts and checks the panel rows. |
| `sample_prepare` | The driver deduplicates the panel and saves its review batches. |
| `quality_review` | With a rubric, the driver reads the review cache once and one execution reviews the sample; without one, the driver decides from conversion and check failures. |
| `full_prepare` | Full mode only: one execution converts, deduplicates and checks every row. |
| `audit_review` | One execution reuses the panel's reviews and applies the source decision to every other prepared row, without model requests. |
| `filter` | One execution writes the filtered audit and its kept rows. |
| `verification` | With controls, the executions of [source verification](#verification-trials). |
| `export` | One execution shuffles the checked rows by task ID into the normalized shards and writes the `normalize`, `review`, `verify` and `final` views, with the unprocessed review rows of an unexpanded sample. |

## Grading environments

A task's agent environment and its grader's environment are declared separately
in `EnvironmentRequirements`. A task may run in ShellSim while its grader needs
an image. An empty `compatible_backends` declares no Shellbox backend; it does
not allow every backend.

| Contract | Backend declaration |
| --- | --- |
| Built-in shell commands and virtual files preserve the requested behavior | ShellSim may be declared after checking command semantics, paths, quoting, pipes and file metadata. |
| Arbitrary Python packages, compiled binaries or image-specific dependencies | Use an image-backed backend and pin the image by digest. ShellSim cannot satisfy a required image. |
| Kernel behavior, subprocesses, networking or special filesystem behavior | Check the specific Shellbox backend's support. |
| Grader has different dependencies from the task | Declare compatibility in the grader's environment. |
| Rows within a source differ in runtime needs | Emit the correct declaration for each row; do not broaden compatibility to fit the launch. |

Compatibility is an author claim. Verification records evidence for the selected
backend and image; a passing sample on gVisor does not certify ShellSim. An image
the selected backend cannot run makes verification inconclusive, not the source
defective.

See the [task curation reference](../../../../../docs/references/task-curation.md)
for the output schema and the experiment's catalog and driver.
