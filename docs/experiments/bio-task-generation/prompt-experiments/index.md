# Prompt experiments

[Planning overview](../index.md) · [Prompt and run versioning](../task-authoring.md#prompt-and-run-versioning)

Record completed prompt experiments here, with their question, configuration,
evidence, interpretation, and next decision. Preserve negative results and the
exact prompt and outputs. Keep proposed changes separate from measured results.
Small text artifacts can accompany the record; link large traces or datasets
from their storage location.

The current [find-units prompt](../prompts/find-units.md) uses one worker, has no
artificial time cutoff, and treats public scientific commands as candidate units,
including extraction and format conversion. The [latest UCSC trial](2026-09-30-catalog-ucsc-luna-high/index.md)
finds the public catalog and accounts for every top-level entry, but stops with
283 entries pending. Source accuracy and completion remain unresolved; these
inventories are provisional inputs for authoring, not validated tasks.

## Evidence preservation and migration

The branch preserves substantive design decisions, prompt revisions, experiment
inputs, outputs, reviews and unresolved findings. It is not a verbatim transcript
of the conversation. Use the [source handoff](../index.md#research-handoff-and-repository-migration)
and [current decisions](../task-authoring.md#current-discovery-decisions) with the
per-run records below when migrating to Open-Athena/biotasks.

| Material | Preservation status |
| --- | --- |
| Exact templates, resolved prompts, worker launch/source-access instructions and source revisions | Versioned per run; hashes recorded where available |
| Worker inventories, final responses, structural checks and independent reviews | Versioned, including negative/partial results; original bytes retained separately when formatting changed them |
| Uncapped DESeq2 CLI comparison | All four configurations, runner metrics and compressed execution events are versioned; the blocked Sol/high run has a failure record instead of a final response |
| Earlier collaboration-worker execution | Available final responses and parent observations are versioned; full event traces and usage were not exposed and cannot be reconstructed from these records |
| CLI wrapper | [Source snapshot](2026-09-30-cli-runner.py.txt) and [launch envelope/hash](2026-09-30-cli-runner.json) captured after the batch; per-run wrapper hashes were not recorded, so this is not proof that its bytes were identical in every run |
| Temporary validation/preparation helpers and source-review caches | Retained only in the source checkout's ignored `artifacts/bio-task-generation/`; they are not part of the branch. Saved check results, source URLs/pins and selected source hashes remain in the records |
| Mutable external documentation and original discussion references | References and retrieval dates are recorded; complete source bodies are not archived. Re-fetching later may produce different contents |
| Conversation and reasoning | No full chat export is included. CLI exports omit reasoning items; private reasoning is not a research artifact |

The records are portable evidence, but historical commands contain absolute paths
to the Marin checkout. Preserve those original snapshots. For a new run, create a
new run ID and directory, adapt paths in copied launch instructions and the runner,
and save the adapted inputs and hashes. Use the original template and repository
pin when the intended comparison requires them. Follow the destination's resource
and service-authorization rules, and record the scope and execution budget for
each new experiment. Never rerun the archived wrapper against an existing evidence directory,
because it writes the run manifest and worker outputs.

The wrapper snapshot is an audit artifact, not a maintained portable pipeline.
Source inspection, structural checks and saved event logs do not establish
scientific correctness, task-authoring success or exact reproducibility of model
outputs. Migration should record a disposition for the local-only material and
check hashes and links after relocation, as required by
[destination issue #3](https://github.com/Open-Athena/biotasks/issues/3).

The [completed discovery migration](https://github.com/Open-Athena/biotasks/issues/2)
provides the destination pattern. Its [pinned archive procedure](https://github.com/Open-Athena/biotasks/blob/6bb14ef5eea9a58b6b704808ff21bdecb8791170/experiments/5-source-discovery/runs/2026-09-30-bucket-archive/README.md)
preserves an exact file allowlist, prepares an archive and manifest, commits the
manifest/upload plan, and verifies anonymous download of both objects and every
member. The [destination storage guide](https://github.com/Open-Athena/biotasks/blob/37271415c4c201ba9dbbda66c203caa4050744c3/docs/storage.md)
uses `hf://buckets/open-athena/biotasks/research/<issue>-<topic>/<capture-date>/<manifest-sha256>/`.
The manifest hash identifies the exact recorded bytes; bucket prefixes are
append-only by convention, without server-enforced version history. Preserve the
source until the archive verification is recorded.

A lightweight source-checkout inventory on 2026-09-30 found 71 files totaling
8,912,438 bytes in ignored `artifacts/bio-task-generation/`, including raw CLI
logs, source caches, helpers and local check records. This is a candidate-file
count at that observation, not a frozen upload allowlist. Review exclusions and
source terms, recover useful helpers as historical source, and archive eligible
remaining evidence using the destination procedure. An exact allowlist, archive
manifest, upload and download verification for these authoring artifacts remain
pending. The original local files are retained.

## Cross-repository comparison on 2026-09-30

The [comparison configuration](2026-09-30-comparison.json) records a two-hour
iteration session across bedtools, Scanpy, MMseqs2, DESeq2, and the UCSC Kent
repository. The first round uses the unchanged single-worker prompt from
`37f8a601d2` in five fresh Luna contexts. Each worker receives a repository and
pinned revision, generic retrieval instructions, and up to ten minutes. Up to
three workers perform lightweight source inspection concurrently; scientific
execution, biological data downloads, and child delegation are excluded.

Reviews compare source coverage, evidence, unit boundaries, tool roles, data
identity, dependencies, and handoff consistency. Repository-specific probes are
saved before launch and withheld from workers. Later rounds will rerun a shared
revision against the same source revisions; any changed setup will be recorded.
Results below from 2026-09-29 remain historical baselines.

The [Bioconductor documentation conventions](https://contributions.bioconductor.org/docs.html)
provide reusable discovery routes: vignette sources in `vignettes/` use R
Markdown, Sweave, or Quarto; exported functions have manual pages and examples;
bundled data have documentation. These conventions locate evidence but do not
define scientific unit boundaries or establish execution in a task environment.
The [package-vignette guide](https://bioconductor.org/help/package-vignettes/)
also describes compiled documents and extracted R scripts. Source and rendered
versions must be matched before combining their evidence.

Independent source inspection, before reading worker outputs, found two useful
probes. In the pinned
[DESeq2 vignette](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/vignettes/DESeq2.Rmd#L304),
observed transcript abundances are paired with artificially assigned condition
labels in an import demonstration; this does not establish a real experimental
contrast. The source also contains hidden and unevaluated chunks. The
[MMseqs2 tutorials](https://github.com/soedinglab/MMseqs2/wiki/Tutorials) include
pathogen investigation, contig taxonomy, and a gut-metagenome workflow that
combines multiple tools. The specific MMseqs2 tutorial probes were identified
after launch and supplement the frozen review criteria. Workers receive neither
these observations nor repository-specific source-selection instructions.

## Round 1 results

| Repository | Units / data records | Parent review |
| --- | --- | --- |
| [bedtools](2026-09-30-round1-bedtools-luna/index.md) | 18 / 5 | Broad command coverage; inconsistent command queue statuses |
| [Scanpy](2026-09-30-round1-scanpy-luna/index.md) | 5 / 6 | Useful tutorial state; PBMC products combined without established shared observations; stale BBKNN status |
| [MMseqs2](2026-09-30-round1-mmseqs2-luna/index.md) | 12 / 5 | Finds pathogen and gut workflows; wrong uninspected paper DOI; stale wiki access statement |
| [DESeq2](2026-09-30-round1-deseq2-luna/index.md) | 8 / 4 | Finds literate sources and artificial labels; combines GENCODE reference with observations; stops early |
| [UCSC Kent](2026-09-30-round1-ucsc-luna/index.md) | 4 / 2 | Preserves utility semantics and separate fixtures; stops early with useful accessible leads |

All five pass structural checks. These checks do not establish semantic accuracy.
The second-round revision requires evidence for exact identifiers and grouped
assets, exact input stages, a coherent final source map, and continuation from
saved checkpoints while inspection time remains. These are general instructions;
no repository names, source hints, or review findings enter worker prompts.
The next round tests the combined revision, so individual clause effects cannot
be isolated. Its output counts will not be used as quality scores.

## Round 2 results

| Repository | Units / data records | Parent review |
| --- | --- | --- |
| [bedtools](2026-09-30-round2-bedtools-luna/index.md) | 13 / 3 | Coherent command map and separate references; tutorial data follow-up remains weak |
| [Scanpy](2026-09-30-round2-scanpy-luna/index.md) | 11 / 8 | PBMC identity repaired; false raw-count-state claim and stale counts; deadline exceeded by 7m25s |
| [MMseqs2](2026-09-30-round2-mmseqs2-luna/index.md) | 11 / 3 | Correct paper DOI and input stages; named references omitted under an unnecessary version gate; early stop |
| [DESeq2](2026-09-30-round2-deseq2-luna/index.md) | 10 / 3 | Hidden tximeta chunk recognized; annotation grouping persists and artificial-label warning is lost |
| [UCSC Kent](2026-09-30-round2-ucsc-luna/index.md) | 5 / 4 | Additional utilities and reference leads; stale final stopping evidence |

The revision has specific gains and regressions; no overall improvement is
established. Round 3 repeats the identical prompt in fresh contexts before
another revision. All structural checks pass, while scientific and source-map
errors remain. The runner supplies a deadline but does not enforce a timeout;
Scanpy's wider coverage is confounded by its overrun. A parent execution gap
from approximately 14:30 to 14:58 UTC is recorded in the comparison configuration.
It is not counted as continuous parent research or review time.

## Identical-prompt repeats and next revision

Round 3 uses the exact round 2 template in five new contexts. The first completed
reviews show that individual improvements are not stable:

- [DESeq2](2026-09-30-round3-deseq2-luna/index.md) restores the artificial-label warning but still groups the independent annotation with quantifications.
- [MMseqs2](2026-09-30-round3-mmseqs2-luna/index.md) restores reference products but misclassifies composed tool use as mixed creation/use and combines independent taxonomy products.
- [UCSC Kent](2026-09-30-round3-ucsc-luna/index.md) preserves useful format/ploidy constraints but combines a call fixture with its independent exclusion annotation and retains a stale pending entry.

- [bedtools](2026-09-30-round3-bedtools-luna/index.md) recovers tutorial data but mislabels analysis composition as mixed and retains stale counts and copied limitations.
- [Scanpy](2026-09-30-round3-scanpy-luna/index.md) restores the counts layer but omits the independent cell-cycle gene list; a parent reminder stops an overrun.

The shared round 4 revision reduces the prompt from 2,080 to 1,628 words and
organizes it around an inspect–record–check loop. It moves checks beside record
creation, explicitly preserves demonstration metadata changes and upstream state,
requires implementation evidence for `mixed`, and admits identifiable incomplete
data products. Counts appear once in the final report. Package/documentation
metadata remain generic discovery routes; there are no repository-specific hints.
The first round 4 launch follows review of the three repeats above, while the
bedtools and Scanpy repeats continue using frozen round 2 copies. Further results
will determine whether this revision should be retained.

## Round 4 results

| Repository | Units / data records | Parent review |
| --- | --- | --- |
| [DESeq2](2026-09-30-round4-deseq2-luna/index.md) | 8 / 3 | Artificial labels and final counts retained; independent annotation still grouped; assisted deadline overrun |
| [MMseqs2](2026-09-30-round4-mmseqs2-luna/index.md) | 7 / 6 | Composed workflows correctly tool use; incomplete named references retained; residual queue inconsistency |
| [UCSC Kent](2026-09-30-round4-ucsc-luna/index.md) | 20 / 4 | Cutoff withdrawn mid-run; broader tool use, but misses public command catalog and stops with useful work pending |

These are exploratory comparisons. The prompt revision changes several clauses
together, and runtime adherence remains imperfect. No individual clause effect
or overall improvement is established from these counts.

## Model and reasoning effort comparison

At the user's request, a [predeclared 2×2 DESeq2 comparison](2026-09-30-model-effort-comparison.json)
uses `gpt-6-luna` and `gpt-6.1-sol`, each at requested `low` and `high` effort.
The initial design used one fresh worker per cell, the same round 4 prompt,
pinned repository, retrieval envelope and an instructed eight-minute maximum. The earlier runs
leave effort unspecified and are not effort controls. Source-level probes and
structural checks are shared; counts alone are not the outcome measure.

This small comparison can expose useful failure differences but cannot establish
a general model ranking. The interface exposes requested settings, not verified
served settings, full traces, token usage or billed cost. Deadline compliance
and parent interventions must be reported with each result.

The historical pilots are now complete: [Luna low](2026-09-30-knobs-deseq2-luna-low/index.md)
produced 5 units and 2 data records; [Sol high](2026-09-30-knobs-deseq2-sol-high/index.md)
produced 57 and 12 after its cutoff was withdrawn. Sol preserves independent
reference identity and discovers source-backed method implementations, but one
filtering-algorithm description is wrong. These pilots cannot isolate model,
effort and runtime effects and are excluded from the fresh comparison.

## User correction: no artificial time cutoffs

The user rejected per-run time limits on 2026-09-30. Active deadlines were
withdrawn; the live prompt and future launch instructions now use source coverage,
repeated leads and genuine access/resource blockers as stopping conditions.
Elapsed time is an outcome. There is no imposed unit quota or replacement cap.
Historical prompts and outputs retain their original limits for auditability.

The [fresh uncapped model/effort comparison](2026-09-30-uncapped-model-effort-comparison.json)
uses the same two models and low/high effort settings with identical inputs.
The capped and transitional pilots above are excluded from that comparison.
The prepared round 4 bedtools/Scanpy runs and capped Luna-high/Sol-low cells
were never launched and were withdrawn. The first two fresh uncapped spawns hit the runner's agent-thread limit. The
comparison now uses the documented `codex exec` interface for all four cells,
with fresh contexts, explicit settings, memories disabled and user configuration
ignored. CLI processes run serially under the shared-node resource lock. Event
logs and reported usage are captured; billed cost remains unknown. Runner
startup failures are recorded separately from model results.

| Uncapped DESeq2 setting | Units / data | Review |
| --- | --- | --- |
| [Luna, low](2026-09-30-uncapped-deseq2-luna-low/index.md) | 6 / 3 | Stops with useful sources pending after 251.27s; annotation grouping, unsupported sample-count inference and sample-name suffix error |
| [Sol, high](2026-09-30-uncapped-deseq2-sol-high/index.md) | 44 / 10, partial | Service biological-risk flag at 3,182.49s; source-backed two-sample revision and separate annotation retained; no completed result |
| [Luna, high](2026-09-30-uncapped-deseq2-luna-high/index.md) | 23 / 5 | Separate annotation and broader operations at 1,453.79s; premature completion, version-scoping and numeric-locator issues remain |
| [Sol, low](2026-09-30-uncapped-deseq2-sol-low/index.md) | 90 / 19 | Broad mapped coverage and coherent source map at 1,827.92s; copied source-kind/dependency metadata and grouped generated fixtures remain |

The first uncapped run preserves the artificial-label warning and adds a
simulation benchmark, but explicitly stops despite available work and successful
retrievals. Its completion failure therefore persists without an external cutoff.
Elapsed time and CLI-reported token usage are recorded as outcomes. A later
source check found that tximportData 1.41.1 has two samples while the release
vignette describes six. Sample-count judgments must match the dependency
revision: inferring a count from the artificial condition vector remains
unsupported, but two samples is not universally incorrect.

The four-cell batch is finished: three completed runs and one service-blocked
run. Luna/high preserves more operations and reference identity than Luna/low,
but both stop with useful inspection pending. Sol/low covers the mapped package
and linked-workflow scope and preserves the sampled scientific-state caveats;
copied metadata and data boundaries still need repair. The blocked Sol/high cell
prevents a completed within-Sol effort comparison. No general model ranking,
optimal effort setting, billed-cost comparison or readiness for hundreds of
repositories follows from this single-repository matrix.

## Public operation catalogs

The user identified the UCSC Linux binary listing after the round 4 worker
finished. The repository README already links that distribution, and its HTTP
index links combined command usage. The worker missed this route and deferred
converters because they did not seem to be distinct analyses. That is a discovery
failure: extraction, conversion and other supporting scientific operations also
qualify as units. Their eventual task difficulty is an authoring decision.

The [live prompt](../prompts/find-units.md) now requires finding public command/API
catalogs and accounting for each entry as an inspected unit, alias, specific
exclusion or pending lead. Catalog entries need semantic inspection before they
become units. This change is general and contains no repository-specific hints.
A [fresh Luna/high trial](2026-09-30-catalog-ucsc-luna-high/index.md) found the
catalog without its URL in the inputs. It produced 60 units and 21 data records,
including all three withheld command examples. All 328 catalog entries have a
disposition: 45 map to units and 283 remain individually pending. It still stops
with accessible work remaining and confuses the versions of two help surfaces.
Catalog accounting improved; reliable completion remains unresolved. The four uncapped model/effort comparison snapshots remain
frozen at their pre-catalog version; they do not test this correction. See the
[UCSC parent review](2026-09-30-round4-ucsc-luna/index.md) for sources and revision gaps.

## Earlier trials

| Date | Experiment | Result | Status |
| --- | --- | --- | --- |
| 2026-09-29 | [Fresh Luna worker finds units in bedtools](2026-09-29-bedtools-luna/index.md) | Three units and one data record; valid structure, incomplete coverage and incorrect data grouping | Completed exploratory baseline |
| 2026-09-29 | [Bedtools repeat with the revised prompt](2026-09-29-bedtools-luna-02/index.md) | Five units and four data records; reaches DNase similarity, still groups references and records schematic data | Completed and reviewed |
| 2026-09-29 | [Bedtools with a source map and clarified data identity](2026-09-29-bedtools-luna-03/index.md) | Five units and seven data records; separate references, explicit source map, partial coverage and a tool-role error | Completed and reviewed |
| 2026-09-29 | [Scanpy transfer trial](2026-09-29-scanpy-luna-01/index.md) | Five units and six data records; useful API/state descriptions, but splits a dataset from its processed derivative | Completed and reviewed |
| 2026-09-29 | [Scanpy with source fallbacks and data reconciliation](2026-09-29-scanpy-luna-02/index.md) | Three tutorial units and three data/reference records; consolidates shared observations, but narrower coverage and unresolved source access | Completed and reviewed; no overall improvement established |
| 2026-09-29 | [Bedtools with an explicit stopping condition](2026-09-29-bedtools-luna-04/index.md) | Four broad units and five data records; role classification retained, coverage and source retrieval still incomplete | Completed and reviewed; no coverage gain established |
| 2026-09-29 | [Bedtools with direct repository-file access](2026-09-29-bedtools-luna-05/index.md) | Five units and three data records; pinned sources, DNase matrix/plots and independent annotation QC; still stops early | Completed and reviewed; execution-setup probe |
| 2026-09-29 | [Bedtools with an entry-level inspection queue](2026-09-29-bedtools-luna-06/index.md) | Seventeen units and two overgrouped data records; broader operation coverage and a continuation queue, but data identity and tool roles regress | Completed and reviewed; partial inventory needs repair |
| 2026-09-29 | [Bedtools inventory reconciliation](2026-09-29-bedtools-reconcile-luna-01/index.md) | Twenty-two units and thirteen data records; repairs grouping, links, roles and workflow boundaries; stale claims and queue count remain | Completed and reviewed |
| 2026-09-29 | [Bedtools reconciliation with consistency checks](2026-09-29-bedtools-reconcile-luna-02/index.md) | Seventeen units and twelve data records; reproduces identity/link/role repairs and corrects counts; broad workflow boundary remains | Completed and reviewed |
| 2026-09-29 | [Scanpy reconciliation transfer](2026-09-29-scanpy-reconcile-luna-01/index.md) | Six units and five datasets; merges PBMC3k stages and splits APIs, but copied API evidence and source interpretation need repair | Completed and reviewed |
| 2026-09-29 | [Scanpy reconciliation with replacement audits](2026-09-29-scanpy-reconcile-luna-02/index.md) | Seven units and five datasets; identity repair retained and workflow descriptions sharpened; copied dependencies and stale source-map IDs remain | Completed and reviewed; parent disagrees with worker readiness |

All trials requested a fresh `gpt-6-luna` context, one child at a time, one
attempt, and zero retries. The runner did not expose the effective reasoning
effort, context/output/compaction limits, served model version, usage, or full
tool traces. Exact prompts, launch instructions where available, raw outputs,
hashes, and separate reviews are retained. Launch and completion timestamps
are parent observations, not measured model runtimes. Absolute deadlines were
set ten minutes after preparation; launch delays reduce the usable window.

The general prompt changed between trials in response to observed failures.
Bedtools 03 and Scanpy 01 used the same template, providing a transfer probe.
Bedtools 04 and 05 used the same template while source access changed. These
are exploratory comparisons with one sample per configuration, no exhaustive
reference inventory, and no measured task-authoring success. Counts describe
the outputs; they are not quality scores. No Harbor task or solver was run.

Across the first eight discovery trials, explicit source access helped recover
evidence and an entry-level queue broadened operation coverage in bedtools 06.
Instructions for data identity and tool roles helped some runs but did not
reliably prevent the same errors from recurring.

A separate reconciliation prompt was tested in four fresh-worker trials, with
two revisions driven by inspected failures. Its final
[template snapshot](2026-09-29-scanpy-reconcile-luna-02/template.md) is preserved
as experiment evidence. Both bedtools trials split unrelated data products and
repair role labels. Both Scanpy trials merge PBMC3k processing stages while
retaining independent PBMC collections. The v2 reconciliation template was
identical in bedtools 02 and Scanpy 01; v3 was tested only in Scanpy 02.

Unit splitting and metadata consistency remain unreliable. The last worker's
audit table says copied dependencies were removed, but its JSON retains them;
its source map also references removed unit IDs. Parent review therefore rejects
the worker's readiness recommendation. Further prose instructions alone did
not resolve these failures. The two-worker approach was not compared at equal
total cost, and no task-authoring success or general reliability across
repositories was established.

Decision on 2026-09-30: retain one discovery prompt and worker, following the
user's preference to avoid a second prompt. The current
[find-units prompt](../prompts/find-units.md) extends the bedtools 06 version's
final self-check with record-specific evidence checks, source-context checks,
and consistent references and counts. It also clarifies that discovery classifies
the source work as tool use, creation, or mixed; the author separately defines
what the solver must do. The separate active reconciliation prompt
has been removed; its trial snapshots and results remain unchanged. The earlier
proposal for a narrower reconciliation worker is superseded. The five round 1 runs above now test this single-worker revision. They preserve
useful coverage but still expose errors in data identity, evidence, and final
source-map consistency.

Record checks on 2026-09-29: scoped `./infra/pre-commit.py` passed for 162
files, the repository documentation-source-link check passed, 242 local Markdown
links resolved, and 98 recorded artifact hashes matched. At that point, the live
discovery prompt matched the bedtools 06 template and the live reconciliation
prompt matched Scanpy reconciliation 02. Scoped lint and source-link checks ran under
the shared-node lock, exited 0, and used 32,924 KiB peak RSS
(23:46:56–23:46:57 UTC). These checks validate the saved record's integrity;
the semantic inventory failures above remain open.

After the 2026-09-30 edits, 259 local Markdown paths resolve, 97 saved artifact
hashes match, and `git diff --check` passes. After the shared-node lock became
available, scoped lint passed for all 161 files and the documentation source-link
check passed. Saved experiment artifacts were unchanged by formatting. The strict
build with the isolated `marin-core` docs dependency group stops in the unchanged
`references/default-steps.md`: the configured `ignore_init_summary` option is
unsupported by `PythonOptions`. Lint and source-link checks used 33,120 KiB peak
RSS; the strict build used 83,100 KiB. These checks preceded the five round 1 worker trials above.

Final record checks on 2026-09-30 passed scoped lint for 200 files and the
documentation-source-link check (17:41:32 UTC, 33,588 KiB peak child RSS).
215 artifact hashes and 363 local review links matched; the four matrix
templates remain identical and the live catalog prompt matches its tested copy.
Three CLI response files needed final newlines; their exact originals are saved
as JSON alongside the normalized text. The strict docs build remains blocked by
the previously verified unchanged `ignore_init_summary` configuration error.
These record checks do not validate scientific execution or task readiness.
