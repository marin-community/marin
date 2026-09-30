# Prompt experiments

[Planning overview](../index.md) · [Prompt and run versioning](../task-authoring.md#prompt-and-run-versioning)

Record completed prompt experiments here, with their question, configuration,
evidence, interpretation, and next decision. Preserve negative results and the
exact prompt and outputs. Keep proposed changes separate from measured results.
Small text artifacts can accompany the record; link large traces or datasets
from their storage location.

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
