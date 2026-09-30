# Prompt experiments

[Planning overview](../index.md) · [Prompt and run versioning](../task-authoring.md#prompt-and-run-versioning)

Record completed prompt experiments here, with their question, configuration,
evidence, interpretation, and next decision. Preserve negative results and the
exact prompt and outputs. Keep proposed changes separate from measured results.
Small text artifacts can accompany the record; link large traces or datasets
from their storage location.

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
proposal for a narrower reconciliation worker is superseded. This revision of
find-units has not been trialed. The next prompt experiment should test this
single-worker version on the testbed with the same saved inputs and review
criteria, recording any changes to source access or budget.

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
RSS; the strict build used 83,100 KiB. No new worker trial was run for this revision.
