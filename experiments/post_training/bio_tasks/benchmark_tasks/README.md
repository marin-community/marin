# Benchmark task inventories

Each JSON file records the inspected tasks from one benchmark release. Scope is
the spreadsheet's **agentic** tab. See the [release index](../benchmark_coverage.json)
for pinned revisions, source hashes, inspected counts and completeness, and the
[catalog](../../../../docs/experiments/bio-task-catalog.md) for the readable inventory.
The [source inventory](../benchmark_sources.json) also lists eligible releases
whose tasks have not yet been enumerated.

Each ID task records its identifier, workflow pattern, required stages, formats
and tools, recipe mappings, evidence, and remaining gaps. Where available, records
also describe output artifacts, scientific decisions and overlap with other
releases. A workflow-family pattern can include more stages than one question
requires; inspect the endpoint before assigning coverage.

To read a compact record, start with its file's `task_defaults`, overlay the
entry in `patterns` selected by `workflow_family`, then overlay the task's own
fields. The task's fields take precedence. Files with fully expanded records
need no overlays. `workflow_patterns`, where present, is a reverse index from
patterns to task IDs. The inspection page expands these records for browsing.

Coverage progresses from `unmapped` to `component-only`, `composed-unvalidated`
and `workflow-validated`. A reviewed prompt or a matching tool name does not
establish workflow coverage. Validation requires an executable connected task
on independent biological observations, with the required artifacts checked.
`restricted/excluded` records are outside training authoring.

Public examples are partial inventories when the full suite is unavailable.
Protocol variants, shared task IDs and registry definitions are not additional
independent workflows. Keep those distinctions when reporting totals.

Existing OOD assignments remain held out. The BioMysteryBench file contains
identifiers only; it supplies no workflow patterns or training mappings.
Benchmark answers and biological fixtures are excluded from training authoring.
