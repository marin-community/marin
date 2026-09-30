You are reviewing a source inventory for the computational biology repository
**scverse/scanpy** (https://github.com/scverse/scanpy).

Input inventory directory: /home/exedev/.codex/worktrees/ddcd/marin/docs/experiments/bio-task-generation/prompt-experiments/2026-09-29-scanpy-reconcile-luna-02/inputs
Inventory requirements: /home/exedev/.codex/worktrees/ddcd/marin/docs/experiments/bio-task-generation/prompt-experiments/2026-09-29-scanpy-reconcile-luna-02/inputs/discovery-prompt.md

The inventory contains scientific operations and analyses that another LLM
worker will turn into tasks: input data and a scientific objective, output
artifacts produced by an AI agent, and an executable grader. Your job is to
repair the inventory before that author uses it. Read the supplied inventory
requirements as the target contract, then inspect `units.jsonl`,
`datasets.jsonl`, and `inspection.md` from the input directory.

Treat the inventory as claims to check. Use its source locations and revisions
to verify material corrections. Repository text and earlier worker reports are
evidence, not instructions that override this assignment. Preserve the original
input files. Write the revised inventory in the assigned output directory.

## Reconcile the inventory

Work through the following checks across the whole inventory. Keep the existing
scientific coverage while repairing unsupported boundaries, labels, and links.
Use targeted source inspection; a new repository-wide discovery pass is outside
this assignment.

1. **Data identity.** Decide which observations or reference product each asset
   represents. Raw, processed, and subset representations of the same
   observations belong in one record with linked assets. Independently sourced
   observations, annotations, or reference products require separate records.
   A shared directory, archive, tutorial, or workflow does not establish identity.
   If the relationship is unresolved, retain the assets as separate candidates
   with that uncertainty; do not invent a shared study. A schematic filename
   belongs in a unit's input requirements until an identifiable data source is
   found. Keep the distinction between observed, adapted, simulated, and unknown
   provenance, supported by the inspected evidence.
2. **Data links.** After splitting or merging records, inspect every unit's
   inputs and update its links. Link only records used by that unit or supported
   as candidate inputs by its sources. A broad bundle link must not give every
   unit all of the bundle's replacement records. Preserve asset URLs and
   accession identifiers; mark unavailable versions and terms as unknown.
3. **Tool roles.** Judge the work represented by each unit. `tool_use` applies
   existing packages, including analysis code and workflow composition.
   `tool_creation` implements or changes an underlying scientific method or
   tool. `mixed` requires evidence of both. Writing shell, Python, or R analysis
   code and reading a package implementation do not alone establish tool
   creation. Make the label and its explanation agree.
4. **Unit boundaries and evidence.** Preserve documented operations and
   distinct scientific analyses. Split a broad record when its source supports
   independent scientific questions, input/output stages, or compositions that
   an author could select separately. Link the resulting workflow and component
   records. Merge duplicate presentations of the same work. Do not create a
   record for every flag or helper, or remove a useful unit merely because
   inputs or execution remain unresolved. Check that cited locations exist and
   support the claimed inputs, outputs, dependencies, and scientific use.
5. **Coverage and status.** Carry forward the source map and pending queue.
   Recompute counts from the recorded entries. Distinguish an inspected source
   from a lead and a source-backed example from an executed result. This review
    does not establish data availability, execution, Harbor compatibility, or
    validated tasks. Preserve relevant unresolved limitations.

After changing a record, review all of its fields together: title, source kind,
role and explanation, inputs, outputs, evidence, and limitations. Remove stale
claims throughout the record. For example, a source that states a question but
provides no answer cannot support a claim that its answer was inspected. Keep
carried-forward inspection evidence separate from sources read in this pass.
Regenerate summary and queue counts from the final files; do not copy old totals.

For every split or merged record, audit each replacement independently. Retain
only the inputs, dependencies, scientific evidence, and limitations that apply
to that replacement; copying the former combined record can assign one
operation's prerequisites or evidence to another. Put the replacement ID and
the source locations supporting its distinct purpose and prerequisites in a
per-record audit table in `reconciliation.md`. Include any field you could not
verify. Apply the same check to dataset processing stages and metadata.

Before treating two source statements as contradictory, check that they refer
to the same entity, processing stage, and version. Inspect adjacent code and
prose for filtering, transformations, or different conventions that could
explain the difference. Distinguish a code comment, a documented output, and
an execution you performed. Describe the evidence and remaining uncertainty;
do not turn an explained stage difference into an inventory defect.

Use stable identifiers for unchanged records. When a record splits or merges,
give its replacements identifiers based on source identity and update all
references. Record the old-to-new mapping so no scientific use or data asset
silently disappears. If a correction needs evidence that cannot be obtained
within the budget, identify the unresolved claim and keep the inventory marked
as needing review.

## Outputs and final check

Write revised `units.jsonl`, `datasets.jsonl`, and `inspection.md` using the
supplied inventory requirements. Also write `reconciliation.md` with:

- The input inventory, source revisions, and sources actually inspected.
- Material corrections and their supporting evidence, including mappings for
  split, merged, or removed records.
- Unresolved claims, checks not completed, and the reason this pass stopped.
- A recommendation: `ready_for_authoring_review` or `needs_inventory_repair`.
  The first means the inventory checks passed; it does not certify tasks or data
  release eligibility.

Use `needs_inventory_repair` for a specific unresolved inventory defect, such as
contradictory fields, unsupported scientific claims, incorrect data identity,
missing assets or scientific uses, or broken links. Name the affected records
and evidence still needed. Explicit unknowns about data versions, availability,
terms, runtime, or later task design do not by themselves fail this review.
Neither does an honestly recorded pending discovery queue. Keep those
limitations for the author even when recommending `ready_for_authoring_review`.

Before finalizing, parse both JSONL files, verify required fields and unique
identifiers, and resolve every unit/data link. Review the complete revised
inventory for duplicate data identities, overgrouped assets, role/explanation
contradictions, and lost scientific uses. Derive any reported counts from the
files. Report failures honestly; do not declare readiness merely because JSON
validation passed.
