# Benchmark task inventories

Each JSON file inventories task IDs or protocol definitions from one benchmark release. Scope is
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

Use missing ID task stages and scientific decisions to prioritize new recipes.
Complete benchmark workflows take priority over increasing repository coverage.
Record the native packages and formats required by each workflow as supporting
coverage, and retain the separate all-50-repository execution objective. Update
the issue's opening ledger when task, validated mapping or native execution
counts change.
The [versioned queue](../workflow_queue.json) records exact proposed target IDs,
conditional coverage gains, overlapping protocols, source gaps and the next
selection. Recompute remaining targets after each completed or rejected candidate;
queue estimates never promote a coverage status by themselves.
The [full design plan](../workflow_plan.json) assigns every inventoried ID record
to endpoint review and records proposed questions, independent input requirements
and verifier designs. These assignments do not change coverage states.
The [BiomniBench-DA endpoint designs](../workflow_designs/biomnibench-da.json) refine
all 50 question IDs into 45 proposed tasks, with output artifacts, numerical decisions,
negative controls and unresolved independent-data requirements. Their IDs are linked
from each inventory record's `endpoint_review`; they add no validated coverage.
The [BixBench endpoint designs](../workflow_designs/bixbench.json) assign all 205
question IDs to 27 prospective workflows and retain scientific adaptation limits,
including ambiguous denominators, invalid count transformations and unsupported
equivalence claims in source formulations.
The [CompBioBench endpoint designs](../workflow_designs/compbiobench.json) assign
100 question IDs to 84 proposals, explicitly separating observed-data adaptations,
simulated source formulations and subjective or software-only endpoints. Metadata
JSON and source TSV hashes are different artifacts and are labeled separately.

The [BioAgent contracts](../workflow_designs/bioagent.json) specify all ten connected
workflows, including ancestor-relative shared variants, conserved functional
annotations, paired community profiles and complete cell-type expression contrasts.
They distinguish real-input adaptations from the source's planted pedigree allele
and simulated transcript counts. These are planned contracts without new coverage.

The [SciGym input review](../workflow_designs/scigym.json) retains structural counts
and hashes for all 350 supplied partial SBML models. All omit reactions and use
anonymized species. Truth-model columns were excluded; no raw benchmark model is
vendored. Input-structure inspection leaves biological endpoints and independent
observed-data adaptations unresolved, with no per-system coverage assignments.

Enumeration, prompt inspection and executable coverage are separate claims.
An ID obtained from a registry can remain uninspected; a reviewed shared protocol
does not mean every instance was reviewed. Release records state whether the
inventory covers the full public release, public examples, or only identifiers.
Unavailable and gated releases remain listed in the source inventory with the
access gap recorded.

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

The [ScholarQA-Bio inventory](scholarqabench-bio.json) enumerates 1,451 public question
IDs and text hashes without copying prompts or inspiring papers. Shared literature
retrieval/synthesis stages are inspected; individual endpoints remain unreviewed.
Its model-based citation attribution is not used as a reward. A deterministic
extraction or retrieval component cannot claim full open-ended synthesis coverage.

The [LAB-Bench literature/database inventory](labbench-literature-database.json)
contains 801 public IDs and question hashes from DbQA, LitQA2 and SuppQA. Its
[family contracts and overlap review](../workflow_designs/labbench-literature-database.json)
retain all fifty Biomni DbQA stem matches without assigning alias credit: twenty
have multiple source matches, and candidate sets were not inspected. All 82 supplementary
question stems and citation records now have an explicit disposition: 58 provisional
targets in sixteen observed-study audit designs and 24 context/interpretation gaps.
Source supplements, answer choices and expected answers remain uninspected.
Independent inputs and acceptance contracts remain open; authored tasks use Harbor.

The full 520-question DbQA schema audit reuses seven existing
[Biomni database contracts](../workflow_designs/biomni-eval1.json). Their target lists
carry explicit benchmark identities and per-record hashes; these are prospective
operation mappings, not aliases or validated coverage. The 160 clinical queries
retain both sequence-input modes and 61 benign/99 pathogenic requested classes.
Candidate choices and biological records remain uninspected.
