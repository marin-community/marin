# Wave-eight curriculum scale-out

Wave eight promoted six subject graphs after one generation and one repair pass. The canonical catalog now has 22
subjects, 107 of 151 inventory guideposts, 697 nodes, 638 capabilities, and 59 groups. Raw discovery tasks, blind
tasks, reviews, and design audits are stored in Loom artifact `curriculum-wave8-evidence`; the current runtime
catalog is registered in `../catalog_artifact.py`. Artifact revision 1 has source JSON SHA-256
`52b9a5907765ca21602dfebef4f4782910f93d6d16b714899105f275fb53dd75` in session channel `s2uyqg34`.

## Setup

This report is part of the historical C-series scale-out; it does not describe the later D-series catalog.

The wave sampled 16 model-visible tasks per subject from TaskTrove Clean `2026.09.10.9`, pinned by Parquet SHA-256
`26d0613cc017f5564d5bc8af29596efefa42f839f0f38842ffe1e6885d20da66`. The broken `nl2bash` source was excluded.
Each subject also received 24 tasks from a curriculum-blind Sol/high generator. Healthcare, hardware, and business
operations used four held-out MMLU test questions at revision
`c30699e8356da336a370243923dbaf21066bb9fe`; answers were omitted. No out-of-distribution question was inspected.

Generation, blind-task generation, holistic review, and blind-fit judgment used `gpt-5.6-sol` with high reasoning.
The provider controlled sampling parameters. The prompts and role isolation rules are in `../workflow.md`.

## Results

| Subject | Version | Nodes/capabilities | Last holistic result | Last blind fit |
| --- | --- | ---: | ---: | ---: |
| DevOps Infrastructure & Cloud | `wave8-2` | 34/29 | 94, low confidence | 18/24 before repair |
| Computer Vision & Document AI | `wave8-2` | 39/35 | 74 before repair | 13/23 before repair |
| Hardware Embedded & Architecture | `wave8-2` | 38/34 | 86 before repair, low confidence | 19/24 before repair |
| Web Backend & Distributed Systems | `wave8-3` | 29/24 | 65 before final repair | 23/24 before final repair |
| Business Operations & Admin | `wave8-2` | 42/35 | 88 before repair | 19/24 before repair |
| Healthcare Informatics | `wave8-2` | 22/17 | 94, low confidence | 22/24 |

Every final graph passes `Curriculum.check_generation_contract(maximum_depth=4)`. The final repairs addressed all
named structural blockers. Four final versions were not sent through another full review after repair; the table keeps
their last independent measurements rather than treating the repair as measured improvement.

## Recurrent findings

The sample exposed three distinct concerns:

1. Capability boundaries still failed when one section shared subject nouns but changed the decisive operation. The
   repairs split construction from diagnosis, retrieval from correspondence, memory budgeting from energy control,
   and document creation from lifecycle reconciliation.
2. Representative probes often named an artifact without supplying the schema, trace, measurements, or capacity
   inputs needed to produce it. The backend repair replaced eight such probe families with self-contained inputs.
3. Task evidence was uneven. TaskTrove supplied strong backend and clinical-knowledge evidence, but little direct
   evidence for semantic vision, digital design, IaC, cloud cost, procurement, or clinical workflow operations.

The third concern describes evidence maturity. The rubric now reports evidence confidence independently from
structural readiness. A score of at least 85 with no structural blocker is `pilot_ready`. Low evidence confidence
directs later sampling. A coherent graph can enter the versioned catalog before its evidence reaches every branch.

## Stopping rule

One repair pass is the default. An uncovered guidepost or invalid schema/generation contract still blocks promotion.
Isolated blind misses, sparse evidence, and provisional mapping quality are recorded for the next version. This avoids
adding one capability per sampled task and keeps scale-out moving while preserving enough provenance to revise a
subject when broader evidence arrives.

At the current density, full coverage projects to about 984 nodes. The observed range is 8 to 51 capabilities per
subject, so this projection is a catalog-level estimate rather than a per-subject target.
