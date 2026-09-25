# Task authoring

[Planning overview](index.md) · [Requirements](requirements.md) · [Validation](validation.md)

Turn inspected scientific use cases into Harbor tasks. Optimize for using established tools: analysis scripts, metadata handling, method configuration and connected scientific decisions. Release-note bug fixes and new algorithm implementation are not the initial target.

## Task units

Use these working definitions while reviewing examples:

- A **workflow archetype** is a recurring scientific job with an input stage, analysis objective and output, such as replicate-aware differential expression from counts.
- A **recipe** specifies how to construct and validate tasks for an archetype, including permitted data/design variations. Its exact granularity remains open.
- A **task instance** is a concrete question, pinned inputs, environment and executable grading contract.

Start with one substantial task per recipe, then add variations that provide useful practice. There is no task-count cap or easy/medium/hard quota. A complete task is complete relative to its starting point: a count-matrix-to-result analysis need not repeat alignment. Connected tasks should require their stages to inform one another.

## Proposal contract

| Field | Required contents |
| --- | --- |
| Question and boundary | Scientific result sought and the supplied starting stage |
| Input data | Observed/adapted/simulated provenance; accessions or simulator/version; units, dimensions, transformations and redistribution terms |
| Required work | Scientific decisions and operations the solver performs |
| Outputs | Complete artifacts, identities, units and necessary intermediate evidence |
| Reference | An input-reading script running the actual packages |
| Verification | Acceptance criteria, tolerances, valid alternatives and incorrect-output controls |
| Resources | Pinned environment, CPU, memory, disk, timeouts and network policy |
| Coverage | Operations, scientific contexts, formats, repositories, study lineage and source links |
| Variation | What can change, validity constraints and the additional practice provided |

The proposal worker should expose unresolved assumptions before construction. Freeze the scientific contract before computing expected outputs. Initially, an explicit analysis protocol is easier to validate than unrestricted method choice; it can still require substantial input handling, experimental design and integration.

## Input variation

| Input source | Transcriptomics example | Contribution |
| --- | --- | --- |
| Observed | Published counts and sample metadata | Experimental variation, annotation issues and confounding |
| Adapted observed | Supported contrasts, valid subsets or different starting stages | Additional work grounded in actual research |
| Simulated | Counts with specified effects, library sizes and dispersion | Controlled variation and known truth for targeted checks |

Prioritize breadth across independent studies, organisms, designs, questions and workflow stages before making many variants of one dataset. Keep shared-study and recipe identifiers so instance counts do not imply independent scientific coverage. No observed/simulated percentage is set.

Every instance needs validation. Preserve biological replication and estimable contrasts where required by the scientific design; do not relabel conditions while retaining the original interpretation or represent fabricated replicates as observed samples. Document simulation assumptions and check relevant properties. New seeds can supply repeated practice but do not add new workflow coverage. Changing identifiers alone supplies little useful diversity.

## Build and orchestration

1. Consume an inspected source record and structured proposal. Deduplicate by workflow, study and scientific question.
2. Give one candidate to an authoring worker in an isolated workspace. It produces instructions, input manifest, pinned environment, executable reference and grader.
3. Submit the candidate to the independently controlled [validation process](validation.md). The author cannot accept its own task by editing a validation record.
4. Record failures and repair or reject the candidate. Release accepted artifacts through the [publication process](storage.md).

Use bounded parallel workers and resumable candidate states when implementation starts. Cache source inputs and environments, avoid duplicate downloads, and record authoring cost, validation time and rejection causes. Model, concurrency, budget and repair limits remain to be decided. This design does not launch workers.

## Prompt and run versioning

Version prompts, output schemas, orchestration and validation rules together. Keep shared [requirements](requirements.md) authoritative and reusable; separate instructions from task-specific structured inputs.

| Prompt role | Output |
| --- | --- |
| Discover | Evidence-backed repository candidates matching search criteria |
| Inspect | Grounded source records with runnable examples and limitations |
| Propose | Scientific task specifications, variations and verification plans |
| Build | Harbor package, input manifest, native reference and executable grader |
| Challenge | Ambiguities, shortcuts and runnable incorrect submissions with expected grader outcomes |

For each run record the prompt revision, resolved prompt, structured inputs, source revisions/hashes, model configuration, orchestration/validation revisions, outputs and results. These enable auditing and reruns, not deterministic model generation. Exclude credentials from records.

Develop prompts using reviewed [transcriptomics examples](examples/transcriptomics.md), then check revisions on those examples before scaling. Independent trial solves use only solver-visible materials and remain separate from challenge inspection that exposes hidden answers. Prompt files and worker implementations are future work; these pages specify their roles and contracts.
