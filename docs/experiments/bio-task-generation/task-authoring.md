# Task authoring

[Planning overview](index.md) · [Requirements](requirements.md) · [Validation](validation.md)

Turn inspected scientific use cases into Harbor tasks. Optimize for using established tools: analysis scripts, metadata handling, method configuration and connected scientific decisions. Release-note bug fixes and new algorithm implementation are not the initial target.

## Task units

Use two independent categorization axes: **scientific context** (for example, transcriptomics) and **operation** (for example, statistical inference). Single-cell can be an additional tag. Assign operations only when a task requires and verifies that work; receiving precomputed clusters does not establish clustering coverage.

Recipes and instances define the generation structure:

- A **recipe** specifies how to construct and validate tasks: input boundary, scientific objective, required stages, permitted variations and a deterministic verification contract. For example, fit a donor-adjusted treatment contrast from paired RNA-seq counts and metadata.
- A **task instance** is a concrete question, pinned inputs, environment and executable grading contract.

Describe the capability being tested in the recipe's scientific objective. Workflow names such as differential expression can be searchable labels; neither capabilities nor workflow families require another taxonomy level. One recipe can exercise several operations, and several recipes can share the same categories.

Review one substantial instance to establish each recipe, then target ten validated, meaningfully varied instances for recipes intended to scale. There is no task-count cap or easy/medium/hard quota. A complete task is complete relative to its starting point: a count-matrix-to-result analysis need not repeat alignment. Connected tasks should require their stages to inform one another.

## Initial instance target

The first instance establishes the scientific contract and verifier. The first ten test whether generation supports the recipe's declared variation axes without redesigning its scientific contract or verifier for each instance. Ten is an initial breadth target, not evidence that a downstream capability evaluation has enough samples.

Use meaningful differences in compatible datasets, sample structures or scientific comparisons. Ten instances need not come from ten independent studies, but report the number of source studies and the variation represented. Cosmetic changes, arbitrary threshold changes or new seeds alone do not demonstrate this breadth. Focused and integrated tasks sharing observations retain shared lineage.

If a recipe cannot reach ten, record the reason: limited eligible inputs, dependence on one tutorial, repeated bespoke repairs or an inherently narrow scientific use case. Useful narrow recipes may remain with a documented exception; prioritize recipes that support repeatable variation for large-scale generation. Beyond ten, expand according to scientific usefulness, available variation, validation reliability and downstream demand. Track eligible candidate inputs separately from released, validated instances.

## Focused and integrated recipes

Recipes can overlap and compose. A quality-control recipe can be used within several complete analyses. Record reusable stages and compatible input/output contracts without requiring a strict recipe hierarchy. See the [Scanpy example](examples/transcriptomics.md#scanpy-focused-and-integrated-recipes).

Focused tasks isolate scientific operations and make failures easier to diagnose. Integrated tasks test coordination across stages, including data representations, identifiers and the consequences of upstream decisions. Each integrated recipe needs its own scientific contract and end-to-end verification; passing its components separately does not establish correctness of their composition.

Internal component reuse does not require exposing a procedural checklist to the solver. Instructions can state a scientific objective and the methodological constraints needed for deterministic verification. Completing a workflow means delivering its defined scientific result; executing every plotting or demonstration cell in a source notebook is not required.

## Deriving recipes from pipelines

Inspect each rule or process as a candidate recipe boundary. For Snakemake, use the rules together with their input/output dependencies and configuration; apply the same procedure to other workflow engines. Record each candidate's inputs, outputs, scientific work, verification approach and resource needs.

| Pipeline component | Candidate treatment |
| --- | --- |
| A rule producing a useful scientific artifact, such as gene counts | Focused recipe with staged upstream inputs |
| Closely coupled rules, such as model fitting and contrast extraction | Integrated recipe spanning those rules; keep separate recipes only where each supports a useful question |
| A connected set of rules with branching or aggregation | Integrated recipe with explicit interfaces and all required inputs |
| Download, indexing or format-handling rule | Usually prepared infrastructure; a separate task is useful only when it tests a substantive input-selection, format or configuration decision |
| Plot or report rule | Grade numerical artifacts underlying the report when they preserve the scientific objective; visual appearance is not a reward |

The rule is a discovery unit; the recipe is the scientific generation contract. One recipe may use several rules, and one configurable rule may suggest several recipes with different scientific assumptions. Avoid enumerating every possible subgraph. Prioritize useful scientific endpoints and reusable intermediate artifacts. Instantiating a rule for many samples produces executions, not automatically distinct recipes.

For each selected boundary, stage compatible upstream artifacts, expose meaningful scientific decisions, define a deterministic verifier and identify variation sufficient for the initial ten-instance target. Record whether the task executes the source workflow or reuses its component packages. See the [STAR–DESeq2 worked example](examples/star-deseq2.md).

## Work item: extraction rules for code and notebooks

Define how to identify candidate recipes from functions, methods, entire classes, individual cells or chunks, connected groups of cells, and complete notebooks or analysis documents. Cover multiple languages and formats, including Python and R, Jupyter notebooks, R Markdown and Quarto. The extraction rules and recipe boundaries for these artifacts remain to be designed.

Compare tasks that ask the solver to use existing code, compose operations or implement specified functionality against the project's scientific-analysis goal. Determine when an artifact supports a useful focused task, when several artifacts should form an integrated task, and when code only supplies setup or implementation details. Include dependencies and hidden state, meaningful instance variation, and deterministic verification in that design.

Use the [pipeline development testbed](index.md#pipeline-development-testbed) to develop and review these rules. Revisit the [Scanpy example](examples/transcriptomics.md#scanpy-focused-and-integrated-recipes) and select additional examples that expose different languages, document formats or code organization. Record source-specific examples as the rules are developed; this work item does not establish recipes or validated tasks.

## Proposal contract

| Field | Required contents |
| --- | --- |
| Question and boundary | Scientific result sought and the supplied starting stage |
| Input data | Observed/adapted/simulated provenance; accessions or simulator/version; units, dimensions, transformations and redistribution terms |
| Data adequacy | Recipe-specific design, coverage and quality checks, with rejection criteria defined before selecting a subset or computing expected outputs |
| Required work | Target capabilities, scientific decisions and operations the solver performs |
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

Subsetting is a standard authoring technique for fitting realistic workflows into the sandbox. Consider reads, samples, features or reference regions according to the task's objective. Preserve paired reads and required design structure; declare the subset and any narrowed scientific scope. Pin the selection procedure, seed where used, source asset and resulting hashes, then recompute expected results on the subset. Measure both resource use and scientific adequacy. Several subsets of one study retain shared lineage, and changing subset seeds alone does not establish recipe breadth.

Define adequacy criteria for the recipe before selecting a subset. Depending on the question, these might check usable counts, retained feature coverage, biological replication or design estimability. Record their results on the delivered inputs and reject or revise inadequate candidates. Criteria need not require significant effects or universal numerical thresholds; do not select subsets merely because they produce a desired conclusion.

## Recipe variation and instance relationships

For each recipe, specify the target capabilities, eligible inputs and prerequisites, permitted variation axes, scientific validity constraints, and how the reference and verifier are constructed for every instance. State which decisions remain constant and which change across instances.

For paired differential expression, a different observed study with a compatible design can supply another instance. A supported contrast within the same study can also supply an instance, but both retain shared study lineage. Starting from transcript quantifications instead of counts adds import and aggregation work and may warrant a separate recipe. Record that boundary explicitly. Cosmetic changes to filenames or thresholds do not establish capability generalization.

Export operation and scientific-context labels, stable recipe and task identifiers, recipe versions, source-study and input-asset identifiers, generation parameters, and derivation links between assets. Record component recipe IDs and versions when composition uses them. A task can use multiple studies or assets. Preserve shared lineage across focused and integrated recipes as well as within them; different starting stages can still derive from the same observations.

This metadata lets downstream users group instances by shared studies, inputs or recipes when designing their evaluations. Task generation supplies the relationships and meaningful variation; split assignment and evaluation protocols remain outside scope. Review a few transcriptomics recipes and their variations before committing to a comprehensive category vocabulary.

## Build and orchestration

1. Consume an inspected source record and structured proposal. Deduplicate by workflow, study and scientific question.
2. Screen source-record license and access information for obvious publication blockers before building. Reject incompatible inputs or hold candidates whose eligibility needs clarification. This lightweight check does not replace final release-eligibility validation.
3. Give one candidate to an authoring worker in an isolated workspace. It produces instructions, input manifest, pinned environment, executable reference and grader.
4. Submit the candidate to the independently controlled [validation process](validation.md). The author cannot accept its own task by editing a validation record.
5. Record failures and repair or reject the candidate. Release accepted artifacts through the [publication process](storage.md).

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

Develop prompts using the [pipeline development testbed](index.md#pipeline-development-testbed), including Scanpy, STAR–DESeq2 and author-familiar paper repositories. Revisit affected cases when the pipeline or prompts change, comparing task framing and verification before scaling. Independent trial solves use only solver-visible materials and remain separate from challenge inspection that exposes hidden answers. Prompt files and worker implementations are future work; these pages specify their roles and contracts.
