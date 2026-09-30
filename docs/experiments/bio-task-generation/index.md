# Computational biology task generation

This is the canonical planning documentation for [issue #9257](https://github.com/marin-community/marin/issues/9257), as of 2026-09-30. It defines the future pipeline; compatibility with the earlier generator is not required. This authoring branch records prompt-development research. Trials of the authoring prompt and newly authored tasks remain pending.

Generate realistic computational biology tasks with deterministic executable rewards, packaged for Harbor and usable by other people's pipelines. Optimize for using bioinformatics software to answer scientific questions. Analysis scripting, metadata reconciliation and workflow configuration belong in scope. Developing new bioinformatics algorithms or fixing package internals is not the initial target.

Task generation includes repository discovery, input curation, model-assisted authoring and task validation. Recipes define meaningful instance variation and preserve shared-data lineage for downstream users. Data splitting, downstream model training, teacher-trace collection, model selection for training and context budgets are outside scope. Models used to author or trial tasks are part of the generation process.

## Read by activity

| Document | Purpose |
| --- | --- |
| [Requirements](requirements.md) | Scientific, sandbox, runtime and deterministic-reward requirements |
| [Repository discovery](discovery.md) | Find and inspect tools, pipelines, tutorials and paper-analysis repositories |
| [Task authoring](task-authoring.md) | Convert sources into tasks and generate meaningful variations |
| [Prompt experiments](prompt-experiments/index.md) | Recorded trials, exact prompts and outputs, findings, and proposed comparisons |
| [Validation](validation.md) | Execute references, challenge graders, trial tasks and decide release readiness |
| [Storage and publication](storage.md) | Public artifacts, release layout, provenance and solver isolation |
| [Transcriptomics examples](examples/transcriptomics.md) | Concrete candidates for reviewing the process |
| [STAR–DESeq2 worked example](examples/star-deseq2.md) | Inspect a source workflow, define focused and integrated recipes, and plan instance variation |

## Research handoff and repository migration

The destination is [Open-Athena/biotasks issue #3](https://github.com/Open-Athena/biotasks/issues/3).
It calls for preserving this research after the integration-baseline migration,
then promoting supported improvements separately. Raw trials belong on a dedicated
research branch; supported templates belong in `src/biotasks/prompts/`. The issue
does not call for merging the research branch wholesale. No destination migration
or promotion is performed by this documentation update.

The source working agreement is:

- `codex/bio-tasks` is the Marin integration branch, starting at planning commit
  [`72008dd682`](https://github.com/marin-community/marin/tree/72008dd68247318a367a840a4f41e27fb15ff7e1).
  Work in separate worktrees and feature branches. Do not push to integration or
  `main`; eventual reviewed draft PRs target `codex/bio-tasks` and reference #9257.
- This session owns `codex/bio-tasks-authoring`, its prompts, task-authoring
  guidance and `prompt-experiments/`. Feature-branch commits and pushes are
  authorized; no PR is requested at this analysis stage.
- The discovery session owns `codex/bio-tasks-discovery` and `01-discovery/`.
  Its source-ranking research has a separate migration scope.

At pushed checkpoint
[`c97c246912`](https://github.com/marin-community/marin/tree/c97c2469120f91931fdec648714a755d20604b47),
all prompt trials discussed through the uncapped model/effort comparison and UCSC
catalog probe were committed, and the checkout was clean. Its delta from the
integration baseline contains 480 changed paths, all under this documentation
directory and none under `01-discovery/`. The migration issue's drafting-time
note about local round-two/round-three additions is superseded by that checkpoint.
This handoff documentation follows it. Freeze the actual branch head when migration
starts and check again for later commits and local additions.

Read [prompt roles and worker handoffs](task-authoring.md#prompt-and-run-versioning),
the [current discovery decisions](task-authoring.md#current-discovery-decisions),
and the [experiment log](prompt-experiments/index.md) before continuing.
The [evidence-preservation limits](prompt-experiments/index.md#evidence-preservation-and-migration)
distinguish versioned outputs and traces from unavailable or local-only material.
Repair destination links and executable paths separately from immutable historical
snapshots; preserve original hashes and record every adaptation.

Use the completed [discovery migration #2](https://github.com/Open-Athena/biotasks/issues/2)
as the preservation precedent. Its [archive record](https://github.com/Open-Athena/biotasks/blob/6bb14ef5eea9a58b6b704808ff21bdecb8791170/experiments/5-source-discovery/runs/2026-09-30-bucket-archive/README.md)
keeps manifests and verification evidence in Git and raw cache bytes in the public
`open-athena/biotasks` HF bucket. Authoring's local-only caches and logs still need
an explicit archive or retention disposition; the source branch alone does not
complete that preservation. No authoring bucket upload is claimed here.

## Direction

Start from recurring scientific workflows found in repositories, tutorials and reproducible studies. Use benchmarks as optional inspiration and checks for omissions; benchmark question frequencies do not determine priorities. Preserve existing out-of-distribution exclusions when borrowing benchmark material. SciGym is excluded.

Use observed biological data as the corpus backbone, with explicitly labeled adaptations and simulations where useful. Include both focused analyses and connected workflows. Prefer breadth across studies, experimental designs and scientific questions before producing many variants of a few datasets.

Track operations and scientific contexts as separate descriptive axes, plus formats, repository use and biological lineage. For example, differential expression combines statistical inference with gene expression. Single-cell, spatial and longitudinal design may be cross-cutting tags. Neither an exhaustive ontology nor a fixed task-count target is required to begin reviewing candidates.

Recipes specify how to generate and verify concrete task instances. Focused recipes can also contribute stages to integrated recipes, with explicit input/output contracts and end-to-end validation. This composition does not add a categorization level; see [task units and composition](task-authoring.md#task-units).

For recipes intended to scale, establish one instance and then target [ten validated, meaningfully varied instances](task-authoring.md#initial-instance-target). Record shared study lineage and document exceptions for useful narrower recipes. This initial target does not cap later generation.

For a selected repository, develop four prompts: **find units → author a task → solve independently → reflect on and validate the task**. Authoring has explicit proposal and construction phases. An LLM reviewer examines solver traces and executable evidence, then recommends acceptance, revision or rejection. Automated checks run between stages, and an independently controlled harness owns acceptance. See [prompt roles and worker contexts](task-authoring.md#prompt-and-run-versioning).

Work should be public: documentation, code, prompts, task inputs, references, graders and validation evidence. Public reference solutions must remain outside solver-visible task inputs. Publication is subject to the source assets' redistribution terms; see [storage](storage.md).

## Pipeline development testbed

Develop a shared task-generation procedure that can apply across hundreds of
repositories. Compare the same prompt revision across different repository types,
using their source material as inputs. Keep repository-specific expectations and
findings in test cases and reviews. For each failure, distinguish unclear general
guidance from execution mistakes, unavailable evidence, and budget limits. Revise
the shared prompt when the diagnosis supports a general improvement, then check
that change on other repository types before claiming it generalizes.

For unit-discovery trials, review source coverage, unit boundaries, data identity,
provenance, and whether another worker can use the inventory. Record manual
expectations before a comparison and keep them outside the worker's inputs.
Compare prompt revisions with the same model, tools, and budget where possible;
record any differences and preserve each run's [experiment record](task-authoring.md#versioned-artifacts).

Revisit these examples when changing discovery, recipe extraction, instance generation or validation. They are reference cases for diagnosing the pipeline and reviewing task framing, not a ranked source list or a requirement to implement every example now.

| Reference case | What to examine when the pipeline changes |
| --- | --- |
| [Scanpy tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html), with our [recipe discussion](examples/transcriptomics.md#scanpy-focused-and-integrated-recipes) | Extract focused and integrated tasks from a teaching workflow; distinguish numerical verification from open-ended interpretation |
| [Snakemake STAR–DESeq2 workflow](https://snakemake.github.io/snakemake-workflow-catalog/docs/workflows/snakemake-workflows/rna-seq-star-deseq2.html), with our [worked example](examples/star-deseq2.md) | Use rule boundaries and dependencies; preserve scientific decisions while composing stages and subsetting data |
| Gonzalo Benegas's [papers](https://gonzalobenegas.github.io/), [Scholar profile](https://scholar.google.com/citations?user=tJbZmiUAAAAJ) and [repositories](https://github.com/gonzalobenegas) | Find useful tasks in paper-specific code with varying documentation and portability; obtain feedback from a researcher familiar with the original scientific intent |
| [Tim O'Donnell's work](https://timodonnell.github.io/) | Add collaborator-familiar papers and software as further cases for reviewing scientific task framing; select specific examples as the pipeline develops |
| [bedtools](https://bedtools.readthedocs.io/en/latest/) and its [usage examples](https://bedtools.readthedocs.io/en/stable/index.html#interesting-usage-examples) | Inspect documented operations such as `map` and `intersect` as units, alongside scientific usage examples that combine operations |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) and its [user guide](https://github.com/soedinglab/MMseqs2/wiki) | Connect sequence-search, clustering and taxonomy workflows to their component commands; distinguish tutorial examples, reusable databases and placeholder inputs |
| [DESeq2](https://github.com/thelovelab/DESeq2) and its [Bioconductor vignette](https://bioconductor.org/packages/3.23/bioc/vignettes/DESeq2/inst/doc/DESeq2.html) | Discover package vignettes through ecosystem conventions; preserve statistical designs, input alternatives, code-chunk state and dataset provenance across tutorial sections |
| [UCSC Genome Browser binaries](https://hgdownload.soe.ucsc.edu/admin/exe/), with their [source code](https://github.com/ucscGenomeBrowser/kent/tree/master/src/utils) | Review tasks built around standalone command-line utilities, including their input/output conventions |

Gonzalo Benegas has substantial experience using bedtools and the UCSC Genome Browser binaries. Use that familiarity to guide manual review of their task framing and expected outputs.

For a proposed pipeline change, compare the resulting question, recipe boundary, input adaptation, meaningful instance variation and deterministic grading contract on the affected cases. Check whether the framing preserves the scientific work and whether source-code limitations are being confused with lack of scientific value. Record concrete examples and unresolved questions for review. Author feedback informs this review; it does not serve as grading-time judgment.

The testbed starts with these cases; it is not yet an automated regression suite and does not provide exhaustive coverage. Inspect and develop them as the relevant pipeline stage takes shape. Versions used for actual comparisons should be pinned; additional cases can be added when they expose a different failure mode.

## Open decisions

- [Extraction rules for code and notebooks](task-authoring.md#work-item-extraction-rules-for-code-and-notebooks): define candidate boundaries for functions, classes, cells or chunks, and complete analysis documents across languages and formats.
- Recipe boundaries, composition and permitted variation axes, to be calibrated through [transcriptomics examples](examples/transcriptomics.md).
- The vocabulary for operations and scientific contexts, and how demand and missing coverage influence selection. No uniform allocation rule or numerical weighting formula is adopted.
- How much retrieval-only or generic statistical work belongs in the corpus, and how much benchmark analysis to retain.
- The size and composition of the [reusable input collection](discovery.md#reusable-input-collection). Around ten datasets per initial scientific area is a provisional idea; study diversity and recipe compatibility guide selection.
- The initial candidate portfolio and the ongoing independent-trial and human-review sampling policy after calibration.
- Authoring model, worker concurrency, budgets and repair limits. GLM is a candidate; no model service is required by a released grader.
- The public release repository name and account, storage quota and final packaging layout. The storage page proposes Hugging Face; no release repository has been created by this plan.

Current prompt work concerns discovery coverage, source fidelity and reliable
handoffs across repositories. The [experiment log](prompt-experiments/index.md)
records remaining failures and the exploratory model comparisons. Authoring-prompt
trials, a reflection prompt, worker orchestration and task execution remain open.
A focused transcriptomics task and a connected analysis from the same observed
study remain proposed authoring cases, not completed validation.

## Earlier work and supporting catalogs

Earlier implementation and planning are preserved at [checkpoint d3f09bbb3b](https://github.com/marin-community/marin/tree/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af). Their rules and status claims are historical, not the specification for this pipeline. No new task validation or release is claimed by this documentation change.

- [Repository adoption inventory](https://github.com/marin-community/marin/blob/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af/docs/experiments/computational_biology_bioinformatics_packages.md): the original 50 repositories, including dated downloads, stars and citations. Discovery has no 50-repository cutoff.
- [Task catalog and validation evidence](https://github.com/marin-community/marin/blob/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af/docs/experiments/bio-task-catalog.md): earlier candidates and executable checks, with readiness varying by task.
- [Benchmark source inventory](https://github.com/marin-community/marin/blob/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af/experiments/post_training/bio_tasks/benchmark_sources.json) and [question inventories](https://github.com/marin-community/marin/tree/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af/experiments/post_training/bio_tasks/benchmark_tasks): auxiliary discovery material; preserve recorded exclusions, including Terminal-Bench-Science, GeneBench, BioMysteryBench and SciCode.

Maintain decisions in the relevant page and unresolved choices here. Keep issue #9257 as a short summary linking to this documentation. Put dense catalogs and run evidence in their own versioned artifacts rather than expanding the issue body.
