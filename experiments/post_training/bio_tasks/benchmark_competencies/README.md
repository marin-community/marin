# Benchmark question review

These files support planning Harbor tasks with executable rewards. Each inventoried ID release has a question review; OOD questions are not imported. A source task can qualify even if its original scorer uses an LLM: missing conventions can be frozen, and an interpretation can be reframed as a check of computed evidence when the scientific purpose survives.

Each benchmark JSON is a manifest linking bounded question files under its same-named directory. Those question files are the editable review records. `taxonomy.json` defines two independent facets:

- **Analytical skill:** an operation with a checkable result, such as association analysis, differential expression or orthology assessment.
- **Biological application:** the setting in which it is used, such as transcriptomics or molecular evolution. Single-cell and spatial context can overlap applications.

A workflow connects skills to an actual question, scientific decisions and output artifacts. Use workflow contracts to select and design new tasks. The facets help find gaps; their Cartesian product is not a task-generation plan. Avoid credit for operations already performed in supplied inputs, or mentioned only as background. Some specialized skills remain biological because flattening differential expression into generic regression would hide important requirements.

## Review status

This planning inventory contains question-specific, source-protocol and first-pass rule-assisted assignments. `annotation_basis` identifies their origin. An optional `category_review` records the scope and rationale of an operation-boundary review. Shared-protocol review applies the same required input/output contract to its instances; it is not individual verifier validation. Original recognition rules have been removed from the current vocabulary because keyword matches cannot distinguish solver work from background or supplied results. Their historical form remains in Git.

The operation-boundary pass covers the available BioML-bench and BioXArena output contracts, SciGym's shared protocol, BioKGBench's three query protocols, and the database/sequence protocols in LAB-Bench and Biomni-Eval1. Other assignments remain drafts, including application labels inferred from heterogeneous question text. Inspect the original question and proposed contract before commissioning a task.

## Assignment boundaries

`taxonomy.json` records the shared assignment policy, checkable outcomes, boundaries for easily confused operations, and application scopes. Use the most specific operation for a required endpoint. Add another skill when it represents a distinct required result or scientific decision. Do not add generic statistics for every count, regression for every differential-expression model, or data preparation for every file load.

| Source requirement | Assign | Do not infer |
| --- | --- | --- |
| Normalize counts, then compute PCA | Normalization; dimensionality reduction | A specific normalization method unless required |
| Predict protein abundance from supplied normalized RNA | Predictive regression; transcriptomics and proteomics applications | Normalization, integration or the grader's RMSE computation |
| Predict differential-expression values for unseen perturbations | Predictive regression; transcriptomics application | Fitting a differential-expression contrast |
| Retrieve a ClinVar classification | Biological database querying; genomic variation application | Recomputing variant consequences |
| Compute a contrast from replicate expression measurements | Differential expression | Generic regression and statistics as automatic extra credit |
| Submit segmentation masks or bounding boxes | Image segmentation or image object detection | Image-level classification alone |
| Choose experiments and recover an SBML model | Computational experiment design; mechanistic modeling and simulation | Evaluation performed only by the benchmark grader |

Applications describe the analyzed biological objects and endpoint. Publications, databases and input formats are source metadata, so **Biological knowledge resources** is no longer an application. An unresolved application is an empty list, displayed explicitly; the task still contributes to skill counts and the eligible denominator. Single-cell and spatial context remain cross-cutting tags. **Spatial analysis** is also a skill when spatial relationships must actually be computed.

For authoring, select a connected workflow by its required operations, input stage, scientific decisions and executable outputs. Change the biological study, comparison or data while preserving those requirements. Category counts help locate candidates; a final-answer match alone cannot prove that a solver exercised every labeled intermediate operation.

Each task has one disposition:

- `reframe`: a proposed route to a deterministic numerical, categorical or artifact check. The page shows the original question, proposed framing, output/check pattern, decisions and scope limits. This is eligibility for review, not proof that a verifier exists, that its work fits the resource budget, or that every claimed skill is exercised.
- `excluded`: no faithful bounded framing established. No skill or application labels are assigned. The reason is revisable; this is not a claim that the question can never be verified.
- `unavailable`: the inventoried identifier has no accessible original instruction in the inspected source. It has no labels or frequency credit.

Freezing a scoring function does not validate a claim of biological causality, clinical suitability or experimental efficacy. For example, expression-supported ligand–receptor scores can be checked; they do not demonstrate signaling. Replacing an unrestricted research plan with execution of an arbitrarily chosen plan changes the objective, so those records remain set aside.

Some output/check patterns are shared across questions and require the exposed scientific decisions to become executable. The original source question remains the endpoint specification. Source-stem-only extracts and shared protocol templates are labeled explicitly. Answer keys, private reference values and fixtures are not included.

## Provenance and counts

`inventory_sha256` binds a review to its versioned source inventory. `source_question_sha256` binds its displayed instruction text. `review_sources` links prior design evidence; `source_group` records a shared study/protocol when available. The site verifies row identities, hashes, dispositions and allowed facet labels during generation.

Frequency counts each eligible source record once per skill. Counts overlap and are not deduplicated across benchmark releases. BixBench and BixBench-Verified overlap; SciGym repeats one protocol across model instances; several sources share studies. A large count can therefore indicate repeated questions rather than broad scientific demand. Existing recipe mappings remain separate from proposed skill assignments and executable validation evidence.

Before building a task, verify the proposed contract against the original question, select independent observed inputs, resolve scientific conventions, specify Harbor resources and implement a native reference plus meaningful incorrect-output controls. Prioritize useful missing workflows and their feasibility; raw counts alone do not determine quotas. Corpus split remains train-only; prediction tasks may require internal held-out labels.

The separation of operations and applications follows [EDAM's operations/topics distinction](https://edamontologydocs.readthedocs.io/en/latest/editors_guide.html). Labels here are local drafts, not official EDAM terms. Data formats and repositories remain independent coverage facets.

## Edit and preview

Edit the benchmark’s question files and, when needed, the shared vocabulary. The manifest lists every question file and the total record count. Changing source instructions requires updating their SHA-256; changing an inventory requires reconciling its review before updating the inventory hash. Do not regenerate hand-reviewed assignments from keyword rules.

```bash
uv run python -m experiments.post_training.bio_tasks.coverage_site
uv run python -m experiments.post_training.bio_tasks.coverage_site --check
```

The standalone [benchmark site](../../../../docs/experiments/bio-benchmarks.html) has global rankings at `#statistics` (skills) and `#statistics/applications`, plus `#benchmarks`, `#benchmark/<encoded-release>` and `#question/<encoded-release>/<encoded-question-id>` routes. It opens on global category statistics and embeds only benchmark review data for public HTMLPreview use. Its navigation contains Categories and Benchmarks. Each benchmark page provides its source links and access notes; OOD exclusions appear below the ID benchmark list. The broader task-generation explorer remains a separate artifact. The generated [BixBench-Verified Markdown](../../../../docs/experiments/bixbench-verified-competencies.md) is a compact starting point for vocabulary review. Sources without a task manifest still have benchmark pages explaining the access gap; no task IDs are invented for them.

Global rankings are also generated in [Markdown](../../../../docs/experiments/bio-benchmark-categories.md), with question counts, eligible-record percentages and number of contributing benchmark releases. Selecting a category in the site shows its per-benchmark counts and within-benchmark percentages.
