# Benchmark question review

These files support planning Harbor tasks with executable rewards. Each inventoried ID release has a question review; OOD questions are not imported. Source tasks can qualify through a faithful executable reframing even when their original scorer uses an LLM. No LLM judge is in scope.

## Planning units

The vocabulary has three independent facets:

| Facet | Meaning | Example |
| --- | --- | --- |
| Workflow | Connected operations and scientific decisions that answer a question | Differential expression between conditions |
| Analytical operation | Work required by the question or its explicit adaptation | Model fitting, effect estimation, hypothesis testing, multiple-testing correction |
| Scientific context | Measurements, biological objects and study setting | Transcriptomics; paired bulk RNA-seq samples |

Use workflows as the primary unit for choosing new tasks. Operations reveal shared requirements and reusable components; scientific context identifies meaningful variants. The context labels are overlapping descriptors, not a hierarchy of biological domains. The source question and decomposition notes retain finer assay and study-design detail. Formats and repositories remain separate metadata.

Split an operation when it changes task construction, a scientific decision or an executable output. Routine file loading and incidental arithmetic do not need separate labels. Some operations, such as sequence alignment, retain a specific input type when that distinction changes how a task is built and checked. This is a practical planning vocabulary, not a claim that operations are irreducible or universally domain-free.

## Input stage and decomposition

A workflow name does not establish which operations the solver must perform. Computing differential expression from replicate counts differs from filtering an existing result table. Querying a ClinVar classification differs from deriving a variant consequence. A prediction submission does not require computing the metric used by the benchmark grader.

BixBench-Verified's 50 questions were reviewed individually for the first decomposition draft. One question (`bix-46-q4`) has no assigned operations because its input stage remains unresolved. Other assumptions appear in each question's `decomposition_note`; some proposed connected workflows explicitly start upstream of supplied result tables and require that adaptation to be recorded. Question review does not certify a runnable verifier.

Other benchmarks retain provisional workflow/context mappings from the previous category review. Existing labels that already describe general operations migrate directly; a specialized workflow label never automatically expands into a fixed operation list. Empty and partial lists are annotation gaps, not evidence that no operations are needed. Global rankings expose these gaps and count all eligible records in their denominator.

`taxonomy.json` defines the three vocabularies and assignment policy. Each benchmark JSON is a schema-3 manifest linking bounded question files under its same-named directory. Question records contain:

- `workflow`: the specific source-family description.
- `workflows`: normalized planning workflow types; a connected task can span several.
- `operations`: explicit analytical requirements, potentially incomplete.
- `scientific_context` and `context_tags`: overlapping scientific descriptors.
- `decomposition_status`: `question-reviewed`, `input-stage-unresolved`, `provisional` or `not-applicable`.
- `decomposition_note`: question-level rationale when reviewed.
- `category_review`: any earlier category-boundary review, kept separate from operation decomposition.

Prior mixed skill/application fields and keyword-recognition rules are available in Git history. Do not regenerate reviewed assignments from keyword matches.

## Eligibility and verification

Each task has one disposition:

- `reframe`: a proposed numerical, categorical or artifact check that retains the scientific purpose. This does not establish an implemented verifier or feasible resource budget.
- `excluded`: no faithful bounded framing established. No categories or frequency credit.
- `unavailable`: no accessible original instruction in the inspected source. No categories or frequency credit.

Freeze missing conventions when doing so preserves the question. A deterministic proxy alone does not establish biological causality, clinical suitability or experimental efficacy. Replacing an unrestricted research plan with an arbitrary fixed procedure changes its objective. Those questions remain set aside unless a faithful framing can be established.

Before authoring, reconcile the input stage, choose independent observed inputs, resolve design and reference conventions, specify Harbor resources, and implement a native reference plus meaningful incorrect-output controls. An endpoint-only reward does not prove that every annotated intermediate operation was exercised. Task generation, rather than model training or trace generation, is the scope of this inventory.

## Provenance and counting

`inventory_sha256` binds each review to its source inventory; `source_question_sha256` binds the displayed original instruction. `review_sources` links prior design evidence and `source_group` identifies shared studies or protocols. Generation checks identities, hashes, dispositions and allowed labels. Source-stem extracts and shared protocols are labeled explicitly. No answer keys or private fixtures are published.

Each eligible record counts once per assigned category. Releases are not deduplicated: BixBench releases overlap, SciGym repeats a protocol across model instances, and other benchmarks share studies. Raw counts and release breadth help locate candidate workflows; neither determines generation quotas or measures independent scientific demand. Operation counts are additionally limited by incomplete decomposition.

The distinction between operations and context is informed by [EDAM's operations and topics](https://edamontologydocs.readthedocs.io/en/latest/editors_guide.html). These are local planning labels, not official EDAM mappings.

## Edit and preview

Edit the question files and shared vocabulary. Reconcile inventory changes before updating their hash. Update the instruction hash only when the original instruction changes.

```bash
uv run python -m experiments.post_training.bio_tasks.coverage_site
uv run python -m experiments.post_training.bio_tasks.coverage_site --check
```

The standalone [benchmark site](../../../../docs/experiments/bio-benchmarks.html) has Categories and Benchmarks navigation. Global rankings open on `#statistics/workflows`, with `#statistics/operations` and `#statistics/scientific_context` as the other facets. Select a category to inspect benchmark contributions. Benchmark and question routes are `#benchmark/<encoded-release>` and `#question/<encoded-release>/<encoded-question-id>`.

Source links and access notes live on each benchmark page. OOD exclusions appear below the benchmark directory. Missing source inventories remain visible without invented question IDs. The broader task-generation explorer is a separate artifact.

The generated [BixBench-Verified Markdown](../../../../docs/experiments/bixbench-verified-competencies.md) contains the question-level decompositions. [Global rankings](../../../../docs/experiments/bio-benchmark-categories.md) are also available as Markdown.
