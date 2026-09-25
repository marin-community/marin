# Benchmark question review

These files support planning Harbor tasks with executable rewards. Each inventoried ID release has a question review; OOD questions are not imported. Source tasks can qualify through a faithful executable reframing even when their original scorer uses an LLM. No LLM judge is in scope.

## Planning units

The vocabulary has two independent axes:

| Axis | Meaning | Example |
| --- | --- | --- |
| Operation family | General analytical work required by the question | Statistical inference |
| Scientific context | Biological measurements, objects or phenomena being analyzed | Gene expression |

The 11 operation families and 19 scientific contexts are defined in
[`taxonomy.json`](taxonomy.json). That version-controlled file is authoritative
for category IDs, names, descriptions and assignment rules. Both HTML explorers
and the Markdown rankings are generated from it. Edit definitions there, then
regenerate the views so descriptions and assignments can be reviewed together.

Use operation–context combinations to identify gaps and design connected tasks.
For example, differential expression can combine Statistical inference with Gene
expression. Normalization belongs to Data preparation when the solver must
perform it. Fine requirements such as hypothesis testing, effect estimation and
multiple-testing correction remain `analysis_tags`; they are not a third ranked
axis. Source benchmark identity supplies provenance.

Contexts overlap and do not form a hierarchy. Single-cell, spatial, longitudinal
and perturbational study characteristics remain context tags. Formats and
repositories are separate metadata. Broad source labels such as immunology or
microbiology do not establish Immune repertoires or Ecology without evidence.
Unresolved mappings remain explicit annotation gaps.

Visualization is an output property. A required figure can have executable
checks on its data, labels, scales or geometry, but aesthetic or communication
quality is outside the reward. No visualization category receives frequency
credit.

## Input stage and decomposition

A workflow name does not establish which operations the solver must perform. Computing differential expression from replicate counts differs from filtering an existing result table. Querying a ClinVar classification differs from deriving a variant consequence. A prediction submission does not require computing the metric used by the benchmark grader.

BixBench-Verified's 50 questions were reviewed individually for the first decomposition draft. One question (`bix-46-q4`) has no assigned operations because its input stage remains unresolved. Other assumptions appear in each question's `decomposition_note`; some proposed connected workflows explicitly start upstream of supplied result tables and require that adaptation to be recorded. Question review does not certify a runnable verifier.

Other benchmarks retain provisional mappings from the previous category review. Fine operations are grouped into families without inferring a complete workflow. A specialized source description never automatically expands into a fixed operation list. Empty and partial lists are annotation gaps, not evidence that no operations are needed. Global rankings expose these gaps and count all eligible records in their denominator.

Each benchmark JSON is a schema-4 manifest linking bounded question files under its same-named directory. Question records contain:

- `workflow`: the specific source-family description.
- `operations`: broad operation families, potentially incomplete.
- `analysis_tags`: finer analytical requirements.
- `output_tags`: required output properties, currently `figure`.
- `context_review_needed`: previous broad labels awaiting a supported context mapping.
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

The standalone [benchmark site](../../../../docs/experiments/bio-benchmarks.html) has Operations and Scientific contexts navigation. Global rankings open on `#statistics/operations`; the other axis is `#statistics/scientific_context`. The benchmark directory remains accessible through the provenance links. Select a category to inspect benchmark contributions. Benchmark and question routes are `#benchmark/<encoded-release>` and `#question/<encoded-release>/<encoded-question-id>`.

Source links and access notes live on each benchmark page. OOD exclusions appear below the benchmark directory. Missing source inventories remain visible without invented question IDs. The broader task-generation explorer is a separate artifact.

The generated [BixBench-Verified Markdown](../../../../docs/experiments/bixbench-verified-competencies.md) contains the question-level decompositions. [Global rankings](../../../../docs/experiments/bio-benchmark-categories.md) are also available as Markdown.
