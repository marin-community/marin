# Kang PBMC IFN-beta objective-task pilot

This campaign turns an independent perturbation dataset into donor-paired,
objectively graded scientific-analysis tasks. It follows the ordinary local path:

```text
pinned source object
→ compact sufficient-statistics extraction
→ deterministic candidate portfolio
→ Taskforge proposal/triage/build/validation/review
→ accepted TaskSpecs
→ TaskCompendium curation
→ TaskTrove/Harbor compatibility export
```

All current outputs are `unreviewed_do_not_publish`. Taskforge structural
acceptance verifies the build and grader contract; it is not a claim that a task
is educationally effective. For the upcoming demo, all 25 candidates proceed
without item-level human filtering so weak tasks and failure modes remain visible.
Human review remains a later publication gate, not a demo gate.

## Source

The study is Kang et al., *Multiplexed droplet single-cell RNA-sequencing using
natural genetic variation* (Nature Biotechnology, 2018):

- DOI: <https://doi.org/10.1038/nbt.4042>
- GEO: `GSE96583`
- design: eight donors, before and after six hours of IFN-beta stimulation
- measurements in the pinned object: 24,673 cells and 15,706 genes

The current smoke uses the TheisLab convenience artifact
`conditions/differential_gene_expression.h5ad` from
`theislab/single-cell-best-practices` revision
`dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5`:

```text
Lamin UID: osmtqcMMwXHulZL10000
size:      123669441 bytes
SHA-256:   229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff
```

A publishable release must reproduce or verify this object's derivation from
`GSE96583` and complete the source-data rights audit.

## Extract sufficient statistics

The extractor verifies the source digest and emits one row per
`donor × condition × cell type × panel gene`:

```text
donor_id,condition,cell_type,n_cells,total_counts,gene,gene_counts
```

It retains twelve predefined genes: ten interferon-response genes plus ACTB and
GAPDH housekeeping comparators. Original donor labels are replaced with stable
campaign pseudonyms.

```bash
SOURCE=$HOME/.cache/lamindb/lamin-eu-central-1/VPwcjx3CDAa2/conditions/differential_gene_expression.h5ad
COUNTS=/tmp/kang-ifnb-counts

uv run --with anndata --with scipy python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/extract_pseudobulk.py \
  --source "$SOURCE" \
  --expected-sha256 229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff \
  --output "$COUNTS"
```

The real extraction produced 1,488 rows covering eight donors and 124 observed
donor-condition-cell-type groups.

## Candidate families

| Family | Scientific contract |
| --- | --- |
| `paired_gene_response` | donor-paired CPM log2 fold change, consistency, and a transcript-level limitation |
| `cell_type_response_comparison` | compare paired response magnitude between two cell types without claiming mechanism |
| `ifn_panel_gene_rank` | rank a predefined response panel within a cell type |
| `ifn_signature_cell_type_rank` | aggregate gene responses within donor before ranking cell types |
| `analysis_unit` | contrast pooled and donor-paired summaries and identify the correct biological replicate |

Every task freezes the normalization, pseudocount, donor inclusion rule,
aggregation, tie policy, rounding, output schema, interpretation vocabulary, and
limitation code. The solver sees the count table, data dictionary, and cited
biology card. Hidden answers are recomputed from those same public resources.

Generate a review release directly:

```bash
PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/generate_tasks.py \
  --counts "$COUNTS/pseudobulk_counts.csv" \
  --source-revision sha256:229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff \
  --count 25 \
  --output /tmp/kang-ifnb-tasks-25
```

## Taskforge authoring lifecycle

Each selected `Candidate` becomes one concrete `TaskProposal`. Taskforge records
the immutable count-table build input, triages the proposal, invokes a static
build program, reconstructs the `TaskSpec`, grades a reference and severe-negative
control, persists all events and drafts, and records a terminal decision.

```bash
PYTHONPATH=. uv run --with-editable lib/taskforge-adapters --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/taskforge_flow.py \
  --counts "$COUNTS/pseudobulk_counts.csv" \
  --source-revision sha256:229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff \
  --count 25 \
  --root /tmp/kang-ifnb-taskforge-root-25 \
  --output /tmp/kang-ifnb-taskforge-release-25
```

The completed smoke produced 25 structurally accepted workpieces with no build
or grader-control rejection.

## Curation and TaskTrove compatibility

```bash
PYTHONPATH=. uv run --project lib/taskcompendium python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/curate_local.py \
  --release /tmp/kang-ifnb-taskforge-release-25 \
  --normalized-output /tmp/kang-ifnb-normalized-25 \
  --tasktrove-output /tmp/kang-ifnb-tasktrove-25
```

The smoke normalized and exported all 25 tasks with zero rejects. The exporter
writes `tasks.parquet` and a manifest with `runtime_verified=false`; it does not
build or execute the Harbor images.

## Private review and calibration

```bash
PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/render_review.py \
  --tasks /tmp/kang-ifnb-taskforge-release-25/accepted_tasks.jsonl \
  --output /tmp/kang-ifnb-taskforge-release-25/private-review.md
```

For a curated release, review every task as keep, revise, or reject. The current
demo intentionally skips that filter and sends all 25 candidates to model
solver/adversary trials. Record solve rates, answer leakage, runtime, token cost,
and failure modes, including evidence that a task itself is poor. Do not collapse
grader validity and educational value into one opaque score.

## CTT solver calibration

The demo keeps trial-level responses rather than only aggregate solve rates. Run
three ordinary GLM-5.3 rollouts per frozen task through one Zephyr shard per task:

```bash
PYTHONPATH=. uv run --with-editable lib/taskforge-adapters --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/kang_ifnb_tasks/calibrate_ctt_zephyr.py \
  --counts "$COUNTS/pseudobulk_counts.csv" \
  --source-revision sha256:229d767ff229cda5b8ec335ead831ef9eee3908bd763055515ce25874428deff \
  --count 25 --k 3 --max-workers 8 \
  --relay-job /muchanem/glm53-relay-tok-02a \
  --relay-service glm-5.3-tok \
  --output s3://marin-us-east-02a/marin/taskforge/kang-ifnb-demo/ctt-glm53-k3/results.jsonl
```

Each result retains the frozen task digest, model and policy identity, individual
binary rewards, full rollout/grade attempt records, token usage, stop reasons,
and append-only ledger entries. Ordinary solver trials are labeled separately
from future adversarial searches. These records support immediate CTT difficulty
and discrimination summaries and preserve the response matrix for later IRT.

## Current limits

- The gene panel is intentionally small and predefined.
- Transcript response is not direct pathway activity, protein activity, antiviral
  protection, or clinical benefit.
- The source convenience artifact and generated count table are not committed.
- The source rights and canonical derivation audits are incomplete.
- Structural acceptance does not replace human review or model calibration.
