# COVID-19 immune-repertoire verifiable-task pilot

This is a small, reproducible demonstration of turning one scientific data object
into ordinary `TaskSpec`s with objective rewards. Deterministic code chooses the
question, computes the hidden answer, and packages an in-process VerifyIt grader.
The resulting candidates can then cross the ordinary Taskforge workpiece lifecycle
and the existing TaskCompendium curation and TaskTrove/Harbor export pipelines.

The first slice covers TRAV1-2 usage in the T-cell repertoire. It is a plumbing
and task-design pilot, not yet a publishable dataset. After reviewing the first
25 generated tasks, we retained the arithmetic families as controls and refocused
the showcase on tasks that combine quantitative evidence, biological knowledge,
and calibrated interpretation. [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md) records
that review and the 50-task target. Every generated manifest is marked
`unreviewed_do_not_publish` until the source-data rights audit is complete.

## Source

The underlying study is Stephenson et al., *Single-cell multi-omics analysis of
the immune response in COVID-19* (Nature Medicine, 2021):

- paper: <https://doi.org/10.1038/s41591-021-01329-2>
- processed data: ArrayExpress `E-MTAB-10026`
- portal: <https://covid19cellatlas.org/>
- analysis code: <https://github.com/scCOVID-19/COVIDPBMC/>

The current smoke starts from the downstream TheisLab object
`air_repertoire/stephenson2021_tcr_qc.h5mu`, produced for the pinned
`theislab/single-cell-best-practices` notebook revision
`dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5`. Its known identity is:

```text
size:    743651887 bytes
SHA-256: 9247763b3be7c89b6f765ed3f4d1c4a0abd68ad7ee06f4cdc78b7882d9660d61
```

For a release campaign, reproduce the derivation from `E-MTAB-10026` rather than
treating the convenience object as the canonical source.

## The seven-step path

```text
1. Pin and audit the source object.
2. Extract minimal sufficient statistics.
3. Validate counts and remove source identifiers.
4. Enumerate meaningful parameterized analyses.
5. Compute each hidden answer deterministically.
6. Package data, instruction, and verifier as TaskSpec.
7. Route candidates through Taskforge, curate them, inspect them, and freeze a release manifest.
```

### 1. Pin the source

Do not begin from a mutable notebook download. Record the accession, object byte
size, SHA-256, notebook/code revision, and rights status. `extract_counts.py`
refuses a source whose SHA-256 differs from the expected value.

### 2–3. Extract and validate

The source-specific extractor emits only one row per pseudonymous donor and
coarse cell type:

```text
donor_id,cell_type_coarse,n_cells,n_primary_vj_called,n_trav1_2
```

It excludes cell-level records, original patient identifiers, and notebook
conclusions. Run it in an isolated environment with explicit versions:

```bash
SOURCE=$HOME/.cache/lamindb/lamin-eu-central-1/VPwcjx3CDAa2/air_repertoire/stephenson2021_tcr_qc.h5mu
OUT=/tmp/covid-repertoire-counts

uv run \
  --with mudata==0.4.1 \
  --with scirpy==0.25.1 \
  --with pandas==3.1.1 \
  python experiments/post_training/capability_driven_envs/covid_repertoire_tasks/extract_counts.py \
  --source "$SOURCE" \
  --expected-sha256 9247763b3be7c89b6f765ed3f4d1c4a0abd68ad7ee06f4cdc78b7882d9660d61 \
  --output "$OUT"
```

The output contains `repertoire_counts.csv` and `provenance.json`.

### 4–6. Generate objectively graded tasks

The generator currently has seven families:

| Family | Objective result |
| --- | --- |
| `aggregate_usage` | pooled called-chain denominator and TRAV1-2 fraction |
| `donor_median_usage` | donor-level median after an explicit support filter |
| `donor_prevalence` | donors exceeding an explicit usage threshold |
| `median_comparison` | donor-level comparison between two cell states |
| `rank_cell_types` | highest median among sufficiently supported states |
| `annotation_triangulation` | combine receptor enrichment, donor consistency, and a biological limitation |
| `analysis_unit` | contrast pooled and donor-level estimates and identify the biological replicate |

Every task also mounts a data dictionary and a cited MAIT biology card. The two
showcase families require applying those materials rather than treating cell
labels as arbitrary strings. Every instruction fixes the denominator, inclusion rule, aggregation, tie rule,
rounding, and JSON schema. The hidden `StructuredExactSpec` answer is recomputed
from the solver-visible CSV; it is not copied from a notebook conclusion.

```bash
RELEASE=/tmp/covid-repertoire-tasks-v1

PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/generate_tasks.py \
  --counts "$OUT/repertoire_counts.csv" \
  --source-revision sha256:9247763b3be7c89b6f765ed3f4d1c4a0abd68ad7ee06f4cdc78b7882d9660d61 \
  --count 50 \
  --output "$RELEASE"
```

This writes:

```text
tasks.jsonl     ordinary TaskCompendium TaskSpecs
manifest.json   input/output digests, task IDs, generator version, release status
```

### 7. Route through Taskforge, curate, inspect, and refocus

`taskforge_flow.py` does not accept serialized task answers as authoritative. For
each candidate it creates a grounded `TaskProposal` and immutable `build_input`,
then runs Taskforge triage, static authoring, deterministic build, objective
positive/negative grader canaries, review, event persistence, and terminal accepted
or rejected state. The build recomputes the task from the count table and stable
candidate identity.

```bash
TASKFORGE_ROOT=/tmp/covid-repertoire-taskforge-workpieces
TASKFORGE_RELEASE=/tmp/covid-repertoire-taskforge-release

PYTHONPATH=. uv run --with-editable lib/taskforge-adapters --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/taskforge_flow.py \
  --counts "$OUT/repertoire_counts.csv" \
  --source-revision sha256:9247763b3be7c89b6f765ed3f4d1c4a0abd68ad7ee06f4cdc78b7882d9660d61 \
  --count 50 \
  --root "$TASKFORGE_ROOT" \
  --output "$TASKFORGE_RELEASE"
```

The release contains `accepted_tasks.jsonl` and a manifest that records source and
output digests, task IDs, validation performed, and the remaining release gates.
The workpiece root retains proposals, rounds, drafts, failures, validation events,
and terminal decisions for resume and audit.

Next, use the existing local Zephyr curation pipeline and TaskTrove/Harbor exporter:

```bash
PYTHONPATH=. uv run --project lib/taskcompendium python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/curate_local.py \
  --release "$TASKFORGE_RELEASE" \
  --normalized-output /tmp/covid-repertoire-normalized \
  --tasktrove-output /tmp/covid-repertoire-tasktrove
```

This validates every accepted `TaskSpec`, preserves its Taskforge ID as a tag,
assigns canonical curation provenance, writes normalized Parquet, and emits the
legacy TaskTrove/Harbor `tasks.parquet` plus manifest. Structured JSON answers are
lowered to a dedicated answer file and remain graded by `StructuredExactSpec`.
The compatibility export builds no images and runs no Harbor runtime; its manifest
therefore records `runtime_verified=false`.

Run the focused tests:

```bash
PYTHONPATH=. uv run --project lib/taskforge pytest \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/tests -q
```

Render a private sheet containing prompts, hidden answers, and the review checklist:

```bash
PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/render_review.py \
  --tasks "$RELEASE/tasks.jsonl" \
  --output "$RELEASE/private-review.md"
```

Taskforge's local deterministic acceptance means only that proposal, build, and
objective-grader contracts passed. It is **not** publication acceptance. Manually
inspect every instruction/answer pair and run model solver/adversary canaries before
promoting a release. The first such review found that biologically named arithmetic
did not actually require biological reasoning; that finding and the resulting
refocus are preserved in `DESIGN_REVIEW.md`. Generation should reject degenerate
candidates rather than fill a requested quota with meaningless variants.

## Scaling to the first 50

The current table supports useful arithmetic, donor-aware aggregation, and a
first annotation-triangulation lesson, but not 50 scientifically diverse showcase
tasks by itself. Scale by adding
new **data products and generator families**, not by paraphrasing these prompts:

- clonotype expansion and clone-size distributions;
- repertoire diversity and rarefaction;
- arbitrary V/J-gene usage and enrichment;
- severity/outcome contrasts with donor-level covariates;
- BCR isotype and mutation summaries;
- paired transcriptome/repertoire consistency;
- explicit QC and sensitivity-analysis tasks.

Each source-derived table should have its own schema, extraction test, and
provenance digest. All tasks from this cohort stay in one dataset split. A
separate cohort or source family is required for held-out evaluation.

## What this demo intentionally omits

- It does not redistribute the source object or generated count table.
- It does not claim that the ArrayExpress or derived-object rights audit is done.
- It does not use an LLM judge; these rewards are objective.
- It does not ask open-ended biological questions with multiple defensible answers.
- It does not turn superficial prompt paraphrases into additional task IDs.
- It does not publish the local TaskTrove compatibility export or mark its runtime verified.
- It does not replace pending human review and model solver/adversary trials with structural Taskforge acceptance.
