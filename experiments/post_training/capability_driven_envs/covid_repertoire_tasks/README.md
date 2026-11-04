# COVID-19 immune-repertoire verifiable-task pilot

This is a small, reproducible demonstration of turning one scientific data object
into ordinary `TaskSpec`s with objective rewards. It is deliberately simpler than
model-authored open-ended Taskforge: deterministic code chooses the question,
computes the hidden answer, and packages an in-process VerifyIt grader.

The first slice covers TRAV1-2 usage in the T-cell repertoire. It is a plumbing
and task-design pilot, not yet a publishable dataset. Every generated manifest is
marked `unreviewed_do_not_publish` until the source-data rights audit is complete.

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
7. Validate, inspect, and freeze a release manifest.
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

The initial generator has five families:

| Family | Objective result |
| --- | --- |
| `aggregate_usage` | pooled called-chain denominator and TRAV1-2 fraction |
| `donor_median_usage` | donor-level median after an explicit support filter |
| `donor_prevalence` | donors exceeding an explicit usage threshold |
| `median_comparison` | donor-level comparison between two cell states |
| `rank_cell_types` | highest median among sufficiently supported states |

Every instruction fixes the denominator, inclusion rule, aggregation, tie rule,
rounding, and JSON schema. The hidden `StructuredExactSpec` answer is recomputed
from the solver-visible CSV; it is not copied from a notebook conclusion.

```bash
RELEASE=/tmp/covid-repertoire-tasks-v1

PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/generate_tasks.py \
  --counts "$OUT/repertoire_counts.csv" \
  --source-revision sha256:9247763b3be7c89b6f765ed3f4d1c4a0abd68ad7ee06f4cdc78b7882d9660d61 \
  --count 25 \
  --output "$RELEASE"
```

This writes:

```text
tasks.jsonl     ordinary TaskCompendium TaskSpecs
manifest.json   input/output digests, task IDs, generator version, release status
```

### 7. Validate and inspect

Run the focused tests:

```bash
PYTHONPATH=. uv run --project lib/taskforge pytest \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/tests -q
```

Before increasing the count, manually inspect every instruction/answer pair and
run solver canaries. Generation should reject degenerate candidates rather than
fill a requested quota with meaningless variants.

## Scaling beyond the first 25

The current table supports useful arithmetic and donor-aware aggregation tasks,
but not a scientifically diverse library of several hundred. Scale by adding
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
