# Bioinformatics notebook Taskforge pilot

This directory inventories the first source candidates and task seeds for testing
beta's LLM-judge and LLM-corrector RL paths. The tasks are intentionally
open-ended scientific analyses. The source notebooks ground reference answers and
rubrics; they are not executable verifiers.

## Source

The initial candidates come from the expert-maintained, Apache-2.0 repository
[`theislab/single-cell-best-practices`](https://github.com/theislab/single-cell-best-practices),
pinned at revision `dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5`.

- `notebooks.jsonl` groups related notebooks by underlying source dataset. The
  group is the split boundary: tasks built from one group may not cross
  train/development/evaluation splits.
- `task_seeds.jsonl` contains 20 candidate task briefs over five source groups.
  A seed fixes the analysis anchor and desired scientific judgment, but does not
  prescribe final wording, a reference answer, a rubric, or judge controls.
  Taskforge authors and calibrates those later.
- `reference_task_designs.md` hand-authors the intended solver experience,
  private reference interpretation, rubric, controls, and build blockers for
  three COVID immune-repertoire seeds. These are target designs for the first
  smoke, not runnable `TaskSpec` objects.

Every row is currently `candidate`. It is not eligible for generation until its
underlying data license, immutable source object, solver-visible capsule, private
evidence, and runtime have been audited. The repository license does not by
itself establish permission to redistribute every referenced dataset.

The notebook and original external dataset are curator-side inputs only. Before
Taskforge runs, a deterministic, human-reviewed extraction creates a compact
authoring packet containing the task brief, minimal solver-visible sufficient
data, narrowly scoped private evidence, and provenance pointers. Taskforge does
not receive the complete notebook or original dataset. The first example
extractor is `extract_covid_mait_packet.py`; generated packets remain uncommitted
while the underlying dataset rights are unreviewed.

## Intended flow

```text
pinned notebook family + source dataset (curator side only)
  -> data/license/runtime audit and deterministic manual extraction
  -> compact authoring packet (brief + solver capsule + scoped private evidence + provenance)
  -> fully hydrated OpenEndedTaskBundle
  -> OpenEndedProposalSource
  -> TaskProposal
  -> Taskforge build and LLM-judge calibration
  -> accepted TaskSpec with a frozen reference and rubric
  -> beta group-relative judge/corrector trajectories
```

The solver sees the generated question, data files, and runtime. It does not see
the source notebook, its outputs, the reference answer, rubric, or private
authoring evidence. Taskforge uses those private materials to construct a
source-grounded `llm_as_judge` contract and to test known-correct, partial, and
materially wrong responses.

## Open-ended reward contract

Taskforge authors one semantic task before any student rollout. Its frozen rubric
combines task-specific scientific requirements, shared scientific-quality
criteria, and prescriptive anti-hacking gates. Candidate text cannot add or alter
criteria. Use deterministic checks for numerical, reproducibility, citation, or
artifact properties whenever the task exposes them; an LLM judge evaluates the
remaining scientific interpretation.

The beta campaign samples a cohort of complete trajectories for the same
`TaskSpec`, blinds their order, and compares them under the same rubric. A bounded
swapped-ring tournament covers every candidate in both positions without a full
quadratic round robin. The resulting scorepad records task and rubric digests,
judge and prompt identity, ordered pairwise ratings, hard-check caps, the
calibrated output-token budget, final rewards, and centered within-task rewards.
Over-budget candidates automatically lose to in-budget candidates; failed hard
checks cap the comparative score. Taskforge validation remains pointwise: cohort
construction and group-relative reward shaping belong to the downstream rollout
campaign, not to `TaskSpec`.

## CoreWeave storage

TaskTrove is an upstream Hugging Face dataset (`open-thoughts/TaskTrove`), not a
separate physical CoreWeave bucket. Its normalized Marin campaign artifacts do
have an existing prefix in the shared bucket:

```text
s3://marin-us-east-02a/marin/rl-data/tasktrove/02923004846e4e73862c20962f823a6d05100e7a/
  campaign-2026.10.06.13/
  campaign-2026.10.06.14/
  campaign-2026.10.06.15/
  campaign-2026.10.06.16/
  campaign-2026.10.06.17/
```

Existing Taskforge artifacts use `marin/taskforge/` in the same shared bucket,
with top-level prefixes including `atlas/`, `cluster_rollout_probe/`, `gen1000/`,
`ladder/`, and `phase11/`.

Keep generation inputs and evidence under Taskforge's existing namespace:

```text
s3://marin-us-east-02a/marin/taskforge/bioinformatics/notebook-pilot-v1/
  manifests/
    notebooks.jsonl
    task-seeds.jsonl
  sources/<notebook-id>/
    private/                 # pinned notebook and executed evidence
    solver/                  # raw solver-visible files
    provenance.json
  capsules/<notebook-id>/
    solver-capsule.tar.zst
    manifest.json
  runs/<run-id>/             # Taskforge event logs and validation evidence
  releases/<release-id>/     # accepted TaskSpecs and release manifest
```

Do not copy an artifact into `marin/rl-data/` merely to make it visible to beta.
The normalized task-curation catalog can reference the immutable Taskforge
release. Publish a separate RL-data artifact only when the accepted TaskSpecs are
lowered into the training wire format.

The proposed `marin/taskforge/bioinformatics/` prefix does not exist yet. The
CoreWeave object-storage credentials are stored in GCP Secret Manager as
`cw-object-storage-key-id` and `cw-object-storage-key-secret`; load them into
`CW_KEY_ID` and `CW_KEY_SECRET` for `fsutil` without printing their values. An
in-cluster CoreWeave job receives corresponding S3 configuration from Iris.

## First gate

Audit one source group at a time. The recommended order is:

1. `visium-adult-mouse-brain`: smallest analysis surface and no model fitting.
2. `covid-immune-repertoire`: compact tables and several distinct judgments.
3. `kang-pbmc-ifnb`: rich linked DGE, enrichment, and communication analyses.
4. `haber-intestinal-infection`: stochastic compositional models and heavier dependencies.
5. `neurips-bmmc-site4-donor8`: raw/filtered matrices, R dependencies, and annotation models.

The first generation smoke should use three approved seeds from one source
family. Repeated student rollouts, rather than hundreds of near-duplicate task
prompts, provide the first few hundred beta training examples.
