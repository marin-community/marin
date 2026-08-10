# Perplexity Gap Analysis

This page documents the `marin.evaluation.perplexity_gap` library, which scores a
model's per-byte perplexity on raw or supervised text slices and compares two
models into a pairwise gap report.

Use it to answer "where is Marin better or worse in bits per byte?" on raw text
slices or supervised target-only text slices. For standard task accuracy and
generation evals, use [Running Evaluations with Marin](../tutorials/run-lm-evals.md).

## Overview

Scoring is tokenizer-independent. Each model tokenizes the same UTF-8 documents
with its own tokenizer, and Levanter projects token losses back onto document
bytes so two models are comparable on the same slice. The reported metric is
bits per byte (`bpb`). In a gap report, positive `gap_bpb` means model A has
higher loss than model B on that slice; negative means model A has lower loss. By
convention Marin is passed as model A, so positive means Marin is worse.

The workflow is two steps:

1. **Score** each model on the datasets with `find_model_perplexity_scores`. This
   runs as a TPU job and writes score artifacts to GCS.
2. **Compare** the two score outputs with `find_model_perplexity_gap`. This reads
   both score directories and writes a gap report.

The implementation lives in:

- `lib/marin/src/marin/evaluation/perplexity_gap.py` — the library documented here.
- `lib/levanter/src/levanter/main/perplexity_gap.py` — the scoring entrypoint.
- `lib/levanter/src/levanter/analysis/model_perplexity.py` — per-document scoring.
- `lib/levanter/src/levanter/analysis/perplexity_gap.py` — the comparison and report writer.

!!! note "Removed scaffolding"

    The one-off dashboard suite, provider registry, and coverage-matrix
    experiment files (`experiments/evals/model_perplexity_gap_suite.py`,
    `experiments/evals/perplexity_gap_registry.py`,
    `experiments/exp_model_perplexity_gap_coverage_matrix.py`) and the
    `marin-community.github.io` dashboard-publishing flow were removed in
    [#6633](https://github.com/marin-community/marin/pull/6633). There is no
    committed experiment that composes these entry points; a caller wires them
    together directly, as shown below.

## Datasets

A dataset is a `RawTextEvaluationDataset`. Build one with `raw_text_dataset` for
raw language modeling (every byte is scored) or `supervised_text_dataset` for
target-only scoring (only the target bytes are scored, while the input bytes
still condition the model).

For raw language-modeling text from a GCS/URL path:

```python
from marin.evaluation.perplexity_gap import raw_text_dataset

datasets = {
    "long_tail_ppl/example/source_a": raw_text_dataset(
        "gs://marin-us-central1/raw/example/source_a/heldout.jsonl.gz",
        tags=("long_tail_ppl", "issue:NNNN", "source:example"),
    ),
}
```

For a Hugging Face source, pass an `HfDatasetSpec`. It carries an optional
`revision`, which is threaded through to pin an immutable snapshot:

```python
from marin.evaluation.perplexity_gap import raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

source = HfDatasetSpec(id="marin-community/example-ppl", name="format_variant", revision="<sha>")
datasets = {
    "synthetic_example_ppl/format_variant": raw_text_dataset(
        source,
        tags=("synthetic_example_ppl", "issue:NNNN"),
    ),
}
```

For supervised target-only scoring, write rows with separate input and target
fields and use `supervised_text_dataset`:

```python
from marin.evaluation.perplexity_gap import supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

source = HfDatasetSpec(id="marin-community/example-ppl", name="format_variant")
datasets = {
    "synthetic_example_ppl/format_variant": supervised_text_dataset(
        source,
        input_key="input",
        target_key="target",
        tags=("synthetic_example_ppl", "issue:NNNN", "loss:target_only"),
    ),
}
```

Raw-text scoring expects JSONL or gzipped-JSONL rows with a `text` field
(`{"text": "..."}`); supervised scoring expects `{"input": "...", "target": "..."}`.
If a source needs ETL, use `raw_download` from `marin.experiment.data` (or a
custom step) to write a small heldout artifact to GCS first.

Dataset keys are part of the report API. Keep them stable, hierarchical, and
explicit — e.g. `structured_text/totto` or `long_tail_ppl/code_ecosystem/stack_v2_json`.
Tags drive report rollups; include at least a family tag and an issue tag, e.g.
`("long_tail_ppl", "issue:5254", "code_ecosystem")`.

Recommended heldout sizing:

- Keep broad probe slices around 1–2 MB compressed per subset unless there is a
  specific reason to go larger.
- Scoring caps at 256 documents per dataset and 32 KiB per document by default
  (`max_docs_per_dataset`, `max_doc_bytes`).
- Put raw artifacts in the same region as the scoring run. Avoid cross-region GCS
  reads for large sources.

::: marin.evaluation.perplexity_gap.RawTextEvaluationDataset

::: marin.evaluation.perplexity_gap.raw_text_dataset

::: marin.evaluation.perplexity_gap.supervised_text_dataset

## Scoring a model

Score a model by constructing a `ModelPerplexityScoreConfig` and calling
`find_model_perplexity_scores` from inside a Fray job (it submits a child TPU job
through the ambient `current_client()` and blocks until it finishes). The model
is a `GapFinderModelConfig` — a checkpoint plus how to load and tokenize it.

```python
from fray.types import ResourceConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    ModelPerplexityScoreConfig,
    find_model_perplexity_scores,
)

marin_scores = ModelPerplexityScoreConfig(
    name="model_perplexity/marin_32b",
    model=GapFinderModelConfig(checkpoint_path="gs://.../marin-32b/step-XXXX"),
    datasets=datasets,
    resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
    output_path="gs://marin-us-central1/analysis/model_perplexity_scores/marin_32b",
)
find_model_perplexity_scores(marin_scores)
```

Build a second `ModelPerplexityScoreConfig` the same way for the comparison model
(e.g. `qwen_scores`) and score it too. Each completed score directory contains:

- `summary.json`
- `scored_documents.parquet`
- `token_counts.parquet`
- `token_counts_summary.json`

::: marin.evaluation.perplexity_gap.GapFinderModelConfig

::: marin.evaluation.perplexity_gap.ModelPerplexityScoreConfig

::: marin.evaluation.perplexity_gap.find_model_perplexity_scores

## Computing the gap

Once both models are scored, compare their score directories with
`find_model_perplexity_gap`. It computes the pairwise report, writes it to GCS,
and logs the summary and report artifact to Weights & Biases.

```python
from marin.evaluation.perplexity_gap import ModelPerplexityGapConfig, find_model_perplexity_gap

find_model_perplexity_gap(
    ModelPerplexityGapConfig(
        name="model_perplexity_gap/marin_32b-vs-qwen3_32b",
        model_a_name="marin_32b",
        model_b_name="qwen3_32b",
        model_a_scores_path=marin_scores.output_path,
        model_b_scores_path=qwen_scores.output_path,
        output_path="gs://marin-us-central1/analysis/perplexity_gap/marin_32b-vs-qwen3_32b",
    )
)
```

The gap step is CPU-only; run it after both score jobs succeed. Each completed
gap directory contains:

- `summary.json`
- `report.md`
- `worst_documents.jsonl`

If the gap step fails after both score jobs have succeeded, do not rerun scoring:
rerun only `find_model_perplexity_gap` against the completed score directories.

::: marin.evaluation.perplexity_gap.ModelPerplexityGapConfig

::: marin.evaluation.perplexity_gap.find_model_perplexity_gap

## Common failure modes

- **Target leakage in target-only rows:** The prompt should end immediately
  before the target. The target bytes should live in the `target` field, not be
  duplicated in `input`.
- **Chat-framed rows:** Base models are sensitive to `User:` / `Assistant:`
  framing. Prefer neutral base-model formats (newline, `=>`, `=`) unless the task
  is explicitly chat-formatted.
- **Mutable HF sources:** Pass a `revision` to `HfDatasetSpec`, or materialize
  the heldout split to a versioned GCS path, when the exact snapshot matters for
  a long-lived comparison.
- **Contamination:** Code and web slices can contain examples a model saw during
  training. Treat row-level surprises as hypotheses and inspect provenance before
  using them as clean capability evidence.
