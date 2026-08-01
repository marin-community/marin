# Qwen teacher size ladder

Status: complete

## Decision

Do not run Qwen3-Embedding-8B in this ladder. Qwen3-Embedding-4B at 256
dimensions has 35 regular-source failures. It fails
`regular_source_collapse` and `multilingual_macro_f1`. The start condition for
8B required all quality gates and at most 34 failures.

Qwen3-Embedding-0.6B at 1,024 dimensions stays the best tested Qwen teacher.
It passes all quality, finite, and unique gates. It has 46 regular-source
failures, so it does not pass the strict zero-failure concentration gate.

Do not train a student from either native 256-dimensional Qwen teacher at this
time. The next controlled test can keep the 1,024-dimensional 0.6B labels. It
can add a train-only 256-to-1,024 alignment head to the 750K student rung. This
keeps the deployed student output at 256 dimensions.

The 1,024-dimensional labels have four times the vector payload of
256-dimensional labels. Quantized payload is 3.072 GB instead of 0.768 GB for
3M rows. It is 20.48 TB instead of 5.12 TB for 20B rows, before metadata and
compression.

## Method

The fixed holdout contains 74,752 documents from 146 data sources. The probe
uses 256 training rows for each source. The evaluation uses the remaining rows.
The cluster audit uses 40 clusters and seeds 42, 43, and 44.

Each teacher receives three 2,000-character document windows. Each window has a
512-token limit. Qwen uses last-token pooling and BF16 inference. The audit
stores 8-bit scalar-quantized vectors with a quantization limit of 0.3.

The 256-dimensional runs use Qwen Matryoshka Representation Learning (MRL).
The model selects the first 256 dimensions of each window before normalization.
The audit then normalizes and pools the three windows. Thus, these vectors are
not slices of the saved 1,024-dimensional document vectors.

The 0.6B model keeps 256 of 1,024 hidden dimensions. The 4B model keeps 256 of
2,560 hidden dimensions. Thus, model size and compression fraction change
together. The batch size is 128 for 0.6B and 32 for 4B.

The fixed gates compare each teacher with Luxical-One. The minimum delta is
-0.02 for overall macro-F1, worst-source recall, and each source group. All
vectors must be finite. Four-decimal uniqueness must be at least 0.99.

The regular-source checks use 143 sources: 28 code, 24 multilingual, and 91
standard sources. The three OOD sources in the holdout do not set this gate.
Each source must have a maximum cluster share of 0.90 or less. Rank and variance
ratios must be at least 0.50. No regular source can fail these checks.

## Results

| Representation | Overall | Code | Multilingual | Standard | Worst recall | Regular failures | Failed gates | Batch | Teacher docs/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Luxical-One | 0.617271 | 0.680893 | 0.795611 | 0.568867 | 0.007812 | — | — | — | — |
| Arctic Medium v2.0 | 0.669148 | 0.799952 | 0.728569 | 0.630702 | 0.046875 | 60/143 | collapse, multilingual | — | — |
| Qwen3-Embedding-0.6B, 1,024d | 0.676645 | 0.800667 | 0.813475 | 0.621594 | 0.031250 | 46/143 | collapse | 128 | 94.4990 |
| Qwen3-Embedding-0.6B, 256d | 0.636255 | 0.761997 | 0.774630 | 0.582559 | 0.015625 | 39/143 | collapse, multilingual | 128 | 94.4994 |
| Qwen3-Embedding-4B, 256d | 0.642998 | 0.756964 | 0.775091 | 0.593466 | 0.019531 | 35/143 | collapse, multilingual | 32 | 29.1872 |

The 4B model improves overall macro-F1 by 0.006743 against the 0.6B
256-dimensional model. Standard macro-F1 increases by 0.010907. Code macro-F1
decreases by 0.005033. Multilingual macro-F1 increases by 0.000461.

Against Luxical-One, the 4B overall delta is +0.025727. Its code delta is
+0.076071, its standard delta is +0.024599, and its multilingual delta is
-0.020519. The multilingual result misses the fixed limit by 0.000519. The
0.6B 256-dimensional result misses this limit by 0.000980.

The 4B model has ten code failures and 25 standard failures. It has no
multilingual source failure. Thirty-four sources fail concentration, two fail
rank, one fails uniqueness, and zero fail variance, with overlap. All vectors
are finite. Exact and four-decimal uniqueness are 0.999759.

The measured 4B teacher rate is 0.309 times the 0.6B rate. Batch size and model
size change together, so this ratio is not an isolated model-size effect. These
rates measure teacher label production. They do not measure student inference.

The 4B model reduced the best 256-dimensional failure count by four. The fixed
condition required a reduction of at least five. This ladder did not show that
a larger Qwen model removes concentration with a fixed 256-dimensional output.

## Reproduction

The model revisions are:

- Qwen3-Embedding-0.6B:
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Qwen3-Embedding-4B:
  `5cf2132abc99cad020ac570b19d031efec650f2b`.
- Qwen3-Embedding-8B, not run:
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

The original ladder commands used one federated H100 and interactive priority:

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-teacher-qwen3-06b-256-h100-001 \
  --priority interactive --gpu H100 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 14400 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-0.6b-256

uv run iris --cluster=marin job run --no-wait \
  --job-name lux-teacher-qwen3-4b-256-h100-001 \
  --priority interactive --gpu H100 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 21600 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-4b-256
```

The harness reuses saved vectors. Thus, a new execution of these commands does
not measure teacher rate again. Use the saved JSON reports for the original
rate values. Use a new output root for a new rate measurement.

The saved reports are:

- 0.6B 1,024-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b/report.json`.
- 0.6B 256-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b-256/report.json`.
- 4B 256-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-4b-256/report.json`.

## Limits

The audit uses one fixed holdout and one model revision for each size. The audit
does not train a student. Source groups use fixed source-name rules, not
document-level language labels. The 512-token window limit can truncate long
documents.

The 4B comparison changes model size, MRL compression fraction, and batch size.
It does not isolate a model-size effect. The 8B candidate stays in the harness
because it records the predeclared ladder and pinned revision. The stop rule
prevented its submission.
