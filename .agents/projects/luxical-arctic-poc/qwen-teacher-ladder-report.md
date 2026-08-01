# Qwen teacher size ladder

Status: complete

## Decision

Do not run Qwen3-Embedding-8B in this ladder. Qwen3-Embedding-4B at 256
dimensions has 35 regular-source failures and fails two fixed gates. The start
condition for 8B required all quality gates and at most 34 failures.

Qwen3-Embedding-0.6B at 1,024 dimensions stays the best tested Qwen teacher.
It passes all quality, finite, and unique gates. It has 46 regular-source
failures, so it does not pass the strict zero-failure concentration gate.

## Method

The fixed holdout contains 74,752 documents from 146 data sources. The probe
uses 256 training rows for each source. The evaluation uses the remaining rows.
The cluster audit uses 40 clusters and seeds 42, 43, and 44.

Each teacher receives three 2,000-character document windows. Each window has a
512-token limit. Qwen uses last-token pooling and BF16 inference. The audit
stores 8-bit scalar-quantized vectors.

The 256-dimensional runs use Qwen Matryoshka Representation Learning (MRL).
The model selects the first 256 dimensions of each window before normalization.
The audit then normalizes and pools the three windows. Thus, these vectors are
not slices of the saved 1,024-dimensional document vectors.

The fixed gates compare each teacher with Luxical-One. The minimum quality
delta is -0.02 for overall macro-F1, worst-source recall, and each source group.
All vectors must be finite. Four-decimal uniqueness must be at least 0.99. No
regular source can fail the concentration, rank, variance, or uniqueness checks.

## Results

| Representation | Overall | Code | Multilingual | Standard | Regular failures | Teacher documents/second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 | — | — |
| Qwen3-Embedding-0.6B, 1,024d | 0.67664 | 0.80067 | 0.81348 | 0.62159 | 46 | 94.50 |
| Qwen3-Embedding-0.6B, 256d | 0.63626 | 0.76200 | 0.77463 | 0.58256 | 39 | 94.50 |
| Qwen3-Embedding-4B, 256d | 0.64300 | 0.75696 | 0.77509 | 0.59347 | 35 | 29.19 |

The 4B model improves overall macro-F1 by 0.00674 against the 0.6B
256-dimensional model. Standard macro-F1 increases by 0.01091. Code macro-F1
decreases by 0.00503. Multilingual macro-F1 increases by 0.00046.

Against Luxical-One, the 4B overall delta is +0.02573. Its code delta is
+0.07607, its standard delta is +0.02460, and its multilingual delta is
-0.02052. The multilingual result misses the fixed limit by 0.00052.

The 4B model has ten code failures and 25 standard failures. It has no
multilingual source failure. Thirty-four sources fail concentration, two fail
rank, and one fails uniqueness, with overlap. All vectors are finite. Exact
and four-decimal uniqueness are 0.99976.

The 4B teacher rate is 0.309 times the 0.6B teacher rate. This rate measures
teacher label production on one H100. It does not measure final student speed.
The 4B job took 50 minutes 3.62 seconds. It had zero task failures and zero
preemptions.

The 4B model reduced the best 256-dimensional failure count by four. The fixed
condition required a reduction of at least five and all quality gates. Model
size alone did not remove source concentration at 256 dimensions.

## Reproduction

Model revisions:

- Qwen3-Embedding-0.6B:
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Qwen3-Embedding-4B:
  `5cf2132abc99cad020ac570b19d031efec650f2b`.
- Qwen3-Embedding-8B, not run:
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

Commands:

```bash
python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-0.6b-256

python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-4b-256
```

Artifacts:

- 0.6B 1,024-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b/report.json`.
- 0.6B 256-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b-256/report.json`.
- 4B 256-dimensional report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-4b-256/report.json`.

## Limits

The audit uses one fixed holdout and one model revision for each size. The audit
does not train a student. Teacher rates do not include remote document reads.
Source groups use fixed source-name rules, not document-level language labels.
The 512-token window limit can truncate long documents.
