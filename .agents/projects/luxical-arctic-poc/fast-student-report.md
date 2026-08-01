# FastTransformer Arctic student report

Status: results complete; peer review addressed

## Decision

The 3M FastTransformer POC succeeds on quality, teacher fidelity, and speed. It
passes nine of ten gates. Do not use this artifact for the 20B-document run
yet, because 59 regular sources fail the strict per-source collapse gate.

The next treatment must reduce per-source cluster concentration while it keeps
the same speed and quality. The current result removes the original broad code
quality failure: code macro-F1 is 0.09832 above stock and no source fails the
variance ratio gate. It does not remove all source-level collapse.

## Controlled inputs

- Teacher: pinned Snowflake Arctic Medium v2.0 vectors from the current fixed
  manifest.
- Evaluation: 74,752 held-out documents from 146 sources. Training does not
  use these rows.
- Training ladder: nested source-balanced samples with 65,536, 750,000, and
  3,000,000 rows.
- Input: one 256-token view made from 256-character head, middle, and tail
  regions.
- Tokenizer: the pinned stock Luxical Rust `ArrowTokenizer` and a
  source-balanced remap.
- Student: two transformer layers, hidden size 256, four attention heads,
  mean/max/min pooling, and a normalized 256-dimensional head.
- Trained parameter count: 9,299,200.
- Objective: Luxical Gram-KL at temperature 3 plus direct cosine alignment at
  weight 1.0.
- Training: three epochs, batch size 4,096, AdamW, peak learning rate 0.0005,
  5% warmup, weight decay 0.05, and gradient clip 1.0.

The quality gates require finite vectors, at least 0.99 four-decimal unique
vectors, no regular per-source composite collapse failures, each probe delta
of at least -0.02, nonnegative Arctic fidelity delta, and at least 0.70 times
stock CPU speed. Arctic fidelity uses within-source pairs.

## Collapse diagnosis

The first compiled Gram-KL-only run collapsed after one epoch. Its effective
rank was 1.37056 and its mean pairwise cosine was 0.98716.

A controlled 2,048-row check found:

| Representation | Effective rank | Mean cosine | Cosine p99 |
| --- | ---: | ---: | ---: |
| Untrained FastTransformer | 26.89108 | 0.95852 | 0.99356 |
| Arctic teacher | 61.30561 | 0.27540 | 0.93112 |

Thus, the prepared documents and Arctic vectors are finite and non-constant.
Gram-KL-only optimization causes this FastTransformer failure. Direct cosine
alignment rejects the weak near-uniform solution and lets rank increase with
scale.

## Speed result

The paired CPU test used 20,000 evenly spaced held-out documents and five timed
runs after warmup.

| Model | CPU documents/second | Ratio to stock |
| --- | ---: | ---: |
| Stock Luxical-One | 8,957.64 | 1.0000 |
| FastTransformer treatment | 23,228.50 | 2.5931 |

One B200-class worker processed 89,405.74 documents per second. The speed
treatment used a 9.65M-parameter provisional full vocabulary. The trained
model has 9.30M parameters, so this is a conservative paired speed test.

## Scaling ladder

| Rung | Macro-F1 | Code | Multilingual | Standard | Within-source fidelity | Regular collapse failures | Min rank ratio | Min variance ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 | 0.82113 | — | — | — |
| 64K | 0.20413 | 0.25530 | 0.37036 | 0.16514 | 0.29437 | 143 | 0.02236 | 0.01473 |
| 750K | 0.63566 | 0.74913 | 0.76573 | 0.58114 | 0.89338 | 66 | 0.35554 | 0.31444 |
| 3M | 0.65785 | 0.77921 | 0.77656 | 0.60384 | 0.90901 | 59 | 0.45300 | 0.52589 |

The 64K rung is a smoke test, not a fair final comparison with stock Luxical.
The 750K rung shows strong scale improvement. It passes eight of ten gates. It
fails multilingual macro-F1 by 0.00988 beyond the allowed loss and has 66
regular source failures.

The 3M rung passes the multilingual gate by 0.00095. It passes all quality,
fidelity, finite, unique, worst-recall, and speed gates. It fails only the
regular-source collapse gate. The 59 failures have 56 concentration reasons,
four uniqueness reasons, and one rank reason, with overlap. No source fails
the variance ratio gate. The minimum rank ratio is 0.45300, below the 0.50
limit. The minimum variance ratio is 0.52589, above its 0.50 limit.

Training-set audit rank rises from 2.72790 at 64K, to 31.15826 at 750K, and to
44.73097 at 3M. Final training loss falls from 0.67426, to 0.33437, and to
0.26548. This trend shows that more data and updates reduce the collapse, but
the 3M rung has not reached the strict source gate.

The reviewed fidelity gate uses only within-source pairs. The 3M delta is
+0.08788. The pooled Spearman values are 0.89423 for the student and 0.86765
for stock, but they are diagnostics and do not set the gate.

## Compute and limits

- Preparation of all 3M rows took 7 minutes 12.85 seconds.
- Hybrid 64K training took 1 minute 11.68 seconds.
- Hybrid 750K training took 1 minute 23.15 seconds.
- Hybrid 3M training took 1 minute 58.64 seconds.
- All successful training jobs used one B200-class worker and interactive
  priority. They had no failure or preemption.
- The fixed view has less text than stock Luxical for long documents.
- The teacher uses a 512-token limit for each 2,000-character window. Code and
  CJK text can reach this limit more often than English prose. This run did not
  measure truncation by category, so the multilingual result is not a clean
  student-capacity result.
- The input joins character regions before WordPiece truncation. Languages
  with many tokens for each character can lose more of the middle or tail.
- The manifest labels its circular block sample as uniform marginal. The final
  partial block is cut from its end, so that label is too strong. The result is
  a source-balanced block sample, not an exact uniform row sample.
- The source inventory URL was overwritten before the corrected manifest was
  built. The manifest records the URL but not its content digest. Each prepared
  source and teacher shard has its own checked digest, but the inventory step
  cannot be rebuilt from the URL alone.
- Code and multilingual groups use source-name rules. They are fixed for this
  evaluation, but they are not content-language labels.
- The worst-source recall gate is a relative delta and can pass near zero. The
  final 3M value is 0.02344. A release evaluation must add an absolute p05
  recall gate.
- The paired speed artifact measures the fixed model and tokenizer treatment.
  It does not include remote document reads.

## Peer-review dispositions

- Rejected: the reported unrelated reversions are not in the diff from current
  `origin/main` at `c29d885cf`. That diff contains only the 24 POC files.
- Rejected: the reported deleted source definitions are present on current
  `origin/main`; this was a stale peer-review base.
- Accepted: record the source-inventory provenance limit.
- Accepted: remove the exact uniform-marginal claim from this report.
- Accepted: record the 512-token teacher truncation confound.
- Accepted: gate Arctic fidelity on within-source pairs. The pooled value stays
  a diagnostic only.
- Accepted: delete the unused results collector and its duplicate thresholds.
- Accepted: record that source groups use name rules.
- Accepted: require an absolute p05 recall gate before release.
- Rejected: optimize completed manifest and survey jobs in this result pass.
  These changes cannot alter the saved artifacts.
- Rejected: the fast-student path has 15 behavior tests, including a compiled
  loss test and a real gradient update.
- Accepted: remove the complete JSON copy from generated HTML. The separate
  JSON artifact is canonical.
- Rejected: edit the append-only logbook or remove prior peer-reviewed reports.
  Repository policy keeps research history append-only.

The accepted findings are addressed. The peer-review skill does not require a
second review loop.

## Reproduction

- Code branch: `research/rav/6850-luxical-arctic-teacher-gates`.
- Preparation job: `/rav/lux-arctic-fast-student-prepare-b200-001`.
- 64K training job: `/rav/lux-arctic-fast-student-train-64k-b200-003`.
- 750K training job: `/rav/lux-arctic-fast-student-train-750k-b200-001`.
- 3M training job: `/rav/lux-arctic-fast-student-train-3m-b200-001`.
- Paired speed JSON:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-full-luxical-one-arrow.json`.
- Evaluation reports:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full/<rung>/report.json`.
