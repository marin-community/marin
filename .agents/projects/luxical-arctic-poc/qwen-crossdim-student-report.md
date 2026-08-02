# Qwen cross-dimension FastTransformer POC

Status: 750K POC complete; peer review addressed

## Decision

Do not scale this treatment to 3M rows. The 750K student passes finite,
unique, Qwen-fidelity, code-quality, worst-recall, and CPU-speed gates. It
fails overall, multilingual, standard, student-only-failure, and variance
gates.

The failed treatment uses a train-only 256-to-1,024 alignment head. Production
inference discards the head and keeps the 256-dimensional FastTransformer.
This design keeps inference speed, but it does not keep enough Qwen geometry.

This run changed the teacher and the direct-cosine path together. Therefore,
it does not isolate teacher identity from the cross-dimension method. The next
control must train on native 256-dimensional Qwen labels without a head.

If that control succeeds, stop the head path. If it fails, test an
orthonormal-row head at 64K. Such a head preserves student dot products and
cannot expand compressed student directions.

## Controlled inputs

- Teacher: `Qwen/Qwen3-Embedding-0.6B` at revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Teacher output: 1,024 dimensions, three 512-token document windows, BF16
  inference, and 8-bit stored vectors.
- Training: 750,000 source-balanced rows from 146 sources.
- Evaluation: 74,752 fixed held-out documents from the same 146 sources.
- Student: the existing 9,299,200-parameter FastTransformer with a
  256-dimensional output.
- Train-only head: 262,144 parameters with shape 256 by 1,024.
- Objective: Gram-KL on raw student and teacher geometry, plus direct cosine
  after the train-only head.
- Training: three epochs and 552 update steps.

Eight label jobs completed without failures or preemptions. The full audit
found exactly 750,000 aligned rows. Every teacher dimension varied. The lowest
source-level unique-vector fraction was 0.92481.

## Results

| Representation | Overall | Code | Multilingual | Standard | Regular failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 | Reference |
| Qwen 0.6B, 1,024d teacher | 0.67664 | 0.80067 | 0.81348 | 0.62159 | 46 |
| Arctic 750K FastTransformer | 0.63566 | 0.74913 | 0.76573 | 0.58114 | 66 |
| Qwen cross-dimension 750K student | 0.58236 | 0.70283 | 0.76387 | 0.51375 | 140 |

The Qwen student loses 0.03491 overall macro-F1 against Luxical-One. Its code
delta is +0.02194. Its multilingual delta is -0.03175, and its standard delta
is -0.05512. The permitted loss is 0.02.

All 46 Qwen teacher failures remain in the student. The student adds 94 new
failures. Thus, the result is not only teacher-failure inheritance.

The student is finite. Its four-decimal unique fraction is 0.99929. Its Qwen
within-source Spearman value is 0.84779, which is 0.11207 above Luxical-One.
The paired CPU speed ratio is 2.59315 against Luxical-One. This value comes
from the existing conservative speed artifact. The current student has fewer
parameters than that measured FastTransformer, but it did not get a new speed
run.

| Source group | Median rank ratio to Qwen | Median variance ratio to Qwen |
| --- | ---: | ---: |
| Code | 0.14605 | 0.21543 |
| Multilingual | 0.12351 | 0.17355 |
| Standard | 0.14135 | 0.25840 |

Only the variance ratio uses the 0.50 gate. Effective rank depends on output
dimension, so the 256-to-1,024 rank ratio is diagnostic. The variance ratio is
dimension-free for normalized vectors.

The high Spearman value shows that the student keeps much of the pair order.
The low variance ratios show that it compresses pair distances into a narrow
region. Arctic 750K student-to-teacher ratios are not available, so this
report does not compare the ratio tables across teachers.

The training audit also detects that region. Final effective rank is 25.35137,
mean cosine is 0.96642, and cosine p99 is 0.99109. The final loss is 0.39807,
down from 0.97988.

## Alignment diagnosis

The trained head has singular values from 0.35161 to 2.38748. Its condition
number is 6.79018. The initialized head condition number was 2.95179.

The condition-number change supports one mechanism: the unconstrained head
can expand small student directions while it aligns outputs with Qwen. The
measurement does not prove sole causality. Teacher identity and the changed
direct-cosine path remain confounded. The raw Gram-KL term also did not prevent
the compressed geometry.

An orthonormal-row head satisfies `P P^T = I`. Therefore, `x P` and `y P`
keep the dot product between `x` and `y`. This constraint removes the measured
amplification path without changing the deployed student.

## Gates

Passed:

- finite vectors;
- at least 0.99 four-decimal unique vectors;
- code macro-F1 delta of at least -0.02;
- worst-source recall delta of at least -0.02;
- nonnegative Qwen within-source fidelity delta;
- CPU speed of at least 0.85 times Luxical-One.

Failed:

- overall macro-F1 delta of at least -0.02;
- multilingual macro-F1 delta of at least -0.02;
- standard macro-F1 delta of at least -0.02;
- at most five student-only failures;
- category median variance ratios of at least 0.50.

The effective-rank ratios remain diagnostics. They do not set a gate because
the student and teacher dimensions differ.

## Operational note

The first CPU audit job failed before data checks because its environment had
no S3 credentials. The H100 audit used the label-job environment and
succeeded. This event did not change data or code.

## Reproduction

- Code commit: `33d52e4d6`.
- Reviewed snapshot: `luxical-qwen-crossdim-student-20260802-v1`.
- Label root:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/teacher-qwen3-embedding-0.6b-1024-train-750k-v1`.
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/full-qwen3-06b-1024-crossdim/750k/training.json`.
- Evaluation report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full-qwen3-06b-1024-crossdim/750k/report.json`.
- Student model SHA-256:
  `a40b37038a1b39f9f0ca20f5c2b47b62bbbaacc3af583ec7262ad9a1c13add22`.

## Peer-review dispositions

- Accepted: state that teacher identity and the cross-dimension method changed
  together. Run the native 256-dimensional Qwen control first.
- Accepted: use only dimension-free variance ratios for the cross-dimension
  geometry gate. Keep effective-rank ratios as diagnostics.
- Accepted: state that Arctic 750K attribution ratios are not available.
- Accepted: identify the CPU speed value as an inherited conservative result.
- Accepted: bind evaluation vectors to teacher and manifest metadata.
- Accepted: remove legacy gate booleans from the new report artifact.
- Accepted: guard empty failure sets and validate embedding rank before shape
  indexing.
- Accepted: exclude the scale-invariant alignment head from weight decay.
- Accepted: give each teacher an explicit artifact suffix.
- Accepted: replace weak gradient tests with component, loss-reduction, and
  invalid-shape behavior tests.
- Accepted: share JSON storage helpers through `ladder_config.py`.
- Retained: `--teacher` stays required because the teacher is a critical
  training input. Future commands must select it explicitly.
- Retained: model and array upload helpers stay separate. They use different
  serializers, and a callback would add indirection without reuse pressure.
