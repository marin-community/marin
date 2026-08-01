# Arctic-distilled Luxical scaling ladder

Status: corrected run complete

## TL;DR

The corrected 3M Arctic-distilled Luxical student is not ready for the
20-billion-document run. It passes eight of ten required gates. CPU speed is
0.995 times Luxical-One, source macro-F1 is 0.0021 higher, and Arctic cosine
fidelity is 0.0582 higher. The name-matched multilingual-source macro-F1 is
0.1060 lower, and 97 of 143 regular sources fail at least one collapse check.

The 0.75M rung passes four of ten gates. Scaling to 3M improves source
macro-F1 by 0.3724 and reduces regular-source collapse failures from 143 to
97. The gain supports a larger rung, but it does not support production use
of the current 3M model.

The corrected run uses uniform source sampling, a separate probe evaluation
split, three clustering seeds, and paired source bootstrap intervals. All
corrected artifacts use manifest SHA-256
`4aea19379cb6b7414d80f0b72c868f239e9247c05c3a703a26b19a059599f211`.

A direct teacher-size test does not support replacing Arctic Embed Medium
v2.0 with Large v2.0. Large improves multilingual macro-F1 by 0.0078 relative
to Medium, but the interval includes zero. Large is lower by 0.0074 overall,
0.0110 on code, and 0.0103 on standard text. The overall, code, and standard
intervals are fully negative.

## Arctic teacher-size comparison

The teacher-size test embeds the same 74,752 held-out documents with
`Snowflake/snowflake-arctic-embed-l-v2.0` at revision
`ac6544c8a46e00af67e330e85a9028c66b8cfd9a`. It keeps the Medium test's three
document windows, 512-token window limit, 256-dimensional truncation, 8-bit
quantization, probe split, clustering seeds, and gate limits.

| Representation | Overall | Code | Name-matched multilingual | Standard |
| --- | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 |
| Arctic Medium v2.0 | 0.66915 | 0.79995 | 0.72857 | 0.63070 |
| Arctic Large v2.0 | 0.66171 | 0.78894 | 0.73634 | 0.62040 |

Large minus Medium has these paired-source results:

| Group | Macro-F1 delta | 95% interval |
| --- | ---: | ---: |
| Overall | -0.00743 | [-0.01200, -0.00300] |
| Code | -0.01101 | [-0.01920, -0.00345] |
| Name-matched multilingual | +0.00777 | [-0.00817, +0.02441] |
| Standard | -0.01030 | [-0.01519, -0.00566] |

Large passes six of eight direct gates. It fails the same two gates as Medium:
`regular_source_collapse` and `multilingual_macro_f1`. Large trails
Luxical-One multilingual macro-F1 by 0.05927. That paired-source interval is
[-0.12500, +0.00438], and the point estimate exceeds the allowed 0.02 loss.

All Large vectors are finite. Its exact and four-decimal unique fractions are
both 0.999759. No regular source fails the rank or variance checks. The minimum
Large-to-Luxical rank ratio is 1.02636, and the minimum variance ratio is
1.51776.

Forty-four of 143 regular sources fail the composite rule, compared with 60
for Medium. The Large counts are 13 of 28 code, zero of 24 name-matched
multilingual, and 31 of 91 standard sources. The overlapping reasons are 43
cluster-concentration failures and one uniqueness failure.

Large reduces per-source concentration failures, but its global code clusters
are worse than Medium. Its largest code cluster share is 0.20752, compared
with 0.17055 for Medium. Its effective code cluster count is 9.86327,
compared with 11.14955. Its code source-cluster NMI is 0.50727, compared with
0.52520.

The Large embedding phase took 1,635.47 seconds on one GB200-class worker. It
processed 45.71 documents per second. Each document uses three windows. This
rate is not a student inference measurement and does not compare teacher
throughput because the test reused the stored Medium vectors.

The fixed source-name groups and 512-token window limit are inherited method
limits. They remain fixed so this test changes only the teacher checkpoint.
The result supports keeping Arctic Medium v2.0. Large does not fix the
multilingual gate and reduces overall, code, and standard probe quality.

Artifacts:

- JSON:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.json`
- HTML:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.html`
- Evaluation job: `/rav/lux-arctic-large-teacher-v2-gb200-001`
- Harness commit: `2dbef9c06`

## Goal

This experiment tests if a Luxical student trained from Arctic teacher vectors
is suitable for text-domain classification. It uses a 0.75M-document rung and
a 3M-document rung. The fixed data has code, standard text, and a small
predeclared OOD group. Source names define a separate multilingual group.

The experiment responds to
[code collapse in issue 6850](https://github.com/marin-community/marin/issues/6850)
and
[language and modality catch-all clusters in issue 6855](https://github.com/marin-community/marin/issues/6855).
It follows the
[Luxical training method](https://github.com/datologyai/luxical#training)
and replaces its original teacher with
[Snowflake Arctic Embed M v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0).

The 3M student is viable only if it passes all required gates:

- All evaluation vectors are finite.
- At least 99 percent of evaluation vectors are unique after rounding to four
  decimal places.
- No regular source has more than 90 percent of its rows in one of 40 global
  clusters. Only the predeclared OOD sources can be exempt.
- Pairwise cosine Spearman correlation with Arctic is not lower than
  Luxical-One.
- Source macro-F1 and worst-source recall are not more than 0.02 below
  Luxical-One.
- Code, name-matched multilingual, and standard-text macro-F1 are each within
  0.02 of Luxical-One.
- CPU throughput is at least 70 percent of Luxical-One. The target is 85
  percent.

The work is complete only when the report contains the corrected results and
reproducible artifact links. An independent peer review must inspect the
report and experiment code. The report must record each finding and its
disposition. All accepted findings must be addressed and checked before the
final handoff.

## Peer review and dispositions

The peer-review skill used this report and the experiment code. One independent
review produced 28 findings. The review did not use a second review loop.

| # | Decision | Disposition |
| ---: | --- | --- |
| 1 | Accept | The sampler uses 64 uniform global row blocks. Every row has equal marginal probability, and each shard probability is proportional to row count. |
| 2 | Accept | The probe now divides the 512 unseen rows into 256 probe-training rows and 256 probe-evaluation rows. |
| 3 | Accept | Model evaluation stops if one vector is not finite. The report treats finiteness as an enforced prerequisite. |
| 4 | Accept | The speed test remains a paired implementation check. The production estimate now states that inputs are bounded document views. |
| 5 | Accept | Arctic fidelity measures the training objective. It is not an independent quality measure. The zero-delta gate stays because the goal specifies it. |
| 6 | Accept | Evaluation checks that logistic regression stops before its 1,000-iteration limit. |
| 7 | Accept | Clustering now uses three seeds. F1 deltas now have paired source bootstrap intervals. One student-training seed remains a limit. |
| 8 | Accept | The report uses “name-matched multilingual-source group.” It separates the baseline gap from the allowed 0.02 margin. |
| 9 | Accept | One shared function defines document windows. The teacher decodes the exact windows that the manifest stores. |
| 10 | Accept | Each teacher Parquet file stores the manifest digest, teacher identity, and teacher revision. Reuse checks all three values. |
| 11 | Accept | Mirror conversion now rejects any canonical bucket other than `marin-us-west2`. |
| 12 | Accept | Manifest shards and the manifest JSON now use atomic writes. |
| 13 | Accept | The audit checks selected-file counts, unique input positions, and stored digests. It does not claim a new read of all raw inputs. |
| 14 | Accept | The unused `in_survey` column is removed. Survey strata use the evaluation ranks directly. |
| 15 | Accept | Worst-source recall stays because the goal specifies it. The evaluation adds fifth-percentile source recall as a stable diagnostic. |
| 16 | Reject | Local read and write functions keep these one-week scripts independent. A shared helper would not change the method or result. |
| 17 | Accept | Student dimensions now come from the checked Luxical-One layer shapes. |
| 18 | Reject | The standalone HTML files help private review. JSON remains the result source. |
| 19 | Partial | Forced model downloads are removed. The pinned constructor stays because the parent constructor cannot accept a model revision. |
| 20 | Accept | The batch-local duplicate-vector check is removed. The complete per-source teacher audit remains authoritative. |
| 21 | Accept | Pair selection now uses vector operations. |
| 22 | Accept | Old smoke scripts and Python cache files are removed from the reproduction directory. |
| 23 | Accept | `model_metrics` now returns only the metric dictionary. |
| 24 | Reject | The short I/O docstrings match the project style and describe storage behavior. Their removal would not change the result. |
| 25 | Accept | Trivial `parse_args` docstrings are removed. |
| 26 | Accept | Survey candidate pairs now use `set[tuple[int, int]]`. |
| 27 | Accept | The training report no longer stores the minimum batch loss. It keeps the first and final batch losses. |
| 28 | Accept | Source path overrides no longer contain unused token-count values. |

Finding 1 and finding 2 invalidate the first quality comparison. Thus, this
report labels the first result as superseded. The corrected run uses a new
artifact root.

The final disposition check found no open accepted item. The corrected code
and artifacts retain all 24 accepted changes. Finding 19 retains its stated
partial disposition: downloads do not force refreshes, and the pinned Arctic
constructor remains because its parent API cannot take a revision. Findings
16, 18, and 24 remain rejected for the reasons in the table. The GB200 thread
limit fix changes only numerical-library resource control; it does not change
the sampler, held-out split, teacher inputs, metrics, or gates.

## Corrected-run inputs

- Corrected manifest build commit: `ebe09a12e`
- Successful evaluation commit: `0b82ba40f`
- Artifact root:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2`
- Sampling: 64 uniform circular row blocks across all rows of each source.
- Probe split: 256 training rows and 256 evaluation rows from the student-heldout
  set.
- Clustering seeds: 42, 43, and 44.
- F1 uncertainty: 10,000 paired source bootstrap samples.

The corrected manifest has these fixed results:

- Manifest SHA-256:
  `4aea19379cb6b7414d80f0b72c868f239e9247c05c3a703a26b19a059599f211`
- Sources: 146
- Source groups: 28 code, 24 name-matched multilingual, 91 standard, and
  3 OOD
- Training rows: exactly 750,000 and 3,000,000
- Held-out evaluation rows: 74,752
- Selected input files: 5,341
- Stack v3 output hash: `32b6fa6f`
- Manifest audit: passed
- Survey documents: 14,600
- Non-constant documents: 100 percent
- Raw and normalized unique documents: 99.9589 percent
- Documents in a near-duplicate pair: 0.3356 percent

Corrected manifest artifacts:

- Manifest:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/manifest.json`
- Audit:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/manifest-audit.json`
- Survey:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/survey/report.html`
- Manifest job: `/rav/lux-arctic-manifest-v2-004`
- Audit job: `/rav/lux-arctic-manifest-audit-v2-001`
- Survey job: `/rav/lux-arctic-survey-v2-001-r1`

The first survey attempt did not import the workspace `marin-dupekit`
package. The replacement job synced `marin-core` and `marin-dupekit`
explicitly. It then completed without a data or native-kernel error.

## Corrected-run results

The corrected 3M student fails two required gates. It is not viable for the
20-billion-document production run under the fixed decision rule.

### Teacher audit and compute

The teacher audit checked 146 source files and 3,074,752 rows against the
corrected manifest digest. Every row has 256 `uint8` values. All 256
dimensions vary in every source. The lowest exact quantized-vector unique
fraction in one source is 89.5610 percent. This value is not the student
99-percent uniqueness gate; the student gate uses floating-point evaluation
vectors rounded to four decimals.

Teacher audit:
`s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/teacher-arctic-v1/audit.json`

One complete teacher shard and six source files from another shard were made
on H100 before the resource change. The remaining teacher work, both student
training runs, and both successful evaluations used GB200 workers. The
teacher jobs used the same pinned Arctic revision, manifest digest,
quantization code, and artifact checks. The mixed teacher hardware is a
reproduction caveat because values close to an 8-bit quantization boundary
can depend on accelerator arithmetic.

The successful evaluations used code commit `0b82ba40f`. This commit limits
BLAS to the fixed eight-thread evaluation budget. The first corrected 0.75M
evaluation exceeded the OpenBLAS thread-region limit on a high-core-count
worker. Its replacement and the 3M evaluation both completed after this fix.

### 0.75M rung

Training completed 186 steps in 6 minutes 5 seconds. Loss changed from
0.00550099 to 0.00310098.

- Model SHA-256:
  `7e5e9202272c27e9c83cc63d048bc4d5ec7f42dd65c3465d3b875af4c902c709`
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/750k/luxical-arctic.npz`
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/750k/training.json`

The 0.75M student passes four of ten required gates:

| Measure | Luxical-One | 0.75M student | Delta or ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| Finite vectors | 100% | 100% | — | Pass |
| Unique vectors at 4 decimals | — | 99.9652% | — | Pass |
| CPU documents/second | 8,369.57 | 8,372.22 | 1.00032x | Pass |
| Source macro-F1 | 0.61727 | 0.24693 | -0.37034 | Fail |
| Worst-source recall | 0.00781 | 0 | -0.00781 | Pass |
| Arctic cosine Spearman | 0.86765 | 0.81600 | -0.05165 | Fail |
| Code macro-F1 | 0.68089 | 0.24992 | -0.43098 | Fail |
| Name-matched multilingual macro-F1 | 0.79561 | 0.17597 | -0.61964 | Fail |
| Standard-text macro-F1 | 0.56887 | 0.28042 | -0.28845 | Fail |
| Regular-source collapse | — | 143 failures | — | Fail |

The paired source bootstrap 95-percent interval for macro-F1 delta is
[-0.40807, -0.33375]. The category intervals are [-0.52201, -0.34508] for
code, [-0.72651, -0.50487] for the name-matched multilingual group, and
[-0.31995, -0.25714] for standard text.

All 143 regular sources fail at least one composite collapse test. The
overlapping counts are 20 cluster-concentration failures, two uniqueness
failures, 143 effective-rank failures, and 143 variance failures. The lowest
regular-source unique fraction is 96.4844 percent. The lowest effective-rank
ratio is 0.04007, and the lowest variance ratio is 0.00628. `biocorpus` puts
all 512 held-out rows in one cluster for at least one clustering seed.

Evaluation reports:

- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/750k/report.html`
- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/750k/report.json`

### 3M rung

Training completed 735 steps in 31 minutes 28 seconds. Loss changed from
0.00558436 to 0.00124420.

- Model SHA-256:
  `395aaa10ff2cbabcff18ceabc8a575e1ea4fb49a0ebd64a894581d48f6b76c5a`
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/3m/luxical-arctic.npz`
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/3m/training.json`

The 3M student passes eight of ten required gates:

| Measure | Luxical-One | 3M student | Delta or ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| Finite vectors | 100% | 100% | — | Pass |
| Unique vectors at 4 decimals | — | 99.9652% | — | Pass |
| CPU documents/second | 8,862.05 | 8,815.10 | 0.99470x | Pass |
| Source macro-F1 | 0.61727 | 0.61938 | +0.00210 | Pass |
| Worst-source recall | 0.00781 | 0.03906 | +0.03125 | Pass |
| Arctic cosine Spearman | 0.86765 | 0.92589 | +0.05824 | Pass |
| Code macro-F1 | 0.68089 | 0.71142 | +0.03053 | Pass |
| Name-matched multilingual macro-F1 | 0.79561 | 0.68956 | -0.10605 | Fail |
| Standard-text macro-F1 | 0.56887 | 0.58850 | +0.01964 | Pass |
| Regular-source collapse | — | 97 failures | — | Fail |

The paired source bootstrap 95-percent interval for total macro-F1 delta is
[-0.01796, 0.02057]. The code interval is [0.00656, 0.05416]. The
name-matched multilingual interval is [-0.21184, -0.01342]. The standard-text
interval is [0.00613, 0.03335]. Thus, the multilingual result is below the
fixed -0.02 gate and its bootstrap interval does not include zero.

The 97 regular-source composite failures contain 55 cluster-concentration
failures, two uniqueness failures, 88 effective-rank failures, and 29
variance failures. Luxical-One has 52 regular sources above the same absolute
90-percent cluster limit. The student's lowest regular-source unique fraction
is 96.4844 percent. Its lowest effective-rank ratio is 0.27527, and its lowest
variance ratio is 0.08335. `biocorpus` puts all held-out rows in one cluster
for at least one seed.

The global 40-cluster distribution does not show a new catch-all cluster. The
student's median largest cluster share is 5.5182 percent, compared with
5.4340 percent for Luxical-One. Its effective cluster count is 36.3332,
compared with 36.1919. Source-cluster normalized mutual information is
0.56278, compared with 0.55471.

Evaluation reports:

- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/3m/report.html`
- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/3m/report.json`

### Scaling and decision

Scaling from 0.75M to 3M improves source macro-F1 by 0.37244, Arctic fidelity
by 0.10989, code macro-F1 by 0.46150, name-matched multilingual macro-F1 by
0.51359, and standard-text macro-F1 by 0.30809. Regular-source composite
collapse failures decrease from 143 to 97. The lowest effective-rank ratio
improves from 0.04007 to 0.27527, and the lowest variance ratio improves from
0.00628 to 0.08335.

Do not use the current 3M student for the 20-billion-document production run.
A useful next ladder rung is 12M documents with the same manifest method and
gates. A second 12M treatment can increase multilingual sampling weight. This
pair separates a scale effect from a mixture effect. If multilingual quality
still fails, rebuild the fixed vocabulary and IDF table from the code and
multilingual mix before another student run.

At 8,815.10 documents per second, 20 billion bounded document views require
26.26 worker-days. With ideal scaling, 100 CPU workers require 6.30 hours.
These values exclude input reads, startup, and output writes. The current
Luxical runtime is CPU based. This POC does not measure an H100, GB200, or TPU
inference path.

Corrected-run jobs:

- Teacher audit: `/rav/lux-arctic-teacher-audit-v2-gb200-001`
- 0.75M training: `/rav/lux-arctic-train-v2-gb200-750k-001`
- 0.75M evaluation: `/rav/lux-arctic-eval-v2-gb200-750k-r1`
- 3M training: `/rav/lux-arctic-train-v2-gb200-3m-001`
- 3M evaluation: `/rav/lux-arctic-eval-v2-gb200-3m-001`

## Fixed inputs for the first run

- Source registry revision:
  `656d77bff319a851cb775e5bef33570ccfd9a9f8`
- Manifest SHA-256:
  `f32689b85f4c0818d610914135263a5f410d6bd7b1098fb02cca5dee90923ba3`
- Ladder code commit:
  `9c88379885ce2342d02a3a03c71aadd1f2964107`
- Manifest:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/manifest.json`
- Sources: 146
- Source groups: 28 code, 24 name-matched multilingual, 91 standard, and 3 OOD
- Stack v3 output hash: `32b6fa6f`
- Training rows: exactly 750,000 and 3,000,000
- Held-out evaluation rows: 74,752
- Teacher: `Snowflake/snowflake-arctic-embed-m-v2.0`
- Teacher revision:
  `95c2741480856aa9666782eb4afe11959938017f`
- Baseline: `DatologyAI/luxical-one/luxical_one_rc4.npz`
- Baseline revision:
  `474cfeb959dd473b3d1cd61da630f566037e69e2`
- Seed: 42

The pinned registry has 147 sources. The current `ghalogs/public` artifact at
output hash `55a2fec7` was absent, so the fixed manifest has 146 sources.
`ghalogs/public` was in the predeclared OOD set but is not part of the
experiment. Stack v3 is present and was sampled from output hash `32b6fa6f`.

The 0.75M rows are an exact subset of the 3M rows. Each source has fixed
training and evaluation ranks. Each teacher vector is the normalized mean of
fixed head, middle, and tail windows. Each window has at most 2,000
characters. Teacher vectors use 8-bit scalar quantization with a limit of 0.3.

## Data checks

The manifest audit passed selected-file provenance, per-source quotas, exact
row counts, split rules, and strict rung nesting. It also passed the manifest
digest check.

The fixed survey read 14,600 documents. It used 80 random, 10 shortest, and 10
longest rows from each source.

- Nonconstant documents: 100 percent
- Raw unique documents: 99.9863 percent
- Normalized unique documents: 99.9863 percent
- Rows in near-duplicate pairs: 0.3082 percent at the fixed 0.80 MinHash
  threshold

Survey report:
`s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/survey/report.html`

## Teacher audit

Job `/rav/lux-arctic-teacher-audit-001` verified all 146 source files and all
3,074,752 teacher rows. Each output has 256 `uint8` dimensions. All 256
dimensions vary in every source. The minimum exact per-source unique fraction
is 89.1407 percent. The audit also checked the exact manifest digest, teacher
identity, teacher revision, source counts, and row counts.

Audit artifact:
`s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/teacher-arctic-v1/audit.json`

## Student configuration

Both students keep the exact Luxical-One inference structure:

- BERT tokenizer
- 2,000,000-item n-gram vocabulary
- Fixed IDF table
- Network dimensions `(2000000, 96, 3072, 3072, 192)`

Only the sparse-to-dense network is initialized again. Training uses three
epochs, a batch size of 12,288, a temperature of 3, a peak learning rate of
0.01, a 5 percent warmup, a 10 percent final decay, global row shuffle, and
AdamW. Both AdamW betas are 0.9, epsilon is `1e-8`, and weight decay is zero.

## Superseded first-run results

The values in this section are provisional. Finding 1 and finding 2 invalidate
their use for a model decision. The section stays as an audit record.

### 0.75M rung

Training job `/rav/lux-arctic-train-750k-002` completed all 186 steps. Loss
changed from 0.0055073155 to 0.0029378380.

- Model SHA-256:
  `7806241aaf7865215d7cc37d5b26e6a596e5dc1050529a6bdde29da131b889a1`
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/750k/luxical-arctic.npz`
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/750k/training.json`

The 0.75M student passes four of ten required gates:

| Measure | Luxical-One | 0.75M student | Delta or ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| Finite vectors | 100% | 100% | — | Pass |
| Unique vectors at 4 decimals | — | 99.9612% | — | Pass |
| CPU documents/second | 8,695.67 | 8,771.55 | 1.00873x | Pass |
| Source macro-F1 | 0.62338 | 0.26189 | -0.36149 | Fail |
| Worst-source recall | 0.00391 | 0 | -0.00391 | Pass |
| Arctic cosine Spearman | 0.86591 | 0.82682 | -0.03909 | Fail |
| Code macro-F1 | 0.69346 | 0.33272 | -0.36074 | Fail |
| Name-matched multilingual macro-F1 | 0.79413 | 0.13614 | -0.65799 | Fail |
| Standard-text macro-F1 | 0.57495 | 0.30121 | -0.27374 | Fail |
| Regular-source cluster concentration | 43 failures | 17 failures | -26 | Fail |

`biocorpus` puts all its held-out rows in one cluster. All 143 regular sources
fail at least one member of the composite collapse check. The overlapping
failure counts are 17 cluster concentration, 143 rank ratio, 143 variance
ratio, and one uniqueness failure. The lowest regular-source unique fraction
is 95.5078 percent, the lowest effective-rank ratio is 0.04242, and the lowest
variance ratio is 0.00572.

Evaluation reports:

- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/750k/report.html`
- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/750k/report.json`

At the measured eight-thread CPU rate, 20 billion bounded document views
require 26.39 worker-days for the 0.75M student. Luxical-One requires 26.62
worker-days. With ideal scaling, 100 workers require about 6.33 hours.
This estimate excludes document I/O, task startup, and output writes.

### 3M rung

Training job `/rav/lux-arctic-train-3m-001` completed all 735 steps. Loss
changed from 0.0055917976 to 0.0012042067.

- Model SHA-256:
  `e6a78c93c0ecea83290095acf7cae4a3338754588b2705cdf4ccde41b17cd8f7`
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/3m/luxical-arctic.npz`
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/3m/training.json`

The 3M student passes eight of ten required gates:

| Measure | Luxical-One | 3M student | Delta or ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| Finite vectors | 100% | 100% | — | Pass |
| Unique vectors at 4 decimals | — | 99.9612% | — | Pass |
| CPU documents/second | 8,570.02 | 8,558.12 | 0.99861x | Pass |
| Source macro-F1 | 0.62338 | 0.63005 | +0.00667 | Pass |
| Worst-source recall | 0.00391 | 0.00781 | +0.00391 | Pass |
| Arctic cosine Spearman | 0.86591 | 0.92687 | +0.06095 | Pass |
| Code macro-F1 | 0.69346 | 0.73355 | +0.04010 | Pass |
| Name-matched multilingual macro-F1 | 0.79413 | 0.70575 | -0.08838 | Fail |
| Standard-text macro-F1 | 0.57495 | 0.59404 | +0.01909 | Pass |
| Regular-source cluster concentration | 43 failures | 49 failures | +6 | Fail |

The first-run multilingual baseline gap is 0.08838. Thus, the value misses the
allowed 0.02 margin by 0.06838.

`glm-5.2-kernelgym-rollouts` puts all its held-out rows in one cluster. In
total, 82 of 143 regular sources fail at least one member of the composite
collapse check. The overlapping failure counts are 49 sources above the 90
percent cluster limit, 65 below the 0.50 rank-ratio limit, 27 below the 0.50
variance-ratio limit, and one below the 99 percent uniqueness limit. The
lowest regular-source unique fraction is 95.5078 percent, the lowest
effective-rank ratio is 0.29290, and the lowest variance ratio is 0.09920. All
three present OOD sources also fail the composite collapse check.

The absolute cluster-concentration gate also rejects Luxical-One on 43 regular
sources. Thirty-three source failures are shared by both models. The 3M
student has 16 student-only failures, and Luxical-One has 10 baseline-only
failures. Thus, the student fails the written gate, but this absolute metric
does not by itself separate representation collapse from a source whose
documents form one valid domain cluster.

The global 40-cluster distribution does not show a new catch-all cluster. Its
largest cluster has 6.8306 percent of rows, compared with 6.2861 percent for
Luxical-One. Its effective cluster count is 35.6267, compared with 35.5421.
Source-cluster normalized mutual information is 0.55995, compared with
0.56282.

Evaluation reports:

- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/3m/report.html`
- `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/3m/report.json`

At the measured eight-thread CPU rate, 20 billion bounded document views
require 27.05 worker-days for the 3M student. Luxical-One requires 27.01
worker-days. With ideal scaling, 100 workers require about 6.49 hours.
This estimate excludes document I/O, task startup, and output writes.

The current Luxical runtime uses CPU NumPy, SciPy, and Numba kernels. This
experiment does not measure a production H100 or TPU inference runtime.

### Superseded gate decision

The first code marked the 3M student as not viable because two required gates
failed. Peer review then invalidated that quality decision.

The collapse failure is conservative. The 90 percent rule also rejects
Luxical-One on 43 regular sources, compared with 49 for the student. The
student still fails the rule as written and also has 65 rank-ratio and 27
variance-ratio failures. The decision does not treat the 49 concentration
failures as 49 proven constant-output failures.

Scaling from 0.75M to 3M changes the result from four passed gates to eight.
It improves source macro-F1 by 0.36816, Arctic fidelity by 0.10004, code
macro-F1 by 0.40083, multilingual macro-F1 by 0.56962, and standard-text
macro-F1 by 0.29283. Regular-source composite collapse failures decrease from
143 to 82. This large scaling gain supports a larger rung, but the present 3M
student is not ready for the 20-billion-document production run.

### First-run recommendation

Do not run the 3M student on the 20-billion-document corpus. First, define a
baseline-calibrated collapse gate that combines uniqueness, effective rank,
variance, and concentration relative to Luxical-One. Then train a larger rung
with more multilingual weight while keeping the same 2-million-item vocabulary
and network dimensions. If the multilingual gap remains, rebuild the fixed
vocabulary and IDF table from the multilingual and code mix, and repeat the
paired speed test.

## Evaluation method

The source probe uses only student-heldout rows. It uses 256 rows per source
for probe training. It uses a different 256 rows for probe evaluation.

The probe reports source macro-F1, per-source recall, and lowest source recall.
It also reports fifth-percentile recall. Group results cover code,
name-matched multilingual, standard, and OOD sources.

The F1 comparison includes paired source bootstrap intervals. Logistic
regression must stop before its fixed iteration limit.

The collapse check uses all 512 student-heldout rows per source. It uses three
fixed clustering seeds. The source concentration gate uses the largest share
from those seeds.

The check reports exact uniqueness, four-decimal uniqueness, total variance,
and effective covariance rank. Student variance and rank are compared with
Luxical-One for each source.

The check also reports global cluster sizes, effective cluster count, and
source-cluster normalized mutual information. These diagnostics test the
global catch-all behavior in issue 6855.

Arctic fidelity uses 100,000 within-source pairs and 100,000 across-source
pairs. It compares pairwise cosine order with Spearman correlation.

The CPU test uses eight threads and 20,000 bounded document views. It warms
both models and alternates measurements for five repetitions. The median rate
gives the speed ratio.

## Incidents and limits

The pinned Arctic custom model has four nonpersistent position and RoPE
buffers. Transformers 5.12.1 did not load valid values for these buffers. The
wrapper rebuilds them from the pinned model configuration and checks the
buffers, all parameters, and startup inference. CPU and GPU diagnostics then
returned finite and distinct vectors. A complete 21,889-document source smoke
test also passed.

Incident record: https://echo.oa.dev/wiki/49

Kueue preempted three of eight teacher shards at 10:51:14 UTC to admit another
workload. Iris scheduled retries. The source files use atomic writes, so each
retry reused complete sources and restarted only its incomplete source. The
three attempts stopped on different source types without a Python, model, or
data error.

One H100 teacher worker disappeared after it had written six complete source
files. Seven replacement H100 jobs then stayed gated because no H100 capacity
was available. Those jobs were stopped. Seven federated, interactive GB200
jobs replaced them. Each GB200 job used one accelerator. Atomic source files
let the shard-4 replacement reuse the six complete H100 files. All seven GB200
jobs finished with zero failures and zero preemptions.

Incident record: https://echo.oa.dev/wiki/52

The first 0.75M training attempt loaded all rows but failed when Arrow joined a
text batch whose data was larger than the 32-bit string offset limit. The
training loader now casts the combined text column to Arrow `large_string`
before shuffled batch selection. A local batch-selection check and the
repository lint and type checks passed before the replacement job started.

The teacher uses only three fixed windows from each document. Thus, it can miss
content outside these windows. The domain labels are source names, not semantic
topic labels. The OOD exemption applies only to collapse checks. It does not
remove OOD rows from the main source probe or Arctic fidelity test.

The held-out evaluation has the same 512-row quota for each source. Its global
cluster balance can show catch-all behavior across sources, but it does not
estimate cluster sizes for the production document mixture. A production-scale
clustering run is outside this one-week POC.

The sampler uses contiguous global row blocks to limit private object reads.
Each block start is uniform across a source. Thus, every row has equal marginal
probability, but rows in one block are correlated.

Luxical-One was trained on about 50 million documents, compared with 0.75
million and 3 million here. This ladder is a fixed-budget viability test. It
is not evidence that Arctic distillation is better or worse than the original
Luxical teacher at equal training scale.

The absolute per-source cluster rule is not a sufficient collapse test.
Luxical-One fails it on 43 regular sources. A source can put most documents in
one valid domain cluster while its vectors remain unique and high-rank. The
report therefore shows the rule together with exact uniqueness, effective
rank, variance, and global cluster balance.

## Reproduction artifacts

The corrected evaluation uses commit `0b82ba40f`. The report records all model,
data, and teacher revisions needed to repeat the run. The corrected artifacts
are:

- Source inventory:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/source_inventory.json`
- Manifest:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/manifest.json`
- Manifest audit:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/manifest-audit.json`
- Survey:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/survey/report.html`
- Teacher audit:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/teacher-arctic-v1/audit.json`
- 0.75M training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/750k/training.json`
- 0.75M evaluation:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/750k/report.json`
- 3M training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/students/3m/training.json`
- 3M evaluation:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/3m/report.json`

The main successful jobs are:

- Manifest: `/rav/lux-arctic-manifest-v2-004`
- Manifest audit: `/rav/lux-arctic-manifest-audit-v2-001`
- Survey: `/rav/lux-arctic-survey-v2-001-r1`
- Teacher audit: `/rav/lux-arctic-teacher-audit-v2-gb200-001`
- 0.75M training: `/rav/lux-arctic-train-v2-gb200-750k-001`
- 0.75M evaluation: `/rav/lux-arctic-eval-v2-gb200-750k-r1`
- 3M training: `/rav/lux-arctic-train-v2-gb200-3m-001`
- 3M evaluation: `/rav/lux-arctic-eval-v2-gb200-3m-001`

The superseded first-run artifacts remain under `manifest-v1`. They are kept
only as an audit record because findings 1 and 2 invalidate their quality
comparison.
