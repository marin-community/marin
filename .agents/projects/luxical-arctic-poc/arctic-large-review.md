# Arctic Embed Large v2.0 teacher review

## Decision under review

Keep `Snowflake/snowflake-arctic-embed-m-v2.0` as the current Arctic teacher.
Do not make student-training labels with
`Snowflake/snowflake-arctic-embed-l-v2.0` based on this test.

## Controlled inputs

- Evaluation rows: 74,752 from the corrected fixed holdout.
- Large revision: `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`.
- Medium revision: `95c2741480856aa9666782eb4afe11959938017f`.
- Both teacher representations use three windows per document, at most 512
  tokens per window, the first 256 MRL dimensions, and the same 8-bit
  quantization and normalization path.
- The test reuses the saved Medium vectors. It embeds only the Large vectors.
- Probe train and evaluation rows, source groups, clustering seeds, bootstrap
  seeds, and gate limits are identical.

## Source probe results

| Representation | Overall | Code | Name-matched multilingual | Standard |
| --- | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 |
| Arctic Medium v2.0 | 0.66915 | 0.79995 | 0.72857 | 0.63070 |
| Arctic Large v2.0 | 0.66171 | 0.78894 | 0.73634 | 0.62040 |

Large minus Medium paired-source results:

| Group | Macro-F1 delta | 95% interval |
| --- | ---: | ---: |
| Overall | -0.00743 | [-0.01200, -0.00300] |
| Code | -0.01101 | [-0.01920, -0.00345] |
| Name-matched multilingual | +0.00777 | [-0.00817, +0.02441] |
| Standard | -0.01030 | [-0.01519, -0.00566] |

Large minus Luxical-One multilingual macro-F1 is -0.05927. Its interval is
[-0.12500, +0.00438]. The point estimate fails the allowed -0.02 limit.

## Gate and collapse results

Large passes six of eight direct gates. It fails
`regular_source_collapse` and `multilingual_macro_f1`.

- Finite fraction: 1.0.
- Exact and four-decimal unique fractions: 0.999759.
- Regular composite failures: 44 of 143.
- Failure groups: 13 of 28 code, zero of 24 name-matched multilingual, and 31
  of 91 standard sources.
- Overlapping reasons: 43 cluster concentration and one uniqueness failure.
- Rank failures: zero.
- Variance failures: zero.
- Minimum Large-to-Luxical rank ratio: 1.02636.
- Minimum Large-to-Luxical variance ratio: 1.51776.

Large reduces the Medium per-source composite failures from 60 to 44. Its
global code distribution is worse than Medium:

| Representation | Largest code cluster | Effective code clusters | Code source-cluster NMI |
| --- | ---: | ---: | ---: |
| Luxical-One | 0.21819 | 10.30169 | 0.42133 |
| Arctic Medium v2.0 | 0.17055 | 11.14955 | 0.52520 |
| Arctic Large v2.0 | 0.20752 | 9.86327 | 0.50727 |

## Compute and limits

The Large embedding phase took 1,635.47 seconds on one GB200-class worker. It
processed 45.71 documents per second, with three windows for each document.
This is not a student-speed measurement. Medium throughput was not measured in
this run.

Source groups use fixed name rules. Each window has a 512-token limit. These
limits are inherited from the Medium test and remain fixed so the checkpoint
is the only intended model change.

## Reproduction

- Harness commit: `2dbef9c06`.
- Result commit: `202481d1e`.
- Job: `/rav/lux-arctic-large-teacher-v2-gb200-001`.
- JSON:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.json`.
- HTML:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.html`.

## Review questions

1. Do the results support the decision to keep Medium?
2. Does the report separate probe quality from the strict concentration gate?
3. Does any result support a claim of rank, variance, constant-vector, or
   modality-wide code collapse in Large?
4. Is a stated limit or comparison missing from the decision?
