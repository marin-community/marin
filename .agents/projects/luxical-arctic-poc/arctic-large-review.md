# Arctic Embed Large v2.0 teacher review

Status: review addressed; corrected run complete

## Decision

Keep `Snowflake/snowflake-arctic-embed-m-v2.0` as the default teacher for the
next Luxical student rung. Large v2.0 gives a measured multilingual gain and
fewer source-concentration failures, but it gives a measured code loss and no
overall gain. Do not start a full Large-label run from this teacher-side test.

## Controlled inputs

- Evaluation rows: 74,752 from the corrected fixed holdout.
- Large revision: `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`.
- Medium revision: `95c2741480856aa9666782eb4afe11959938017f`.
- Both teacher representations use three windows for each document, at most
  512 tokens for each window, the first 256 MRL dimensions, and the same 8-bit
  quantization and normalization path.
- Both representations use the pinned Luxical pooling path, eager attention,
  explicit position IDs, float32 inference, and the same token staging.
- The test reuses the saved Medium vectors. It embeds only the Large vectors.
- Probe rows, source groups, clustering seeds, bootstrap seeds, and gate limits
  are identical.

## Corrected results

| Representation | Overall | Code | Name-matched multilingual | Standard |
| --- | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 |
| Arctic Medium v2.0 | 0.66915 | 0.79995 | 0.72857 | 0.63070 |
| Arctic Large v2.0 | 0.66810 | 0.79003 | 0.74798 | 0.62683 |

Large minus Medium paired-source results:

| Group | Macro-F1 delta | 95% interval |
| --- | ---: | ---: |
| Overall | -0.00105 | [-0.00616, +0.00408] |
| Code | -0.00992 | [-0.01922, -0.00150] |
| Name-matched multilingual | +0.01941 | [+0.00156, +0.03893] |
| Standard | -0.00388 | [-0.00908, +0.00120] |

Large minus Luxical-One multilingual macro-F1 is -0.04763. Its interval is
[-0.11048, +0.01441]. The point estimate fails the allowed -0.02 limit.

Large passes six of eight direct gates. It fails
`regular_source_collapse` and `multilingual_macro_f1`.

- Finite fraction: 1.0.
- Exact and four-decimal unique fractions: 0.999759.
- Regular composite failures: 51 of 143.
- Failure groups: 15 of 28 code, zero of 24 name-matched multilingual, and 36
  of 91 standard sources.
- Failure reasons: 50 cluster concentration and one uniqueness failure.
- Rank failures: zero.
- Variance failures: zero.
- Minimum Large-to-Luxical rank ratio: 1.08243.
- Minimum Large-to-Luxical variance ratio: 1.44775.

Large reduces the Medium composite failures from 60 to 51. Its median global
code results are:

| Representation | Largest code cluster | Effective code clusters | Code source-cluster NMI |
| --- | ---: | ---: | ---: |
| Luxical-One | 0.21819 | 10.30169 | 0.42133 |
| Arctic Medium v2.0 | 0.17055 | 11.14955 | 0.52520 |
| Arctic Large v2.0 | 0.13958 | 11.49555 | 0.52480 |

These results do not show rank, variance, constant-vector, or modality-wide
code collapse in Large. The strict failure is per-source concentration.

## Compute and limits

The Large embedding phase took 1,490.43 seconds on one GB200-class worker. It
processed 50.15 documents per second, with three windows for each document.
The complete source loop took 1,659.20 seconds. This is not a student-speed
measurement. Medium throughput was not measured in this run.

Source groups use fixed name rules. Each window has a 512-token limit, so this
test does not use Large's full context limit. These limits stay fixed for the
controlled checkpoint comparison.

No Large-distilled student was trained. Teacher-side probe and collapse
results cannot prove student fidelity, quality, or speed.

## Peer-review dispositions

The independent review found that the first Large run did not use the same
embedding implementation as Medium. It also found that gate counts could hide
the teacher tradeoff and that the report did not make the missing student test
clear.

- Accepted: use the shared pinned pooling path, eager attention, explicit
  position IDs, float32 inference, and the same token staging. The corrected
  run uses this path.
- Accepted: mark the first result as superseded and write corrected artifacts
  to a new path.
- Accepted: select the teacher from paired quality intervals, not from the
  count of passed gates.
- Accepted: state that Large improves multilingual quality and concentration,
  while Medium has better code quality.
- Accepted: state that no Large-distilled student or student-speed test exists.
- Accepted: fix the embedding timer and separate it from the source-loop time.
- Accepted: record the 512-token context limit and all embedding metadata.
- Accepted: remove Medium-inapplicable collapse gates from the direct
  Large-minus-Medium block.
- Accepted: use “failure reasons,” because the 50 concentration failures and
  one uniqueness failure sum to the 51 failed sources.
- Rejected: move one-use HTML rendering and constants into shared helpers.
  This cleanup does not change the method or result.

The accepted findings are addressed. The peer-review skill does not require a
second review loop.

## Reproduction

- Corrected harness commit: `beb363f2b`.
- Job: `/rav/lux-arctic-large-teacher-v2-gb200-002`.
- JSON:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0-v2/report.json`.
- HTML:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0-v2/report.html`.

The superseded first result remains at the old `teacher-arctic-l-v2.0` path as
an audit record.
