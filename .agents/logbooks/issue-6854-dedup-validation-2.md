# Issue #6854 fuzzy-dedup false-positive validation, volume 2

Status: running

Coordinating issue: https://github.com/marin-community/marin/issues/6854

Previous volume:
[issue-6854-dedup-validation.md](issue-6854-dedup-validation.md).

## Objective

Continue the exhaustive baseline-versus-treatment semantic review on the full
115-source, 103,716,988-document DataKit 100B testbed. The pinned arms,
evaluation contract, artifact roots, and results through 153,122 reviewed pairs
are recorded in the previous volume.

## Experiment log

### 2026-07-26T20:19:04Z — 179,480 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2019-v587` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 2,304.
  Its 128 baseline pairs contain 59 false positives and 69 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  331 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `3e774aca04a638adcd64d1f2212e841d121b527d636b690a2bd025060be52402`.

- Across the stable 1,413-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,833 pairs, 91,425 false positives, 52,408 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,480 pairs, 109,880 false positives, 69,600 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,432)`, and p3 `(104, 128)`. P2's next batch has 170 review units
  and 340 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:17:33Z — 179,352 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2016-v586` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 2,176.
  Its 128 baseline pairs split evenly between 64 false positives and 64 true
  duplicates, with no unresolved outcomes. Four pairs were chunked and 124
  were direct. All 457 judgments and request attempts were valid on their
  first attempt. The outcome Parquet SHA-256 is
  `bfe1709eec83feabecef5db3957d271b8272acda9ba9ee0bec9a761be42a8c3e`.

- Across the stable 1,412-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,705 pairs, 91,366 false positives, 52,339 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,352 pairs, 109,821 false positives, 69,531 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,304)`, and p3 `(104, 128)`. P2's next batch has 158 review units
  and 316 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:15:42Z — 179,224 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2014-v585` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 0. Its
  128 baseline pairs contain 115 false positives and 13 true duplicates, with
  no unresolved outcomes. Twenty-one pairs were chunked and 107 were direct.
  All 2,853 judgments and request attempts were valid on their first attempt.
  The outcome Parquet SHA-256 is
  `b9147dfc6fd7bc201dc4b1e9904fa80849177c28ae0752c89e80eb8a4c96bd0d`.

- Across the stable 1,411-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,577 pairs, 91,302 false positives, 52,275 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,224 pairs, 109,757 false positives, 69,467 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,176)`, and p3 `(104, 128)`. P3's next batch has 1,143 review
  units and 2,286 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:13:27Z — 179,096 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2012-v584` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 2,048.
  Its 128 baseline pairs contain 59 false positives and 69 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  273 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `be71d176844b594d183be464fe011e512d6c07b7fd7196df2edf8a0000e453e9`.

- Across the stable 1,410-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,449 pairs, 91,187 false positives, 52,262 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,096 pairs, 109,642 false positives, 69,454 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,176)`, and p3 `(104, 0)`. P2's next batch has 225 review units
  and 450 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:11:35Z — 178,968 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2010-v583` independently
  revalidated four p2 decision-file 71 checkpoints at semantic offsets 1,536,
  1,664, 1,792, and 1,920. Their 512 baseline pairs contain 237 false
  positives and 275 true duplicates, with no unresolved outcomes. One pair
  was chunked and 511 were direct. All 1,099 judgments and request attempts
  were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `7c39e2a058b88f5ce81f3caced250d5dba090f03b1c47d74d4461e2e806d6cda`,
  `d10c49ab160f7580234e0656e874c02ae7eba424934d79e93c6159cfd4210b08`,
  `9eda8c73f637366f16061e754ddc51c534306107a7ba10e647419bc4fef07221`,
  and
  `8bc8ad953e0d5b63eb67d9a4c6752b0a46ab9b7f661ecbd8b4c3cdc172415d2e`.

- Across the stable 1,409-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,321 pairs, 91,128 false positives, 52,193 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 178,968 pairs, 109,583 false positives, 69,385 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,048)`, and p3 `(104, 0)`. P2's next batch has 132 review units
  and 264 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:06:47Z — 178,456 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2005-v582` independently
  revalidated two p2 decision-file 71 checkpoints at semantic offsets 1,280
  and 1,408. Their 256 baseline pairs contain 197 false positives and 59 true
  duplicates, with no unresolved outcomes. All pairs were direct, and all 527
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `e9689c1ef0b685388e3aa2764b0a3ae6b0ba2493f42948170e69ca46266b1577`
  and
  `fedb8265dd0c938f0b38fa5145de0c201f575f754de45c91e91d29c547aa0784`.

- Across the stable 1,405-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 142,809 pairs, 90,891 false positives, 51,918 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 178,456 pairs, 109,346 false positives, 69,110 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 1,536)`, and p3 `(104, 0)`. P2's next batch has 150 review units
  and 300 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:03:45Z — 178,200 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2002-v581` independently
  revalidated two p2 decision-file 71 checkpoints at semantic offsets 1,024
  and 1,152. Their 256 baseline pairs contain 236 false positives and 20 true
  duplicates, with no unresolved outcomes. All pairs were direct, and all 516
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `4eb4f75af49f646460a794e9b49cdc273e73dff036331fbf0f1e232b23257256`
  and
  `8dc7fe786e6d279e426a8119ae9dc06063d35156664a44946a4cfa7bda9eaedd`.

- Across the stable 1,403-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 142,553 pairs, 90,694 false positives, 51,859 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 178,200 pairs, 109,149 false positives, 69,051 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 1,280)`, and p3 `(104, 0)`. P2's next 128-pair batch is entirely
  direct and requires 256 minimum model requests. All four batch-priority
  2-H100 workers continue serving requests. Their 12 root, broker, and GPU
  pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:00:25Z — 177,944 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1957-v580` independently
  revalidated p1 decision-file 39 semantic offset 0 and p2 decision-file 71
  offsets 768 and 896. Their 384 baseline pairs contain 306 false positives
  and 78 true duplicates, with no unresolved outcomes. The p1 checkpoint
  contains 29 chunked and 99 direct pairs; both p2 checkpoints are fully
  direct. All 4,092 judgments and request attempts were valid on their first
  attempt.

- The outcome Parquet SHA-256 values are
  `f92b308b22700c5c95307baf0a7c0ae65f983c8dbc26bdd4ddea18b55810d74f`,
  `829d88482ee6cda7e1cbcd2bfadcbd301d8222882c2d3a855389749e6d65eafb`,
  and
  `30f582dffb675e956a6c256dd80184dc5b36b503a270f2a945379221c96cc14d`.

- Across the stable 1,401-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 142,297 pairs, 90,458 false positives, 51,839 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 177,944 pairs, 108,913 false positives, 69,031 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 1,024)`, and p3 `(104, 0)`. P1's next batch has 538 review units
  and 1,076 minimum model requests; p2's is entirely direct with 256 minimum
  requests. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T19:56:58Z — 177,560 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1956-v579` independently
  revalidated p2 decision-file 71 semantic offset 640. Its 128 baseline pairs
  contain 89 false positives and 39 true duplicates, with no unresolved
  outcomes. Two pairs were chunked and 126 were direct. All 301 judgments and
  request attempts were valid on their first attempt. The outcome Parquet
  SHA-256 is
  `f43309952121ceac3e0b95b96c44f60fef9c335413eedcc7d27a444a3cda90be`.

- Across the stable 1,398-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,913 pairs, 90,152 false positives, 51,761 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 177,560 pairs, 108,607 false positives, 68,953 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 0)`,
  p2 `(71, 768)`, and p3 `(104, 0)`. P2's next 128-pair batch is entirely
  direct and requires 256 minimum model requests. All four batch-priority
  2-H100 workers continue serving requests. Their 12 root, broker, and GPU
  pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T19:53:42Z — 177,432 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1952-v578` independently
  revalidated three p2 decision-file 71 checkpoints at semantic offsets 256,
  384, and 512. Their 384 baseline pairs contain 264 false positives and 120
  true duplicates, with no unresolved outcomes. Nine pairs were chunked and
  375 were direct. All 1,707 judgments and request attempts were valid on
  their first attempt.

- The outcome Parquet SHA-256 values are
  `a2790d900eabeafde892063cb519f13061490a16425d09a297234e49c2a1e02d`,
  `ac2b2f1c652252469ea8da5957b556d9e0351be42cee461a1aeb170bea91989d`,
  and
  `a1fc099dd1a788d02d30c189628ac9d3a8420933f685a2bd3efa6ae949a33f23`.

- Across the stable 1,397-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,785 pairs, 90,063 false positives, 51,722 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 177,432 pairs, 108,518 false positives, 68,914 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 0)`,
  p2 `(71, 640)`, and p3 `(104, 0)`. P2's next 128-pair batch has only 143
  review units and 286 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T19:41:55Z — 177,048 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1940-v574` independently
  revalidated the final five treatment pairs in p0 decision-file 7 at semantic
  offset 5,888 and 128 baseline pairs in p2 decision-file 71 at offset 128.
  Their 133 pairs contain 99 false positives and 34 true duplicates, with no
  unresolved outcomes. Seventeen pairs were chunked and 116 were direct. All
  1,616 judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `a0c4b8ecb1db558ba8b4cff2eee535e965888fc1527da9f527623dbd528a07e8`
  and
  `54da5c61249b603c3b5f67d12a141608376a50ab846b62bcd36413643ffaf522`.

- Across the stable 1,394-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,401 pairs, 89,799 false positives, 51,602 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 177,048 pairs, 108,254 false positives, 68,794 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 0)`,
  p2 `(71, 256)`, and p3 `(104, 0)`. P0's next batch contains 114,125,475
  characters and 21 oversized pairs; p2's contains 15,796,542 characters and
  eight oversized pairs. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T19:39:28Z — 176,915 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1932-v567` independently
  revalidated two p0 decision-file 7 checkpoints at semantic offsets 5,632 and
  5,760. Their 256 treatment pairs contain 149 model false positives, 105
  model true duplicates, and two unresolved outcomes. All pairs were direct.
  The audit reread 525 judgments and 536 request attempts: 520 attempts were
  valid, 16 were invalid, and six judgments required retries.

- The outcome Parquet SHA-256 values are
  `d9aecf809b3c9ed03edf9e24ba1f1ed334facee2053e134ec2e0db7aef298ac3`
  and
  `943ba2a3a9d9f60dcc9f67ac9bcbef706deaa7b0862c9906828706868f56b2c7`.

- Complete-text inspection resolves both ambiguities as true duplicates:

  - row 9,117 compares 11,386- and 11,379-character social-media SFT
    examples. All 276 lines, the question, reasoning, and answer are identical
    except that the member ends with a text-wrapped boxed `C` and the canonical
    boxes `C` directly. Character, line, and word-sequence similarities are
    0.999693, 0.996377, and 0.999740. Member/canonical SHA-256 values are
    `8081d0f6e06d2f4f0ac28208834a0f864e072a56579ffd39c8ad944400414b05`
    and
    `ecdfddf42e04419f57aeb648f2f3993924fbec5852a4f0fd318125503adb88e3`.
  - row 9,118 compares 12,912- and 12,905-character e-waste SFT examples.
    All 316 lines, the question, reasoning, and answer are identical except
    for the same text-wrapper difference around boxed `G`. Character, line,
    and word-sequence similarities are 0.999729, 0.996835, and 0.999737.
    Member/canonical SHA-256 values are
    `ebd1d8615455ac208bf1956ebfe4863e01569a2bf845da88aad270fc2f56c096`
    and
    `fe99efb824e39d5c07db42df5f79ef1649b1408b46d16c79d2341b1458f582ee`.

- `/rav/datakit-6854-inspect-row9117-1933-v568` and
  `/rav/datakit-6854-inspect-row9118-1934-v569` persisted the complete pairs
  and diffs with inspection SHA-256 values
  `bdcf3cdb544d2cb13697cfd1f33177527a52c49a8228207b161267d651ca949b`
  and
  `9a0039c130bdb8d50a4826ff13e03018311eb8a6404d37f2d6a9970104329c72`.
  Their semantic-judgment SHA-256 values are
  `e6cf23b675b858c9542fbbfc3e704822453854d41e149f89f3612a5832f0af39`
  and
  `f8655e1dfb560dc2a0be96dc985dd8a27b47a47e1d3e3883be5995be069860ef`.

- `/rav/datakit-6854-publish-row9117-1936-v571` and
  `/rav/datakit-6854-publish-row9118-1935-v570` wrote immutable
  true-duplicate records. Separate jobs
  `/rav/datakit-6854-verify-row9117-1938-v572` and
  `/rav/datakit-6854-verify-row9118-1939-v573` independently reread the source
  pairs, semantic checkpoints, inspections, deterministic Parquet bytes, and
  completion markers. Their manual-Parquet SHA-256 values are
  `d1a2bdc9f2cbe480bf14dfd779800eb52ab7735b88768b4fe0e8ca7f7295c235`
  and
  `257bdac32b1fa7a649c05760174cf8163865bf201315e87ca20cb3d9ab2de97a`;
  their marker SHA-256 values are
  `102990986e74c32037961c4e6163da6c638d92a954aada5583ec003a39c0dc31`
  and
  `924930227d4a397f95e0581431044c1ab8a3f510e8088c83b12bb8b61d77d8e6`.

- Across the stable 1,392-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 35,642 pairs, 18,450 false positives, 17,192 true duplicates;
  - combined: 176,915 pairs, 108,155 false positives, 68,760 true duplicates.

- The next audit frontiers are p0 `(7, 5,888)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:30:55Z — 176,659 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1930-v566` independently
  revalidated p0 decision-file 7 semantic offset 5,504. Its 128 treatment
  pairs contain 46 false positives and 82 true duplicates, with no unresolved
  outcomes. All pairs were direct, and all 268 judgments and request attempts
  were valid on their first attempt. The outcome Parquet SHA-256 is
  `86cae05168a54ea706e09252392d47e1fc4571b5013293fb65b4383036094998`.

- Across the stable 1,390-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 35,386 pairs, 18,301 false positives, 17,085 true duplicates;
  - combined: 176,659 pairs, 108,006 false positives, 68,653 true duplicates.

- The next audit frontiers are p0 `(7, 5,632)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:28:50Z — 176,531 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1928-v565` independently
  revalidated two p0 decision-file 7 checkpoints at semantic offsets 5,248 and
  5,376. Their 256 treatment pairs contain 76 false positives and 180 true
  duplicates, with no unresolved outcomes. All pairs were direct, and all 527
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `b1becc8791fd455ef110154cff130467a48d38ec567e783da8411662d61f93da`
  and
  `35c0bd374943f8e839bf924640f2db30b1d3c31c55e4316747d4224054fd0889`.

- Across the stable 1,389-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 35,258 pairs, 18,255 false positives, 17,003 true duplicates;
  - combined: 176,531 pairs, 107,960 false positives, 68,571 true duplicates.

- The next audit frontiers are p0 `(7, 5,504)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:26:50Z — 176,275 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1926-v564` independently
  revalidated p0 decision-file 7 semantic offset 5,120. Its 128 treatment
  pairs contain 37 false positives and 91 true duplicates, with no unresolved
  outcomes. All pairs were direct, and all 265 judgments and request attempts
  were valid on their first attempt. The outcome Parquet SHA-256 is
  `309efbcf0ebf32041f10722414413f361ec9d1f9e5e883163331963de8a1ea65`.

- Across the stable 1,387-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 35,002 pairs, 18,179 false positives, 16,823 true duplicates;
  - combined: 176,275 pairs, 107,884 false positives, 68,391 true duplicates.

- The next audit frontiers are p0 `(7, 5,248)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:25:15Z — 176,147 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1924-v563` independently
  revalidated two p0 decision-file 7 checkpoints at semantic offsets 4,864 and
  4,992. Their 256 treatment pairs contain 234 false positives and 22 true
  duplicates, with no unresolved outcomes. All pairs were direct, and all 517
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `d7a265ec926711c6daa70fb4b6abaa6c346c88ad3a27c98c7425d1ee9c363cef`
  and
  `b059eeec0dcb7eb068aff99469966c10f1713f5cc43e6d1b41d25709176bec70`.

- Across the stable 1,386-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 34,874 pairs, 18,142 false positives, 16,732 true duplicates;
  - combined: 176,147 pairs, 107,847 false positives, 68,300 true duplicates.

- The next audit frontiers are p0 `(7, 5,120)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:22:56Z — 175,891 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1922-v562` independently
  revalidated p0 decision-file 7 semantic offset 4,736. Its 128 treatment
  pairs contain 71 false positives and 57 true duplicates, with no unresolved
  outcomes. All pairs were direct, and all 269 judgments and request attempts
  were valid on their first attempt. The outcome Parquet SHA-256 is
  `36ae78ee64e58e9097d49c0aaff96df881bbc3d3ac52a8c221f47f4eb7237fdc`.

- Across the stable 1,384-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 34,618 pairs, 17,908 false positives, 16,710 true duplicates;
  - combined: 175,891 pairs, 107,613 false positives, 68,278 true duplicates.

- The next audit frontiers are p0 `(7, 4,864)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:21:25Z — 175,763 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1920-v561` independently
  revalidated p0 decision-file 7 semantic offset 4,608. Its 128 pairs contain
  101 false positives and 27 true duplicates, with no unresolved outcomes.
  The baseline contributes 75 false positives and eight true duplicates; the
  treatment contributes 26 and 19. Two pairs were chunked and 126 were direct.
  All 573 judgments and request attempts were valid on their first attempt.
  The outcome Parquet SHA-256 is
  `7b4c2d732624c16d7e999db33a7ead95b6d7e7c438017c1fc0feaeced4a13864`.

- The audit also measured each next unsealed batch from the complete texts.
  P1's 128 pairs contain 129,039,872 characters, 29 oversized pairs, and one
  12,827,562-character pair. P2 contains 52,385,231 characters, 16 oversized
  pairs, and one 10,366,673-character pair. P3 contains 91,317,419 characters,
  21 oversized pairs, and one 8,521,935-character pair. These exhaustive
  chunked reviews explain their long checkpoint intervals; their inference
  response counters continue moving.

- Across the stable 1,383-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,273 pairs, 89,705 false positives, 51,568 true duplicates;
  - treatment: 34,490 pairs, 17,837 false positives, 16,653 true duplicates;
  - combined: 175,763 pairs, 107,542 false positives, 68,221 true duplicates.

- The next audit frontiers are p0 `(7, 4,736)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:08:20Z — 175,635 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1907-v555` independently
  revalidated p0 decision-file 7 semantic offset 4,480. Its 128 baseline pairs
  contain 113 false positives and 15 true duplicates, with no unresolved
  outcomes. All pairs were direct, and all 262 judgments and request attempts
  were valid on their first attempt. The outcome Parquet SHA-256 is
  `9ffecf32a28349ed811ead655f2e279e5d44fc39968461563d673acdde8c44a5`.

- Across the stable 1,382-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,190 pairs, 89,630 false positives, 51,560 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 175,635 pairs, 107,441 false positives, 68,194 true duplicates.

- The next audit frontiers are p0 `(7, 4,608)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:06:47Z — 175,507 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1905-v554` independently
  revalidated two p0 decision-file 7 checkpoints at semantic offsets 4,224
  and 4,352. Their 256 baseline pairs contain 146 false positives and 110 true
  duplicates, with no unresolved outcomes. All pairs were direct, and all 524
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `7393a4760b9faa25af5b7d371209a7315a9429305bd90e179c2a7ad2e080c18d`
  and
  `f26ef2fba84846056ac1318ef90dc03a7fec8fcaef87b24e19a6c0c20d8c3d39`.

- Across the stable 1,381-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 141,062 pairs, 89,517 false positives, 51,545 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 175,507 pairs, 107,328 false positives, 68,179 true duplicates.

- The next audit frontiers are p0 `(7, 4,480)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:04:57Z — 175,251 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1902-v553` independently
  revalidated six p0 decision-file 7 checkpoints at semantic offsets 3,456
  through 4,096. Their 768 baseline pairs contain 582 false positives and 186
  true duplicates, with no unresolved outcomes. Two pairs were chunked and 766
  were direct. All 1,627 judgments and request attempts were valid on their
  first attempt.

- The outcome Parquet SHA-256 values are
  `c1f3629c992465fc4147eb6d13bfe338f84952e7b406eb353555c1600f6d5255`,
  `e085fd0fe372c893c1390f85831148b2545f046284150d88eda1c802ecd739b2`,
  `9bbc069e3aaf5ce3e0de44cc5106f08944511e8a64540165390aa6d04d427c93`,
  `4d9237b27ce2f227f7fd6fc1aea6b0c80d1b511c1d8b86eb6af6c46c9cb929d5`,
  `daf90ccbc143b9cfebf166a1396c2791d615d9b55d9c6b663549609d3212dbef`,
  and
  `d7f2922b0d55d0278e68bab5f2b172ddaa5c4c795d58bfd0a3ac921089df41ff`.

- Across the stable 1,379-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 140,806 pairs, 89,371 false positives, 51,435 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 175,251 pairs, 107,182 false positives, 68,069 true duplicates.

- The next audit frontiers are p0 `(7, 4,224)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T19:01:00Z — 174,483 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1853-v546` independently
  revalidated three p0 decision-file 7 checkpoints at semantic offsets 3,072,
  3,200, and 3,328, plus p2 decision-file 71 semantic offset 0. Their 512
  baseline pairs contain 296 model false positives, 214 model true duplicates,
  and two unresolved outcomes. Thirty pairs were chunked and 482 were direct.
  All 4,175 judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `30d8ecf6950b64a06d8a3958ddb7cade9eb92592a4390daa92c55b09760b3072`,
  `764fe1e5484b6e0577d54475f53e39d7131c44a162eb44a59158644629316b39`,
  `5f3e4cd6ebe567613afbc8090586cb1dc99338c4c67f63e04670e57e8392fc31`,
  and
  `7f57e55c02140a51b13ed7cfecb2959426c3bd82a2966e76ba0757260a9bd2bf`.

- Complete-text inspection resolves p0 row 5,471 as a false positive. The
  4,739/6,677-character Nectar mattress pages share a corrupted review
  scaffold, but the member alone instructs readers to visit
  `NectarSleep.com` for discount details and frames the record as a consumer
  coupon page. The canonical discusses the company website only for warranty
  details. Character, line, and word-sequence similarities are 0.588297,
  0.342857, and 0.519685. Member/canonical text SHA-256 values are
  `8f5ad9c7d0753c3dd7df604170a9fbc71bf2770c33aa0f035c60d219562649cd`
  and
  `b824721714ab59634414dc91587a2899ec1fef5fc4934206ea7c449b8e3f07c6`.

- Complete-text inspection resolves p0 row 5,818 as a false positive. The
  1,048/827-character college SEO pages share two sentence scaffolds, but the
  member adds a complete instruction that a corporate internship can improve
  career prospects and provide work experience, plus additional program and
  adult-education payloads. This exceeds low-value institution slots and
  matches the earlier false-positive boundary recorded against the same
  canonical SHA-256
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
  The member SHA-256 is
  `d3e0766e2886c8181b5d864fda12b56d8083a3f47a1fdb2c48af0005188f618b`;
  character, line, and word-sequence similarities are 0.693333, 0.333333,
  and 0.633452.

- `/rav/datakit-6854-inspect-row5471-1855-v547` and
  `/rav/datakit-6854-inspect-row5818-1856-v548` persisted the complete pairs
  and diffs with inspection SHA-256 values
  `8e35ccee9b2dad2e49731f9a5cdef083f7577c815b87fdde7b25324f37ab6fc5`
  and
  `8922d67d32cd8ed7585ce1fd8c19d0ef58f5097ed68f2ff5a363f3e217644f2b`.
  Their semantic-judgment SHA-256 values are
  `cf7f7dd1eaba85cd40d1934272334904f2cba3c351e894801a4e2b4a21f63114`
  and
  `c1e73059fa105159493c6c3d1d87f2405ed40b745d67312a1cf0739b47d9ed99`.

- `/rav/datakit-6854-publish-row5471-1858-v550` and
  `/rav/datakit-6854-publish-row5818-1857-v549` wrote the immutable
  false-positive records. Separate jobs
  `/rav/datakit-6854-verify-row5471-1859-v551` and
  `/rav/datakit-6854-verify-row5818-1900-v552` independently reread the source
  pairs, semantic checkpoints, inspections, deterministic Parquet bytes, and
  completion markers. Their manual-Parquet SHA-256 values are
  `f19ab283004b611ea09ed8ca2455e080d1d9d47f27128622ad4f39a74dbc6388`
  and
  `89adb1edc1424c441f475e1ba91de7ea2e79d7a180e954c4c0793567158ffe59`;
  their marker SHA-256 values are
  `a9d505b9c429e5fa04c8bd6fb2a8aee4ae152c88e86ca20d7eee48c00bc5526b`
  and
  `25f8ddb54de178a4094bff66bc971faf31734c8c6a07f030b54e21fb8dfe98d6`.

- Across the stable 1,373-checkpoint snapshot, all 198 unresolved model
  outcomes are covered by 155 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 140,038 pairs, 88,789 false positives, 51,249 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 174,483 pairs, 106,600 false positives, 67,883 true duplicates.

- The next audit frontiers are p0 `(7, 3,456)`, p1 `(39, 0)`,
  p2 `(71, 128)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:52:45Z — 173,971 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1848-v542` independently
  revalidated four p0 decision-file 7 checkpoints at semantic offsets 2,560
  through 2,944. Their 512 pairs contain 252 model false positives, 259 model
  true duplicates, and one unresolved outcome. Three pairs were chunked and
  509 were direct. All 1,217 judgments and request attempts were valid on
  their first attempt.

- The outcome Parquet SHA-256 values are
  `bb9645cce9c357a12e9f31a71f6f2bc72a59af1a025686aa524dd8d77ca65385`,
  `7164f0ab2326169b2c67df7812261c0ac9ca1fd9bcece550f478ac7a271c2ebd`,
  `4bcbad84cdfa3effdb2de90343ca067eb7bc05832e0e803ff4b098881c514827`,
  and
  `e8c73c56688139b09e039962dd65d4ae222bf8536ade5cdb2d725787c3c63daa`.

- Complete-text inspection resolves row 5,053 as a true duplicate under the
  established low-value-template boundary. Both 29-line casino SEO pages
  contain the same sentence-spun paragraphs about casino selection, games,
  and responsible gambling. Differences are synonym spins and malformed Thai
  keyword slots: the member inserts `คาสิโน` ("casino"), while the canonical
  inserts `แทงบอล` ("sports betting"). The model judgments incorrectly
  described both tokens as member-only; the complete texts show one in each
  document. Neither token adds a distinct fact or instruction in context. The
  4,307/4,333-character records have character, line, and word-sequence
  similarities 0.799769, 0.551724, and 0.706957. Member/canonical SHA-256
  values are
  `8f310f0b0e02f7d5e3bd34083e680cc29cf005b04d084d80ba80172f9ec04b9a`
  and
  `4a47f4ad86945f7a9ef16467514e35bd0cc9ab4b7a4106186d2191dcfd04f68d`.

- `/rav/datakit-6854-inspect-row5053-1849-v543` persisted the complete pair
  and diff with inspection SHA-256
  `e21c15f632f67e9904b8798d6b219f4c68e10db42ecd559f3d222fa37d4973b3`.
  `/rav/datakit-6854-publish-row5053-1851-v544` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row5053-1852-v545`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The
  semantic-judgment, manual-Parquet, and marker SHA-256 values are
  `de4af28e9bb0d0d96bb88ec2700fc29de0e30d60cdde638d58ccc2daaaf8460f`,
  `006015d544d19ca510a13851d2f7ac50242ce02b18c02602d6cd53a84b7c8758`,
  and
  `8d552dd7d64a29eae8fdcd4c588963f1781d57a4a2df7a2df904cf05de585219`.

- Across the stable 1,369-checkpoint snapshot, all 196 unresolved model
  outcomes are covered by 155 true-duplicate and 41 false-positive manual
  records. The adjusted totals are:

  - baseline: 139,526 pairs, 88,491 false positives, 51,035 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 173,971 pairs, 106,302 false positives, 67,669 true duplicates.

- The next audit frontiers are p0 `(7, 3,072)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:47:10Z — 173,459 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1840-v535` independently
  revalidated nine checkpoints: six p0 decision-file 7 checkpoints at semantic
  offsets 1,792 through 2,432, and the final three p3 decision-file 103
  checkpoints at offsets 5,504, 5,632, and 5,760. Their 1,129 pairs contain
  550 model false positives, 577 model true duplicates, and two unresolved
  outcomes. Three pairs were chunked and 1,126 were direct. The audit reread
  2,523 judgments and 2,529 request attempts: 2,520 attempts were valid, nine
  were invalid, and three judgments required retries.

- The outcome Parquet SHA-256 values are:

  - p0:
    `2957ce09e606c135c57056c2324f2dc82b1e68f66bc3efc4738666cc1fd10a55`,
    `b1429558788b6d6d00aeb17cf85ccda7b01cefeb7888b0413f911e0e0a0fcad1`,
    `6f3518fa5f5f59b5f1ec81380f3b22429147db72467a35f40bbc03c736a3a835`,
    `fc6f2dc0246426f3652bee2666b88cd9032b527329ebc5e3f2e45e4e74a0fe24`,
    `3a4ccf8348dfa408da368cd51436c616ec919d3ef1e4e7a7386756ac0f7710ff`,
    and
    `0eeedb9c5bf827d9335c7e1e745c13318377b241e02483ed125dd2691e3588f6`;
  - p3:
    `cde773f3140f8f37f3437b075b1224cfe0d51ec6309dc22e4fc6a4226042a44b`,
    `922b42922dbf2ab8468821d05019fbdb7e1adae9682a4d7b99bfb713cbf9a97e`,
    and
    `151992ee53ac3670cb59bbefb134995d212b3fea2df61cfcc91d5ab1bdb75cb6`.

- Complete-text inspection resolves p0 row 3,386 as a false positive. The
  5,312/4,837-character BetterHelp pages share a sentence-spun review
  scaffold, but the member alone cites what 20,000-plus people mention in
  reviews, explicitly advises contacting a local emergency hotline, and adds
  matching and time-zone guidance. These are substantive claims and advice,
  not merely the page's SEO title and brand slots. Character, line, and
  word-sequence similarities are 0.745886, 0.360656, and 0.696629.
  Member/canonical text SHA-256 values are
  `96794dcfa37532db6197c7d4d2d22d5f1f589306758b1c563f4b476dd48706b6`
  and
  `2fda9594f514b18ab24999d6a83a0c071100a57c5bb84e51064d5da0ab61ebae`.

- Complete-text inspection resolves treatment row 8,912 as a true duplicate.
  Both 306-line SFT records have the same comparative-negligence question,
  choices, reasoning, conclusion, and final choice H. Their sole changed line
  is the final boxed answer with or without a LaTeX text wrapper. The
  14,631/14,638-character records have character, line, and word-sequence
  similarities 0.999761, 0.996732, and 0.999793. Member/canonical text
  SHA-256 values are
  `5e9399fec624da65655707cca62d11481fc9ff27edd76e37f4284c74aba2fc0f`
  and
  `bf6c454d888d93e98d1bd7bf2a9b89d1a38829f71db1d0191d3a1b879936a7f8`.

- `/rav/datakit-6854-inspect-row3386-1842-v536` and
  `/rav/datakit-6854-inspect-row8912-1843-v537` persisted the complete pairs
  and diffs with inspection SHA-256 values
  `a2a6dd59f7d795a2a138a94bebecc422f900f4dd98827b99a07eba649f3cfa02`
  and
  `7b7dc8b12e5769392fa133fc0f64b9123ec9296a5d80114a3ed9c4937b13d40b`.
  Their semantic-judgment SHA-256 values are
  `edf97f35d7b19dc7a299bfa35440706d52c7e4a583c75507556d24423d8fed01`
  and
  `d1fd511fb37158deaa521a041ee4d111f8d245e20507ac179651c8e349314d5b`.

- `/rav/datakit-6854-publish-row3386-1844-v538` and
  `/rav/datakit-6854-publish-row8912-1845-v539` wrote the immutable manual
  records. Separate jobs `/rav/datakit-6854-verify-row3386-1847-v541` and
  `/rav/datakit-6854-verify-row8912-1846-v540` independently reread the source
  pairs, semantic checkpoints, inspections, deterministic Parquet bytes, and
  completion markers. Their manual-Parquet SHA-256 values are
  `24b1059c0b7c4a4972f24e5b4dcb0e7d07c2afa2058cd024275b7b3cc9bce72b`
  and
  `481057d7e4011d7e975a3f6752c8feaa147af28be625813c5afa9554bc9b900a`;
  their marker SHA-256 values are
  `be4e4249a6c6fc5b169cd1f71b46fe26936abaed5f41ddf284f136d39efb9b9f`
  and
  `5885cfd60ec9816e4bead1f7b1787a74f727976df5635ac162269cf03a5aa7a0`.

- Across the stable 1,365-checkpoint snapshot, all 195 unresolved model
  outcomes are covered by 154 true-duplicate and 41 false-positive manual
  records. The adjusted totals are:

  - baseline: 139,014 pairs, 88,239 false positives, 50,775 true duplicates;
  - treatment: 34,445 pairs, 17,811 false positives, 16,634 true duplicates;
  - combined: 173,459 pairs, 106,050 false positives, 67,409 true duplicates.

- The next audit frontiers are p0 `(7, 2,560)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(104, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:39:30Z — 172,330 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1838-v534` independently
  revalidated 12 checkpoints: six p0 decision-file 7 checkpoints at semantic
  offsets 1,024 through 1,664, and six p3 decision-file 103 checkpoints at
  offsets 4,736 through 5,376. Their 1,536 direct pairs contain 965 false
  positives, 571 true duplicates, and no unresolved outcomes. All 3,153
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are:

  - p0:
    `7563d970b7e4aabe96b1b037c3bedc3bebde9d979aa36381770ac624dfa615a8`,
    `e9a12b79dc046c2d88b26142051b1360db5ed513e9116d573ce9c32f57d29337`,
    `20683e332189e45e2a8457eb37ac99f391289687d7014d33e4e2957bdde471df`,
    `08053e8d75dc0a6f2a499876590ad2faf778aac1fe0009ad2f709ee6e2e34161`,
    `a36b8037cb10c497a92e87eec3e5f37bd138c6ae8d49a80e82b280d3ac1efe19`,
    and
    `aa1806afd552ce0900a4fc55239512f7f03c802144930b13ab84a16cca054876`;
  - p3:
    `e2f1eac0afa5a6e0fc351befe9b4dd3b3aa6a78a76a5e7d644f7f5e5479138b7`,
    `559fb66aaa7e0b6eefe6ee5d618babbeceb7f99b686987e5e8175b6a528ff5b3`,
    `fc606e61f51bc28fb22e7e90c2863118a479c049be971416e7b14c8376e32828`,
    `690c28571a2c09ac33efeb83bc47aceeaf4efa840b2fc1680925ef2eab480824`,
    `f81be181e201d0109daa43899d9ac8a96bd45382e29e44bfe98c5ead96678565`,
    and
    `a49f812a325b78feb2ead0ebcacb062c3180a57d020b40d64afa6a34e54e5b64`.

- Across the stable 1,356-checkpoint snapshot, all 193 unresolved model
  outcomes remain covered by 153 true-duplicate and 40 false-positive manual
  records. The adjusted totals are:

  - baseline: 138,246 pairs, 87,887 false positives, 50,359 true duplicates;
  - treatment: 34,084 pairs, 17,612 false positives, 16,472 true duplicates;
  - combined: 172,330 pairs, 105,499 false positives, 66,831 true duplicates.

- The next audit frontiers are p0 `(7, 1,792)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 5,504)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:36:30Z — 170,794 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1822-v518` independently
  revalidated four checkpoints: p0 decision-file 7 semantic offsets 768 and
  896, and p3 decision-file 103 semantic offsets 4,480 and 4,608. Their 512
  pairs contain 372 model false positives, 135 model true duplicates, and five
  unresolved outcomes. Four pairs were chunked and 508 were direct. The audit
  reread 1,157 judgments and 1,177 request attempts: 1,148 attempts were
  valid, 29 were invalid, and 10 judgments required retries.

- The outcome Parquet SHA-256 values are:

  - p0:
    `a3cd53c64914aa3a259fbc5f525a7abd1081f23aae41443bccb36be836d0e2af`
    and
    `190cc7fe5be52db2e58d74711e5ba83b21d631c9611e872d2c22a3fed169cc1c`;
  - p3:
    `f8572eb1453dc1fd307af7a3fb505fc26d6957486ba95eea55af2e05e2eeb758`
    and
    `b31954e49499397f9fc4f5f853e6f6d616c3b073252979c92a91d04003bd5ab2`.

- Complete-text inspection resolves all five ambiguities as true duplicates:

  - p0 row 1,532 contains the same MathOverflow binomial-transform question,
    definitions, positivity conjecture, supporting facts, derivative identity,
    heuristic, author, and date. Differences are LaTeX formatting, headings,
    and a non-substantive generated summary. The 1,510/1,718-character records
    have character, line, and word-sequence similarities 0.786865, 0.186047,
    and 0.838499. Member/canonical SHA-256 values are
    `2b35d68001c4085fc4af835f5fff74da680fbb4b2ee57e00091e35a0b649da22`
    and
    `2ac76eaee4972955498d05a900679580bcaf15d12e2f16b2823a8a3a11ed038e`.
  - p3 rows 7,532, 7,538, 7,542, and 7,545 are otherwise identical SFT
    records whose sole changed line is the final boxed answer with or without
    a LaTeX text wrapper. They cover emotional intelligence, WTO renewable
    energy subsidies, life expectancy, and team cohesion, respectively. Their
    character similarities are 0.999161, 0.999838, 0.999757, and 0.999334;
    their member/canonical character counts are 4,176/4,169,
    21,583/21,576, 14,406/14,399, and 5,260/5,253.
  - In row order, the four SFT member/canonical SHA-256 pairs are
    `7bebc87158a22973a7bb65f8ca321d46589d7ba44c80dac446cd03d1e7ec9c16` /
    `bdcc42ec9168b8f36c2e176ed755e351d224cfeb20346d4e4ad2a8e8e928a110`,
    `cc8de6e10afb62dd7879b4921e7af91693d61b670e3b473fcded5f4ccc73d759` /
    `ee1f5094dbba22e2d4782746c946c24c4b92fef014340a5dd0fcfdea3c69ec06`,
    `59d3f629855e29e17b08a8499132720a9f1b08214fc005d99484927573907a64` /
    `c77ed3b7cbbd34b6f899132a848bc9241fe29bdf7b35fbeee131c568c83169fb`,
    and
    `4851f487eb74ad62d5dffb14a9dfd81384bba0334e4893c67a76246fa4ca1e29` /
    `1b5a8d02437b4637a9a87e968c4861cd7cb42732956be799451bbe1fd5dbee4c`.

- The inspection jobs were
  `/rav/datakit-6854-inspect-row1532-1823-v519`,
  `/rav/datakit-6854-inspect-row7532-1824-v520`,
  `/rav/datakit-6854-inspect-row7538-1825-v521`,
  `/rav/datakit-6854-inspect-row7542-1826-v522`, and
  `/rav/datakit-6854-inspect-row7545-1827-v523`. Their inspection SHA-256
  values in the same order are
  `28bfc624430e62274a3c4273b598fdfd6d844a7816f85b73731a76c552073ddc`,
  `d5d322f246e7c1c2d8dc7bb6010da1f5fa8e961324c3e582875b4df29b59b6b3`,
  `bab3c6f3847f96be39b52b860f0933f66af02e7c7c1db03f59978a6c850b09c5`,
  `3d0dea8028d0d66005347737782e483ac95ff3e75270199ec4d8ed0e418ea623`,
  and
  `a4006978d4f0c8149908fbd482b7ca61a3f44e340bb5eb810d164fe56f62919a`.
  The corresponding semantic-judgment SHA-256 values are
  `595d7cd14bd58141b2875928430063d547af10252a17d2dd9db93990a0879c10`,
  `6e6012a495bdbb20965fa7da44349436db298898614305432a0649a07b07a47f`,
  `a7d1f6eb8e96b5891c8a1d2440c1b7d95a049dbfdde3a028d9b77e136cfa3912`,
  `81d2d23ea0b197a7979ac7288879539dcde9f116f801c857f418bec95399f223`,
  and
  `53804c1b471205dcedee728a6304aa1554dca3058e8c8b4cf7b172dd8d17e571`.

- The publication jobs were
  `/rav/datakit-6854-publish-row1532-1828-v524`,
  `/rav/datakit-6854-publish-row7532-1829-v525`,
  `/rav/datakit-6854-publish-row7538-1830-v526`,
  `/rav/datakit-6854-publish-row7542-1831-v527`, and
  `/rav/datakit-6854-publish-row7545-1832-v528`. Separate verify-only jobs
  `/rav/datakit-6854-verify-row1532-1837-v533`,
  `/rav/datakit-6854-verify-row7532-1836-v532`,
  `/rav/datakit-6854-verify-row7538-1835-v531`,
  `/rav/datakit-6854-verify-row7542-1834-v530`, and
  `/rav/datakit-6854-verify-row7545-1833-v529` independently reread the
  source pairs, semantic checkpoints, inspections, deterministic Parquet
  bytes, and completion markers.

- In row order, the manual-Parquet SHA-256 values are
  `f332db501a4853ca46f70345e92dfd9a1ad0f706409676d3865a1c335201643e`,
  `409708b71fe09532650892188c1f3b76320f53e6a716f4fed98ff453ef3dc372`,
  `fb727d115c7397ba8ff1b141cbf7c7b52dcc6eaaafb57d30e7d6f2e1a53ae0ba`,
  `5cc65dfb3e87bfdebaee62069384169b491c615981c1781d57851b9070132165`,
  and
  `605b3eb2312395fa37eae0133de8adc336c8df51401104a71b66c63598c2e887`.
  Their marker SHA-256 values are
  `b611dff5cbde9d54a5d35e623d6d6bae3e21a80e6947f755396482170a714212`,
  `eb1b63df6dc7c14a88e3baaa894c59e4d787b6b4e35eb5b300482c18cf6d209b`,
  `79a0b4da949e57ad25779d9a21a14fd3212c85e47205fcfe90d3a20f0d6e5ad2`,
  `25b209a5b354ec1e940ded51292861727fac7bf23fffdfd567002f83579a792c`,
  and
  `5fba4905723ae09da04121f17c03079a44091381bef2a64feee0ca40ad52b897`.

- Across the stable 1,344-checkpoint snapshot, all 193 unresolved model
  outcomes are covered by 153 true-duplicate and 40 false-positive manual
  records. The adjusted totals are:

  - baseline: 137,478 pairs, 87,300 false positives, 50,178 true duplicates;
  - treatment: 33,316 pairs, 17,234 false positives, 16,082 true duplicates;
  - combined: 170,794 pairs, 104,534 false positives, 66,260 true duplicates.

- The next audit frontiers are p0 `(7, 1,024)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 4,736)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:20:55Z — 170,282 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1816-v514` independently
  revalidated five baseline checkpoints: p0 decision-file 7 semantic offsets
  384, 512, and 640, and p3 decision-file 103 semantic offsets 4,224 and
  4,352. Their 640 direct pairs contain 412 model false positives, 227 model
  true duplicates, and one unresolved outcome. The audit reread 1,319
  judgments and their 1,321 request attempts: 1,318 attempts were valid, three
  were invalid, and one judgment required retries.

- The outcome Parquet SHA-256 values are:

  - p0:
    `a43667efbf5b64d1d278301d1fa733e15e9a6720081244da39d7bd5694f5f745`,
    `dcd43256e3aa3d5755ccbcb4ab05291eb8baf252e637d3fec49c0b87360d8b0a`,
    and
    `f2d3773ee4014a5f52b29a2f20025a7597d124c0aab0bc6ef5d237c8fd37d86c`;
  - p3:
    `25337c8c04c047e096f230b4c31b953763dac52cb5130951c30d722a3d087d70`
    and
    `5edd0b41284a89dafed7505601b809f81783ef89613001ffb1b471a4a350759e`.

- Complete-text inspection resolves the p3 ambiguity as a false positive.
  `part-00103-of-00128.parquet:7317` compares SFT records that share only a
  generic answer-format instruction. The member asks for the center of a
  circle and derives `(3, -1)`; the canonical asks for `54 × 46` and derives
  `2,484`. Their user requests, reasoning, intermediate values, and answers
  are distinct. The 501/485-character records have character, line, and
  word-sequence similarities 0.701826, 0.421053, and 0.627219.
  Member/canonical text SHA-256 values are
  `5937574f173529b6c5d3f8dc81db3a47a0a479f963302f798b4ad0c86e24c2bb`
  and
  `987b60cb2ac229ddf85800ae1271c587ef6e72ae2d9c9078fd832a86c4c23c24`.

- `/rav/datakit-6854-inspect-row7317-1817-v515` persisted the complete pair
  and diff with inspection SHA-256
  `b7a6befdd1b11f5065c4a394ffb24de805321e39ee8571f4e9d876f2cc6eb309`.
  `/rav/datakit-6854-publish-row7317-1818-v516` wrote the immutable
  false-positive record, and `/rav/datakit-6854-verify-row7317-1820-v517`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `275abd4b61906af51ffab50efbf444421c5d083579cf24cf3f5036b221342f09`,
  `7bb3f314dda4255da6df1fe7ee5edf2c7b8d1156c1b31c1a33a6b74db0ffd630`,
  and `169adafdcb2298390ea678bc4848f89fcb435bf6524f143c812195e3b8fa2294`.

- Across the stable 1,340-checkpoint snapshot, all 188 unresolved model
  outcomes are covered by 148 true-duplicate and 40 false-positive manual
  records. The adjusted totals are:

  - baseline: 137,017 pairs, 86,958 false positives, 50,059 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 170,282 pairs, 104,162 false positives, 66,120 true duplicates.

- The next audit frontiers are p0 `(7, 768)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 4,480)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:15:50Z — 169,642 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1811-v510` independently
  revalidated p0 decision-file 7 semantic offset 256 and p3 decision-file 103
  semantic offset 4,096. Their 256 baseline pairs contain 175 model false
  positives, 80 model true duplicates, and one unresolved outcome. Twelve
  pairs were chunked and 244 were direct. All 1,883 judgments were valid on
  their first request attempt. The p0 and p3 outcome Parquet SHA-256 values
  are
  `73f89e2b858c0664d4e0fd89bad407814f4321c755af5398b4b11a8c911f35e4`
  and
  `bcf86583271f3e6baf67312114ef5bf79c3d6435c504266a24410f79781359b7`.

- Complete-text inspection resolves the p0 ambiguity as a true duplicate.
  `part-00007-of-00128.parquet:773` compares sentence-spun copies of the same
  singing-myth article. Both carry the same argument, Michael Jordan and
  American Idol anecdotes, tone-deafness explanation, and first-person
  vocal-coach story. Differences such as soccer/netball, Babe Ruth/Pele,
  Peter/Perry, and synonymous wording are low-value slot substitutions; the
  member's title only restates the article. The 2,502/2,589-character records
  have character, line, and word-sequence similarities 0.859949, 0.080000,
  and 0.840164. Member/canonical text SHA-256 values are
  `060609ebba0565ab71b6a612cbeb133576379c5b3825a8e582fda4b8e13bf178`
  and
  `e3b4f65990716b72c48fa37cc3d5b9cade00f6d77ec251039c51b4388a1ca91f`.

- `/rav/datakit-6854-inspect-row773-1813-v511` persisted the complete pair and
  diff with inspection SHA-256
  `e09315bd87d3e98d4aa9a9ca90b0f1680d05c95973bfe1a5cb068132cfcde494`.
  `/rav/datakit-6854-publish-row773-1814-v512` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row773-1815-v513`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `6988f1159e3dd3fe7b45f7da90fbef2cf48a9356b7f249feda73c583a8c0d24e`,
  `f95ceec8a97ce1a7b5f7b259fa974729eb1f4e04e4a579b4572e1873b32b4894`,
  and `3536eda8f4e36be6c99bbcccf6357a28bbf3966c2fadf52f57a1628a0e05c834`.

- Across the stable 1,335-checkpoint snapshot, all 187 unresolved model
  outcomes are covered by 148 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 136,377 pairs, 86,545 false positives, 49,832 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 169,642 pairs, 103,749 false positives, 65,893 true duplicates.

- The next audit frontiers are p0 `(7, 384)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 4,224)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:11:00Z — 169,386 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1810-v509` independently
  revalidated p3 decision-file 103 semantic offset 3,968. Its 128 direct
  baseline pairs contain 98 false positives, 30 true duplicates, and no
  unresolved outcomes. All 264 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `967bc619675a981d6cd1d222c02df8d6b3d16abfcc7974c7a3c7ac3c701e3bcf`.

- Across the stable 1,333-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 136,121 pairs, 86,370 false positives, 49,751 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 169,386 pairs, 103,574 false positives, 65,812 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 4,096)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:09:20Z — 169,258 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1808-v508` independently
  revalidated p3 decision-file 103 semantic offsets 3,712 and 3,840. Their 256
  baseline pairs contain 185 false positives, 71 true duplicates, and no
  unresolved outcomes. One pair was chunked and 255 were direct. All 604
  judgments were valid on their first request attempt. The outcome Parquet
  SHA-256 values are
  `58ab505bdffb935a75430ce55093cd8efb2e50e4535dd09dddb8d51fc9888f38`
  and
  `d7cdc115207b0b8a6c03e264efa6559613d85ae58360d523fd3c711713951969`.

- Across the stable 1,332-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,993 pairs, 86,272 false positives, 49,721 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 169,258 pairs, 103,476 false positives, 65,782 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,968)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:06:15Z — 169,002 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1805-v506` independently
  revalidated p3 decision-file 103 semantic offset 3,584. Its 128 direct
  baseline pairs contain 98 false positives, 30 true duplicates, and no
  unresolved outcomes. All 262 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `77f2aa77789c69500db87c6eb1274301f362178d7cd5f43ecbb4bd2e3caf435f`.

- Across the stable 1,330-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,737 pairs, 86,087 false positives, 49,650 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 169,002 pairs, 103,291 false positives, 65,711 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,712)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:04:45Z — 168,874 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1804-v505` independently
  revalidated p3 decision-file 103 semantic offset 3,456. Its 128 direct
  baseline pairs contain 79 false positives, 49 true duplicates, and no
  unresolved outcomes. All 270 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `7010ad3d8509a21c8660bc9f7c29014d049d1bc1687edc3f77ed63ee272bca89`.

- Across the stable 1,329-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,609 pairs, 85,989 false positives, 49,620 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,874 pairs, 103,193 false positives, 65,681 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,584)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:03:20Z — 168,746 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1803-v504` independently
  revalidated p3 decision-file 103 semantic offset 3,328. Its 128 direct
  baseline pairs contain 55 false positives, 73 true duplicates, and no
  unresolved outcomes. All 270 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `eef9475b07066a27445cbfe6b8ef03fb49511b00b97ed7eab4fab74136ef7396`.

- Across the stable 1,328-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,481 pairs, 85,910 false positives, 49,571 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,746 pairs, 103,114 false positives, 65,632 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,456)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:01:45Z — 168,618 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1801-v503` independently
  revalidated p3 decision-file 103 semantic offset 3,200. Its 128 direct
  baseline pairs contain 69 false positives, 59 true duplicates, and no
  unresolved outcomes. All 286 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `e6795e41872d706c0465d4145e2186f711def30bd48d56636f8c684e08709151`.

- Across the stable 1,327-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,353 pairs, 85,855 false positives, 49,498 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,618 pairs, 103,059 false positives, 65,559 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,328)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T18:00:00Z — 168,490 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1800-v502` independently
  revalidated p3 decision-file 103 semantic offset 3,072. Its 128 baseline
  pairs contain 55 false positives, 73 true duplicates, and no unresolved
  outcomes. One pair was chunked and 127 were direct. All 335 judgments were
  valid on their first request attempt. The outcome Parquet SHA-256 is
  `2af8b69dc81d3a924a7f4df0186e58afa305717593c8d953d74fed9221a2a967`.

- Across the stable 1,326-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,225 pairs, 85,786 false positives, 49,439 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,490 pairs, 102,990 false positives, 65,500 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,200)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:58:25Z — 168,362 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1758-v501` independently
  revalidated p3 decision-file 103 semantic offset 2,944. Its 128 direct
  baseline pairs contain 57 false positives, 71 true duplicates, and no
  unresolved outcomes. All 276 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `1129bdf982e92cb2956a3ea08bb9bad3c63dc3e07e7d2b81bcb91746787da1b7`.

- Across the stable 1,325-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 135,097 pairs, 85,731 false positives, 49,366 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,362 pairs, 102,935 false positives, 65,427 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 3,072)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:56:55Z — 168,234 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1756-v500` independently
  revalidated p0 decision-file 7 semantic offset 128 and p3 decision-file 103
  semantic offset 2,816. Their 256 baseline pairs contain 143 false positives,
  113 true duplicates, and no unresolved outcomes. Fifteen pairs were chunked
  and 241 were direct. All 1,857 judgments were valid on their first request
  attempt.

- The p0 and p3 outcome Parquet SHA-256 values are
  `aaf3c60e14beff6d121ce77433ed0978f3bfa720d659c0f62205dafc8cc806bd`
  and
  `349d331f62f052e9c9ae2e6ad0fd3eb938b4a7900e78709a3d22b8063fdd97d0`.

- Across the stable 1,324-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 134,969 pairs, 85,674 false positives, 49,295 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 168,234 pairs, 102,878 false positives, 65,356 true duplicates.

- The next audit frontiers are p0 `(7, 256)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 2,944)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:55:10Z — 167,978 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1755-v499` independently
  revalidated p3 decision-file 103 semantic offset 2,688. Its 128 baseline
  pairs contain 48 false positives, 80 true duplicates, and no unresolved
  outcomes. One pair was chunked and 127 were direct. All 321 judgments were
  valid on their first request attempt. The outcome Parquet SHA-256 is
  `442defb510f536cc28d3bcebdead1b61acaac1751017e6bf07233a747730d22c`.

- Across the stable 1,322-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 134,713 pairs, 85,531 false positives, 49,182 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 167,978 pairs, 102,735 false positives, 65,243 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 2,816)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:53:40Z — 167,850 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1757-v498` independently
  revalidated six p3 decision-file 103 checkpoints spanning semantic offsets
  1,920 through 2,560. Their 768 baseline pairs contain 365 false positives,
  403 true duplicates, and no unresolved outcomes. One pair was chunked and
  767 were direct. All 1,696 judgments were valid on their first request
  attempt.

- In checkpoint order, the outcome Parquet SHA-256 values are
  `5666a2feffe820e71acb5eba60bdc5796b1a2b1edf4d8749cf0d610241659f28`,
  `7607eb1c8b87e53f5eeb476566acac79f95426987559def45f07f0a8ea79efcb`,
  `7536c5400915bed4fe6a49702eb16c9ee6d2327f2341e9dd050b122a49cd0b38`,
  `3be034499362a6d40c5e5a87c2ee78034b9a8eaa232f6f20197e36ee04c30989`,
  `9ebfc4ddc70421fed651098c728746511a5a7f9fbad319106d23abdd5d5b560f`,
  and
  `6d8d7698f0d465d5e974ae5abd2808cae5359ccc0bb1f2220c5a9a5cef15584c`.

- Across the stable 1,321-checkpoint snapshot, all 186 unresolved model
  outcomes remain covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 134,585 pairs, 85,483 false positives, 49,102 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 167,850 pairs, 102,687 false positives, 65,163 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 2,688)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:51:45Z — 167,082 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1743-v485` independently
  revalidated three checkpoints: p1 decision-file 38 semantic offsets 5,760
  and 5,888, and p3 decision-file 103 semantic offset 1,792. Their 326 pairs
  contain 194 model false positives, 128 model true duplicates, and four
  unresolved outcomes. All pairs were direct. The audit reread 669 judgments
  and their 689 request attempts: 659 attempts were valid, 30 were invalid,
  and 10 judgments required retries.

- The outcome Parquet SHA-256 values are:

  - p1:
    `b35c27443a5919f86470978bc27b5b77c2f0839be3cc30f098c85e45174b955e`
    and
    `61d0fc6858ba90bdff6f1c0d664e9ab742af50d2262a070cff8e3385c95e5628`;
  - p3:
    `94032028e8de234f95145c61d2102b73603406ba6b9bf234652d7cefa16fc556`.

- Complete-text inspection resolves all four treatment ambiguities as true
  duplicates:

  - row 9,143 compares 218-line SFT records with identical Australian
    Indigenous-rights questions, choices, reasoning, conclusions, and
    answers. Its sole changed line is `\boxed{D}` versus
    `\boxed{\text{D}}`. The 12,036/12,043-character records have character,
    line, and word-sequence similarities 0.999709, 0.995413, and 0.999747.
    Member/canonical text SHA-256 values are
    `8e3eb3a4f159f5a675dcd6b0dd3181f4df83acc9ca2fd40ffc50904f7d893f32`
    and
    `4287b738415078f66f232fbbd9f41ad64c93d07bfd1852aef860dc24f7750632`.
  - row 9,169 compares 56-line SFT records with identical
    emotional-intelligence questions, choices, reasoning, conclusions, and
    answers. Its sole changed line is `\boxed{\text{C}}` versus
    `\boxed{C}`. The 4,176/4,169-character records have character, line, and
    word-sequence similarities 0.999161, 0.982143, and 0.999258.
    Member/canonical text SHA-256 values are
    `7bebc87158a22973a7bb65f8ca321d46589d7ba44c80dac446cd03d1e7ec9c16`
    and
    `bdcc42ec9168b8f36c2e176ed755e351d224cfeb20346d4e4ad2a8e8e928a110`.
  - row 9,170 compares 167-line SFT records with identical memory-assessment
    questions, choices, reasoning, conclusions, and answers. Its sole changed
    line is `\boxed{\text{J}}` versus `\boxed{J}`. The
    8,811/8,804-character records have character, line, and word-sequence
    similarities 0.999603, 0.994012, and 0.999632. Member/canonical text
    SHA-256 values are
    `b6eb09948a59300d79794e6ae0d91dcaa5085505094397948670a7d1fc3b3572`
    and
    `dd02384ba621cb1aa80713d1711a762fddc0f2fe000b3ccdf3d526dbe711e2eb`.
  - row 9,174 compares 73-line SFT records with identical
    electron-microscopy questions, choices, reasoning, conclusions, and
    answers. Its sole changed line is `\boxed{\text{H}}` versus
    `\boxed{H}`. The 5,789/5,782-character records have character, line, and
    word-sequence similarities 0.999395, 0.986301, and 0.999456.
    Member/canonical text SHA-256 values are
    `add527135c9a99634be976006bd5e80b9e6498756f26be95c2a76178cf14f411`
    and
    `9c179831aa37ec208f97d6c8cdd1722acac9ee50c4dca878d9c4c51042c31936`.

- The inspection, semantic-evidence, manual-Parquet, and marker SHA-256
  values are:

  - row 9,143:
    `a9a9ab5e593c5e9556e59ef12c689540deec380f0d139fbc14f27f3d56d99e6b`,
    `0a15e2e5df103c3ffc649bd5451675f10e32b3078ceb201495be1a894503d5a2`,
    `0a38326b1b84153cc787083dcabc071dcacbc3e4321c1bf59c32cecc2ffec918`,
    and
    `cbc3dabc77da93b2ed9ff8e754ef725c7c48ef0f4e1ad0a1e372ee73beee6906`;
  - row 9,169:
    `c34b71e5b3d7e7549d1afa648f13f7d3f985e28a5414c014c02e2c309aaca8e5`,
    `06dcdd51a325c83ba4fe9e1d2b72ff98d8ef315fee7fc6341dbf175ac473d029`,
    `12693b7b424c1a6c9ea0ab705502d090a2d3042d9ec2468fe2930cf5179f9529`,
    and
    `15a1bfa65ceac53da9a9e073bea00e6767842ffc13381a451893107ac927ac73`;
  - row 9,170:
    `719811a0f938827f758a18560deb48c37b1cf25930a01774092e157deb08424f`,
    `8f34da77dcb31df786fec5452e4a98cfd4da3d16500aa147687fb2add25ffa88`,
    `484998a0ae504a0121a8544d9c88cfd0c710bdd26bafef9b44d2cabe57043c84`,
    and
    `771144714767e2abafb91d8fafb1afe33b57d8a2f4a38d4f539d39b5ae1875f6`;
  - row 9,174:
    `1ecbb7a77a8a1686db43b6b8a1a9f5c6d3bf853556ff02734251140ef4159861`,
    `37e183771e98367792d74a3af356b4c944d10effe6c262ef6469406ad2b85855`,
    `4238ec1490a442bbff9526d5787d42357da6ca58fd196d8250beb163aa8d0aa8`,
    and
    `c2aacf02942d057c775373feda99ad9510a5d727b4c2c30137d83e04979534e4`.

- Inspection jobs `/rav/datakit-6854-inspect-row9143-1744-v486`,
  `/rav/datakit-6854-inspect-row9169-1745-v487`,
  `/rav/datakit-6854-inspect-row9170-1746-v488`, and
  `/rav/datakit-6854-inspect-row9174-1747-v489` persisted the complete
  documents and diffs. Publish jobs
  `/rav/datakit-6854-publish-row9174-1748-v490`,
  `/rav/datakit-6854-publish-row9170-1749-v491`,
  `/rav/datakit-6854-publish-row9169-1750-v492`, and
  `/rav/datakit-6854-publish-row9143-1751-v493` wrote the immutable manual
  records. Independent verification jobs
  `/rav/datakit-6854-verify-row9143-1752-v494`,
  `/rav/datakit-6854-verify-row9169-1753-v495`,
  `/rav/datakit-6854-verify-row9170-1754-v496`, and
  `/rav/datakit-6854-verify-row9174-1755-v497` reread the source pairs,
  semantic checkpoint, inspections, deterministic Parquet bytes, and
  completion markers.

- Across the stable 1,315-checkpoint snapshot, all 186 unresolved model
  outcomes are covered by 147 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,817 pairs, 85,118 false positives, 48,699 true duplicates;
  - treatment: 33,265 pairs, 17,204 false positives, 16,061 true duplicates;
  - combined: 167,082 pairs, 102,322 false positives, 64,760 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(39, 0)`,
  p2 `(71, 0)`, and p3 `(103, 1,920)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:41:30Z — 166,756 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1741-v484` independently
  revalidated p1 decision-file 38 semantic offset 5,632 and p3 decision-file
  103 semantic offset 1,664. Their 256 pairs contain 106 false positives, 150
  true duplicates, and no unresolved outcomes. One pair was chunked and 255
  were direct. All 585 judgments were valid on their first request attempt.

- The p1 and p3 outcome Parquet SHA-256 values are
  `669be5e2c5a222bfefd99618df3d0695aac1609e9dbb21d3e6c8eed517ef580d`
  and
  `72975f209e5bc6a865d5fa07a5f0271cac99bfab1061e1d1055fdb54807f9a1e`.

- Across the stable 1,312-checkpoint snapshot, all 182 unresolved model
  outcomes remain covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,689 pairs, 85,059 false positives, 48,630 true duplicates;
  - treatment: 33,067 pairs, 17,069 false positives, 15,998 true duplicates;
  - combined: 166,756 pairs, 102,128 false positives, 64,628 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 5,760)`,
  p2 `(71, 0)`, and p3 `(103, 1,792)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:39:55Z — 166,500 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1739-v483` independently
  revalidated p1 decision-file 38 semantic offset 5,504 and p3 decision-file
  103 semantic offset 1,536. Their 256 pairs contain 101 false positives, 155
  true duplicates, and no unresolved outcomes. Two pairs were chunked and 254
  were direct. All 623 judgments were valid on their first request attempt.

- The p1 and p3 outcome Parquet SHA-256 values are
  `a72e5939cc594c8c96cdd8e2abda7754fe5e2eae1feb8127e4903d2712667711`
  and
  `8d22839ae99f6a5ae265b6c890f1cc5fd5b46fdd40b1a0bec0543b87e12ffaac`.

- Across the stable 1,310-checkpoint snapshot, all 182 unresolved model
  outcomes remain covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,561 pairs, 85,001 false positives, 48,560 true duplicates;
  - treatment: 32,939 pairs, 17,021 false positives, 15,918 true duplicates;
  - combined: 166,500 pairs, 102,022 false positives, 64,478 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 5,632)`,
  p2 `(71, 0)`, and p3 `(103, 1,664)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:38:05Z — 166,244 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1737-v482` independently
  revalidated two direct checkpoints: p1 decision-file 38 semantic offset
  5,376 and p3 decision-file 103 semantic offset 1,408. Their 256 pairs
  contain 147 false positives, 109 true duplicates, and no unresolved
  outcomes. All 527 judgments were valid on their first request attempt.

- The p1 and p3 outcome Parquet SHA-256 values are
  `ca3f26db70341299f189dd96adb4a0dc4a1a3e1a66c155cfe6f011b08fa64aa8`
  and
  `8900d5d51e65fade42d1a149052938b1e550416312e865c304db8cf513dcfbba`.

- Across the stable 1,308-checkpoint snapshot, all 182 unresolved model
  outcomes remain covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,433 pairs, 84,947 false positives, 48,486 true duplicates;
  - treatment: 32,811 pairs, 16,974 false positives, 15,837 true duplicates;
  - combined: 166,244 pairs, 101,921 false positives, 64,323 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 5,504)`,
  p2 `(71, 0)`, and p3 `(103, 1,536)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:36:25Z — 165,988 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1735-v481` independently
  revalidated three direct checkpoints: p1 decision-file 38 semantic offsets
  5,120 and 5,248, and p3 decision-file 103 semantic offset 1,280. Their 384
  pairs contain 224 false positives, 160 true duplicates, and no unresolved
  outcomes. All 786 judgments were valid on their first request attempt.

- The outcome Parquet SHA-256 values are:

  - p1:
    `411123d9e2125c526b89beede5517a101a00185cf4538acd84aafa282b5b33fd`
    and
    `4789b0efb7033c4d8133776e0d67b2ea70801d9bb5715244364bdadabd6b8f02`;
  - p3:
    `ef7e04f2b049faba5342ffd0c955f326c9f16adecffe6a362ed8b55f4b22c520`.

- Across the stable 1,306-checkpoint snapshot, all 182 unresolved model
  outcomes remain covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,305 pairs, 84,831 false positives, 48,474 true duplicates;
  - treatment: 32,683 pairs, 16,943 false positives, 15,740 true duplicates;
  - combined: 165,988 pairs, 101,774 false positives, 64,214 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 5,376)`,
  p2 `(71, 0)`, and p3 `(103, 1,408)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:34:20Z — 165,604 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1732-v480` independently
  revalidated eight checkpoints: p1 decision-file 38 semantic offsets 4,736,
  4,864, and 4,992, and p3 decision-file 103 semantic offsets 640 through
  1,152. Their 1,024 pairs contain 759 false positives, 265 true duplicates,
  and no unresolved outcomes. Six pairs were chunked and 1,018 were direct.
  All 2,343 judgments were valid on their first request attempt.

- The outcome Parquet SHA-256 values are:

  - p1:
    `c32bc4df146c060bf4a143036ed2ccc3e291ec107ba130ebfb3e76435cf333ad`,
    `84bd6775b5ed735c7f4deb17111b08e6c7708bd0e8e9525f060e22980d661dbe`,
    and
    `b67b3370e3b83a54084a9151352ba85606482b4d9c47013c3279b6174d2bad8b`;
  - p3:
    `667252c724175fcb5893bd5d86127cbd562ee27fa850f8b6082f6fe07a7a241f`,
    `0dca6f60088fd1311f105bd5b28f3d596e1b04020b4dc5f00d2e45fe78e2f2fb`,
    `136edc3b6d93ce5b622d29d9a04978b6fb79e938b938348dc269545409c9247f`,
    `36423ffb9e9c1f4e8a86b7c65b8fbf3d9ab612a7df6984fd1ec33a951f9f4b6a`,
    and
    `f7182d84d6752bfbaefce4e20072e8b41991bc5065d4947725cf21e4361987b2`.

- Across the stable 1,303-checkpoint snapshot, all 182 unresolved model
  outcomes remain covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 133,177 pairs, 84,713 false positives, 48,464 true duplicates;
  - treatment: 32,427 pairs, 16,837 false positives, 15,590 true duplicates;
  - combined: 165,604 pairs, 101,550 false positives, 64,054 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 5,120)`,
  p2 `(71, 0)`, and p3 `(103, 1,280)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:32:00Z — 164,580 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1728-v476` independently
  revalidated five baseline checkpoints: p1 decision-file 38 semantic offsets
  4,352, 4,480, and 4,608, and p3 decision-file 103 semantic offsets 384 and
  512. Their 640 pairs contain 440 model false positives, 199 model true
  duplicates, and one unresolved outcome. Two pairs were chunked and 638 were
  direct. The audit reread all 1,632 judgments and their 1,637 request
  attempts: 1,631 attempts were valid, six were invalid, and three judgments
  required retries.

- The outcome Parquet SHA-256 values are:

  - p1:
    `fb2f8f0484e20211efccf5350f00ebde622c866b1448d8b29b04e85bcec20767`,
    `059bf072daea1b90b6a1fc2e18a478980c11ef681ba5f9454dbcdd10b555d27c`,
    and
    `fce351a252c3040eeef907d1d4dee520be4d18da025ffb63224b83d78a383590`;
  - p3:
    `7aee4a95dabd191921f481d0dbfa6b9220837fd573ab1b6b62ada8ccac6b91fa`
    and
    `cac35eedd64fd12debdf137da4c969e12803cf95bce604322e56b24e57510476`.

- Complete-text inspection resolves the baseline ambiguity as a true
  duplicate. `part-00038-of-00128.parquet:7500` compares 332-line SFT records
  with identical historical-figures questions, choices, reasoning about every
  candidate, conclusions, and answers. The sole changed line is
  `\boxed{C}` versus `\boxed{\text{C}}`. The 16,656/16,663-character records
  have character, line, and word-sequence similarities 0.999790, 0.996988,
  and 0.999815. Member/canonical text SHA-256 values are
  `ef7b40f621668f57b795c2994aad3c1e20cdac2f6fd4e31a53e6a3121b8b8d3a`
  and
  `3a4659343099e47f6f40772bd924808fd8bd9f61844d2a491c56dd0c7e7cd671`.

- `/rav/datakit-6854-inspect-row7500-1729-v477` persisted the complete pair
  and diff with inspection SHA-256
  `85aa1a88be725e81ca09bf4a367595d293fcb295feba537c9cc2ed77b545e9f3`.
  `/rav/datakit-6854-publish-row7500-1730-v478` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row7500-1731-v479`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `9002b32223980dc8a799505be4d158505c53ad5cbe3c730991f1c873dc0321a6`,
  `5da5d7a4be1c5fa3fdb9eb18dfc622451da1cfeb3e1aa7c4613789be02eab8cc`,
  and `1abd25112e460ae9c61e3dc17dc57c85ed4ea9a309b7c385c3356f24bc941bc0`.

- Across the stable 1,295-checkpoint snapshot, all 182 unresolved model
  outcomes are covered by 143 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 132,530 pairs, 84,225 false positives, 48,305 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 164,580 pairs, 100,791 false positives, 63,789 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 4,736)`,
  p2 `(71, 0)`, and p3 `(103, 640)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:24:35Z — 163,940 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1719-v469` independently
  revalidated p1 decision-file 38 semantic offsets 4,096 and 4,224 and p3
  decision-file 103 semantic offset 256. Their 384 baseline pairs contain 279
  model false positives, 103 model true duplicates, and two unresolved
  outcomes. Eleven pairs were chunked and 373 were direct. The audit reread
  all 2,003 judgments and their 2,011 request attempts: 1,999 attempts were
  valid, 12 were invalid, and four judgments required retries.

- The outcome Parquet SHA-256 values are:

  - p1:
    `19c76b69ce68199557cf5be5f401fa920a10bc4f74ae0e5b3e85e1749073fefa`
    and
    `bd4da0e978ea0d9a020df9c5745688c167dfc20359ec37fcbc4f4887e5413ff7`;
  - p3:
    `c88f1771f594c6977b489f6a299c8f0de67863119e5684bbbd5fe197215cee3f`.

- Complete-text inspection resolves both baseline ambiguities as true
  duplicates:

  - `part-00038-of-00128.parquet:7413` compares 333-line SFT records with
    identical positive-reinforcement questions, choices, reasoning,
    behavioral-psychology facts, conclusions, and answers. The sole changed
    line is `\boxed{E}` versus `\boxed{\text{E}}`. The
    17,545/17,552-character records have character, line, and word-sequence
    similarities 0.999801, 0.996997, and 0.999823. Member/canonical text
    SHA-256 values are
    `3b58e4c8f919fa3d2845ba34441621fa2965093b95b4925582c9d40a38de6b28`
    and
    `f889337ce22248d60330d62534a2edfc0cef8b432027052ee6f4bbfa63cdd718`.
  - `part-00038-of-00128.parquet:7447` compares 389-line SFT records with
    identical vaccination questions, choices, reasoning, statistics,
    conclusions, and answers. The sole changed line is `\boxed{J}` versus
    `\boxed{\text{J}}`. The 20,343/20,350-character records have character,
    line, and word-sequence similarities 0.999828, 0.997429, and 0.999865.
    Member/canonical text SHA-256 values are
    `5dea2d66f28687ed1109fcf014bc8c48a9a87d72496739cbb0f3b68a97ebcd31`
    and
    `b73ae5af67cbedc077fae9beb8bded0459d71167a43a893926ba0d135aed92ac`.

- The row-7413 inspection, semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `ab3b78ae00d632f57b85a67f1ffec5a75e57723e1ed169dcf613bd9ad9643d54`,
  `03a660dd24588424d2c1d1e384d6188c9e4711a13317dbebcbbade91ccfda4d4`,
  `8cf53941daaaaf4ddddb1561e1d0eff051118d83e80d11963c3201c942f62e8a`,
  and `eee442731ea9af3d679ec9ac149e7b5bb74e0cc5bba4c2eb84bf7d53f614a36f`.
  `/rav/datakit-6854-inspect-row7413-1720-v470`,
  `/rav/datakit-6854-publish-row7413-1723-v473`, and
  `/rav/datakit-6854-verify-row7413-1724-v474` persisted and independently
  verified those artifacts.

- The row-7447 inspection, semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `d705fa789ea0a59a863d54c4eac0ab7c3812a30a63d74caaff551b921f671adc`,
  `e048fa85acfa289aefc81643ef0b3cd471e99d15d40a7233aa4a0c94c0ec1f85`,
  `01629cd795ebea05fc2d37bee1c13f5e253339762c7dd55d551e32643748c13f`,
  and `367dfb3a08d474a668e96ac906eadbbb39cd30cd9676e86f8ac8d6c8f8602f7b`.
  `/rav/datakit-6854-inspect-row7447-1721-v471`,
  `/rav/datakit-6854-publish-row7447-1722-v472`, and
  `/rav/datakit-6854-verify-row7447-1725-v475` persisted and independently
  verified those artifacts.

- Across the stable 1,290-checkpoint snapshot, all 181 unresolved model
  outcomes are covered by 142 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 131,890 pairs, 83,785 false positives, 48,105 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 163,940 pairs, 100,351 false positives, 63,589 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 4,352)`,
  p2 `(71, 0)`, and p3 `(103, 384)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:16:10Z — 163,556 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1717-v468` independently
  revalidated p0 decision-file 7 semantic offset 0 and p1 decision-file 38
  semantic offsets 3,840 and 3,968. Their 384 baseline pairs contain 314
  false positives, 70 true duplicates, and no unresolved outcomes. Twenty-six
  pairs were chunked and 358 were direct. All 2,477 judgments were valid on
  their first request attempt.

- The outcome Parquet SHA-256 values are:

  - p0:
    `ce21f32f23cfa959c7611d09e5bf6e38010590c844f36edae76255cd05083e0a`;
  - p1:
    `6652be863d40495c56559d9f92b350f5b5c5a39f5a5c929e75cb4dc533fb3a60`
    and
    `99176669637fce7b12eb6b7b07fdcdc92148d06f44b25a2f5e4254d818d310b5`.

- Across the stable 1,287-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 131,506 pairs, 83,506 false positives, 48,000 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 163,556 pairs, 100,072 false positives, 63,484 true duplicates.

- The next audit frontiers are p0 `(7, 128)`, p1 `(38, 4,096)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:13:35Z — 163,172 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1715-v467` independently
  revalidated p1 decision-file 38 semantic offset 3,712. Its 128 baseline
  pairs contain 99 false positives, 29 true duplicates, and no unresolved
  outcomes. One pair was chunked and 127 were direct. All 323 judgments were
  valid on their first request attempt. The outcome Parquet SHA-256 is
  `3410e2ccdedb207dda7bdb47be0ccc9dbb1637b5be23dedee881aaa1dbfd0b4b`.

- Across the stable 1,284-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 131,122 pairs, 83,192 false positives, 47,930 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 163,172 pairs, 99,758 false positives, 63,414 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,840)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:11:50Z — 163,044 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1713-v466` independently
  revalidated p1 decision-file 38 semantic offset 3,584. Its 128 direct
  baseline pairs contain 97 false positives, 31 true duplicates, and no
  unresolved outcomes. All 264 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `8d763b94ad9ad07378e127a7b1ccaa823ef23a0dbea969c23af55f194dcfe429`.

- Across the stable 1,283-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 130,994 pairs, 83,093 false positives, 47,901 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 163,044 pairs, 99,659 false positives, 63,385 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,712)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:09:10Z — 162,916 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1711-v465` independently
  revalidated p1 decision-file 38 semantic offset 3,456. Its 128 direct
  baseline pairs contain 84 false positives, 44 true duplicates, and no
  unresolved outcomes. All 273 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 is
  `87bfd0c26722a913b085fb0adb6089fdd4fde647c0567ec3c84bb2f1cb296133`.

- Across the stable 1,282-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 130,866 pairs, 82,996 false positives, 47,870 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 162,916 pairs, 99,562 false positives, 63,354 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,584)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:07:05Z — 162,788 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1709-v464` independently
  revalidated p1 decision-file 38 semantic offsets 3,200 and 3,328. Their 256
  direct baseline pairs contain 117 false positives, 139 true duplicates, and
  no unresolved outcomes. All 557 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 values are
  `fc9b3adc5c133250583e34dd140cf4422e475f5e3d1c33c45c26f754ed4f56b1`
  and `13ceb7aa3e1e2809ab49efcf42ab37ebd8ca958f9f0d08decf0911f2b367ba9a`.

- Across the stable 1,281-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 130,738 pairs, 82,912 false positives, 47,826 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 162,788 pairs, 99,478 false positives, 63,310 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,456)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:04:20Z — 162,532 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1707-v463` independently
  revalidated p1 decision-file 38 semantic offset 3,072 and p3 decision-file
  103 semantic offset 128. Their 256 baseline pairs contain 146 false
  positives, 110 true duplicates, and no unresolved outcomes. Eighteen pairs
  were chunked and 238 were direct. All 1,528 judgments were valid on their
  first request attempt. The outcome Parquet SHA-256 values are
  `d189726364fc68caa419099756cbd93a7950e354978edb99eb610946416de289`
  and `a49246e4d02d1b6f0e2ac1fec55d06e1f0c2e4a0b5ad45203a8cbe0cdf3737fd`.

- Across the stable 1,279-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 130,482 pairs, 82,795 false positives, 47,687 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 162,532 pairs, 99,361 false positives, 63,171 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,200)`,
  p2 `(71, 0)`, and p3 `(103, 256)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T17:01:30Z — 162,276 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1705-v462` independently
  revalidated p1 decision-file 38 semantic offsets 2,816 and 2,944. Their 256
  baseline pairs contain 116 false positives, 140 true duplicates, and no
  unresolved outcomes. One pair was chunked and 255 were direct. All 592
  judgments were valid on their first request attempt. The outcome Parquet
  SHA-256 values are
  `cb418fb200122043426c74534d58056233143de3d9509aca786b9e3d5de6b6f2`
  and `f5d5a2eceb797a7f0514a89a75499ff32e373d0ce07d4de2955afca61a7cbf82`.

- Across the stable 1,277-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 130,226 pairs, 82,649 false positives, 47,577 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 162,276 pairs, 99,215 false positives, 63,061 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 3,072)`,
  p2 `(71, 0)`, and p3 `(103, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:58:45Z — 162,020 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1703-v461` independently
  revalidated four p1 decision-file 38 checkpoints spanning semantic offsets
  2,304 through 2,688. Their 512 baseline pairs contain 247 false positives,
  265 true duplicates, and no unresolved outcomes. Two pairs were chunked and
  510 were direct. All 1,231 judgments were valid on their first request
  attempt. The outcome Parquet SHA-256 values are
  `9f51b748bb0b4f2ca73992c1218b60b1c3adc9f9ae8aad00cbc546d36f08d9bd`,
  `f9b837e220dbc1ba6e9c38ceaffa8ed8b74fe85fc1c8414e4426d3283147f65e`,
  `cecd311ed59ac84429a2da0198e92fe523ed75c04923590bc40be26acd681bc4`,
  and `a2675c2b8287936f34ab36f94efcc40f8875fdca1636a5a5fa16e3e3351fa3fa`.

- Across the stable 1,275-checkpoint snapshot, all 179 unresolved model
  outcomes remain covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 129,970 pairs, 82,533 false positives, 47,437 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 162,020 pairs, 99,099 false positives, 62,921 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 2,816)`,
  p2 `(71, 0)`, and p3 `(103, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:57:00Z — 161,508 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1655-v454` independently
  revalidated five checkpoints: p1 decision-file 38 semantic offsets 1,920,
  2,048, and 2,176 and p2 decision-file 70 semantic offsets 5,632 and 5,760.
  Their 583 pairs contain 317 model false positives, 264 model true
  duplicates, and two unresolved outcomes. Two pairs were chunked and 581
  were direct. The audit reread all 1,424 judgments and their 1,432 request
  attempts: 1,422 attempts were valid, ten were invalid, and five judgments
  required retries.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p1:
    `7d7d4551af7bcc4e93a91a5fc89c28273d2e433f937d9fa7095e768b10dc6553`,
    `23ecf803cb3c7167fb9b543907b607864b4a1d3e23941a2bf20c6ba2e574d791`,
    and
    `c4a7c7d2ee7b20ffe4250c8eca14e7cfe193a80e63b212dc5de11539ead8b9fa`;
  - p2:
    `da70c1832c76d6421b911f51698ddeed18bb54055bfb248f38dfce50d1a760c4`
    and
    `587582d4744b06faa1b5aafdb17a5a40282441e05b255b6eb9a6d4b4fceb1330`.

- Complete-text inspection resolves both treatment ambiguities as true
  duplicates:

  - `part-00070-of-00128.parquet:8953` compares 74-line SFT records with
    identical memory-consolidation questions, choices, reasoning,
    neuroscience facts, conclusions, and answers. The sole changed line is
    `\boxed{E}` versus `\boxed{\text{E}}`. The 4,532/4,539-character records
    have character, line, and word-sequence similarities 0.999228, 0.986486,
    and 0.999274. Member/canonical text SHA-256 values are
    `2a1f7cab12334d9048903fd84b6486c2d0572f3647b24144a4533b0ccbf6010c`
    and
    `a5883f3d95e57452b2b4d4b8305596b21ee7a5d1032bb53806044231a72352c8`.
  - `part-00070-of-00128.parquet:8972` compares 72-line SFT records with
    identical moral-relativism questions, choices, reasoning, metaethics
    facts, conclusions, and answers. The sole changed line is
    `\boxed{\text{C}}` versus `\boxed{C}`. The 6,069/6,062-character records
    have character, line, and word-sequence similarities 0.999423, 0.986111,
    and 0.999471. Member/canonical text SHA-256 values are
    `1db6b64a04ff06098ba12ab4aaaa49ec6f61181ceeacd6c9ddaa758f788cbd40`
    and
    `071d1c09490ce5624f2b2e1bb0ee9da930c7d0075678beb3217f73c2c8e5af47`.

- The row-8953 inspection, semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `dad6ad8e517c30598db8657b2552d9e6f2be8e2b3fa38789f2348b6bc84bf4d0`,
  `f1da252a7ecd4d29cda54e468d6344b824d820fffba7bdca4d5d933434f193fe`,
  `c77dd00703fc181f9c2d985b080daad189d02c32a1586af0e54dcf9943c22a1f`,
  and `f9432429b31afd1ecda0579192ffa8aadb76c5d1dae275d80ada4920439a27c5`.
  `/rav/datakit-6854-inspect-row8953-1656-v455`,
  `/rav/datakit-6854-publish-row8953-1658-v457`, and
  `/rav/datakit-6854-verify-row8953-1701-v460` persisted and independently
  verified those artifacts.

- The row-8972 inspection, semantic-evidence, manual-Parquet, and marker
  SHA-256 values are
  `1c64326b8aca9de1b4898e67e465703d64e6f3ded228ce60dfcdc548e5db2adb`,
  `09da662fa2758bbfff744e699344aa51e78aa2481f93a0c901b9471d4d826874`,
  `0f678053817067ed19dba2a0b20ae68829ff72074edae7b7f1c221a809039c0c`,
  and `148e9576e29de8ac64f1fcc39f0959e6ab2eabd0d1d703567427493a1a79fe74`.
  `/rav/datakit-6854-inspect-row8972-1657-v456`,
  `/rav/datakit-6854-publish-row8972-1659-v458`, and
  `/rav/datakit-6854-verify-row8972-1700-v459` persisted and independently
  verified those artifacts.

- Across the stable 1,271-checkpoint snapshot, all 179 unresolved model
  outcomes are covered by 140 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 129,458 pairs, 82,286 false positives, 47,172 true duplicates;
  - treatment: 32,050 pairs, 16,566 false positives, 15,484 true duplicates;
  - combined: 161,508 pairs, 98,852 false positives, 62,656 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 2,304)`,
  p2 `(71, 0)`, and p3 `(103, 128)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:49:30Z — 160,925 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1650-v450` independently
  revalidated p1 decision-file 38 semantic offset 1,792 and p2 decision-file
  70 semantic offsets 5,376 and 5,504. Their 384 direct pairs contain 148
  model false positives, 235 model true duplicates, and one unresolved
  outcome. All 803 judgments were valid on their first request attempt. The
  outcome Parquet SHA-256 values are
  `e76d6d3ab272e1a480e0ab83661d39888cf35a5d95e47e0ea07b8cb98e663117`,
  `86a34eb2efed42c5d542618c6c9c325bba536a3e24a394c22dc6322ac7f89fda`,
  and `046a5204ed53e38db9b213d0bebc73230cb2259413fd5766d5e5485c2ba6c00e`.

- Complete-text inspection resolves the drinking-water article ambiguity as a
  true duplicate. The member and canonical are sentence-level variants with
  the same hydration facts, health risks, and recommendations. Member-only
  strings `soda drink machine`, `bottle carefully`, and `get more info` are
  malformed SEO fragments. Its closing paragraph restates the canonical's
  existing warning about low water consumption and recommendation to add
  flavor, so deleting the member loses no member-exclusive substantive fact.
  The canonical's appended Q&A does not affect that deletion test. The
  3,354/3,517-character records have 13/8 lines and character, line, and
  word-sequence similarities 0.806869, 0.095238, and 0.746060.
  Member/canonical text SHA-256 values are
  `43cb8da70a6969333e502b826f5eaedb022a3e3e110b253be2928546e8b0f179`
  and
  `788bb999fa2d4f0fa8e007b8173c5ca7cee4bd49a762f65088d8b559aa94e00d`.

- `/rav/datakit-6854-inspect-row2626-1651-v451` persisted the complete pair and
  diff with inspection SHA-256
  `da19637ffeaf9a9e0fff08fd68a28f431352776a382fc4a9d35cf6a3346434bc`.
  `/rav/datakit-6854-publish-row2626-1652-v452` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row2626-1653-v453`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `c34c1d22e28705de10b33bd9f207281d020c5c3688c6b1d6d34319c8b5e0b60a`,
  `2effd60f96da986cef6fd51fa55437d57d760754bdb45da81959cc4d9a1c5688`,
  and `bdec29075c860c846abe294c222bcd16e2bbcf90c80c0b592810cd78ecaebaaf`.

- Across the stable 1,266-checkpoint snapshot, all 177 unresolved model
  outcomes are covered by 138 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 129,074 pairs, 82,102 false positives, 46,972 true duplicates;
  - treatment: 31,851 pairs, 16,433 false positives, 15,418 true duplicates;
  - combined: 160,925 pairs, 98,535 false positives, 62,390 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 1,920)`,
  p2 `(70, 5,632)`, and p3 `(103, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T16:42:45Z — 160,541 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1648-v449` independently
  revalidated p1 decision-file 38 semantic offsets 1,536 and 1,664 and p2
  decision-file 70 semantic offset 5,248. Their 384 direct pairs contain 148
  false positives, 236 true duplicates, and no unresolved outcomes. All 798
  judgments were valid on their first request attempt. The outcome Parquet
  SHA-256 values are
  `561ad61402bf538528e3bef9f2251539621ea988853a8ab18b240e50bff7b44a`,
  `58a3e24d277f2171fbfc7c91c578d9be724ba8f95b8f7f208cc5243b07e70aea`,
  and `afc4be3481cbda25e16533aec86a56e73cfa50abea52d62d2f98b9b0d9b1eddc`.

- Across the stable 1,263-checkpoint snapshot, all 176 unresolved model
  outcomes remain covered by 137 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 128,946 pairs, 82,039 false positives, 46,907 true duplicates;
  - treatment: 31,595 pairs, 16,348 false positives, 15,247 true duplicates;
  - combined: 160,541 pairs, 98,387 false positives, 62,154 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 1,792)`,
  p2 `(70, 5,376)`, and p3 `(103, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T16:40:30Z — 160,157 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1646-v448` independently
  revalidated p1 decision-file 38 semantic offset 1,408 and p2 decision-file
  70 semantic offset 5,120. Their 256 direct pairs contain 145 false
  positives, 111 true duplicates, and no unresolved outcomes. All 531
  judgments were valid on their first request attempt. The outcome Parquet
  SHA-256 values are
  `9ac3567e25047d7a8a11c0b9ff9d7f40ae3240dc97ab830802da67139d911cdf`
  and `8d904316d3746c1e9dd9ae3dccfdd1ec0dafde3e1ddde0441b9a4c8e47da276a`.

- Across the stable 1,260-checkpoint snapshot, all 176 unresolved model
  outcomes remain covered by 137 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 128,690 pairs, 81,928 false positives, 46,762 true duplicates;
  - treatment: 31,467 pairs, 16,311 false positives, 15,156 true duplicates;
  - combined: 160,157 pairs, 98,239 false positives, 61,918 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 1,536)`,
  p2 `(70, 5,248)`, and p3 `(103, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T16:39:00Z — 159,901 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1644-v447` independently
  revalidated ten checkpoints: five p1 decision-file 38 checkpoints spanning
  semantic offsets 768 through 1,280 and five p2 decision-file 70 checkpoints
  spanning semantic offsets 4,480 through 4,992. Their 1,280 pairs contain
  992 false positives, 288 true duplicates, and no unresolved outcomes. Three
  pairs were chunked and 1,277 were direct. All 2,693 judgments were valid on
  their first request attempt.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p1:
    `07a7ef2ba108dc3dcb26ec0aa4fd329ae417b01995205c52ce64f8e2eafc27f4`,
    `7fa6a2564d3181854703f1979a4ec4285883673d60d1c55fe849f02b1aa9d0ae`,
    `a7c30b9c318c4c377d92121cc84b36a11433cf01e1bff1a1dfb9ccdd96d1d139`,
    `cf7e17ee304cc6a35739ef1f65d33ac18ae8068ae3b3e0e2f888e903e72b064c`,
    and
    `9b1ccdeb8d6a6087bdcd3a8bff5cdc7d87ca1e9bde19a5b3c12ba30357ba5fd2`;
  - p2:
    `841b455e0101d3bf92875ca67310e97207c0130a12d3e9439cb7a6859761be4e`,
    `2bb5bd484cd9d38415f0e0a61f20101dc9b63b9649a45ed597793ab87caeca80`,
    `67e7e54ec690c7922fc258cc785e1bd3b1828aea171dc42919413cd67aa3d192`,
    `d625c078bb0f2da1647566fd9fe517c0cd1c0779be0e861aac21663038981f95`,
    and
    `45acd40ed245f6f7897ecba8705ad849bab7763eefad34236a6fdbc1dd083f4f`.

- Across the stable 1,258-checkpoint snapshot, all 176 unresolved model
  outcomes remain covered by 137 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 128,562 pairs, 81,828 false positives, 46,734 true duplicates;
  - treatment: 31,339 pairs, 16,266 false positives, 15,073 true duplicates;
  - combined: 159,901 pairs, 98,094 false positives, 61,807 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 1,408)`,
  p2 `(70, 5,120)`, and p3 `(103, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T16:36:30Z — 158,621 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1640-v443` independently
  revalidated seven baseline checkpoints: p1 decision-file 38 semantic
  offsets 384, 512, and 640; p2 decision-file 70 semantic offsets 4,096,
  4,224, and 4,352; and p3 decision-file 103 semantic offset 0. Their 896
  pairs contain 648 model false positives, 247 model true duplicates, and one
  unresolved outcome. Twenty-nine pairs were chunked and 867 were direct.
  The audit reread all 3,828 persisted judgments and their 3,830 request
  attempts: 3,827 attempts were valid, three were invalid, and one judgment
  required retries.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p1:
    `e2255db577872a729f29009c363cb7f089a81f6508d5ea211db0f01538383368`,
    `8a2433d3c7de0e2a87d2c7c9e8680b77ff69a10f565c2303ddd8f58f7168a431`,
    and
    `c32f79f7ccfa8632dd4ff97db0ee9dcdff91d03ab0629fe54891a3acd257616c`;
  - p2:
    `17fcf16345cba05f99ff9d3b411c0994511c3fd3dfc158ae956d3906aca3df50`,
    `bd8a345d8474396d41e2d1140d4088a894a91ce6ef0ae0a2a1e603b38760e626`,
    and
    `caeb0eb9a234d38ed3c2a819928d3da47ae2d93af740edbebe6044757515a1db`;
  - p3:
    `a8251771a07fd0af78f64e91e50db0c314cc4d3cc9c09899bc24c63170047aa7`.

- Complete-text inspection resolves the ambiguity as a true duplicate.
  `part-00001-of-00003.parquet:7338` compares 365-line SFT records with
  identical questions, choices, reasoning, historical facts, conclusions, and
  answers. Their only difference is final-answer markup: `\boxed{A}` versus
  `\boxed{\text{A}}`. The 17,559/17,566-character records have character,
  line, and word-sequence similarities 0.999801, 0.997260, and 0.999839.
  Member/canonical text SHA-256 values are
  `1d27d997c0047099e4e8d0dd2e958823c8fdf5edf70afb05d240755cd5e18e7f`
  and
  `b983f7264a6b1cf161a1a93f79e3cf498ec6f88357ca6c725200b67cd9f67551`.

- `/rav/datakit-6854-inspect-row7338-1641-v444` persisted the complete pair and
  diff with inspection SHA-256
  `c23eddebf4a5d939b4c184f7562e9945af39483a1f5efc4ad90b58fd0e2ee59d`.
  `/rav/datakit-6854-publish-row7338-1642-v445` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row7338-1643-v446`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `573db93cf1e967cfbe5b86f056957a68b00b78f075622fbe3285186f4de90411`,
  `331ff196c60625dc5cdcac3df6c782533a1374d543020b42b8309b02693cbbbd`,
  and `445bb584e44689efb861c5b829512c5156f3c17ec21e51bc8774fa8a038731dd`.

- Across the stable 1,248-checkpoint snapshot, all 176 unresolved model
  outcomes are covered by 137 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 127,791 pairs, 81,192 false positives, 46,599 true duplicates;
  - treatment: 30,830 pairs, 15,910 false positives, 14,920 true duplicates;
  - combined: 158,621 pairs, 97,102 false positives, 61,519 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 768)`,
  p2 `(70, 4,480)`, and p3 `(103, 128)`. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T16:29:00Z — 157,725 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1636-v439` independently
  revalidated p1 decision-file 38 semantic offset 256 and five p2
  decision-file 70 checkpoints spanning semantic offsets 3,456 through 3,968.
  Their 768 baseline pairs contain 572 model false positives, 195 model true
  duplicates, and one unresolved outcome. Six pairs were chunked and 762 were
  direct. All 2,061 judgments were valid on their first request attempt.
- In checkpoint order, the outcome Parquet SHA-256 values are
  `fc42716f50e29155521cc4c5e3605fe70362b1ddc81d9b1058d75211a69d96b8`,
  `ed09e471e130301851506044d65ed1d3b0d65a3e9a2fdc20827c0ba08b014dd8`,
  `52b42bb2911608108a5bf02b4c7204b7815879e532ffdc436eef87bab2fb3207`,
  `f3fb86f1aba4748390e7fa2fad77a9fbc860d723b3645666dfcb72b7246b0c02`,
  `5164ca016f0fd2cf58793c5051c91cd89807d765ad3c6be5ddd246b8845d3f89`,
  and
  `96772e752871cc5e8174cca0c6f28643791dd14b985fa76a896d38e660e40fe1`.

- Complete-text inspection resolves the synthetic-text ambiguity as a true
  duplicate. The member is a sentence-level paraphrase of the canonical with
  the same electrical-safety advice, switch wiring details, ceiling-fan
  instructions, and circuit-breaker warning. Its only unique strings are
  malformed SEO/image metadata (`#6407`, `1056x9`, and title fragments), not
  distinct actionable content. The 3,739/3,245-character records have 17/15
  lines and character, line, and word-sequence similarities 0.707904,
  0.437500, and 0.652210. Member/canonical text SHA-256 values are
  `bd228d9538aec218157df7ac92dc69d894d3258a135e82fabf1c36ab35a21287`
  and
  `25e885aaab0033f1cd6c05d4736768911004be12a4e19dcf1bfe517e9cc32d2f`.

- `/rav/datakit-6854-inspect-row6258-1637-v440` persisted the complete pair and
  diff with inspection SHA-256
  `59e2a8ab404567fb38aa7f7bbbe7f10036bd47335d3bd915a0defc80d6f3c232`.
  `/rav/datakit-6854-publish-row6258-1638-v441` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row6258-1639-v442`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `ec6f4cc0de3475f94d0ad9633b682ffa5607b95dfe0285b9505f754c3833025c`,
  `32f133c67cb4418985fd8e3384594c00256058fa877a5b608c3f40c9cd277957`,
  and `76d13125afd58455a5edc77d33e7e97651436578d2b741bc9df498cce257c279`.

- Across the stable 1,241-checkpoint snapshot, all 175 unresolved model
  outcomes are covered by 136 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 126,895 pairs, 80,544 false positives, 46,351 true duplicates;
  - treatment: 30,830 pairs, 15,910 false positives, 14,920 true duplicates;
  - combined: 157,725 pairs, 96,454 false positives, 61,271 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 384)`,
  p2 `(70, 4,096)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:22:50Z — 156,957 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1632-v435` independently
  revalidated p1 decision-file 38 semantic offset 128 and p2 decision-file 70
  semantic offset 3,328. Their 256 baseline pairs contain 140 model false
  positives, 115 model true duplicates, and one unresolved outcome. Twenty-two
  pairs were chunked and 234 were direct. All 2,231 judgments were valid on
  their first request attempt. The outcome Parquet SHA-256 values are
  `9de5ee0e60e36ca9906c24167be796e5b25cb6f911588c8a35188866d086bd31`
  and `76be7da0148044907c1bdc0c79831be5e29e213f06f43ed58e2e876f0a1a633c`.

- Complete-text inspection resolves the Wikiteam ambiguity as a true duplicate
  under the established low-value-template boundary. Both user-talk pages are
  the same FamilySearch Wiki welcome template. Their variations are username,
  signer, timestamp, capitalization, one blank line, and semantically
  equivalent contributor-help link text; those instance slots add no
  substantive training content. The 415/414-character documents have 20/19
  lines and character, line, and word-sequence similarities 0.878166,
  0.717949, and 0.809160. Member/canonical text SHA-256 values are
  `228c7e08604193460d84474a8e7b9ad786cc35fdde03c0df23a6eaa32b57b92b`
  and
  `6a44946d0660ab2dac1961d401a52d06766132522e2ec5adf8f86ab73146f9b2`.

- `/rav/datakit-6854-inspect-row292-1633-v436` persisted the complete pair and
  diff with inspection SHA-256
  `2712f374ed7309ad23543a91d227d8f7b078d3b0224d954d8d1f7584d84729a4`.
  `/rav/datakit-6854-publish-row292-1634-v437` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row292-1635-v438`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `04a0576d00fcaa2997540a83512bd2226660b22bbb1f717cb1370892bddaa3b4`,
  `1849452e23b818cb62d2e1652dc5aa5f5771411514a842d1bee06cab8759e8ed`,
  and `b51cf1fc1fb170aa24e61bac96e3f447672dbdc06f59f25c1344aee797cb26a8`.

- Across the stable 1,235-checkpoint snapshot, all 174 unresolved model
  outcomes are covered by 135 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 126,127 pairs, 79,972 false positives, 46,155 true duplicates;
  - treatment: 30,830 pairs, 15,910 false positives, 14,920 true duplicates;
  - combined: 156,957 pairs, 95,882 false positives, 61,075 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 256)`,
  p2 `(70, 3,456)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:15:40Z — 156,701 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1631-v434` independently
  revalidated six p2 decision-file 70 checkpoints spanning semantic offsets
  2,560 through 3,200. Their 768 baseline pairs contain 352 false positives,
  416 true duplicates, and no unresolved outcomes. One pair was chunked and
  767 were direct. All 1,685 judgments were valid on their first request
  attempt.
- In checkpoint order, the outcome Parquet SHA-256 values are
  `de4b4783efb412ad588f15d65f2191c4a4242178e05a8584328beb1361a048ca`,
  `ed87c9760666c08b25fd8b093ebccf7d3adb2eddf8fbcf7688b6e8986b4c8131`,
  `dd245c277506370e4d24ab92075298978213fb0bb08e6ee807e37610b2b0d894`,
  `eaddea2fe545fa43ac89a731a483d999097fda80ab841fb4af9e058893f4fb09`,
  `be1c289b26e4d8d7d32aa8b559dd19447d258d38bd6c13b5e4cd349a656105b7`,
  and
  `b29b7f99174043c15f35ae8c8138c07329d6b336e65897dba3cb9ee69a0f03c8`.
- Across the stable 1,233-checkpoint snapshot, all 173 unresolved model
  outcomes remain covered by 134 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 125,871 pairs, 79,832 false positives, 46,039 true duplicates;
  - treatment: 30,830 pairs, 15,910 false positives, 14,920 true duplicates;
  - combined: 156,701 pairs, 95,742 false positives, 60,959 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 128)`,
  p2 `(70, 3,328)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:13:40Z — 155,933 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1627-v430` independently
  revalidated four p0 checkpoints spanning the remaining 507 pairs in
  decision-file 6 and six p2 decision-file 70 checkpoints spanning semantic
  offsets 1,792 through 2,432. Their 1,275 pairs contain 590 model false
  positives, 684 model true duplicates, and one unresolved outcome. Two pairs
  were chunked and 1,273 were direct. Nine invalid responses affected four
  retried judgments; 2,706 attempts were valid across 2,715 requests.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p0:
    `347b99712dad4fbc6509164e40a9e41f8d471db701d8a93ea277884a82e57670`,
    `ea0be2b8f73263a316313b2ab3744b1ebf5b14227b1b22623abe684d107c5ce6`,
    `28e29871c2fb29119eaff6030ab4d299725bca2b5f459cd0741517174b5e1841`,
    and
    `b465eabe0606a282c99f1487582504e652ddfad86b42cb0a1050e923ed88e8e3`;
  - p2:
    `0ac271f35550206c20d2d05f56ea90c04367c0fe23e52289db95d2aa74f9479c`,
    `9da41cbcbd9136be12eb6938e452d74495b7b05faecf90ae6cc56c0f42297bd0`,
    `faf3a59021b5654af3c50dbe1007971442f293d977690e322750754518381f16`,
    `435a298c4993b842fabe790bc85aa06f93950ba3b0bee8615b50dfd64f3c72c7`,
    `1da54655f1ccfe1c988b9ba9db1f5ab47f8a6bacbbb05f26f3183d4801a473df`,
    and
    `08601932f4c0d8240e03897c7a6ec8b46af470825be4dd2845742b6fde0db6c3`.

- Complete-text inspection resolves the treatment ambiguity as a true
  duplicate. `part-00006-of-00128.parquet:8983` compares 122-line SFT records
  with identical questions, choices, reasoning, historical details,
  conclusions, and answers. The sole changed line is
  `\boxed{\text{B}}` versus `\boxed{B}`. The 5,998/5,991-character records
  have character, line, and word-sequence similarities 0.999416, 0.991803,
  and 0.999522. Member/canonical text SHA-256 values are
  `ab38a41346d7e55b41ae7eddb93ccbe56ba74dd1fcc2af9e45bf0a9ead0fc680`
  and
  `a10fc588fa39e8cffd60326f417b458b7b778177439c8f13a9eee16a90d337b7`.

- `/rav/datakit-6854-inspect-row8983-1628-v431` persisted the complete pair and
  diff with inspection SHA-256
  `14cad1f37242bf65c143a7cdc4189d4645bc6ce5c728753dd630073c0425cf44`.
  `/rav/datakit-6854-publish-row8983-1629-v432` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row8983-1630-v433`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `b9bfef3337def67aa1f7bf35fa58287c85eaa80404cd2ba00f6aaa7331e78b64`,
  `ae66cc595a32c16243a213ac4cb76a559e5d95a21f3bd7077d934cd9aff3ea3f`,
  and `1160566caf4972c068ad642a5fc202e6318f38e238bf9cffd265b8141998490c`.

- Across the stable 1,227-checkpoint snapshot, all 173 unresolved model
  outcomes are covered by 134 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 125,103 pairs, 79,480 false positives, 45,623 true duplicates;
  - treatment: 30,830 pairs, 15,910 false positives, 14,920 true duplicates;
  - combined: 155,933 pairs, 95,390 false positives, 60,543 true duplicates.

- The next audit frontiers are p0 `(7, 0)`, p1 `(38, 128)`,
  p2 `(70, 2,560)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T16:07:20Z — 154,658 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-1623-v426` independently
  revalidated 12 checkpoints: six from p0 decision-file 6 semantic offsets
  4,608 through 5,248 and six from p2 decision-file 70 semantic offsets 1,024
  through 1,664. Their 1,536 pairs contain 1,025 model false positives, 510
  model true duplicates, and one unresolved outcome. Two pairs were chunked
  and 1,534 were direct. All 3,184 judgments were valid on their first request
  attempt.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p0:
    `f16c3dbed28ae34921508155ec59873ff89325b7dc0ce4129d145f6372decee5`,
    `d59fc56a4004fc3e83285366266bc3ff6291c122ffa9eb4d537e9d5cfe715c95`,
    `1306110f3143abae7dcf0fd03a6b99b67c5973fa32fb3b505dc380980aee9173`,
    `c102cb53bc2243263400c02a1b0da0c056f01b445208424df8845a1c3380749a`,
    `33e6bf16f93b97695db8b8a5de3c3ba26e0bfb34d754620d5939803589844ba7`,
    and
    `08a0fcbac4167cc54f0f083794487c4d8f105d3635f078fa619625c4efa84f3d`;
  - p2:
    `3d345f11adea380b42c856d7928b33dcaa3e7d7a96c59a9cdbe55ed7329c824b`,
    `a554c38a4706a7825a1be876d715363d372b44f4c70938b1fd8e299b4fdd0c41`,
    `8789d0e4a42c181fe1ec23598eee4dc7e2bb987fb40cb253f941a2c3b273ca51`,
    `e2b74c421884e42af32cca928d91ccb6d864f1cc35f71c11caa62355c0c5271d`,
    `1b185b47dc2b2a00c5c1ac586c8825e4d820f076d505da305d8b5888d6d1e0a7`,
    and
    `9bf7bce0ee2b8bfb435d41df4d867eee087035a2ea2252b5d53cf001212c31cd`.

- Complete-text inspection resolves the ambiguity as a false positive.
  `part-00070-of-00128.parquet:2521` compares 32-line Halsey and Northrop
  family-tree pages. They share Ancestry boilerplate but have different
  surnames, tree counts, census changes and rates, 1940 populations, and four
  demographic facts. Character, line, and word-sequence similarities are
  0.808271, 0.562500, and 0.770833. The 1,066/1,062-character member and
  canonical SHA-256 values are
  `e4f6e42c315402eb7f999a8bab2297b2288a5663a34b2c1f9618470193abbc77`
  and
  `d8862607bb758393fc473ddd6695e035597efd95b96b4dec5d4976926bbbb35e`.

- `/rav/datakit-6854-inspect-row2521-1624-v427` persisted the complete pair and
  diff with inspection SHA-256
  `c9680ba0c11f88dca7f21f60519b85c8fc8662f11a17577b24b19a9fc5a2eae7`.
  `/rav/datakit-6854-publish-row2521-1625-v428` wrote the immutable
  false-positive record, and `/rav/datakit-6854-verify-row2521-1626-v429`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `220fc2c5445bf72917fdc2750b8283b39479b9113bf743bf0d1e1feb58fa53fc`,
  `89e7b8d0d79834937e7d085a48b598957cb2e6e23f9a80df52ed4e097270500c`,
  and `8f70693c7ac6afda2239112f39094d3ea39269b006dfa6af751e8d33b9e8cc9d`.

- Across the stable 1,217-checkpoint snapshot, all 172 unresolved model
  outcomes are covered by 133 true-duplicate and 39 false-positive manual
  records. The adjusted totals are:

  - baseline: 124,335 pairs, 79,138 false positives, 45,197 true duplicates;
  - treatment: 30,323 pairs, 15,662 false positives, 14,661 true duplicates;
  - combined: 154,658 pairs, 94,800 false positives, 59,858 true duplicates.

- The next audit frontiers are p0 `(6, 5,376)`, p1 `(38, 128)`,
  p2 `(70, 1,792)`, and p3 `(103, 0)`. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.
