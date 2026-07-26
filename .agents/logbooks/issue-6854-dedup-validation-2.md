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
