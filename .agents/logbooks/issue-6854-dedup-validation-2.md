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
