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

### 2026-07-27T00:38:00Z — 200,556 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0031-v752` independently
  revalidated eight baseline checkpoints: p0 decision-file 8 offsets 1,536
  through 1,920 and p2 decision-file 72 offsets 3,328 through 3,712. Their
  1,024 pairs contain 550 model false positives, 472 model true duplicates,
  and two unresolved outcomes. Two pairs were chunked and 1,022 were direct.
  All 2,213 judgments and request attempts were valid on their first attempt.
  The outcome Parquet SHA-256 values are:

  - p0:
    `760281140a14ae8999d7687b64f38b35da82777467dcbb669b92f63cfc20f9e0`,
    `e44b8aa0ab2ba129cea21e8140cd35008bf2827681ad0c64d01f8af828deedf9`,
    `dd9a9ad9b8be334777145250bfd8e35e98fb63789ff207a63729dfd5f731071e`,
    and
    `a8a0d80584a5ea06915e358d126e670c62f081a9e12fa6e20da795eccb4503d2`;
  - p2:
    `5e0ee06ac199a0350a7879c5d1f21a740fc63b2ef7dcf1a91474d0340376aee5`,
    `0718acae5bfbab6e41f02a8e8c9393c6bc4f9ad05ba39fcaf2b12aee7c2001c4`,
    `989a760ac552ab23e9e6e5e35708ce55be13f97f0d5700c488106878f10a7283`,
    and
    `1a7c10c05fa4334d916f6ccd9f6554630f19113804c8d434d8d241f972678074`.

- Complete-text inspection resolves baseline
  `part-00008-of-00128.parquet:2428` as a false positive. The texts share the
  core language-learning study, but the 13-line member uniquely includes a
  related-story finding that prenatal testosterone exposure makes male
  language-development delays twice as likely as female delays. The
  seven-line canonical has no equivalent of that scientific claim. Character,
  line, and word-sequence similarities are 0.876611, 0.200000, and 0.852679.
  The 1,389/1,326-character member and canonical SHA-256 values are
  `57d4f7ec3892dc4ecee64ecbfae57d457c35144a399231f8d7f50e5f938e1b83`
  and
  `3484ad500de92350a1a17fe7f0ec1cfbcb752ba708b1f33265cf876d25ac044c`.
  `/rav/datakit-6854-inspect-row2428-0033-v753` persisted the pair and diff
  with inspection SHA-256
  `aab68612ea72da5061e6b74c9e6e1517cbbd3ffbf8328524350c5c047fd57fd0`.
  `/rav/datakit-6854-publish-row2428-0035-v755` wrote the immutable record,
  and `/rav/datakit-6854-verify-row2428-0036-v758` independently verified it.
  The semantic-evidence, manual-Parquet, and marker SHA-256 values are
  `07a625799fa670133a0f2a0bce948bab79b15192d6da63d428210ff3c139caa0`,
  `1d5ee12fe86ac2cbe3676c373b2ff641e7583040288f84712db5b1d73c25b5e5`,
  and `da5ebee37601854819cca3792bf92bc4c26f1b20258d26e3ed1d861dde09067a`.

- Complete-text inspection resolves baseline
  `part-00072-of-00128.parquet:5608` as a true duplicate. Both texts are the
  same geo-targeted rotational-molding SEO template with Dover substituted
  for Clayton. They share the phone number, historical anecdotes, process
  description, and four manufacturing steps; other differences are location
  slots, synonyms, and broken text fragments. Character, line, and
  word-sequence similarities are 0.785859, 0.400000, and 0.703030. The
  3,048/2,892-character member and canonical SHA-256 values are
  `92ee154ca10cc8fda51ac14b0d92f6b0a52839bd956bb0abd5bdaf6751ca371f`
  and
  `bcd52a6bd90a701bcc1860cf7b9a2e6fdbb2fadded894774ac54fbb1f2a7bff1`.
  `/rav/datakit-6854-inspect-row5608-0033-v754` persisted the pair and diff
  with inspection SHA-256
  `74393adabf0b611bd257b1a08cff764465b3ff1b87fb2da8a52aa768cf5528f5`.
  `/rav/datakit-6854-publish-row5608-0035-v756` wrote the immutable record,
  and `/rav/datakit-6854-verify-row5608-0036-v757` independently verified it.
  The semantic-evidence, manual-Parquet, and marker SHA-256 values are
  `e88c34f359727f5a9a8a858dabddc8580c4d493ceee9eb55dab5ac7b329c0ccd`,
  `177ebab29188a3edb83c6e00f8a1b501bc9409c889ed58ed7570cc4adaee91d9`,
  and `9fa5d5240824231ff519ceeb5f34902e777f3c50d4cb75639ed69f130cb3b575`.

- Across the stable 1,579-checkpoint snapshot, all 234 unresolved model
  outcomes are covered by 181 true-duplicate and 53 false-positive manual
  records. The adjusted totals are:

  - baseline: 161,307 pairs, 102,457 false positives, 58,850 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 200,556 pairs, 122,773 false positives, 77,783 true duplicates.

- The next audit frontiers are p0 `(8, 2,048)`, p1 `(40, 128)`, p2
  `(72, 3,840)`, and p3 `(105, 0)`. P0 and p2 each have another pending
  128-pair checkpoint. P1 and p3 continue processing unusually large code
  documents. All four batch-priority 2-H100 workers remain active.

### 2026-07-27T00:30:00Z — 199,532 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0025-v748` independently
  revalidated five baseline checkpoints: p0 decision-file 8 offsets 1,152,
  1,280, and 1,408, and p2 decision-file 72 offsets 3,072 and 3,200. Their
  640 pairs contain 448 model false positives, 191 model true duplicates, and
  one unresolved outcome. Two pairs were chunked and 638 were direct. All
  1,442 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 values are:

  - p0:
    `984f09add835fab2e9d126793b359b0dd1f6f352195735e86d1a3a652f2fbc4a`,
    `9cc8ec2229f3478a849ab1c7e42f11cb2730fb3d7b78b146575db523d2b802ea`,
    and
    `144a0b095d773832c62b44ba5de438f99167f14afc51a75e64a5cf1410297522`;
  - p2:
    `5a75ba5f4ed4b32c9b82fa6e0c87034354b465dd5de48070d1a8e2f66ceedd95`
    and
    `4ed5b568e28279cc0e7ec29be3107cbe942b6b82e9400bd5e3ad336cdfea5a17`.

- Complete-text inspection resolves baseline
  `part-00072-of-00128.parquet:5232` as a true duplicate. The 174-line member
  and 169-line canonical are corrupted paraphrases of the same Papaya Global
  payroll and Deel-comparison SEO article. Both follow the same progression
  through cross-border payments, relocation policy, workforce payments,
  EOR/PEO options, product pricing, and the final Deel/Papaya comparison.
  The member-specific `Llc 750 Estate Drive` phrase is a repeated keyword
  injection; the remaining differences are broken, truncated, or reordered
  copies rather than a distinct factual payload. Character, line, and
  word-sequence similarities are 0.667009, 0.413994, and 0.598593. The
  22,876/21,933-character member and canonical SHA-256 values are
  `70bd6a1e74b2c011400d59977452df03cd2f068f1e787be74d047952d747c7e6`
  and
  `290c90b5384dda0528093e6dbe8f5560e883cb5d22c4bc6e4496880e65a35681`.

- `/rav/datakit-6854-inspect-row5232-0026-v749` persisted the complete pair and
  276-line diff with inspection SHA-256
  `80682fabef763e1ba649dcce2c4bbda20fe985ac859291e74bacf3c4fa983ca2`.
  `/rav/datakit-6854-publish-row5232-0029-v750` wrote the immutable
  true-duplicate record, and `/rav/datakit-6854-verify-row5232-0030-v751`
  independently reread the source pair, semantic checkpoint, inspection,
  deterministic Parquet bytes, and completion marker. The semantic-evidence,
  manual-Parquet, and marker SHA-256 values are
  `b04e506d840af63b0bdf0b41811b3525c341163e4ca181cd68c9fbc51e13570a`,
  `ee6e92637f9560fd362ac636df700cd25217cab8bc71299c8d636d567a2fd45a`,
  and `813d144291a4110d47b74db30fdeb032ce148e049fa45e92a0b60de33fee0b34`.

- Across the stable 1,571-checkpoint snapshot, all 232 unresolved model
  outcomes are covered by 180 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 160,283 pairs, 101,906 false positives, 58,377 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 199,532 pairs, 122,222 false positives, 77,310 true duplicates.

- The next audit frontiers are p0 `(8, 1,536)`, p1 `(40, 128)`, p2
  `(72, 3,328)`, and p3 `(105, 0)`. P0 and p2 each have another pending
  128-pair checkpoint. P1 and p3 continue processing unusually large code
  documents. All four batch-priority 2-H100 workers remain active.

### 2026-07-27T00:23:00Z — 198,892 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0021-v747` independently
  revalidated four baseline checkpoints: p0 decision-file 8 offsets 896 and
  1,024, and p2 decision-file 72 offsets 2,816 and 2,944. Their 512 pairs
  contain 312 false positives and 200 true duplicates, with no unresolved
  outcomes. One pair was chunked and 511 were direct. All 1,133 judgments and
  request attempts were valid on their first attempt. The outcome Parquet
  SHA-256 values are:

  - p0:
    `471e2b287e222f4f9dcd618c477ba58650381131a6cab9fcf89bdb50410b8ebf`
    and
    `5639495fdbcca195cc81d5ba26ba31ca996d51dd78d12e64638ffa6610439c71`;
  - p2:
    `172cb7030539a9ee47304225cfc2be60dcfe2a89da59987fadefbfaadceb6038`
    and
    `24fd4f8e0db4d4c9a48498c04474b4c9b28fed804e598d79ba7698db9a6ca704`.

- Across the stable 1,566-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 159,643 pairs, 101,458 false positives, 58,185 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 198,892 pairs, 121,774 false positives, 77,118 true duplicates.

- The next audit frontiers are p0 `(8, 1,152)`, p1 `(40, 128)`, p2
  `(72, 3,072)`, and p3 `(105, 0)`. P0 and p2 each have another pending
  128-pair checkpoint. P1 and p3 continue processing unusually large code
  documents. All four batch-priority 2-H100 workers remain active.

### 2026-07-27T00:19:00Z — 198,380 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0018-v746` independently
  revalidated two direct baseline checkpoints: p0 decision-file 8 offset 768
  and p2 decision-file 72 offset 2,688. Their 256 pairs contain 145 false
  positives and 111 true duplicates, with no unresolved outcomes. All 538
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 values are
  `7e53d3a5254c243455dfda8574ff659f84ddf720fcbfa7046805cb06b43b4886`
  and
  `d41b16baceef2934a7343c8621723f043893730c84da4b12d96069b6771c9b64`.

- Across the stable 1,562-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 159,131 pairs, 101,146 false positives, 57,985 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 198,380 pairs, 121,462 false positives, 76,918 true duplicates.

- The next audit frontiers are p0 `(8, 896)`, p1 `(40, 128)`, p2 `(72, 2,816)`,
  and p3 `(105, 0)`. P0 and p2 each have 128 direct pairs next, requiring 256
  minimum model requests apiece. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-27T00:17:00Z — 198,124 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0016-v745` independently
  revalidated two baseline checkpoints: p0 decision-file 8 offset 640 and p2
  decision-file 72 offset 2,560. Their 256 pairs contain 147 false positives
  and 109 true duplicates, with no unresolved outcomes. Three pairs were
  chunked and 253 were direct. All 661 judgments and request attempts were
  valid on their first attempt. The outcome Parquet SHA-256 values are
  `abb71144651a6169eef00a395e84f0ba6e3a4a56ed154b0f8e6c24b241c50173`
  and
  `e6eb14e05cb374c4178ce38762c19d8692286a223b1b37e3640d4dd7f4698dae`.

- Across the stable 1,560-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 158,875 pairs, 101,001 false positives, 57,874 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 198,124 pairs, 121,317 false positives, 76,807 true duplicates.

- The next audit frontiers are p0 `(8, 768)`, p1 `(40, 128)`, p2 `(72, 2,688)`,
  and p3 `(105, 0)`. P0 and p2 each have 128 direct pairs next, requiring 256
  minimum model requests apiece. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-27T00:15:00Z — 197,868 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0013-v744` independently
  revalidated two baseline checkpoints: p0 decision-file 8 offset 512 and p2
  decision-file 72 offset 2,432. Their 256 pairs contain 138 false positives
  and 118 true duplicates, with no unresolved outcomes. One pair was chunked
  and 255 were direct. All 598 judgments and request attempts were valid on
  their first attempt. The outcome Parquet SHA-256 values are
  `7cd2dca6530071cb18431f0e6cf607dc93fa3f8fd4e08097061e59849a7f5b09`
  and
  `ea4bca0ddc7c415a92686b2378775626da4ec034c2b35011548ba71e0b4c5323`.

- Across the stable 1,558-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 158,619 pairs, 100,854 false positives, 57,765 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 197,868 pairs, 121,170 false positives, 76,698 true duplicates.

- The next audit frontiers are p0 `(8, 640)`, p1 `(40, 128)`, p2 `(72, 2,560)`,
  and p3 `(105, 0)`. P0's next batch contains 128 direct pairs requiring 256
  minimum model requests; p2's next batch has three oversized pairs and
  requires 380 requests. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-27T00:12:00Z — 197,612 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0011-v743` independently
  revalidated two direct baseline checkpoints: p0 decision-file 8 offset 384
  and p2 decision-file 72 offset 2,304. Their 256 pairs contain 154 false
  positives and 102 true duplicates, with no unresolved outcomes. All 529
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 values are
  `a3b46ee98dc1d9a06f3e86e5b4de26fa63a8e94e5d51a65f38cbeaee2e5dad5a`
  and
  `e4a5eb8d48610f503d1c0a90c42b4846bbe6f7fa73b051d41a15e5d841b6ec0f`.

- Across the stable 1,556-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 158,363 pairs, 100,716 false positives, 57,647 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 197,612 pairs, 121,032 false positives, 76,580 true duplicates.

- The next audit frontiers are p0 `(8, 512)`, p1 `(40, 128)`, p2 `(72, 2,432)`,
  and p3 `(105, 0)`. P0's next batch contains 128 direct pairs requiring 256
  minimum model requests; p2's next batch has one oversized pair and requires
  320 requests. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-27T00:10:00Z — 197,356 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0009-v742` independently
  revalidated four baseline checkpoints: p0 decision-file 8 offset 256 and p2
  decision-file 72 offsets 1,920 through 2,176. Their 512 pairs contain 260
  false positives and 252 true duplicates, with no unresolved outcomes. Twelve
  pairs were chunked and 500 were direct. All 2,024 judgments and request
  attempts were valid on their first attempt. In checkpoint order, the outcome
  Parquet SHA-256 values are
  `af779920e4d47793e295d8bef46449c9768ccfb0920b08b17473fc97c9896a78`,
  `c0618df3bcac9a1f4dda3f1a140041f9ecd8985553088175605467dd4696574b`,
  `43089ac6d71670a6f765133a520a69d2daba125aa8432f707300eec3625a6ff5`,
  and
  `1915a34337956bb7437ae135d08908f9194c92b7543cb63213d87a52ea5c20ae`.

- Across the stable 1,554-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 158,107 pairs, 100,562 false positives, 57,545 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 197,356 pairs, 120,878 false positives, 76,478 true duplicates.

- The next audit frontiers are p0 `(8, 384)`, p1 `(40, 128)`, p2 `(72, 2,304)`,
  and p3 `(105, 0)`. P0 and p2 each have 128 direct pairs next, requiring 256
  minimum model requests apiece. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-27T00:07:00Z — 196,844 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0004-v741` independently
  revalidated three baseline checkpoints: p1 decision-file 40 offset 0 and p2
  decision-file 72 offsets 1,664 and 1,792. Their 384 pairs contain 242 false
  positives and 142 true duplicates, with no unresolved outcomes. Twenty-nine
  pairs were chunked and 355 were direct. All 4,040 judgments and request
  attempts were valid on their first attempt. In checkpoint order, the outcome
  Parquet SHA-256 values are
  `3900d3fef2cbdda30c77224e4b908ae6401202566878d402e9cca2c28bdc32f9`,
  `190531facb802237ceaa454a447315f8214f91e827325528ad132f20595158f5`,
  and
  `8c4dac46d53fae1a5a6981fdd97678d36e124ff7a0e3b51f8962aec1d8e250d4`.

- Across the stable 1,550-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 157,595 pairs, 100,302 false positives, 57,293 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 196,844 pairs, 120,618 false positives, 76,226 true duplicates.

- The next audit frontiers are p0 `(8, 256)`, p1 `(40, 128)`, p2 `(72, 1,920)`,
  and p3 `(105, 0)`. P1's next baseline batch has 14 oversized pairs and
  requires 1,586 minimum model requests; p2's next batch contains 128 direct
  pairs requiring 256 requests. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-27T00:03:00Z — 196,460 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-0002-v740` independently
  revalidated two p2 decision-file 72 checkpoints at semantic offsets 1,408
  and 1,536. Their 256 direct baseline pairs contain 201 false positives and
  55 true duplicates, with no unresolved outcomes. All 527 judgments and
  request attempts were valid on their first attempt. The outcome Parquet
  SHA-256 values are
  `3d242389c5f369b66f113997e35cd17564705676eac89c22a4654d39d82fa057`
  and
  `72793564c1f48447316ce52dd4abe69d707e6cf9a93ce5f97cca615d7930f1f5`.

- Across the stable 1,547-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 157,211 pairs, 100,060 false positives, 57,151 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 196,460 pairs, 120,376 false positives, 76,084 true duplicates.

- The next audit frontiers are p0 `(8, 256)`, p1 `(40, 0)`, p2 `(72, 1,664)`,
  and p3 `(105, 0)`. P2's next baseline batch contains 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-27T00:01:00Z — 196,204 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2359-v739` independently
  revalidated five baseline checkpoints: p0 decision-file 8 offset 128 and p2
  decision-file 72 offsets 896 through 1,280. Their 640 pairs contain 508
  false positives and 132 true duplicates, with no unresolved outcomes.
  Twenty-seven pairs were chunked and 613 were direct. All 4,526 judgments and
  request attempts were valid on their first attempt. In checkpoint order, the
  outcome Parquet SHA-256 values are
  `8cefa2640e1eddac0213262a94c62f391fbc26652fa031376f5afec595d51593`,
  `dab5b0891fbd387048e4f627e1e599eee072db9f01976a31437ba345ebe6a907`,
  `4db6f6cca6903e10e4b5c2ffe1570bffa5ec2f1843308aa55315864a957f003d`,
  `54938b77df845d4a849d19ffbc0d996774563d5280209e35ba4a10b71a5c0ec5`,
  and
  `3b274b55b48006732d49b7924c170a031d9408d4dc83deda70f8de2f0a5b2b61`.

- Across the stable 1,545-checkpoint snapshot, all 231 unresolved model
  outcomes remain covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 156,955 pairs, 99,859 false positives, 57,096 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 196,204 pairs, 120,175 false positives, 76,029 true duplicates.

- The next audit frontiers are p0 `(8, 256)`, p1 `(40, 0)`, p2 `(72, 1,408)`,
  and p3 `(105, 0)`. P0's next baseline batch has 11 oversized pairs and
  requires 1,212 minimum model requests; p2's next batch contains 128 direct
  pairs requiring 256 requests. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T23:58:00Z — 195,564 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2353-v735` independently
  revalidated two p2 decision-file 72 checkpoints at semantic offsets 640 and
  768. Their 256 direct baseline pairs contain 164 model false positives, 91
  model true duplicates, and one unresolved outcome. The 535 judgments
  required 538 request attempts: 534 were valid and four invalid JSON attempts
  affected two retried judgments. The outcome Parquet SHA-256 values are
  `9f93339ce6fe7174234ea51d5e4448b163599aa3a3699bf7f0d5aa4ba42156a1`
  and
  `12c6a4aa80f1d0089e30f805f848fb874e63f8a572f27db0557612a0436b9e3a`.

- Complete-text inspection resolves the baseline ambiguity as a false
  positive. The 701/714-line TensorFlow references share long C++ API
  catalogs, but the member uniquely contains a Keras Dense-model example, an
  MSE derivation, and detailed Placeholder shape-attribute documentation. The
  canonical instead contains addition, matrix-multiplication, and Conv3D
  attribute examples. Character, line, and word-sequence similarities are
  0.777123, 0.090459, and 0.763013. Member/canonical text SHA-256 values are
  `8092a2338ebbd9e864ac0248fde658e6eff82c28bd3beb3d91047140acff784f`
  and
  `d00b75e89cdd863f9a057f1d501eb6dbaa594486564a89277931aa30c78090b9`.

- `/rav/datakit-6854-inspect-row1205-2354-v736` persisted both complete texts,
  their complete 1,409-line diff, and all three model judgments with inspection
  SHA-256
  `161e90ed66feb7ad10702d991d3eee913a9e4cd3b301dd6b9d295bf4713847dc`.
  `/rav/datakit-6854-publish-row1205-2357-v737` published the hash-bound
  false-positive record, and
  `/rav/datakit-6854-verify-row1205-2357-v738` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The semantic-judgment, manual-record, and marker SHA-256
  values are
  `32568dd182f39f174f365fd0b2243b2edc0b59ef3a6969f2ad65e3d5177c6e10`,
  `765248921dd8643939aac3398a0d97487a629e4a1130564013ed188d9d820383`,
  and
  `d6b6ebd61cf6b0daf463403a50b1518c8514259c11a6337a264ed1af291fb845`.

- Across the stable 1,540-checkpoint snapshot, all 231 unresolved model
  outcomes are covered by 179 true-duplicate and 52 false-positive manual
  records. The adjusted totals are:

  - baseline: 156,315 pairs, 99,351 false positives, 56,964 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 195,564 pairs, 119,667 false positives, 75,897 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 896)`,
  and p3 `(105, 0)`. P2's next baseline batch has two oversized pairs and
  requires 292 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:50:00Z — 195,308 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2349-v734` independently
  revalidated the p2 decision-file 72 checkpoint at semantic offset 512. Its
  128 direct baseline pairs contain 86 false positives and 42 true duplicates,
  with no unresolved outcomes. All 261 judgments and request attempts were
  valid on their first attempt. The outcome Parquet SHA-256 is
  `0c8af414b52ca1ae8a971e72d6c584392d9c27b5c3751979d9af76b5f777f92b`.

- Across the stable 1,538-checkpoint snapshot, all 230 unresolved model
  outcomes remain covered by 179 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 156,059 pairs, 99,186 false positives, 56,873 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 195,308 pairs, 119,502 false positives, 75,806 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 640)`,
  and p3 `(105, 0)`. P2's next baseline batch contains 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:48:00Z — 195,180 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2347-v733` independently
  revalidated the p2 decision-file 72 checkpoint at semantic offset 384. Its
  128 baseline pairs contain 97 false positives and 31 true duplicates, with
  no unresolved outcomes. Ten pairs were chunked and 118 were direct. All
  1,079 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `1d0ce77352accbe03a12ed7cf5c3a5664e23315ce13193dc5a13476745921b20`.

- Across the stable 1,537-checkpoint snapshot, all 230 unresolved model
  outcomes remain covered by 179 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,931 pairs, 99,100 false positives, 56,831 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 195,180 pairs, 119,416 false positives, 75,764 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 512)`,
  and p3 `(105, 0)`. P2's next baseline batch contains 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:37:00Z — 195,052 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2336-v731` independently
  revalidated the p2 decision-file 72 checkpoint at semantic offset 256. Its
  128 baseline pairs contain 85 false positives and 43 true duplicates, with
  no unresolved outcomes. Ten pairs were chunked and 118 were direct. All 784
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `d90996a76f57e53c365223748f97977e21a565d808770a819ef07c3c7da61127`.

- Across the stable 1,536-checkpoint snapshot, all 230 unresolved model
  outcomes remain covered by 179 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,803 pairs, 99,003 false positives, 56,800 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 195,052 pairs, 119,319 false positives, 75,733 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 384)`,
  and p3 `(105, 0)`. P2's next baseline batch contains ten oversized pairs
  and requires 1,066 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T23:35:00Z — 194,924 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2330-v727` independently
  revalidated three checkpoints: p2 decision-file 72 offset 128 and p3
  decision-file 104 offsets 5,760 and 5,888. Their 324 pairs contain 222 model
  false positives, 101 model true duplicates, and one unresolved outcome.
  Fifteen pairs were chunked and 309 were direct. The 3,211 judgments required
  3,217 request attempts: 3,209 were valid and eight invalid JSON attempts
  affected three retried judgments. The outcome Parquet SHA-256 values are
  `abbffcc21d84394cef35055e7fc14a385cd9e0fbc9e49348d7c96865597601e1`,
  `184ddfa930137d6892a9737730df3ec6e6b282221ebec25a90844d1011df774b`,
  and
  `8e08eb758b264d37a8df4b4ee8e7d13b786c2f6088c12a1eb16594642bc5a9af`.

- Complete-text inspection resolves the treatment ambiguity as a true
  duplicate. The 268-line analytical-chemistry examples have identical
  questions, options, reasoning, conclusions, and answers. The only difference
  is the final LaTeX spelling: `\boxed{\text{H}}` versus `\boxed{H}`. The
  11,019/11,012-character records have character, line, and word-sequence
  similarities 0.999682, 0.996269, and 0.999723. Member/canonical text SHA-256
  values are
  `e254ff25f602057b042a6b8cc905a5a2c04279d02b967c1ccfd3a1fa34a20ae0`
  and
  `3e3479af114f8007e3553b56239d445dab83d7394ed146d869207896193ddd8b`.

- `/rav/datakit-6854-inspect-row9150-2332-v728` persisted both complete texts,
  their complete diff, and all three model judgments with inspection SHA-256
  `90e2f1f224753ab16f861cffb0fa5002650b5962c9db09b087427107a56cadd4`.
  `/rav/datakit-6854-publish-row9150-2333-v729` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row9150-2334-v730` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The semantic-judgment, manual-record, and marker SHA-256
  values are
  `64250637ece06c960700b6ae07a1ab249a0d89c957ff61ab806542ed1c51f736`,
  `48f819539029ea690cfb04f4ac79e40704b59eb707f3dab8f211390a54f381b7`,
  and
  `652b4fec8e08909a620f0828732cd31c99a8303393371b71d677bb25fdd74e8f`.

- Across the stable 1,535-checkpoint snapshot, all 230 unresolved model
  outcomes are covered by 179 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,675 pairs, 98,918 false positives, 56,757 true duplicates;
  - treatment: 39,249 pairs, 20,316 false positives, 18,933 true duplicates;
  - combined: 194,924 pairs, 119,234 false positives, 75,690 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 256)`,
  and p3 `(105, 0)`. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T23:29:00Z — 194,600 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2322-v723` independently
  revalidated two p3 decision-file 104 checkpoints at semantic offsets 5,504
  and 5,632. Their 256 direct treatment pairs contain 84 model false
  positives, 171 model true duplicates, and one unresolved outcome. All 538
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 values are
  `538b034aeaafa988decd5b3af4542967a0583d2113eec6846aa1d342991ae6bd`
  and
  `5f38b01840b00ccf8e13984fb166461bd491ddc362b17924c4fb3da983b0f0b0`.

- Complete-text inspection resolves the treatment ambiguity as a true
  duplicate. The 1,036/1,038-character records are incoherent college and
  career SEO spam with the same sentence scaffold. Their differences are
  substituted fields for subjects, institutions, locations, jobs, and
  programs; under the audit boundary these low-value entity slots are not
  substantive facts. Character, line, and word-sequence similarities are
  0.774349, 0.250000, and 0.748344. Member/canonical text SHA-256 values are
  `f32571fd12815668beff5c7e60e6de193a7642591e5cfd35ce0175fe399109e2`
  and
  `500412d7e3b5cd41a0c03e652ab846758e5ec49e0bc6f6e0c11788da792b2ea7`.

- `/rav/datakit-6854-inspect-row8820-2324-v724` persisted both complete texts,
  their complete diff, and all three model judgments with inspection SHA-256
  `6f666e9a4cfe567cf35f06403cbe02d2d36967755033cbe645d8bef42f793e43`.
  `/rav/datakit-6854-publish-row8820-2327-v725` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row8820-2328-v726` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The semantic-judgment, manual-record, and marker SHA-256
  values are
  `152c88fa80ed572f8035b5c728c6528bdb039234ba4ddcb4c907c9d2da01c65a`,
  `6b130bc2aa1463dbaf10cfb6b225273b056f6211de947afc63fc16ce4da9804d`,
  and
  `53c2e2caa445a9b74e4ea9ac3e23ea46b4a8cdb045c40f72f7cd91cacffe669a`.

- Across the stable 1,532-checkpoint snapshot, all 229 unresolved model
  outcomes are covered by 178 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,547 pairs, 98,836 false positives, 56,711 true duplicates;
  - treatment: 39,053 pairs, 20,176 false positives, 18,877 true duplicates;
  - combined: 194,600 pairs, 119,012 false positives, 75,588 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 5,760)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T23:21:05Z — 194,344 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2320-v722` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 5,376.
  Its 128 direct treatment pairs contain 31 false positives and 97 true
  duplicates, with no unresolved outcomes. All 263 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `5c757dc464b02dffbfe4d234f8d241bc84717f9b0de58ce6a049fc2b8c1f4456`.

- Across the stable 1,530-checkpoint snapshot, all 228 unresolved model
  outcomes remain covered by 177 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,547 pairs, 98,836 false positives, 56,711 true duplicates;
  - treatment: 38,797 pairs, 20,092 false positives, 18,705 true duplicates;
  - combined: 194,344 pairs, 118,928 false positives, 75,416 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 5,504)`. P3's next treatment batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:19:10Z — 194,216 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2318-v721` independently
  revalidated two p3 decision-file 104 checkpoints at semantic offsets 5,120
  and 5,248. Their 256 direct treatment pairs contain 104 false positives and
  152 true duplicates, with no unresolved outcomes. All 520 judgments and
  request attempts were valid on their first attempt. The outcome Parquet
  SHA-256 values are
  `46388aaaed20825e44776f62475dd970a0041ccdb132229123ca434cb079659b`
  and
  `92c830fa5b436001ec9de1f55adcb87dd6f6c00c963fd7ba205debbdec022f6f`.

- Across the stable 1,529-checkpoint snapshot, all 228 unresolved model
  outcomes remain covered by 177 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,547 pairs, 98,836 false positives, 56,711 true duplicates;
  - treatment: 38,669 pairs, 20,061 false positives, 18,608 true duplicates;
  - combined: 194,216 pairs, 118,897 false positives, 75,319 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 5,376)`. P3's next treatment batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:17:00Z — 193,960 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2316-v720` independently
  revalidated three p3 decision-file 104 checkpoints at semantic offsets 4,736
  through 4,992. Their 384 treatment pairs contain 284 false positives and 100
  true duplicates, with no unresolved outcomes. One pair was chunked and 383
  were direct. All 802 judgments and request attempts were valid on their
  first attempt. The outcome Parquet SHA-256 values are
  `30ea982fbbe02e07621367ce4ee1a2d5691dac836a1f89b463c9c8e0b69238c6`,
  `b31afaeedf2683875558d1f4b2864f7f222b4226796d68a8f9c90a9c6405a6f0`,
  and
  `5723e9c2a0405d3fd563f1ecb69b20f462efe639efa0961eab01a8149c58851f`.

- Across the stable 1,527-checkpoint snapshot, all 228 unresolved model
  outcomes remain covered by 177 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,547 pairs, 98,836 false positives, 56,711 true duplicates;
  - treatment: 38,413 pairs, 19,957 false positives, 18,456 true duplicates;
  - combined: 193,960 pairs, 118,793 false positives, 75,167 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 5,120)`. P3's next treatment batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:15:05Z — 193,576 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2311-v716` independently
  revalidated two p3 decision-file 104 checkpoints at semantic offsets 4,480
  and 4,608. Their 256 baseline pairs contain 218 model false positives, 37
  model true duplicates, and one unresolved outcome. One pair was chunked and
  255 were direct. The 540 judgments required 546 request attempts: 537 were
  valid, and nine invalid JSON attempts affected three retried judgments. The
  outcome Parquet SHA-256 values are
  `09626f41b19c2b07c0c35a0368877b9704e480904d8344b1ea1a994e723b9984`
  and
  `a6f9abf5e6ddd16402933cb99e73edab2db145b65a8fe0faa17bae7e421efd3d`.

- `/rav/datakit-6854-inspect-row7696-2312-v717` read both complete
  contract-law examples and all model evidence. Both records contain the same
  119-line question, options, reasoning, conclusion, and answer; their sole
  changed line is `\boxed{\text{H}}` versus `\boxed{H}`. Character, line, and
  word-sequence similarities are 0.999467, 0.991597, and 0.999537, so the
  manual label is true duplicate. All three model passes failed JSON parsing;
  the manual decision uses the complete hash-bound texts. Member/canonical
  text SHA-256 values are
  `3701b5115d4c279180001e5a2de16cb1386771e376ab547ed5eeb8e440ff460c`
  and
  `bc0a960d54adcf2c33475d8ff1f2d9ba691990d6a22b59307d789aa45290f889`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `35d16e6b096e2474f1de91ab017faf8c5ff1494f609e5cc97f983ad79740e85e`
  and
  `0162c893d749c251fc26c86b97255bbabb39ab2e20a8ca52e8143cdca1bf2f7e`.
  `/rav/datakit-6854-publish-row7696-2313-v718` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row7696-2314-v719` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The manual-record and marker SHA-256 values are
  `ee0186ec835410511f7cabf4a77fb7b3ea35571ef72ad52d726036136d34e1a0`
  and
  `6a9ef046723767f9d05a23124a7cac147034029412fcbb54e0941ae7206a89c2`.

- Across the stable 1,524-checkpoint snapshot, all 228 unresolved model
  outcomes are covered by 177 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,547 pairs, 98,836 false positives, 56,711 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 193,576 pairs, 118,509 false positives, 75,067 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 4,736)`. P3's next batch contains treatment pairs, including
  one oversized pair, and requires 270 minimum model requests. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T23:10:15Z — 193,320 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2306-v712` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 4,352.
  Its 128 direct baseline pairs contain 73 model false positives, 54 model
  true duplicates, and one unresolved outcome. The 260 judgments required 265
  request attempts: 258 were valid, seven invalid JSON attempts affected three
  retried judgments. The outcome Parquet SHA-256 is
  `f51672760e7299f66a96cf6758fc171184b0f1b8972fa424a743edccf9da3459`.

- `/rav/datakit-6854-inspect-row7456-2307-v713` read both complete CSMA/CD
  examples and all model evidence. Both records contain the same 170-line
  question, options, reasoning, protocol explanation, conclusion, and answer;
  their sole changed line is `\boxed{J}` versus `\boxed{\text{J}}`. Character,
  line, and word-sequence similarities are 0.999593, 0.994118, and 0.999651,
  so the manual label is true duplicate. Member/canonical text SHA-256 values
  are
  `acdbfabab7f666a98e184ded43c7417d5b0da7f626788e11fb11621f31ae11f5`
  and
  `d76725323a8baa20733b4730d871fbd8d7a96b1b57621b6f7d1fdcccddadb50e`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `d47af1139010fcfe4873568d74d98219b7bab273df531dbd34468faee91e002d`
  and
  `4e0c0416cd6495afb499c31e769ae2b888c3ee3a56ee972f956d4f3cf11f73f5`.
  `/rav/datakit-6854-publish-row7456-2308-v714` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row7456-2309-v715` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The manual-record and marker SHA-256 values are
  `4ee58f5414be64e34379bb5798b5536c249540cd7da77d2260e24ca3a5fdab1b`
  and
  `70c96c95e138e0af286e3902027718591c965a501a5feb00858fe3e2ac758afb`.

- Across the stable 1,522-checkpoint snapshot, all 227 unresolved model
  outcomes are covered by 176 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,291 pairs, 98,618 false positives, 56,673 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 193,320 pairs, 118,291 false positives, 75,029 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 4,480)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:05:30Z — 193,192 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2304-v711` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 4,224.
  Its 128 direct baseline pairs contain 90 false positives and 38 true
  duplicates, with no unresolved outcomes. All 262 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `dc831bafffa35b92efda7840a68443163c692318fe5209e14799c3fd95f9e28b`.

- Across the stable 1,521-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,163 pairs, 98,545 false positives, 56,618 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 193,192 pairs, 118,218 false positives, 74,974 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 4,352)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:03:35Z — 193,064 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2302-v710` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 4,096.
  Its 128 direct baseline pairs contain 92 false positives and 36 true
  duplicates, with no unresolved outcomes. All 272 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `b2aca614fee875a96d6ab0035398d34e0fb37fbd6453428aba14664f8bcbe2d9`.

- Across the stable 1,520-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 155,035 pairs, 98,455 false positives, 56,580 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 193,064 pairs, 118,128 false positives, 74,936 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 4,224)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:02:00Z — 192,936 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2301-v709` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,968.
  Its 128 baseline pairs contain 97 false positives and 31 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  309 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `7a7731b0568fda667c85878b24458c1300e6d107dbf20ef82219a02cdcb14501`.

- Across the stable 1,519-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,907 pairs, 98,363 false positives, 56,544 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,936 pairs, 118,036 false positives, 74,900 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 4,096)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T23:00:25Z — 192,808 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2259-v708` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,840.
  Its 128 direct baseline pairs contain 102 false positives and 26 true
  duplicates, with no unresolved outcomes. All 259 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `b1493bbca4c7586f4a71a4f4f11157a854d8ed050159f0e32a7c74705ab48b5b`.

- Across the stable 1,518-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,779 pairs, 98,266 false positives, 56,513 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,808 pairs, 117,939 false positives, 74,869 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,968)`. P3's next baseline batch has 128 pairs, including one
  oversized pair, and requires 294 minimum model requests. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T22:58:40Z — 192,680 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2258-v707` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,712.
  Its 128 baseline pairs contain 91 false positives and 37 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  307 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `ba6356ddd2399acb016c5606cf05e8e08c8baae23e410c644a069c5150fdac62`.

- Across the stable 1,517-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,651 pairs, 98,164 false positives, 56,487 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,680 pairs, 117,837 false positives, 74,843 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,840)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:56:00Z — 192,552 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2258-v705` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,584.
  Its 128 baseline pairs contain 107 false positives and 21 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  309 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `ad3d93bc87d48ea674c9fa256091ea2e458ff8a9a7ed96f37af89de5fb7e072f`.

- Across the stable 1,516-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,523 pairs, 98,073 false positives, 56,450 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,552 pairs, 117,746 false positives, 74,806 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,712)`. P3's next baseline batch has 128 pairs, including one
  oversized pair, and requires 302 minimum model requests. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T22:54:30Z — 192,424 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2256-v704` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,456.
  Its 128 direct baseline pairs contain 66 false positives and 62 true
  duplicates, with no unresolved outcomes. All 270 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `3aa402b0aa572a4e4f8ff06d96637ab048a89b9980ad8f5ff661bf2588f72af7`.

- Across the stable 1,515-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,395 pairs, 97,966 false positives, 56,429 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,424 pairs, 117,639 false positives, 74,785 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,584)`. P3's next baseline batch has 128 pairs, including one
  oversized pair, and requires 306 minimum model requests. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T22:53:10Z — 192,296 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2254-v703` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,328.
  Its 128 direct baseline pairs contain 56 false positives and 72 true
  duplicates, with no unresolved outcomes. All 272 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `2c07eeb0d3b8517185462abfd182d5a13df0500f85410dded327be05ac0243fc`.

- Across the stable 1,514-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,267 pairs, 97,900 false positives, 56,367 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,296 pairs, 117,573 false positives, 74,723 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,456)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:51:45Z — 192,168 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2252-v702` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 3,200.
  Its 128 direct baseline pairs contain 74 false positives and 54 true
  duplicates, with no unresolved outcomes. All 275 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `c694d023f5b146294a24eb4cd04e89bb6ec2753bbb97e77cf94dd01d0db07b35`.

- Across the stable 1,513-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,139 pairs, 97,844 false positives, 56,295 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,168 pairs, 117,517 false positives, 74,651 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,328)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:50:10Z — 192,040 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2250-v701` independently
  revalidated two p3 decision-file 104 checkpoints at semantic offsets 2,944
  and 3,072. Their 256 baseline pairs contain 128 false positives and 128 true
  duplicates, with no unresolved outcomes. Two pairs were chunked and 254
  were direct. All 668 judgments and request attempts were valid on their
  first attempt. The outcome Parquet SHA-256 values are
  `8f68129d873a99f92f5c41d84541cb2a1fa9937ef2212ae6b9112a98fe54eaa3`
  and
  `a9e0359af22d35c15f4792bc19fb98a5c09df6737d649c1259a1baeff280f8cf`.

- Across the stable 1,512-checkpoint snapshot, all 226 unresolved model
  outcomes remain covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 154,011 pairs, 97,770 false positives, 56,241 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 192,040 pairs, 117,443 false positives, 74,597 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 3,200)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:48:30Z — 191,784 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2244-v697` independently
  revalidated three p3 decision-file 104 checkpoints at semantic offsets 2,560
  through 2,816. Their 384 baseline pairs contain 177 model false positives,
  206 model true duplicates, and one unresolved outcome. One pair was chunked
  and 383 were direct. All 896 judgments and request attempts were valid on
  their first attempt. The outcome Parquet SHA-256 values are
  `b82e095b17fb37368194b7a0b51fe908abba48180154831edc9f1f478e851525`,
  `481e3ff680cf3c7b2963645751739bbea4ab13fd2c618cb0a6c150f8fe2015b5`,
  and
  `a26bfbf9ba957d9983345e9e4429d04fd352126e77f9df2a74b3daa77856442e`.

- `/rav/datakit-6854-inspect-row4724-2246-v698` read both complete keto
  advertorials and all three model judgments. The 7,991-character member and
  8,200-character canonical have character, line, and word-sequence
  similarities 0.736335, 0.484211, and 0.671647. They are sentence-spun
  copies with the same diet claims, Rachel Roberts program description,
  named testimonials, and sales narrative. `Chicken Recipes` versus
  `Cocktails` is an injected SEO keyword rather than topical content. The
  model tiebreak cited a five-item benefit list as member-only, but complete
  inspection shows that list only in the canonical. The member adds no
  distinct recipe, cocktail, factual claim, or advice, so the manual label is
  true duplicate. Member/canonical text SHA-256 values are
  `6d4ab6aa4e99075f44be92dedd22cc07abcd470a4945b4097ad8628ad8b522e0`
  and
  `0cd1d367b6951316b678c0904375093bebe77db50dfee852c2d5268559bc5fc0`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `1dcda5f1511aa08702eddcfb819c3ca8590b402848aa31ad325033bb07a3fac9`
  and
  `38079e094aa86b824694862fe7041bcb699a21c665ad8903bc167a61ac7a57ca`.
  `/rav/datakit-6854-publish-row4724-2247-v699` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row4724-2248-v700` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The manual-record and marker SHA-256 values are
  `27db4ef3aeccf9673efc85bdf3bd69ae5df8509109cb2ee839fe5b7d4c7e04ef`
  and
  `c5c1ccf4368ea76cec0f34a175fd0592f008cc15e3c00b6845e381536c4d6f4d`.

- Across the stable 1,510-checkpoint snapshot, all 226 unresolved model
  outcomes are covered by 175 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 153,755 pairs, 97,642 false positives, 56,113 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 191,784 pairs, 117,315 false positives, 74,469 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 2,944)`. P3's next baseline batch has 128 pairs, including two
  oversized pairs, and requires 366 minimum model requests. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T22:43:15Z — 191,400 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2237-v693` independently
  revalidated four p3 decision-file 104 checkpoints at semantic offsets 2,048
  through 2,432. Their 512 baseline pairs contain 236 model false positives,
  275 model true duplicates, and one unresolved outcome. Three pairs were
  chunked and 509 were direct. All 1,305 judgments and request attempts were
  valid on their first attempt. The outcome Parquet SHA-256 values are
  `831d325b2a4a5d03c7bf515a2364bbd5effb0c49a3812739385020c9fd5077b5`,
  `751686761b046cd4865927f3fb08416db2022824e5c7a23f100f3bdf2846adbe`,
  `e5dc0ffa1a9c80e489110a0a9fd5e4d058d212760bced0092f96c6e00e4860b1`,
  and
  `325b119ad683b5c788da5eb38154154d7788371f635ccb078709516cc07f5a5a`.

- `/rav/datakit-6854-inspect-row3575-2240-v694` read both complete
  worksheet-SEO texts and all three model judgments. The 409-character member
  and 537-character canonical have character, line, and word-sequence
  similarities 0.632135, 0.307692, and 0.552147. Both records are the same
  malformed SEO shell. Their changed grade/title, date, time, image count, and
  link caption are low-value instance and page-metadata slots; neither record
  contains the worksheet itself, and the member adds no distinct factual or
  instructional payload. The manual label is true duplicate. Member/canonical
  text SHA-256 values are
  `1d9d05deb0ab21f8ecc6c9c7f967a1f56ace967b1d902fbb55d18e1e06a69c35`
  and
  `f10a03bbc4c19ee4c15c085423e5fc1becc4d06ebe847e963495fd43d7b57aec`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `5e349aa2127c2ebc8a588618086f532be668da08d0e5bc11a21c044214c7a207`
  and
  `4aa895fc9b5c2a61a29b63066ce8068f20229b0d67bd70cac2d0192645c4c5b3`.
  `/rav/datakit-6854-publish-row3575-2241-v695` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row3575-2242-v696` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The manual-record and marker SHA-256 values are
  `1f3c7a0465f0203055d073af9d2660de115bae08da5be86bf171239be08d5967`
  and
  `16e9ef129a158b3d8c970f7b822ec6fd7783a91f32490dfefd9080a89a170f62`.

- Across the stable 1,507-checkpoint snapshot, all 225 unresolved model
  outcomes are covered by 174 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 153,371 pairs, 97,465 false positives, 55,906 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 191,400 pairs, 117,138 false positives, 74,262 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 2,560)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:37:00Z — 190,888 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2231-v689` independently
  revalidated the p3 decision-file 104 checkpoint at semantic offset 1,920.
  Its 128 direct baseline pairs contain 57 model false positives, 70 model
  true duplicates, and one unresolved outcome. All 275 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `c893a8dbe0a025826a839c038e57bf7d5fd89c5c5a1a2ee5452e2be741ae4a20`.

- `/rav/datakit-6854-inspect-row2886-2232-v690` read both complete lawn-care
  texts and all three model judgments. The 3,884-character member and
  3,446-character canonical have character, line, and word-sequence
  similarities 0.715689, 0.080000, and 0.645849. They share a sentence-spun
  lawn-care article, but the member adds a complete paragraph claiming that
  lawn cleanliness prevents disease and family death, attracts visitors, and
  protects against insects and parasites. Those member-only propositions
  exceed the low-value location and business-footer slots, so deleting the
  member loses distinct substantive content. The manual label is false
  positive. Member/canonical text SHA-256 values are
  `23e3ca8b884205c37325c639c00373cedee92f467a4d7670ea9700c3fe3f0ba0`
  and
  `3f81f41ce459abbaf54560a3a548f2f31c39d034cfc27358f78c70c61afe7a5d`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `3ee8cf51c73a0b06c0f89b965231ffafecf52cc3c96e64d4c9a59262d6389eb5`
  and
  `83491e79d9efcc19077c8ad8780dd00553069ee601dd356f40b6ec6759a748e0`.
  `/rav/datakit-6854-publish-row2886-2234-v691` published the hash-bound
  false-positive record, and
  `/rav/datakit-6854-verify-row2886-2236-v692` independently reread the source
  pair, semantic checkpoint, inspection, deterministic Parquet bytes, and
  completion marker. The manual-record and marker SHA-256 values are
  `5ececc9b7560081630636768c7500885a3b5f827d6cdd78c4fb94c6df3e52c57`
  and
  `168316f249a7b298382f5e94f3e5ce95c8d93b70cf96773f34c4e16fc41933a1`.

- Across the stable 1,503-checkpoint snapshot, all 224 unresolved model
  outcomes are covered by 173 true-duplicate and 51 false-positive manual
  records. The adjusted totals are:

  - baseline: 152,859 pairs, 97,229 false positives, 55,630 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 190,888 pairs, 116,902 false positives, 73,986 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 2,048)`. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T22:29:40Z — 190,760 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2228-v687` and
  `/rav/datakit-6854-audit-next-checkpoints-2229-v688` independently
  revalidated p3 decision-file 104 semantic offsets 1,664 and 1,792. Their 256
  direct baseline pairs contain 115 false positives and 141 true duplicates,
  with no unresolved outcomes. All 537 judgments and request attempts were
  valid on their first attempt. The outcome Parquet SHA-256 values are
  `5c598380de92fb90f1d2661c1229408f71d201efb8346266c2519e379c6989a3`
  and
  `3f82eba957bb76aa28929bae9bb40c5bf6bdc2e14aa35128f3a0e97a6387b0ad`.

- Across the stable 1,502-checkpoint snapshot, all 223 unresolved model
  outcomes remain covered by 173 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 152,731 pairs, 97,171 false positives, 55,560 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 190,760 pairs, 116,844 false positives, 73,916 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 1,920)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:27:00Z — 190,504 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2226-v686` independently
  revalidated p3 decision-file 104 semantic offset 1,536. Its 128 direct
  baseline pairs contain 61 false positives and 67 true duplicates, with no
  unresolved outcomes. All 267 judgments and request attempts were valid on
  their first attempt. The outcome Parquet SHA-256 is
  `a22ed6c19803228ddabd035a584de323eb139c24ea893a704e2f8a04888c05b5`.

- Across the stable 1,500-checkpoint snapshot, all 223 unresolved model
  outcomes remain covered by 173 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 152,475 pairs, 97,056 false positives, 55,419 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 190,504 pairs, 116,729 false positives, 73,775 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 1,664)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:25:30Z — 190,376 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2225-v685` independently
  revalidated three p3 decision-file 104 checkpoints at semantic offsets
  1,152 through 1,408. Their 384 direct baseline pairs contain 357 false
  positives and 27 true duplicates, with no unresolved outcomes. All 784
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `b71deab63151109241ef1445c913a3db6cb9d43799faf25d37a364f369d8c751`,
  `b2aa4965d8ff280d9e276e7d7fcde2a0c402e602aca3eea60e522a0c283c3b8c`,
  and
  `f9e9d017da2e8f55788a4c37cdf77ab11d072f48916869e437d9ec1ab4a068c1`.

- Across the stable 1,499-checkpoint snapshot, all 223 unresolved model
  outcomes remain covered by 173 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 152,347 pairs, 96,995 false positives, 55,352 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 190,376 pairs, 116,668 false positives, 73,708 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 1,536)`. P3's next baseline batch has 128 direct pairs
  requiring 256 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T22:23:20Z — 189,992 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2223-v684` independently
  revalidated six p3 decision-file 104 checkpoints at semantic offsets 384
  through 1,024. Their 768 baseline pairs contain 528 false positives and 240
  true duplicates, with no unresolved outcomes. Three pairs were chunked and
  765 were direct. All 1,637 judgments and request attempts were valid on
  their first attempt.

- The outcome Parquet SHA-256 values are
  `a3df325267590f11cc1cb7f2ff948326101b43130cfc20c193b14157ee6e37e7`,
  `845b86b1dc4ae955b53de95f0277dfc54d48ad1594fc65a38d61ff5d7e13e6c0`,
  `d8e36599082c7a58cefa9e883d04104e1275c22c5c3243c0ba256d1222c13402`,
  `565eb4ec665bf5d907e1862e631c6ea409063c558c90ed2ea3fc66ffab595518`,
  `6161a7a76d45dfd4275930ae8e49ca83a6a5d026b401fd7cc7819362a5c25886`,
  and
  `bfc64e55542d05fd2f97440a1e3d44cd4f2c692361340972dc26b4f65644d051`.

- Across the stable 1,496-checkpoint snapshot, all 223 unresolved model
  outcomes remain covered by 173 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 151,963 pairs, 96,638 false positives, 55,325 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 189,992 pairs, 116,311 false positives, 73,681 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 1,152)`. The pending p0, p1, and p2 batches require at least
  3,194, 3,176, and 2,466 model requests respectively. All four batch-priority
  2-H100 workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T22:21:00Z — 189,224 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2208-v668` independently
  revalidated six p1 decision-file 39 checkpoints spanning semantic offsets
  5,248 through 5,888 and p3 decision-file 104 offset 256. Their 789 pairs
  contain 364 model false positives, 420 model true duplicates, and five
  unresolved outcomes. The arm split is 128 baseline pairs with 88 false
  positives and 40 true duplicates, plus 661 treatment pairs with 276 false
  positives, 380 true duplicates, and five unresolved outcomes. Sixteen pairs
  were chunked and 773 were direct. The 2,893 judgments required 2,906 request
  attempts: 2,887 were valid, and 19 invalid JSON attempts affected seven
  retried judgments.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p1:
    `3f1bf90bb6af180c5b952a0e50358da45a030c17798412fe7f64312c07fddbc9`,
    `19ac1dd32c05c6a00888751df7b07975943aa63c4d5d2afee0c2d22305736b06`,
    `ba34d4848c1f7f0587ac0ea055200012e3d2267d8c7752cd695d0f240138de0b`,
    `f256be537acd6cdde4b4bb4d0dbaca7c9cd7a3cd8e6a57f7513677dd7c940ac5`,
    `972eeda2631512ce87f36497c36ab8a10c2d63f896542e01118e8c70ef50317f`,
    and
    `962083ca34f1e3c69986d83841400b087f807c54e60dc3e265abdf94b0510682`;
  - p3:
    `2f9ffb4166b4a111dec1ca444ec2ff40e78cfb29438b7504edcba80029f2ecfd`.

- Five separate inspection jobs read, hash-bound, and persisted every complete
  unresolved pair. All five are true duplicates:

  - Row 8,460 compares the same Carnegie Mellon cold-study release. The
    2,778-character member adds a PRNewswire dateline, source attribution,
    and malformed logo snippet; the 3,420-character canonical preserves the
    article and adds related-story text and article-derived Q&A. The
    member-only material is source metadata and page chrome rather than a
    distinct fact payload or training example. Character, line, and
    word-sequence similarity are 0.823814, 0.076923, and 0.827309.
  - Row 8,546 compares the same jammer article. The 2,168-character member
    adds a cart notification, share controls, title expansion, and equivalent
    date spelling; the 2,919-character canonical preserves the article and
    adds article-derived Q&A. The member-only material is page chrome rather
    than substantive content. Similarity is 0.828779, 0.488889, and 0.819188.
  - Rows 9,000, 9,033, and 9,034 are complete SFT records with identical
    questions, options, reasoning, conclusions, and answers. Their only
    differences are `\boxed{B}` versus `\boxed{\text{B}}`,
    `\boxed{\text{G}}` versus `\boxed{G}`, and
    `\boxed{\text{B}}` versus `\boxed{B}`. Their member/canonical character
    counts are 7,799/7,806, 5,073/5,066, and 10,035/10,028.

- The inspection jobs, in row order, are
  `/rav/datakit-6854-inspect-row8460-2210-v669`,
  `/rav/datakit-6854-inspect-row8546-2210-v670`,
  `/rav/datakit-6854-inspect-row9000-2211-v671`,
  `/rav/datakit-6854-inspect-row9033-2211-v672`, and
  `/rav/datakit-6854-inspect-row9034-2211-v673`. Their inspection SHA-256
  values are
  `4d1547c350e67d6c0b525f1db87da64285c95af01634934299a2c4675067a8c9`,
  `40d0a942955fdedbb213f48aa885e3e90553b956261139f5c98fed9960f9d8f6`,
  `ad8c82c2a5dd8e884e0a64bc284779712c22b99344a5629518fb28c0648c1175`,
  `c5850d3a88620484de5649db84feaac9101580388bc80e49b1f429b18d144940`,
  and
  `aed04c2cfc0bc4e5c6e82bc7a945e876a6825814bdc88049da664e0192312a25`.
  Their semantic-judgment SHA-256 values are
  `9be6c068401d2d5ea4e7065497c99cedc570e1369a20ce7494857ac21ad2a29b`,
  `f27cf8564136fb4e8d4b1d26b52978ecb7bc81ec7012606fa81139dae2965b2e`,
  `274a571e8cc86891e278e89234a6b4ff1f5f3ff80f6c94f1796a17dabf62a0a2`,
  `e5b965464fc91d72645d979c7d859695f50d8d42309b7a833cd11d63cb9759ef`,
  and
  `f20839152e5f4cb5aa721a1c3c1dfdc82e14160e7928f2b6d4e990c8953b062d`.

- The publish jobs wrote five manual Parquet and completion-marker pairs. In
  row order, their manual SHA-256 values are
  `c06044f0d84494906a1e5a722af611e99765dbe04363132c6f3a75e286835dfc`,
  `5cbc8f0f127a241bbce428cf08e5a3a4b62c65a35fbb7989916059addceff110`,
  `6334cfaa65c8dff1b1ed01492895bcc277c62c3f86cd581337ecd7bbeb14e031`,
  `fbbfc263eb0f1f9db473926398e198c2d669c82b564596e30e04a760f524ba35`,
  and
  `e202a958a00a5d06f0c5a20395e3a120bf4542edb67c87dc88cf021798352336`.
  Their marker SHA-256 values are
  `e04abcbb9f07b941cd14ac83151927433ce2fb4900de178297c80ec345245725`,
  `3e4beb611b5a5aa83863e6e1f8fcdd8d6c14467a9e75604d9524699b304b0753`,
  `c48f76262e0157a4d5689c59fe434a616b742451c0dbb6738c649527002c066a`,
  `35a0c1b1b782160604ce05b67aa05aa00fb33c736bc90817159d95e70c770961`,
  and
  `445b4fda67cd319b640335b9cfb49ae8a83c853d57e69b008a38fac5c2b48ccd`.
  The matching verify-only jobs
  `/rav/datakit-6854-verify-row8460-2217-v679`,
  `/rav/datakit-6854-verify-row8546-2218-v680`,
  `/rav/datakit-6854-verify-row9000-2218-v681`,
  `/rav/datakit-6854-verify-row9033-2219-v682`, and
  `/rav/datakit-6854-verify-row9034-2219-v683` independently fetched and
  verified the exact artifacts.

- Across the stable 1,490-checkpoint snapshot, all 223 unresolved model
  outcomes are covered by 173 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 151,195 pairs, 96,110 false positives, 55,085 true duplicates;
  - treatment: 38,029 pairs, 19,673 false positives, 18,356 true duplicates;
  - combined: 189,224 pairs, 115,783 false positives, 73,441 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(40, 0)`, p2 `(72, 128)`,
  and p3 `(104, 384)`. P3's next baseline batch has 128 direct pairs requiring
  256 minimum model requests. All four batch-priority 2-H100 workers continue
  serving requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T22:07:00Z — 188,435 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2159-v661` independently
  revalidated eight checkpoints: p0 decision-file 8 offset 0, six p1
  decision-file 39 checkpoints spanning offsets 4,480 through 5,120, and p3
  decision-file 104 offset 128. Their 1,024 pairs contain 737 model false
  positives, 285 model true duplicates, and two unresolved outcomes. The arm
  split is 495 baseline pairs with 393 false positives, 100 true duplicates,
  and two unresolved outcomes, plus 529 treatment pairs with 344 false
  positives and 185 true duplicates. Thirty-five pairs were chunked and 989
  were direct. The 7,430 judgments required 7,442 request attempts: 7,424 were
  valid, and 18 invalid JSON attempts affected six retried judgments.

- In checkpoint order, the outcome Parquet SHA-256 values are:

  - p0:
    `6d48ba488a964a93cfc6907bb7f6bc7e4bb3eaf9286fd023e8a2dc0f3ff5d6ea`;
  - p1:
    `10d12fc965d1fa58230c710e0ca61a35099ff99faf5ed94e4b21a45075d9306c`,
    `d58a4f0d425e566c619eab26fb8ad202d06f8f471a87d041e84c3ff6e754b7e3`,
    `f23d3f3d04c4be1691630cec3b2a1c11346e8ef25f3875c99be01bfa451c0cc0`,
    `8834372b01487642a043e39afcf2e6f16b9359b64a03e41aa1edc69114925906`,
    `6ad7e8911ac24fd2c70ba56c4874be3197979edd7baab9b4f1e4eace3fe3b884`,
    and
    `e0d64b99b43a5d940ba04cb9607c015d524bfa73c215751bf4c16d7a5dd477c7`;
  - p3:
    `e18207a770baec184b11931bc4b90b1c5bfff473d2364f91606b0b6d7413dc05`.

- Two separate inspection jobs read, hash-bound, and persisted every complete
  unresolved pair. Both are true duplicates:

  - Row 7,620 is an 8,862-character member and 8,855-character canonical with
    character, line, and word-sequence similarity 0.999605, 0.993333, and
    0.999635. The complete high-rise-design examples differ only by the final
    `\boxed{\text{J}}` versus `\boxed{J}`.
  - Row 7,622 is a 6,923-character member and 6,916-character canonical with
    similarity 0.999494, 0.990741, and 0.999493. The complete
    software-integration examples differ only by the final
    `\boxed{\text{G}}` versus `\boxed{G}`.

- The inspection jobs for rows 7,620 and 7,622 are
  `/rav/datakit-6854-inspect-row7620-2203-v662` and
  `/rav/datakit-6854-inspect-row7622-2203-v663`. Their inspection SHA-256
  values are
  `48d6d3496744d2a71c5d07012bbff793e080f9aeaab3652b94613bcf940d259e`
  and
  `643266da492c3c86075048145bf6a08ed5c983d4caa58fa276ae664bb9f787dc`.
  Their semantic-judgment SHA-256 values are
  `db43afb6bd88e2ef1d37fd58b722cd834b5d6d03d81550d6c51b134a63e75ae8`
  and
  `5f5ed734ccb9ea98207ba62c35f2dd9f9d343dd98ec22d0c51a02a75489905ea`.

- `/rav/datakit-6854-publish-row7620-2205-v665` and
  `/rav/datakit-6854-publish-row7622-2204-v664` wrote the manual Parquet and
  completion-marker pairs. Their manual SHA-256 values are
  `9c988d481a30f5fab34492cfceba0f26cf5f09bdbc34c04a48bfea52e2cc39de`
  and
  `ec5be89b62f8dcf56aa8866823b87276b9040e1fb4f01db1a653455d8e2f81e1`.
  Their marker SHA-256 values are
  `b170cd8bf9a46fc149e275d4601522b253c61332451a0b5cd6ee0985e20fda7d`
  and
  `8a8f0d5333154fccb2316834f52e33aa1b53bdb73dd38d4ee319a25ba4a8060d`.
  `/rav/datakit-6854-verify-row7620-2206-v666` and
  `/rav/datakit-6854-verify-row7622-2206-v667` independently fetched and
  verified the exact artifacts.

- Across the stable 1,483-checkpoint snapshot, all 218 unresolved model
  outcomes are covered by 168 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 151,067 pairs, 96,022 false positives, 55,045 true duplicates;
  - treatment: 37,368 pairs, 19,397 false positives, 17,971 true duplicates;
  - combined: 188,435 pairs, 115,419 false positives, 73,016 true duplicates.

- The next audit frontiers are p0 `(8, 128)`, p1 `(39, 5,248)`, p2
  `(72, 128)`, and p3 `(104, 256)`. The pending p0, p2, and p3 batches require
  at least 3,194, 2,466, and 1,498 model requests respectively. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:58:10Z — 187,411 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2145-v651` independently
  revalidated five p1 decision-file 39 checkpoints at semantic offsets 3,840
  through 4,352 and p2 decision-file 72 offset 0. Their 768 baseline pairs
  contain 559 model false positives, 206 model true duplicates, and three
  unresolved outcomes. Sixteen pairs were chunked and 752 were direct. The
  2,448 judgments required 2,461 request attempts: 2,444 were valid, and 17
  invalid JSON attempts affected eight retried judgments.

- The outcome Parquet SHA-256 values are:

  - p1:
    `a9ffebe483ec353572ba61d5683b2fa3eba50e0b3b11d168f368d85e296aadb5`,
    `2c9aaa2062502a293acc8985566e5166bc2b324da99ae15cfaa567c5f7f7ce7b`,
    `b574c9b69fa450dbf3ba5ca7b95f87d1c2c19ef622205ccb89e5a8ad7b38d543`,
    `31b766f1cb9f1b396a5b301c980eb5c41031d90db1947e65c0ae4d81879db32a`,
    and
    `b195caf3a42201d25bf12e7ee8948b380f9f35b03b5484f91ca0690c0bff0427`;
  - p2:
    `62e9ea1a64b31374d402f357ffca962f6cd130ee05c384c08dba15cf8b09972e`.

- Three separate inspection jobs read, hash-bound, and persisted every
  complete unresolved pair. All three are true duplicates:

  - Row 7,394 is a 15,941-character member and 15,948-character canonical
    with character, line, and word-sequence similarity 0.999780, 0.997067,
    and 0.999820. The complete Davisson-Germer examples differ only by the
    final `\boxed{J}` versus `\boxed{\text{J}}`.
  - Row 7,397 is a 10,666-character member and 10,673-character canonical
    with similarity 0.999672, 0.993976, and 0.999704. The complete
    conditioning examples differ only by the final `\boxed{A}` versus
    `\boxed{\text{A}}`.
  - Row 7,402 is a 9,461-character member and 9,468-character canonical with
    similarity 0.999630, 0.994898, and 0.999665. The complete
    diabetes-prevention examples differ only by the final `\boxed{C}` versus
    `\boxed{\text{C}}`.

- The inspection jobs for rows 7,394, 7,397, and 7,402 are
  `/rav/datakit-6854-inspect-row7394-2147-v652`,
  `/rav/datakit-6854-inspect-row7397-2147-v653`, and
  `/rav/datakit-6854-inspect-row7402-2148-v654`. Their inspection SHA-256
  values are
  `da845e1e0ffb5735d5421dcf4516bd1685c89f272617790a7401166a6a58184a`,
  `113815c967a1d1d6c0a70535ed27114c0dae5c1485468f8a9f13ebfc262f90b6`,
  and
  `9f3fa5737231bf407ffee33f801e5edb8ca90321a9f74abb124089e047faf247`.
  Their semantic-judgment SHA-256 values are
  `40f04cc71d87a91bb2bf3f1b71ef4e09839c9586fed046e3260acf1d3375de42`,
  `c1bdd4365b2501a7cd8c2380b70f9ece9076c4a27d01ed99ff11f5dc17206e33`,
  and
  `cbf08d836aeb70035ac7f79f2a2f818e2168b57e6504d739b930030b445be765`.

- `/rav/datakit-6854-publish-row7394-2154-v655`,
  `/rav/datakit-6854-publish-row7397-2155-v656`, and
  `/rav/datakit-6854-publish-row7402-2155-v657` wrote the three manual Parquet
  and completion-marker pairs. Their manual SHA-256 values are
  `799bc0ccb52991a44c9a3d9602c5de8837176e62ae42c2314f9f81734313d9f6`,
  `d81d3bb64a327c101a17f892ec41000f2a354f031244aee4581bfbfbb41880bf`,
  and
  `08e8430f68498ac9b8dcde0ae63a0f2e082f330664b8b7af617c52e26d8be126`.
  Their marker SHA-256 values are
  `497d7d84e10afc847d16134d322b20be34fd6018372a8b8369c68fab7b08f829`,
  `17da41eb16088369cac5b31c7ac928ff625ed4cd7122e9fc7f0b724a66343544`,
  and
  `fca2ca7f60c292314f4dbaca24f4f4a664d3df72419b8240a42af9263bae1426`.
  `/rav/datakit-6854-verify-row7394-2157-v660`,
  `/rav/datakit-6854-verify-row7397-2156-v659`, and
  `/rav/datakit-6854-verify-row7402-2156-v658` independently fetched and
  verified the exact artifacts.

- Across the stable 1,475-checkpoint snapshot, all 216 unresolved model
  outcomes are covered by 166 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 150,572 pairs, 95,629 false positives, 54,943 true duplicates;
  - treatment: 36,839 pairs, 19,053 false positives, 17,786 true duplicates;
  - combined: 187,411 pairs, 114,682 false positives, 72,729 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 4,480)`, p2 `(72, 128)`,
  and p3 `(104, 128)`. P1's next baseline batch has 128 review units and 256
  minimum model requests with no oversized pairs. P2's next batch has 1,233
  review units and 2,466 minimum requests, including 15 oversized pairs. All
  four batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:43:45Z — 186,643 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2135-v641` independently
  revalidated five p1 decision-file 39 checkpoints at semantic offsets 3,200
  through 3,712. Their 640 baseline pairs contain 390 model false positives,
  247 model true duplicates, and three unresolved outcomes. Two pairs were
  chunked and 638 were direct. All 1,458 judgments and request attempts were
  valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `154e536e59e7e0ce6d3a83362ff3b5b55d7a362aeb1e3f4deb08c9d704f87d88`,
  `1db6f0136193a41ee18d174d4ec4392a3b7e4fac0e48bee32df11e0efecd89b0`,
  `26f66042077ccd5c199e1e1768d7cf320e7bc8314dbee8190a98769127df5915`,
  `ed99c6c2a642ab936edd95fdb12167ddca21eefacbdad796d9519a840a14115b`,
  and
  `8587a8d50618888f304d723a39e4d0c6068e63ca2fd432d146d6dde005ff295a`.

- Three separate inspection jobs read, hash-bound, and persisted every complete
  unresolved pair:

  - Row 5,587 is a 3,932-character member and 4,074-character canonical with
    character, line, and word-sequence similarity 0.757682, 0.48, and
    0.671587. The thermostat pages share a rewritten installation scaffold,
    but the member uniquely references a Y Plan wiring video and a solved
    no-common-wire problem. The canonical instead focuses on hooking a Nest
    thermostat to a humidifier. The manual label is false positive.
  - Row 5,749 is a 5,142-character member and 5,444-character canonical with
    similarity 0.709239, 0.4375, and 0.659280. The builder pages share a
    rewritten scaffold but advertise different services and places. The member
    uniquely describes Langley Corner property refurbishment, planning, and
    architectural services; the canonical describes Moss End kitchen
    renovation. The manual label is false positive.
  - Row 6,004 is a 6,474-character member and 6,338-character canonical with
    similarity 0.817827, 0.472222, and 0.784865. The CheaperForex articles
    share a rewritten scaffold, but the member uniquely claims that advanced
    algorithms and tools analyze trends and maximize profits. It also adds a
    concluding claim about dedication, perseverance, and continuous education.
    The manual label is false positive.

- The inspection jobs for rows 5,587, 5,749, and 6,004 are
  `/rav/datakit-6854-inspect-row5587-2137-v642`,
  `/rav/datakit-6854-inspect-row5749-2137-v643`, and
  `/rav/datakit-6854-inspect-row6004-2137-v644`. Their inspection SHA-256
  values are
  `7c6037be9fafdae65194428ce40d248bef84d029a71a159515c3d48898ad8517`,
  `675de76308a7e3b29e4e6593e6857e1390f2e295fecc93450d45630157b82a2e`,
  and
  `b55d7593b50915fae2e6d47fdafd0e010c39322387b1adf812b4ddcdbbedb21b`.
  Their semantic-judgment SHA-256 values are
  `8b09e6ee09c18341a27c5de9867decb8f29bc0500a80c952d81af17a4cb301a3`,
  `570790d190de7d95e7cc0b5f1ab643e45071a980b8d26a4963b3adc11d91dea5`,
  and
  `cf0e1bde5f37a530f8cdd8e7caed7990cac9105a0dd95972fd529ab7b6303e0c`.

- `/rav/datakit-6854-publish-row5587-2140-v647`,
  `/rav/datakit-6854-publish-row5749-2139-v646`, and
  `/rav/datakit-6854-publish-row6004-2139-v645` wrote the three manual Parquet
  and completion-marker pairs. Their manual SHA-256 values are
  `fc0a8a5aec12c455fade079d706e13fee8e8f9320b6baa188a26c73e65a31351`,
  `9ebb1f92d178a5b2923c4ce23b435d5ac2442447fe9b61b2d3945f60cadbd019`,
  and
  `e63ae61c50600eebf0f6eb594b77b1031d3271295f1c1c95ffaf5020a15e4158`.
  Their marker SHA-256 values are
  `22221fec105b51ca8ecddc50387e09689ef186b3f8ff4367d631a957a24dfef0`,
  `315db723f9b7d62525533cd4261dbb30966d69897947a85d5025ca6eef31bb22`,
  and
  `a94f056827ac27bcf895317d8bacb663413652ce231f4d98f91cac6181328e4c`.
  `/rav/datakit-6854-verify-row5587-2141-v648`,
  `/rav/datakit-6854-verify-row5749-2142-v649`, and
  `/rav/datakit-6854-verify-row6004-2142-v650` independently fetched and
  verified the exact artifacts.

- Across the stable 1,469-checkpoint snapshot, all 213 unresolved model
  outcomes are covered by 163 true-duplicate and 50 false-positive manual
  records. The adjusted totals are:

  - baseline: 149,804 pairs, 95,070 false positives, 54,734 true duplicates;
  - treatment: 36,839 pairs, 19,053 false positives, 17,786 true duplicates;
  - combined: 186,643 pairs, 114,123 false positives, 72,520 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 3,840)`, p2 `(72, 0)`,
  and p3 `(104, 128)`. P1's next baseline batch has 153 review units and 306
  minimum model requests, including one oversized pair. P2's next batch has
  468 review units and 936 minimum requests, including 13 oversized pairs. All
  four batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:34:20Z — 186,003 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2131-v640` independently
  revalidated six p1 decision-file 39 checkpoints at semantic offsets 2,432
  through 3,072. Their 768 baseline pairs contain 371 false positives and 397
  true duplicates, with no unresolved outcomes. Four pairs were chunked and
  764 were direct. All 1,845 judgments and request attempts were valid on their
  first attempt.

- The outcome Parquet SHA-256 values are
  `cf97e49b24c55fab44a4cd41801e556c29a193d2a1548b60ba9e325698989656`,
  `5e4636a36765b07dbf507287e206dfc16fab305efcf7bcc5824f47ba17dfa6d5`,
  `4cba41fca79c2412c59f0b70785edec191e45c7c5f402977f8df0fcc95fd8f85`,
  `596d95cb35e83eb0e0566a41c37220414a638caebfe008ec39d480a9b4159e7c`,
  `5bbe862638e7cd012f57dbd874d5df78f9152a02e8a4d84659acc78e6140c875`,
  and
  `af9b2ed251756783fdba76c3f917d383443892c53629036a66e5547257e8c4b5`.

- Across the stable 1,464-checkpoint snapshot, all 210 unresolved model
  outcomes are covered by 163 true-duplicate and 47 false-positive manual
  records. The adjusted totals are:

  - baseline: 149,164 pairs, 94,677 false positives, 54,487 true duplicates;
  - treatment: 36,839 pairs, 19,053 false positives, 17,786 true duplicates;
  - combined: 186,003 pairs, 113,730 false positives, 72,273 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 3,200)`, p2 `(72, 0)`,
  and p3 `(104, 128)`. P2's next batch has 468 review units and 936 minimum
  model requests, including 13 oversized pairs. All four batch-priority
  2-H100 workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:30:20Z — 185,235 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2115-v624` independently
  revalidated four p1 decision-file 39 checkpoints at semantic offsets 1,920
  through 2,304 and six p2 checkpoints spanning decision-file 71 offsets 5,120
  through 5,760. Their 1,275 pairs contain 531 model false positives, 739 model
  true duplicates, and five unresolved outcomes. The arm split is 512 baseline
  pairs with 250 false positives and 262 true duplicates, plus 763 treatment
  pairs with 281 false positives, 477 true duplicates, and five unresolved
  outcomes. Three pairs were chunked and 1,272 were direct. The 2,841 judgments
  required 2,853 request attempts: 2,836 were valid, and 17 invalid JSON
  attempts occurred across six retried judgments.

- The outcome Parquet SHA-256 values are
  `2639d7e0221305162c4fc4c30222bbd8a8aff01c704f8903950d57aef3f6ee76`,
  `bc615594a8885362b7befc5f24cf97a0736a3edcedfad4e7816f3c7207cee634`,
  `8241bca3907d5662570a058302feb65d1239cbd66684bb70cf8e05a1441a3253`,
  `c4b1f0f7602c38c9a9fa9fa7824cbdede3dc461b3faf2e3736600039fd46b6be`,
  `5eb520a24e97ca4855f1b0b152ed6c794ee58e8b83923f00e3cca09a0a6d8c63`,
  `f0415545a50484f9d644b56c78d0f98a8e6d5016fccfc8b03df5e60956eef11d`,
  `0431ce0a33a0224b54dbbdf862d51c4a86d8a3f5e23ba26627b41329e008efbc`,
  `0ed62fc9042f36eaf907b3d881581e6965f3d068396d77b60e25c5890bce7dcc`,
  `6c2f3ef19b994cb7798fe3b96a626f0cc0d8622ae48d7920432a0e309417fc2c`,
  and
  `f1f36c92bd184a0e2c790b1b43d241acab67c1960d365da5148dbf80340d5a12`.

- Five separate inspection jobs read, hash-bound, and persisted every complete
  unresolved pair:

  - Row 8,461 is a 104-character member and 84-character canonical with
    character, line, and word-sequence similarity 0.851064, 0, and 0.85. They
    are overlapping truncations of the same sentence. The member's trailing
    "basis are made and more" and the canonical's leading "but" do not form a
    separate fact, request, or answer. The manual label is true duplicate.
  - Row 8,640 is a 1,681-character member and 2,064-character canonical with
    similarity 0.768491, 0.076923, and 0.737190. The moving pages share an SEO
    template but describe Mound City, Missouri and Waggoner, Illinois. The
    member uniquely contains zip 64470 and an identity-theft warning; the
    canonical instead contains corporate-moving advice and a quiz. The manual
    label is false positive.
  - Row 9,015 is an 11,214-character member and 11,221-character canonical with
    similarity 0.999688, 0.996441, and 0.999740. The complete hypertension
    answers differ only by `\boxed{B}` versus `\boxed{\text{B}}`. The manual
    label is true duplicate.
  - Row 9,016 is a 5,541-character member and 5,548-character canonical with
    similarity 0.999369, 0.986486, and 0.999389. The complete supply-chain
    answers differ only by `\boxed{C}` versus `\boxed{\text{C}}`. The manual
    label is true duplicate.
  - Row 9,041 is a 9,333-character member and 9,326-character canonical with
    similarity 0.999625, 0.994505, and 0.999673. The complete
    Aristotelian-metaphysics answers differ only by `\boxed{\text{A}}` versus
    `\boxed{A}`. The manual label is true duplicate.

- The inspection jobs for rows 8,461, 8,640, 9,015, 9,016, and 9,041 are
  `/rav/datakit-6854-inspect-row8461-2117-v625`,
  `/rav/datakit-6854-inspect-row8640-2117-v626`,
  `/rav/datakit-6854-inspect-row9015-2118-v627`,
  `/rav/datakit-6854-inspect-row9016-2118-v628`, and
  `/rav/datakit-6854-inspect-row9041-2119-v629`. Their inspection SHA-256
  values, in the same order, are
  `a5bf803c93873587cffc1885de68eaf6addbed1d6399a486b2d0c3e52f1681cb`,
  `e9e147c28dc8b41b5ca85a04fbcb81c584222eceb2916b6170908d78a3668231`,
  `97096eb82c405071e73f76e5354d74f2fc9bd1f88b0f8836dabbd946b9e8d02b`,
  `5373e791bcc0bfd230d25195ad6db76edce01250c33dc4ecba29e996896314a5`,
  and
  `74bf9e3e9ad78ab09c3471926140f4e5a307c5d5d6e7fb29fd6831fe19fbbcec`.
  Their semantic-judgment SHA-256 values are
  `386e76940c2dccfc8e2677b8a02a7ea9d3819ed60f9e703e8cc1690420a618d0`,
  `5e3d1159ff55ba6bf8b42f87ea624a3d8ac4e7bb13e97e25ba318384047fb901`,
  `358a13739d1bb52632b110ac976dd994a7e84b866671e302eee6c19d9021fcd6`,
  `8514811a3ff135198c46d64c3f84f541ccbb614fe169af6059b2e7d4c6aa319c`,
  and
  `b800171ebe1788beb4fcf1d2a9f9953ac0fb045ea2e0b1d0a139d8c4ee16ddc0`.

- The five publish jobs wrote manual Parquet and completion-marker pairs. Their
  manual SHA-256 values are
  `58720e4febef41b5059dbee187c39a7fa7836d197d537ca41bf1113d7fc52978`,
  `0530f1397cbd01e42ffcab9209ff1161297d52c7fe1fbb33acbeba6b41c6d9fd`,
  `330b39e8edd435413ffb69779fbbbfa8ade7e96eae9ff1a661c48d6aeb989158`,
  `8d46281c33926dfbdc94e6269bcb7eaddd90ca676e23dccf72121f81ea623698`,
  and
  `d2d270e5f5f5242b70a1a441b045f7cd258795439eede0e0a4530dd26506bdc0`.
  Their marker SHA-256 values are
  `fc78a3394913fa4bc243a4eb9881d7d77042e84c771d5f2cc65d540c2abc31f2`,
  `7249138731b73fff7a9968f0f57758649602024eb40d8d964389d434538a085f`,
  `d002b65d5ddd83f595620b28166126204ecfec0436f494948392d7ccf43894ed`,
  `61ae7b752b6cfb038a3083532a556a36a621ec5fe7a22650c6af23f3559871b0`,
  and
  `8050c315aa686c07f492a0f4ff934d635355f71f8d5fd5b6f65c29c25707b7dc`.
  `/rav/datakit-6854-verify-row8461-2129-v639`,
  `/rav/datakit-6854-verify-row8640-2128-v638`,
  `/rav/datakit-6854-verify-row9015-2128-v637`,
  `/rav/datakit-6854-verify-row9016-2127-v636`, and
  `/rav/datakit-6854-verify-row9041-2127-v635` independently fetched and
  verified the exact artifacts.

- Across the stable 1,458-checkpoint snapshot, all 210 unresolved model
  outcomes are covered by 163 true-duplicate and 47 false-positive manual
  records. The adjusted totals are:

  - baseline: 148,396 pairs, 94,306 false positives, 54,090 true duplicates;
  - treatment: 36,839 pairs, 19,053 false positives, 17,786 true duplicates;
  - combined: 185,235 pairs, 113,359 false positives, 71,876 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 2,432)`, p2 `(72, 0)`,
  and p3 `(104, 128)`. P1's next baseline batch has 155 review units and 310
  minimum model requests, including one oversized pair. All four
  batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:13:45Z — 183,960 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2107-v620` independently
  revalidated p1 decision-file 39 semantic offsets 1,664 and 1,792 and p2
  decision-file 71 offsets 4,864 and 4,992. Their 512 pairs contain 359 model
  false positives, 152 model true duplicates, and one unresolved outcome. The
  arm split is 256 baseline pairs with 129 false positives, 126 true
  duplicates, and one unresolved outcome, plus 256 treatment pairs with 230
  false positives and 26 true duplicates. One pair was chunked and 511 were
  direct. All 1,086 judgments and request attempts were valid on their first
  attempt.

- The outcome Parquet SHA-256 values are
  `95d209f45615e1d8e4da28e2fc67c61f545f89a9552c919cc56379dd92cdf517`,
  `744780a4ae269d2dc49ec985f14324938f502da73f89b4c9645168126f0c0eb0`,
  `659fa4fef3ed07b5439fd8e43e5562a9d3585edcabab56e3dcf6896296e66da3`,
  and
  `cf31a9805bd9ef72923d5ed95f1895dc5654e00631b83957d760dd6e7350d80a`.

- The unresolved baseline pair is decision-file 39 row 2,836 at semantic
  offset 1,792. `/rav/datakit-6854-inspect-row2836-2108-v621` independently
  read and hash-bound the complete 3,848-character, 33-line member and
  3,643-character, 18-line canonical. Their SHA-256 values are
  `9140b8e4681141a6bbdbc72f689fbdb977b55e4f1b5c9914b215201647da9c08`
  and
  `e1428a7ff1e45add01e277e8bcdd83146e528793decfaa2ccd41fa2a09602ea7`.
  Character, line, and word-sequence similarity are 0.920037, 0.588235, and
  0.911686. Both contain the same University of Toronto salt-tolerance
  article. The canonical ends with two questions derived from that article;
  the member instead ends with a different recycled-water item. Although the
  latter is truncated after "safe to reuse for", its wastewater-treatment and
  conservation proposition is intelligible and absent from the canonical, so
  deleting the member would lose distinct information. The manual label is
  therefore false positive.

- The inspection and semantic-judgment SHA-256 values are
  `91b5313fc13e33a23f0436bd7668f8807546aab871726ee5d4edfbb0bd766856`
  and
  `2a3c0cfca94657c71d0cea4d1fc572c631f8338e6b66f4424890b6e7ef8ce334`.
  `/rav/datakit-6854-publish-row2836-2110-v622` published the manual Parquet
  and marker with SHA-256 values
  `4c8b1b39c6f46329952c714f5aad6141f7da1e932be76efa7491df31fe54b6b4`
  and
  `270d3a6a40abb5035cf19e6ca5d31fae8ea5b066eebb2cbfd6f9ca6f130262b2`.
  `/rav/datakit-6854-verify-row2836-2111-v623` then independently fetched and
  verified both exact artifacts.

- Across the stable 1,448-checkpoint snapshot, all 205 unresolved model
  outcomes are covered by 159 true-duplicate and 46 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,884 pairs, 94,056 false positives, 53,828 true duplicates;
  - treatment: 36,076 pairs, 18,771 false positives, 17,305 true duplicates;
  - combined: 183,960 pairs, 112,827 false positives, 71,133 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,920)`,
  p2 `(71, 5,120)`, and p3 `(104, 128)`. P1's next baseline batch has 152
  review units and 304 minimum model requests, including one oversized pair.
  P2's next treatment batch is entirely direct with 256 minimum requests. All
  four batch-priority 2-H100 workers continue serving requests. Their 12 root,
  broker, and GPU pods remain Ready with zero Kubernetes restarts.

### 2026-07-26T21:05:45Z — 183,448 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2105-v619` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 4,736.
  Its 128 direct-review treatment pairs contain 63 false positives and 65 true
  duplicates, with no unresolved outcomes. All 266 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `490fa39f9bce14873c7101a8e933475492c02eb89703a2587df4e9b49aa86199`.

- Across the stable 1,444-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,628 pairs, 93,926 false positives, 53,702 true duplicates;
  - treatment: 35,820 pairs, 18,541 false positives, 17,279 true duplicates;
  - combined: 183,448 pairs, 112,467 false positives, 70,981 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,664)`,
  p2 `(71, 4,864)`, and p3 `(104, 128)`. P2's next treatment batch is
  entirely direct with 256 minimum model requests. P1 is still processing its
  one oversized pair. All four batch-priority 2-H100 workers continue serving
  requests. Their 12 root, broker, and GPU pods remain Ready with zero
  Kubernetes restarts.

### 2026-07-26T21:04:03Z — 183,320 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2104-v618` independently
  revalidated p1 decision-file 39 semantic offset 1,536 and p2 decision-file
  71 semantic offset 4,608. Their 256 pairs contain 148 false positives and
  108 true duplicates, with no unresolved outcomes. The arm split is 211
  baseline pairs with 125 false positives and 86 true duplicates, plus 45
  treatment pairs with 23 false positives and 22 true duplicates. Four pairs
  were chunked and 252 were direct. All 913 judgments and request attempts
  were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `695a91c3d79a44b511cbaef9fba11ebd90e9fe2550bfe74bb36ac81b260be5c5`
  and
  `3d46aeab10055767abcbe4c8cebc3c014a8c3aba3c8ad84dc2a230103d2d2747`.

- Across the stable 1,443-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,628 pairs, 93,926 false positives, 53,702 true duplicates;
  - treatment: 35,692 pairs, 18,478 false positives, 17,214 true duplicates;
  - combined: 183,320 pairs, 112,404 false positives, 70,916 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,664)`,
  p2 `(71, 4,736)`, and p3 `(104, 128)`. P1's next batch has 137 review units
  and 274 minimum model requests; p2's is entirely direct with 256 minimum
  requests. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T21:02:37Z — 183,064 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2102-v617` independently
  revalidated two p1 decision-file 39 checkpoints at semantic offsets 1,280
  and 1,408. Their 256 direct-review baseline pairs contain 209 false
  positives and 47 true duplicates, with no unresolved outcomes. All 523
  judgments and request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `4e1de3f47304246d1f5fc32b5ecd5e06634586bb13482de1b8c615c85ed9bedb`
  and
  `fec19ab00b194224ef6879c2b117d2cddc8bdf853ea0cb6f04f1b45166a8e1f4`.

- Across the stable 1,441-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,417 pairs, 93,801 false positives, 53,616 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 183,064 pairs, 112,256 false positives, 70,808 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,536)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch is entirely direct
  with 256 minimum model requests. P2 remains in its 293-review-unit oversized
  batch. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T21:01:00Z — 182,808 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2101-v616` independently
  revalidated the p1 decision-file 39 checkpoint at semantic offset 1,152.
  Its 128 direct-review baseline pairs contain 120 false positives and eight
  true duplicates, with no unresolved outcomes. All 259 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `c2612377061928bfa4539c1e49d1e83fd116ec926c56975d4852ef16711358c3`.

- Across the stable 1,439-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,161 pairs, 93,592 false positives, 53,569 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 182,808 pairs, 112,047 false positives, 70,761 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,280)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch is entirely direct
  with 256 minimum model requests. P2 remains in its 293-review-unit oversized
  batch. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:59:15Z — 182,680 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2059-v615` independently
  revalidated the p1 decision-file 39 checkpoint at semantic offset 1,024.
  Its 128 direct-review baseline pairs contain 117 false positives and 11 true
  duplicates, with no unresolved outcomes. All 263 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `5d109f8e093e36d3b0e3c015aeb775606ed19936d5ac339c88597f8afcf89df6`.

- Across the stable 1,438-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 147,033 pairs, 93,472 false positives, 53,561 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 182,680 pairs, 111,927 false positives, 70,753 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,152)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch is entirely direct
  with 256 minimum model requests. P2 remains in its 293-review-unit oversized
  batch. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:57:37Z — 182,552 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2057-v614` independently
  revalidated the p1 decision-file 39 checkpoint at semantic offset 896. Its
  128 baseline pairs contain 75 false positives and 53 true duplicates, with
  no unresolved outcomes. Two pairs were chunked and 126 were direct. All 308
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `c4ac6a7af2c0d66958c96018083fd4e6741be8b3a6e15ba5e3fd3ea29ae527c5`.

- Across the stable 1,437-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 146,905 pairs, 93,355 false positives, 53,550 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 182,552 pairs, 111,810 false positives, 70,742 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 1,024)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch is entirely direct
  with 256 minimum model requests. P2 remains in its 293-review-unit oversized
  batch. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:56:00Z — 182,424 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2055-v613` independently
  revalidated two p1 decision-file 39 checkpoints at semantic offsets 640 and
  768. Their 256 direct-review baseline pairs contain 170 false positives and
  86 true duplicates, with no unresolved outcomes. All 529 judgments and
  request attempts were valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `cd6a740389fdbcf8c31b86f59d3b8fc5862bec3c34fcc442fc5cf1a601aa6611`
  and
  `64dcc7e067124faa1b0c77ba634469d1c8d658ea06b883974e725f1751544ac7`.

- Across the stable 1,436-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 146,777 pairs, 93,280 false positives, 53,497 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 182,424 pairs, 111,735 false positives, 70,689 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 896)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch has 146 review units
  and 292 minimum model requests. P2 is still processing its two oversized
  pairs within a 293-review-unit batch. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:54:15Z — 182,168 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2050-v609` independently
  revalidated p1 decision-file 39 semantic offsets 384 and 512 and p2
  decision-file 71 offsets 4,352 and 4,480. Their 512 baseline pairs contain
  350 model false positives, 161 model true duplicates, and one unresolved
  outcome. Two pairs were chunked and 510 were direct. P1's 618 judgments were
  all valid on their first attempt. P2 had nine invalid JSON attempts across
  three retried judgments, all belonging to the unresolved pair; the remaining
  519 attempts were valid.

- The outcome Parquet SHA-256 values are
  `a3e4dd2ec7cf8feaae4ef181c1ea71c6fe7b43515892497c392cda37f08bb534`,
  `3af023a4a7856cb3e380ec932e1861e53e4db6df1d066a9cb822b65e87e8a461`,
  `bd8b66ad51759699a3d76946d5c93305770b1ddf94b12f84cce95b3784a1b424`,
  and
  `9dbc9ab7b44ee10aa1e0ec11bff83bedb5020b4a3f990525679c59328fc25fec`.

- `/rav/datakit-6854-inspect-row7633-2051-v610` read both complete SFT texts
  and all semantic evidence. The 5,712-character member and 5,705-character
  canonical each have 76 lines and contain the same customer-loyalty problem,
  choices, reasoning, and answer. Their character, line, and word-sequence
  similarities are 0.999387, 0.986842, and 0.999409. The complete unified diff
  has one changed line: the member ends with `\boxed{\text{C}}` and the
  canonical with `\boxed{C}`. Deleting the member loses no substantive
  content, so the pair is a true duplicate. The member/canonical text SHA-256
  values are
  `061f4f2164a2ccf82fa361a77727c8708d50a566157374a3d442f2c778bbf307`
  and
  `124f1d5603ae142c41c19533240c29208e477491598464a8192b396e06ad179e`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `68468090ebe74658f1257ca62557fa046cbe063efb8894aa2569a28e09fd1103`
  and
  `da3afd678852aa0d82d8a2c81b5dbb97f86dcd362e20be68d8284913c1294c57`.
  `/rav/datakit-6854-publish-row7633-2053-v611` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row7633-2054-v612` independently reread the source
  pair, semantic checkpoint, inspection, manual Parquet bytes, and completion
  marker. The manual-record and marker SHA-256 values are
  `2ccd1c7f8c55130bd791c61f2e43b27ec01bc31b5686f4f8b3367931c37b9866`
  and
  `2bd7f4f589d33ef05b3b7b9faf957c7c7b64a52e68822655261e83f0aca82e7c`.

- Across the stable 1,434-checkpoint snapshot, all 204 unresolved model
  outcomes are covered by 159 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 146,521 pairs, 93,110 false positives, 53,411 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 182,168 pairs, 111,565 false positives, 70,603 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 640)`,
  p2 `(71, 4,608)`, and p3 `(104, 128)`. P1's next batch is entirely direct
  with 256 minimum model requests; p2's has 293 review units and 586 minimum
  requests. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:49:04Z — 181,656 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2045-v605` independently
  revalidated p1 decision-file 39 semantic offset 256 and p2 decision-file 71
  semantic offset 4,224. Their 256 baseline pairs contain 174 model false
  positives, 81 model true duplicates, and one unresolved outcome. Ten pairs
  were chunked and 246 were direct. P1's 917 judgments were all valid on their
  first attempt. P2 had four invalid JSON attempts across two judgments; one
  judgment recovered on retry, while the unresolved pair's loss and tiebreak
  judgments remained valid and agreed.

- The outcome Parquet SHA-256 values are
  `083322947160ff1a7f20930f16c55c757ffa6fb60bc3bd6630de80a800f9eea7`
  and
  `ecfef7256e657ed99db098ae9dc40896881770f864ac220a7cb2c3170d9aa593`.

- `/rav/datakit-6854-inspect-row7349-2046-v606` read both complete SFT texts
  and all semantic evidence. The 16,823-character member and
  16,830-character canonical each have 402 lines and contain the same
  employment-law problem, choices, reasoning, and answer. Their character,
  line, and word-sequence similarities are 0.999792, 0.997512, and 0.999820.
  The complete unified diff has one changed line: the member ends with
  `\boxed{H}` and the canonical with `\boxed{\text{H}}`. Deleting the member
  loses no substantive content, so the pair is a true duplicate. The
  member/canonical text SHA-256 values are
  `d70e30c84170a94c6851efdf8acab1290465f24d73264ad584e2d34b60125bc8`
  and
  `28a91bbced0fc95df731b39fe14e2911e66e6b950ce406cc1fcdef2d35b631ca`.

- The persisted inspection and semantic-judgment SHA-256 values are
  `5a97cf8d0e2f75a2f82de54a5f9eb0b41aa37b56defc29082c631b760563c54d`
  and
  `7c40ad05dead83c8b6b7100da984f27d679f49117161e84d8f539af0ebed0717`.
  `/rav/datakit-6854-publish-row7349-2048-v607` published the hash-bound
  true-duplicate record, and
  `/rav/datakit-6854-verify-row7349-2049-v608` independently reread the source
  pair, semantic checkpoint, inspection, manual Parquet bytes, and completion
  marker. The manual-record and marker SHA-256 values are
  `5154db2b48e43b6774df3ed04b8a718382a4def69ae69a9938268b13f9d787ce`
  and
  `dc5c568f5146c2308d1ba0c544356599237d149c0641392bfd48720347972978`.

- Across the stable 1,430-checkpoint snapshot, all 203 unresolved model
  outcomes are covered by 158 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 146,009 pairs, 92,760 false positives, 53,249 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 181,656 pairs, 111,215 false positives, 70,441 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 384)`,
  p2 `(71, 4,352)`, and p3 `(104, 128)`. P1's next batch has 157 review units
  and 314 minimum model requests; p2's is entirely direct with 256 minimum
  requests. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:43:58Z — 181,400 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2043-v604` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 4,096.
  Its 128 direct-review baseline pairs contain 102 false positives and 26 true
  duplicates, with no unresolved outcomes. All 274 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `63761f007edce2acc0943ab0e2533efb4de9156581c8875c559c83c73e836f18`.

- Across the stable 1,428-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 145,753 pairs, 92,586 false positives, 53,167 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 181,400 pairs, 111,041 false positives, 70,359 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 256)`,
  p2 `(71, 4,224)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:42:28Z — 181,272 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2042-v603` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 3,968.
  Its 128 direct-review baseline pairs contain 98 false positives and 30 true
  duplicates, with no unresolved outcomes. All 264 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `363bfab4c17f52b42f53371439bc9b7db9a4cb37a5fa00da8be884df68021954`.

- Across the stable 1,427-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 145,625 pairs, 92,484 false positives, 53,141 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 181,272 pairs, 110,939 false positives, 70,333 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 256)`,
  p2 `(71, 4,096)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:41:07Z — 181,144 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2041-v602` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 3,840.
  Its 128 direct-review baseline pairs contain 90 false positives and 38 true
  duplicates, with no unresolved outcomes. All 264 judgments and request
  attempts were valid on their first attempt. The outcome Parquet SHA-256 is
  `5f9861e0fda2746b87d0af6d7985180f283c7251c43bf2b05389dfe8b5035c77`.

- Across the stable 1,426-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 145,497 pairs, 92,386 false positives, 53,111 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 181,144 pairs, 110,841 false positives, 70,303 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 256)`,
  p2 `(71, 3,968)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:39:42Z — 181,016 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2040-v601` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 3,712.
  Its 128 baseline pairs contain 100 false positives and 28 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  320 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `0dec78b0df00ecc55c85251b3b5def1c481d2b0e49f3925d21a2edee0efd4c9a`.

- Across the stable 1,425-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 145,369 pairs, 92,296 false positives, 53,073 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 181,016 pairs, 110,751 false positives, 70,265 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 256)`,
  p2 `(71, 3,840)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:38:17Z — 180,888 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2037-v600` independently
  revalidated p1 decision-file 39 semantic offset 128 and p2 decision-file 71
  semantic offset 3,584. Their 256 baseline pairs contain 185 false positives
  and 71 true duplicates, with no unresolved outcomes. Fourteen pairs were
  chunked and 242 were direct. All 1,395 judgments and request attempts were
  valid on their first attempt.

- The outcome Parquet SHA-256 values are
  `40d52700fea873af1bf43a7138a4cf35d66b8cc51b20f95eed80eedb469ed7b8`
  and
  `1c913de5817c38bca6400f9516cd6fa78c6434850928f72f6ef118f3e8ef8f93`.

- Across the stable 1,424-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 145,241 pairs, 92,196 false positives, 53,045 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 180,888 pairs, 110,651 false positives, 70,237 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 256)`,
  p2 `(71, 3,712)`, and p3 `(104, 128)`. P1's next batch has 454 review units
  and 908 minimum model requests; p2's has 157 review units and 314 minimum
  requests. All four batch-priority 2-H100 workers continue serving requests.
  Their 12 root, broker, and GPU pods remain Ready with zero Kubernetes
  restarts.

### 2026-07-26T20:36:18Z — 180,632 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2035-v599` independently
  revalidated three p2 decision-file 71 checkpoints at semantic offsets 3,200,
  3,328, and 3,456. Their 384 baseline pairs contain 232 false positives and
  152 true duplicates, with no unresolved outcomes. Two pairs were chunked and
  382 were direct. All 905 judgments and request attempts were valid on their
  first attempt.

- The outcome Parquet SHA-256 values are
  `8256c670fd47b695edd0654c01bfa59ecc1a5641db895ad75345d906dc7a42f1`,
  `b853be8a7bb1148186b30bbbcba6af0ec4b9aead78e1452d0cb5260c42abfe9a`,
  and
  `d00fece38ac20830561f0629b02a36ae932eef80692d2fb094d0562e9b75290e`.

- Across the stable 1,422-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 144,985 pairs, 92,011 false positives, 52,974 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 180,632 pairs, 110,466 false positives, 70,166 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 3,584)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:34:33Z — 180,248 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2031-v595` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 3,072.
  Its 128 direct-review baseline pairs contain 52 model false positives, 75
  model true duplicates, and one unresolved outcome. All 279 judgments and
  request attempts were valid on their first attempt. The outcome Parquet
  SHA-256 is
  `3103cb0d3cfff42ca357b9d9bd6fa301022b36890ad882317c4d5b3dbd83b521`.

- `/rav/datakit-6854-inspect-row5061-2032-v596` read both complete
  same-source college-SEO texts and all three model judgments. The
  1,417-character member and 827-character canonical have SHA-256
  `e727c8ed9315faabe77bd98dcf796ef911049cff108e6b059d81812def15cb99`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
  Their complete character, line, and word-sequence similarities are
  0.595365, 0.250000, and 0.525074. The pair shares admissions, employer,
  biomedicine, and coursework scaffolds, but the member separately states
  that an advanced degree lasts a lifetime and merits several years of work,
  business internships improve prospects and provide experience, and
  certificate and continuing-adult-education programs are available. Those
  additions exceed low-value institution and program slots, so deleting the
  member loses distinct propositions and the pair is a false positive.

- The persisted inspection and semantic-judgment SHA-256 values are
  `d7c6b54cc26e0241b446c42c3327a792a09f0589ac7315a24d64f685c2358c2d`
  and
  `9cf29b8a016af473ecbe264bae6607568753d0898f9b930123ab478d18963e0e`.
  `/rav/datakit-6854-publish-row5061-2033-v597` published the hash-bound
  false-positive record, and
  `/rav/datakit-6854-verify-row5061-2034-v598` independently reread the source
  pair, semantic checkpoint, inspection, manual Parquet bytes, and completion
  marker. The manual-record and marker SHA-256 values are
  `b782acb3603cc5cd921c558f567c52d87dd7a943ba55ecafb7037f7c0d24d69e`
  and
  `1f2057890c5b7c422369d3ec9be07ff3672aa8e87c5bf26a415c3dcc019e6037`.

- Across the stable 1,419-checkpoint snapshot, all 202 unresolved model
  outcomes are covered by 157 true-duplicate and 45 false-positive manual
  records. The adjusted totals are:

  - baseline: 144,601 pairs, 91,779 false positives, 52,822 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 180,248 pairs, 110,234 false positives, 70,014 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 3,200)`, and p3 `(104, 128)`. P2's next batch has 153 review units
  and 306 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:29:40Z — 180,120 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2029-v594` independently
  revalidated two p2 decision-file 71 checkpoints at semantic offsets 2,816
  and 2,944. Their 256 baseline pairs contain 110 false positives and 146 true
  duplicates, with no unresolved outcomes. Two pairs were chunked and 254 were
  direct. All 706 judgments and request attempts were valid on their first
  attempt.

- The outcome Parquet SHA-256 values are
  `ede7409c24b186c16210803168445c8c361f2afba6635d09ee5399492177007c`
  and
  `63f961cf79f6488990a289ae76cd3232f39907a1faaf66b31dd1fe0a1c3d03d1`.

- Across the stable 1,418-checkpoint snapshot, all 201 unresolved model
  outcomes are covered by 157 true-duplicate and 44 false-positive manual
  records. The adjusted totals are:

  - baseline: 144,473 pairs, 91,726 false positives, 52,747 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 180,120 pairs, 110,181 false positives, 69,939 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 3,072)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

### 2026-07-26T20:27:39Z — 179,864 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2023-v590` independently
  revalidated two p2 decision-file 71 checkpoints at semantic offsets 2,560
  and 2,688. Their 256 direct-review baseline pairs contain 124 model false
  positives, 131 model true duplicates, and one unresolved outcome. All 552
  judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 values are
  `10af3b1415d40d80502d39c1b41f466247959320e0d871e502b78e1878a3a7a7`
  and
  `ae90356deb04dfa81d5b91cdc9b5b838e00686e8321bf30c96531aec93b71e90`.

- `/rav/datakit-6854-inspect-row4167-2024-v591` read both complete
  same-source college-SEO texts and all three model judgments. The
  1,504-character member and 827-character canonical have SHA-256
  `d9832ecb0c10eeb4aa6bafa6c0c4dea06abbcc1e278869618a061be5d4650912`
  and
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
  Their complete character, line, and word-sequence similarities are
  0.606607, 0.222222, and 0.583333. The pair shares admissions, employer,
  biomedicine, and coursework scaffolds, but the member adds complete claims
  about on-campus recruiting, biotechnology registration and
  mechanical-engineering requirements, the Army Corps of Engineers, and
  browsing biotechnology programs. Those additions exceed low-value
  institution and program slots, so deleting the member loses distinct
  propositions and the pair is a false positive.

- The persisted inspection and semantic-judgment SHA-256 values are
  `e7614f3817d61504522fdffaf47d6042015a3181aaac3c4a2c5ccea01cda569b`
  and
  `16a5f5058c9bb0a8da015fee257544daf6f826fe4527d2a5d120d46a13aaab85`.
  `/rav/datakit-6854-publish-row4167-2026-v592` published the hash-bound
  false-positive record, and
  `/rav/datakit-6854-verify-row4167-2026-v593` independently reread the source
  pair, semantic checkpoint, inspection, manual Parquet bytes, and completion
  marker. The manual-record and marker SHA-256 values are
  `b7527069504e1baec8ae5980484fd76aaeccb917c2513d474d2f37a461f22fcc`
  and
  `a4b71fd5a52ebfb32b02474c254b24ea5f5e7bfbffd84cf2f503cf28f72eada4`.

- Across the stable 1,416-checkpoint snapshot, all 201 unresolved model
  outcomes are covered by 157 true-duplicate and 44 false-positive manual
  records. The adjusted totals are:

  - baseline: 144,217 pairs, 91,616 false positives, 52,601 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,864 pairs, 110,071 false positives, 69,793 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,816)`, and p3 `(104, 128)`. P2's next batch has 202 review units
  and 404 minimum model requests. All four batch-priority 2-H100 workers
  continue serving requests. Their 12 root, broker, and GPU pods remain Ready
  with zero Kubernetes restarts.

### 2026-07-26T20:20:37Z — 179,608 pairs verified

- `/rav/datakit-6854-audit-next-checkpoints-2020-v588` independently
  revalidated the p2 decision-file 71 checkpoint at semantic offset 2,432.
  Its 128 baseline pairs contain 66 false positives and 62 true duplicates,
  with no unresolved outcomes. One pair was chunked and 127 were direct. All
  364 judgments and request attempts were valid on their first attempt. The
  outcome Parquet SHA-256 is
  `9b40fa2085dc60c421f68c8470797b4fac07805971480e36484927ac266a23fc`.

- Across the stable 1,414-checkpoint snapshot, all 200 unresolved model
  outcomes are covered by 157 true-duplicate and 43 false-positive manual
  records. The adjusted totals are:

  - baseline: 143,961 pairs, 91,491 false positives, 52,470 true duplicates;
  - treatment: 35,647 pairs, 18,455 false positives, 17,192 true duplicates;
  - combined: 179,608 pairs, 109,946 false positives, 69,662 true duplicates.

- The next audit frontiers are p0 `(8, 0)`, p1 `(39, 128)`,
  p2 `(71, 2,560)`, and p3 `(104, 128)`. P2's next batch is entirely direct
  and requires 256 minimum model requests. All four batch-priority 2-H100
  workers continue serving requests. Their 12 root, broker, and GPU pods
  remain Ready with zero Kubernetes restarts.

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
