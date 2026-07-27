# Issue #6854 fuzzy-dedup false-positive validation, volume 3

Status: running

Coordinating issue: https://github.com/marin-community/marin/issues/6854

Previous volume:
[issue-6854-dedup-validation-2.md](issue-6854-dedup-validation-2.md).

## Objective

Continue the exhaustive baseline-versus-treatment semantic review on the full
115-source, 103,716,988-document DataKit 100B testbed. The pinned arms,
evaluation contract, artifact roots, and results through 286,326 reviewed pairs
are recorded in the previous volumes.

## Experiment log

### 2026-07-27T15:56:02Z — 288,118 pairs verified; all 339 ambiguities covered

- `/rav/rav-datakit-6854-reconcile-manual-1555-v1328` reconciled 2,269
  semantic checkpoints and all 339 unresolved model outcomes. No manual
  outcome is missing. The snapshot covers 288,118 of 755,281 candidates
  (38.15%). Applying 74 false-positive and 265 true-duplicate manual decisions
  gives:

  - baseline: 230,087 pairs, 146,087 false positives, and 84,000 true
    duplicates;
  - treatment: 58,031 pairs, 30,063 false positives, and 27,968 true
    duplicates;
  - combined: 176,150 false positives and 111,968 true duplicates.

  False-positive rates are 63.4921% for baseline and 51.8051% for treatment,
  an 11.6870 percentage-point reduction. The immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1555.json`
  with SHA-256
  `b4286d3743496ab94a42fba0d265121d7f095a1d3755b466e95968840f19b807`.
- Seven audit waves independently reread the 14 newly reconciled checkpoints.
  Their 1,792 baseline pairs contain 1,053 model false positives, 736 model
  true duplicates, and three unresolved outcomes. All 5,258 judgments were
  valid on their first attempt; 24 pairs were chunked and 1,768 were direct.
  Outcome Parquet SHA-256 values are:

  - p0 decision-file 12 offsets 128, 256, and 384:
    `f12b2db8e4de8fb854d79d3d67a1a2efa6a9319396dab474e9277afa07825b22`,
    `d130ce8c098dc41166a9aabfed7ff1f2c4c5932817a59114082f53cb4c193444`,
    and
    `040ac4b740b85b08df4d41e2cacf999fe2fa40f673a550a910f02bd045bef933`;
  - p3 decision-file 108 offsets 2,560 through 3,072:
    `e15ff850e95dacc681ac4e6d8aa7279a13d745f5813e77018c9f44ac3f2aed75`,
    `2b3fa72df890ec4f44c9f82db487753488f90e49c764356cae8efa4a4f9ebe3c`,
    `59c55910ecfe84a6dddfbf6d3e2f71db16bc8788f269a57688d8d598f964de91`,
    `345cf3bfb42ee78a3339a07166f655f0b9b8d8dafbe973f3348a744c579a281e`,
    and
    `6a4393b091adb6c5d8308a557c5ebbb7739aff7b140670ef11f4fbd2f66ad196`;
  - p3 offsets 3,200 through 3,840:
    `0f83832fd3592d3670db2246feef79aefa1bfaeaf4cd7b40ce80a30273df9149`,
    `df1eefeb505978377030ed3ded6fd22e286a9981b60ff8de266f3d22736c49f6`,
    `35970c8f9a819134a2865944ba0a53d3c7a5027d6aabbd56fc2e8939a8237c66`,
    `607381121f1686d17fdd055ffe33d56be9f1b49b11239e1eb761f37014ab49e3`,
    `598b30d86f7573fa55bb96e51d81098723d120c266167b7224b96d9184addb23`,
    and
    `ef0ee7ba7b778401c565de5d86483732c377329287ba8471c6edee93dcb6eddb`.
- Full-text review resolves rows 5,291 and 5,362 as false positives against
  the same 827-character college-SEO canonical. Row 5,291 uniquely contains
  chemistry recruiting, registration-deadline, organic-lab,
  admissions-request, and program-browsing statements. Row 5,362 adds
  certificate and adult-education content plus a corporate-internship claim.
  Those propositions exceed low-value institution and program slots.
  Member SHA-256 values are
  `9427bd0e75aa8e9d144657d7a8e4e3637ceb4a7b157bf752c1b99a9a9c776b9c`
  and
  `e2151417e2ed14b2455a36144e40994bf393b5e3f44dc91088954ed579099d05`;
  the shared canonical SHA-256 is
  `4893ce3e5b496530e4b22e31e43f7505e575b1b5c377c62acb8ae20ccb5c1c4e`.
  Inspection SHA-256 values are
  `8fc3cd445cc8d9f78a8c2d142e7356e9f93867bb1e40177ae8fefa74c0028f1b`
  and
  `d31d00eb6b8a49b234146d6b5d7a667e335aca4d589104a1145b3a782083d352`;
  semantic-judgment SHA-256 values are
  `c01f242737423ad14bac30f109e9b952466f5ce45572f10ebb7407809d232ad6`
  and
  `0588bc3265488c00eab657708f7ef2b6b163c1664d357d82bc59147cfb2772de`;
  manual-Parquet SHA-256 values are
  `fddf95bab54a95c4ac695a291fe08ab8e81436a897717d71427ed93016080387`
  and
  `727c34bd2045be6a9c4e4fd5472daee591aba8adbbde59f50a388e49bdb99130`;
  marker SHA-256 values are
  `d22980413b2502d8f6b3b9af99454ac0198919a197ad485fdd9d2ca6f2f589d4`
  and
  `748c80416266eb8001cff9ca5970a1452d3cb76ddb646a093bf8b57a2a982d04`.
- Full-text review resolves row 6,355 as a true duplicate. Every member event,
  entity, quotation, action, and conclusion appears in the canonical. Two
  model votes incorrectly attributed the canonical-only `non-issue` sentence
  to the member. Remaining differences are wording and sentence grouping.
  Member/canonical SHA-256 values are
  `5770609ebbd1ab6f3e58204170eedcb2c9cd4bf8f6bc301ef265595ad674382a`
  and
  `27834970b22ef0d949c0c69a49ab229f94534d40aa39a9580bc029dd0053ba47`.
  Inspection, semantic-judgment, manual-Parquet, and marker SHA-256 values are
  `bc48e6ad83e41378398abd881946e334c9f15070b03b762ac6d532cc4d5f5fad`,
  `05e19942d445f3e47e9fde20febef16efeb459b31b8e79b7d7eba371f40ba204`,
  `e404e67200929943ee4ad3b231d384fa4cb77d96a3c4d7b9923e1cbcf31b5c75`,
  and
  `a8e82c946073940aa6e63c2fe04bb9416bab2275875048d135f112530633a147`.
- Next audit frontiers are p0 `(12, 512)`, p1 `(44, 0)`, p2 `(76, 128)`,
  and p3 `(108, 3,968)`. P1 remains on its 122,322,523-character checkpoint
  and p2 on its 21,249,569-character checkpoint. All four batch-priority
  2-H100 jobs continue running; their 12 pods are Ready with zero restarts.
