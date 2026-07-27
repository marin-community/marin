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

### 2026-07-27T16:12:08Z — 291,062 pairs verified; four formatting-only duplicates

- `/rav/rav-datakit-6854-reconcile-manual-1610-v1342` reconciled 2,292
  semantic checkpoints and all 343 unresolved model outcomes. The snapshot
  covers 291,062 of 755,281 candidates (38.54%). Applying 74 false-positive
  and 269 true-duplicate manual decisions gives:

  - baseline: 232,550 pairs, 147,851 false positives, and 84,699 true
    duplicates;
  - treatment: 58,512 pairs, 30,418 false positives, and 28,094 true
    duplicates;
  - combined: 178,269 false positives and 112,793 true duplicates.

  False-positive rates are 63.5782% for baseline and 51.9859% for treatment,
  an 11.5922 percentage-point reduction. The immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1610.json`
  with SHA-256
  `5efa577e93515c7f85a670bbc491ef53170733aadd0b2dff60ad49de47a6b1c4`.
  The known historical/shadow artifact counts remain unchanged: two duplicate
  manual-Parquet keys, three outcome-hash mismatches, and 146 orphan manual
  Parquets. Marker-bound decisions are complete and internally consistent.
- `/rav/rav-datakit-6854-audit-all-checkpoints-1602-v1329` independently
  reread nine checkpoints containing 1,152 baseline pairs: 747 model false
  positives, 401 model true duplicates, and four unresolved outcomes. Twelve
  pairs were chunked and 1,140 direct. There were 3,141 judgments in 3,161
  attempts, of which 3,133 were valid; 11 judgments needed retries. Outcome
  SHA-256 values are:

  - p0 decision-file 12 offsets 512 through 896:
    `418289559d3757c0333fdae641fa13e82561ce4645131c2f091e0b7672aab87c`,
    `be13a1bd9df4dc0cf461ec0788f8b4f97faf6e1a992be6dd2ed0fb0a976e157c`,
    `cec09ab6a72b2ba61d77be5f63a84e9e449ae203d61148bc69d437b5edb11c73`,
    and
    `824bc04f1a5e674a2ff1ad535ccbad9867b9f597b77e6e035c01ebb7da84a2be`;
  - p2 decision-file 76 offset 128:
    `7a566327b42bfa5220f3b1b347d1bcd4d5705d2e9c76421be886ab312be51959`;
  - p3 decision-file 108 offsets 3,968 through 4,352:
    `6773d3e117de158609cf113f3fe07999583228acae635c3bc145a478524c0043`,
    `54adb8b4548ec53120db3d7b8686d410cd78351fd3d39165fbe4c5b0b36e56fc`,
    `dd4d91f465ace99147ee7a44d5d56b4fa67b5856210f9df4b0be82c78d667f68`,
    and
    `96c5d986586753d2fb0e88c38db7432880dbe40e6d1e4070e51b3b0ce20a24ce`.
- Full-text inspection resolves all four ambiguities as true duplicates.
  Across rows 7,333, 7,394, 7,606, and 7,612, respectively, the pairs have
  194, 94, 128, and 68 lines. Every line is identical except the final answer,
  where `\boxed{X}` and `\boxed{\text{X}}` express the same answer. Exact
  sequence comparison covered the complete texts. Member/canonical,
  inspection, semantic-judgment, manual-Parquet, and marker SHA-256 values are:

  - row 7,333:
    `f38e27997016d4074b133442e3f06eba0d46579124051c0f031eef6da2a150cb` /
    `62342f2aefc77e0b673bc8a6f01e6ffa0fb2a79b08b96e0c08724e0c87ddf32f`,
    `7ef7559dadb8c8edbb16142977e533ec61e8113eb5a45f796332e035a1c0fa9a`,
    `2e2740f2f23da8cf730823d78db1fd833142eb0435495d4fb0c9fd72485c1351`,
    `29bbaa77e7a330ddff7328e94bb527322656157699f86fee9f966f431438be82`,
    and
    `f8796bb4a26d532feff884dbb7bb67edecb64e4cc25c221679f90ddabf0937b5`;
  - row 7,394:
    `c22719c5816390b5391f567a77686b0118ed190568c841e48c348e1a35129a8f` /
    `348bdfa22c2dcd497668011576605763e659009aca106f451788004f09126e25`,
    `0b505ad719fe13d9a81299193271fefe065e61c1c1b7789ba642f0548efb6d54`,
    `599f5caf5f76c164bff5c7262a4c54df014d5c1721b29fab52f479d4032384f0`,
    `58cf04415b9d4ff1ce04d312a9dddf04eb96466eb89ff61298c7773a62c9fc95`,
    and
    `0b58f9232a89076cfd15f8eb8f270795c42d655e4287cd2f08f1554e79617af1`;
  - row 7,606:
    `15cfb3b06942bca56743ddda28c1738c9174eae23806985c794500e4e2c9cdd2` /
    `f36a85dfaf06ed612f97588d8f4107dd63f8454cf42c7ab0d7fe2113ff82f465`,
    `e9ddcb76c1d0eca7b87f33119dfa56f4927b279a4863cd24224aef2937d410d0`,
    `46f45ef41f737eab52107687ee3c7d7bf40cbd1e7928dcbb3fab65ec3d395def`,
    `88a40cb888af7b06c8a249b5b801a8437a5efc670494e0930556148c98fd1ee7`,
    and
    `712ff088011d5e24b332777d52cb55b967648b1cb7991b66029cd2d429f57cba`;
  - row 7,612:
    `ca36448949bb1cdce13ff7827866b1f42716e075e655bc4bd7d50d5b9e536d13` /
    `7d3df671af58301d5f4f864c5b341b64dd0eaabcde19ce8f6b5d99a825e206f6`,
    `47f057781f62845b8f11114d10a9234a32ec63168bbbd3f3049362c218cdd1b7`,
    `9bffbadd57de284b60599764daadce6fae37214c5ebe3692c5d5a2caf04e76d6`,
    `efb783128ad625d14ac9c51f2d193f4c013df6991b933ac56f2b8c96c1228856`,
    and
    `57f75f3c4478d3945278e09394765bf2d20d21918bfcb1d7db0151aebbe12750`.

  Four independent verify-only jobs reread the source rows, semantic outcomes,
  full-text inspections, manual Parquets, and completion markers and reproduced
  every byte and hash.
- `/rav/rav-datakit-6854-audit-fast-1611-v1343` then independently reread 15
  newer checkpoints containing 1,920 pairs: 1,439 model false positives and
  481 model true duplicates, with no unresolved outcomes. Eight pairs were
  chunked and 1,912 direct. There were 4,509 judgments in 4,510 attempts; all
  judgments were valid after one retry. The arm split is 1,439 baseline pairs
  (1,084 false positives and 355 true duplicates) and 481 treatment pairs (355
  false positives and 126 true duplicates). Outcome SHA-256 values are:

  - p0 decision-file 12 offsets 1,024 through 1,664:
    `8b7910c751f4317f8b69dd6bf720bee542e6c0d2f45b953d9b2140d6812ee7d3`,
    `e71c2d8c94dd5b771e5f0f2f7bed469c204fc32aa72cd091174f16fb6ab6c9aa`,
    `0d2c76a95d62c5addb382997d88f27e9d3b46527eeb2dd6908adac1aa24cf768`,
    `39dc6e710055cc722289d90d741269487ff9d2e04c7218f49ce6a5967acde41b`,
    `86b341e7640d4da335fc9628754ad23464e4dca96c268256f3747f72a6cff581`,
    and
    `224999b9c9011b555a788a58d3c8ed034612cd3bdb8a141a68b23f141ef8ba6e`;
  - p2 decision-file 76 offsets 256 through 640:
    `93a6271bf59d918fd5d964491a809d6dbd5ad8e32c841cbb26bc87d6de5b6e01`,
    `e9ddf40439d10fb9005bc3d1bf482989f48b21c13c8a19881373a48996c9ae3a`,
    `e2c6da4835662625b73b726b93751c024e1e1ebb0805204db2e1e0c0098a8e2d`,
    and
    `a79b62a5a46ffb68886e1be0773d0ac134081e642cb52b776f47448853d3595d`;
  - p3 decision-file 108 offsets 4,480 through 4,992:
    `0ca6fda58488333865dabed63115e1cfab80ede05a8af8e947a91d4bc8695bc4`,
    `d202f3aedc9a5d10c258aa468527024c1a26d853ea34028489d4f19ed27745dc`,
    `472bbff52c87f062fca8bac45bffdfe10606c441d220b0e4a28107d042505d44`,
    `6283729c04af98e5379729ade11e2bb8fc8367f012a5376b532410da05c7dc62`,
    and
    `a4a483bdde0e7614b634d3a0c320fc4d92e0308acbd08036f148c5d19de97787`.

  The next audit frontiers are p0 `(12, 1,792)`, p1 `(44, 0)`, p2
  `(76, 768)`, and p3 `(108, 5,120)`. The p1 giant checkpoint and four
  2-H100 batch-priority jobs remain healthy; all 12 pods are Ready with zero
  restarts.
