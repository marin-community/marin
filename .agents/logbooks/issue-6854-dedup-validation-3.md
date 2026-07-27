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

### 2026-07-27T18:13:18Z — 301,372 pairs verified; all 362 ambiguities covered

- `/rav/rav-datakit-6854-reconcile-manual-1635-v1375` first established a
  fully reconciled 295,453-pair checkpoint. It reread and hash-verified 2,327
  semantic checkpoints and covered all 351 unresolved model outcomes. Applying
  74 manual false-positive and 277 manual true-duplicate decisions gives:

  - baseline: 236,262 pairs, 149,809 false positives, and 86,453 true
    duplicates;
  - treatment: 59,191 pairs, 30,668 false positives, and 28,523 true
    duplicates;
  - combined: 180,477 false positives and 114,976 true duplicates.

  False-positive rates are 63.4080% for baseline and 51.8119% for treatment,
  an 11.5961 percentage-point reduction. The immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1635.json`
  with SHA-256
  `2877b49f3739fc554bae8e40906f0cd9849aefcc23b8b78fcb1e49caf3c43dfb`.
- Eight full-text decisions close the ambiguities between the prior
  291,062-pair checkpoint and the 295,453-pair checkpoint. All are true
  duplicates:

  - baseline row 8,399 is the same 27-line article with one appended
    related-story teaser. Member/canonical, inspection, semantic-judgment,
    manual-Parquet, and marker SHA-256 values are
    `11a9711f2f786fce3068053576fcee2f24cfdf3064ee1042f57fb857025fdc2d` /
    `1d3d36e4f0c3539139dc79a938512fe5e35b28405eedf88a6b72445b5fbd8b2c`,
    `5fdf7668b6bbf227701eb8b6978334be8a4a0ea26e1014b63694289cd5c1be74`,
    `b5012dc99af8121f37b5a875b379ad0dcb989f531893bb7c3a8e5be0c13df3da`,
    `6790f0a834358c559e60c84802f5c36e752201f192ea9cc3ba74eabb221470e4`,
    and
    `c09fd4b09c2f5adc53da5be368f0b42d2390d3c8d77b79d343937696f501027c`;
  - baseline row 8,656 is a low-value college SEO template whose factual
    differences are institution and program slots. Hashes in the same order
    are
    `df94ab854683c7832b1f3dda801a64f6ad3e5c8dbeb6dab3f0f708330e1ffc57` /
    `8d55778ed0cda0da08f5da52cfdd6fa8e7451ecdb77c66a4d95e53b6d5717be2`,
    `f4ca901169f5bd02a2c5165984a5c2ab93fcee59f28336420ce712b9d3530770`,
    `322a25ce084071bf7d962e53206232dbb1ea52060085d47354306f5782bef503`,
    `b6b4a726e3e40a14256e182b6ede32ae42b6b7438e78f87d4ec8575c420d742c`,
    and
    `eeb5899ce093b4c3bb43a5ea60beea4e331c0d9fd8316d07a97917e59d613930`;
  - baseline rows 8,984, 8,999, and 9,000 have 184/183, 163/162, and 82/81
    identical lines. Their only differences are equivalent boxed-answer
    formatting. The five-hash tuples are:
    `75195b9283184aeb3b4e3cd2804ab0ede31a98c540da6c5d57f7fef935e58eb1` /
    `ce739514410bd8cf07d37aae2a9269e10a5d02a7ef9bd3603486ffd13717194c`,
    `971194a894af7aa291d43ec0981ad61f45abbb7bd4c5d8fcd8679f744edfaf85`,
    `1b059234982d44cfbbafadfeb8c60a0d6cc38ec202ffc3795aef2030837ae340`,
    `0b5687211e0a065ac03daa8c0c1f79bc1be73c5c4eb0e5cbc2ae2d23e4ebfc59`,
    `8df7ac9390df3a389d101ed644ca7ce1cfa81eb2f5c7301db5f5e3d790826e60`;
    `838e34c32d63f6d2f6aa856fc36fefe6b475f321b69bcdfea54cc8dd2efc33ae` /
    `190b37e987e314ae2a4528fc5260b47dcce074a2a79bb6d5f58176e3b27f0624`,
    `705c7b722ac70f812dec5d036bca2ca33c2f70db69194169010a97a9c1622b34`,
    `ac671c37e0514e1c5809372dc95074a040a9ae64afb688e8fc0b61b7e6276290`,
    `654c6a1aebb997e947df876de01bbcd2d257b28b400d14ccb0b7acbc40e9710f`,
    `0434c55855469f899728a822e178e6f1d2840d7c6a15ead87421a7ab1586f194`;
    and
    `768c73ab60abb7b54a144f9ddf372093507ca7b8e75237bb11c5d490a3ab0dc2` /
    `0e385ab7c6d60f0fd842f7298cb4a47eb46dc2506cd07f90da79a5c7004daca7`,
    `518bfa9a046a945c06a932c9a14937002fbcb5bd41c7bbbfc7ef21159a1357bd`,
    `c4a1b4668077f7c336aac13a2651e4bf445643a0cf19fffc59d2a9b4f1081742`,
    `9d8b88b94bc01b4fbd523cccb4e3b6c0c03bdaa02505bd002492d5536c33a930`,
    `a4eb72ec7eb38b91f108d4d0bb295cc913a9aca21c2e5ea7b5680b3c872e5f2e`;
  - treatment row 2,517 is a baby-name/numerology SEO template. Hashes are
    `6c0e7b92810649e6dd3285edb929d95ee2b41501529240062469298cc186c1bf` /
    `65c7cf0cff9c8d186a15c8c82f41d5683434ad494aa8fb3fd0661635d0de65bf`,
    `a599221c5ec53f7fbf82bfe1c5c8a07532e955b32f7ff67452df70f84fb54b89`,
    `372bc33d9407d38f2cee4caa849474492b6e1430b2d331fc5e498cce1ed6b537`,
    `26a9212ce14e097012d88b7702a71f05c44393839634e55f19da1e467243d2fb`,
    and
    `ad7890280b3a147c62dfc68103e1a0c85b4b0da30793f180f53a59105963732b`;
  - treatment row 4,876 is an AncientFaces surname template. Hashes are
    `7d8b48223bf044e47c995e965674040c51a032cdf4a09bb853da2169afbdb34e` /
    `601a3dde5681f75a466ffb662db60199e2247144c9597fa901cf865f914fa2ec`,
    `29ef365b9ced58065f21253188554454c4eb550e170095dff95dfd717c48f7b5`,
    `2025816facb53128ef0e4cf53fa84ae2b8adf41b02a2027d0bcf36cc6677b180`,
    `997745b17d8d68f2bb00db586ab2d2d93c4a3b6c1c6fa7c73342521abdf8d5c6`,
    and
    `8c26696e8d9d984e81fe6425db7d1c2d8b9e0fbe875a0826fb6e0cf47b44ed64`;
  - treatment row 3,427 is the same medical article with a related-article
    teaser. Hashes are
    `3afc633a533663cdb814cfd288b071a72ebaa903a094b5727099c3cd53d1abee` /
    `3a685187aa3ce0a194dbcdba0e8e8794efde7f09bc0eb4feb459efac2f4f15d9`,
    `a16403e939c7544478bc0034177b56e7656062957403975b3888750eeccd3564`,
    `b2688a487080f784fdafa33db775e12b17a88c8ee6e22bc1410c399e4570b9b4`,
    `ecaf87cd3e1ef3dfaf189e5ac95a3585d4a562a4ac1ab0cfbb7a7089761e8b92`,
    and
    `5d8572b865355336722aed0c977545cbdbe79ddd32801dbb5f758dbdac2b0417`.

  Eight independent verify-only jobs reread the source rows, outcome Parquets,
  inspection artifacts, manual decisions, and markers and reproduced every
  byte and hash.
- The next 5,919 completed pairs introduced 11 unresolved model outcomes. Full
  persisted-text review labels nine true duplicates and two false positives:

  - baseline row 5,575 is a sentence-spun satellite-internet SEO article. The
    location, provider, headings, and synonyms fill the pinned low-value
    template slots. Member/canonical, inspection, judgment, manual, and marker
    SHA-256 values are
    `8187caa4716e67dd4054c2431448748e5208ecdabb25c84d3d532e05d0752355` /
    `a57b09fab5f4c05b5da866505d9bed30c0f4124507a769b9e32292ae33fb5683`,
    `ace90a39c3b677f59b85c7e3af33e12890bf27e45dcf791232dec0d8a3dba36e`,
    `be1a19d8ec2d5ab946c20d16124bcb4cef2cd662240faba97b52c7f7e781f5ca`,
    `b6329a51d94c31acc803ea9f4e9b9db5b5b836897a11e2f9b790e9b7b4625c6f`,
    and
    `94b5e786a9d562b7e9b0b767e241cc7ef0bfe87239194ab35d8f70f422049310`;
  - baseline row 5,682 is a false positive. Both pages use a Grammarly SEO
    scaffold, but the member adds a first-person claim about using the product
    to learn French and additional feature claims. Hashes are
    `54bf41539f9c5c81c512035fa10cc8e0d10c7f8ed74f048f27119fdd12510bf7` /
    `b96e0239ebb9ad11b2e660ad8dd032cc93c93a2c3c3b91365461a2d6a9235286`,
    `76b7f6a614871bcb94b36d6eee20173d9fcdd5a234f3aa68eb203d4385ade714`,
    `f8f15413f5c2fdec3adfbed55dc404a9e88707fc83817d7111892d4991b1d126`,
    `2aecf1af21207940c498ee8e23456567879cdb9df0df536342c281fed1de81d1`,
    and
    `7b62e420a7d7b37ca4094270a62a9ecd8eeeed3888d7b9f19cb9463aa310df32`;
  - baseline row 7,341 is a false positive. The member asks for a symbolic
    product whose answer is `120a^10`; the canonical asks for `54 * 46`, whose
    answer is 2,484. Hashes are
    `0ed8a4dc5ecaccb711f71f9ae7ddb25a493eff750fe3ee2eb5c1c2b9a463ff76` /
    `987b60cb2ac229ddf85800ae1271c587ef6e72ae2d9c9078fd832a86c4c23c24`,
    `6906d3d3e63220fa41b0d52be0ce2e6d77953636f38385b6437059de40b8b4c0`,
    `b923b7be31f4cade56a395d54db8daab91c9227599363723d052c9c19836a6c4`,
    `65b45eba328b7e9857ad5287fba5be10c49c6607c378de609457151b4011d86d`,
    and
    `d41ffebce521d7c955d982e5f156a068971a6448890577b61c4bd83364c3d564`;
  - baseline rows 7,379, 7,637, 7,604, 7,609, and 7,610 are formatting-only
    duplicates. Exact line comparison finds 219/220, 113/114, 135/136, 61/62,
    and 169/170 identical lines; the remaining line changes only
    `\boxed{X}` to `\boxed{\text{X}}`. Their hash tuples are:
    `89fa521c9217666806d7730e0188f71708b04689eed368ca558573560a985a0d` /
    `12b9429ab49a78795ed4c1bc2c332e60829d83f31eae942b4c5bb8bddeb69157`,
    `041aa5ca35eb5fd61750818b634ddb6df0329ef893270fe0fa5882004ec07957`,
    `9bff97158d5c44906fcda4b26ea5a8362076814414571c3e2f4c9b267c6261ae`,
    `343a10e4c1396f6fc1540055bc0331831af228de60760c9ff9d9f14cebbba43e`,
    `4d1d4f251f4103d598f1a64730d4434c437ff1223f3c6d60fb2756c4587235b6`;
    `3f549f8a4cab6bf126a0fd61b70697d9ae46521eca1a4b29a5ccff96fa136106` /
    `1cdbdf0b62292cf9de7b2cc755faecca00c0fddd9440569db5156c25a1a18b90`,
    `0e4f0d3834d63c81ce374c6b5a02f2e6a6ce717ec8f95efb07f16ac1dfc430e1`,
    `c71bdd5c6db459c72e8de79867398dd5218ead5302da0d3fb29c6e9900f2e131`,
    `3d120dd8d367ed754e06cc73b61f447bf247e4dece38dcf0d1ad97f2d8a8498a`,
    `b3718038dc761873215a832a17c26a2f1543e8d1f176a1dac1698ea39c1ec328`;
    `b782c230ebd353b1d694cb887251244ff0a2874b581097959ac3c80f489db5c2` /
    `8ae79f6516a907b12582162256a756b62abdc516afc374e4648fb2fefe8f0db2`,
    `862620c3a82092bdf3ff307467b04931412bc999724a2c7e81fcfa366c6c49d7`,
    `08fd269bcea4550a1c267cb84b3d221030cf1145c9fc9543ed917c2dc840807f`,
    `952c38d07bad25eb6976d0a7011f9f6242913183758ad1f556c8a242ab0577c6`,
    `65d7c71285d85c5e2a4e83110f42cd2886dd25544ed69d788af31bcf5d6087c4`;
    `003ca93bcb4d39dc7bab32252fb85abd7d5d8c14ea305bdd22d842c7fa7959d9` /
    `9eef48b4627b82cb278cf126ea1af2df36ea968292a01caf1a86644f2ca07584`,
    `4a48aba1695f222223b580a1b6a67b9dec2a6f966b0f82efd6ed97d29682094f`,
    `b7bec3a2616fb0360d386914dd3131689580b1c061f37edbf866383e0890fbdb`,
    `c1c6719fcfb7dd5fa62cc955ba3b7b7daca3ac5bf019a2e25dfc85893acd5580`,
    `59b196303e41a2fc450f032cda5a6f824bbcf3ef06a23c1538930e803a6aa7a7`;
    and
    `b2d764511ba264b5309a05758de58e78a4a3316ed774703edee71003f63f02f7` /
    `617c3dda6eb72f452a30066d8b10ef62c6b488d075cb68dc01ba52925d9e6f3d`,
    `0c952eb18603d30de59714fb78cb3c2f8d3c2d19c63313235bce4c79cf37ee32`,
    `796eb17ae643336e6eefd923b40038f163aa7af7cbe409da9f5d32092ebed88b`,
    `ac9ef20ebd6d239b626fb8e6088359859fec7c7c88f690509b308359b6998502`,
    `4945d47e3e4af7dd7bc968a7f0d1a73403932e8be259d765090cf957e0c21cbf`;
  - treatment rows 9,033 and 9,069 are formatting-only duplicates with
    388/389 and 61/62 identical lines. Their hash tuples are
    `5dea2d66f28687ed1109fcf014bc8c48a9a87d72496739cbb0f3b68a97ebcd31` /
    `b73ae5af67cbedc077fae9beb8bded0459d71167a43a893926ba0d135aed92ac`,
    `b8307ed46da1fa905f5150a16da22c757f23cf872147a9374154b0251479b798`,
    `58af7ec921145ee12e55a3085bc54a1dd4b8454d1eb81c4046c67b5b32610b78`,
    `d896537efdbb3cca6f7f0e2717931e58ba0156c85d737cd8f71aa1bb455479ca`,
    `1c22af81a361336d57c061421353dbf05df31449f04065ce75721808ffc83efb`;
    and
    `c289064b8f6e62deeba2c81d2970fc0cee9c3fd84c15aa84ddf89a3091acb7fe` /
    `dc0ff1775e481d8ea36f43c724f8fb83b578eb6305213689e81513659fc15be3`,
    `528db82be74a67279d4adcaaeee2ba3244a99df2a36be585e179cffd1750e33a`,
    `f4c987ac67de1202c2bf42b9a50497ade38804b2c11fefe0ab603d45e05f090f`,
    `24dcdcf56aa281ae40a921d84c80e3d62f401b3ec8d35cb42269af008c4aa767`,
    `4631000f71450799f04565d97e3452579205501473fdbf9a7fb69a105f46a474`;
  - treatment row 9,056 is the same raw pair as baseline row 7,604 and has
    the same formatting-only difference. Its inspection, judgment, manual, and
    marker SHA-256 values are
    `099db2f006ba8e4d64c27bb59cd91123d0092a53a1bcfa39dadff123d9a56be9`,
    `d19f311211bf8f0a726df97d857382bc88927388c74d3ae0ee9aab533a447c45`,
    `57306cb0fd7a72621cf624411ee09bdcbea50ecb499fb575aa9a0373d78b1b2b`,
    and
    `5f393b9e2f1d425ed12b19fd0c13ee3a251cf9fc06d8df860643dd03e4e9dae9`.

  `/rav/rav-datakit-6854-verify-row5575-v1402` through
  `/rav/rav-datakit-6854-verify-row9056-v1412` independently reread every
  source row and artifact and reproduced all 11 decisions and hashes.
- `/rav/rav-datakit-6854-reconcile-manual-1808-v1413` then ran:

  ```text
  scratch/iris-cli-env/bin/iris \
    --config lib/iris/config/cw-rno2a.yaml job run --no-wait \
    --enable-extra-resources --cpu 4 --memory 32g --disk 20g \
    --priority batch --extra marin-core:cpu \
    --job-name rav-datakit-6854-reconcile-manual-1808-v1413 -- \
    python experiments/datakit/scripts/dedup_ab_reconcile_manual_tmp.py \
    --snapshot 20260727-1808
  ```

  It verified 2,374 semantic checkpoints and all 362 unresolved outcomes.
  There are no missing manual outcomes. Applying 76 false-positive and 286
  true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 239,769 | 152,302 | 87,467 | 63.5203% |
  | treatment | 61,603 | 31,950 | 29,653 | 51.8644% |
  | combined | 301,372 | 184,252 | 117,120 | 61.1378% |

  The baseline-minus-treatment gap is 11.6559 percentage points. The snapshot
  covers 39.9020% of the 755,281 semantic candidates. The immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1808.json`
  with SHA-256
  `fe3fbdb153b6ed4e908be07469e97435b4bbc3809c5ab5d5a0d646689560dc5d`.
  The path-manifest SHA-256 values for semantic markers, manual markers, and
  manual Parquets are
  `2c52c1ff845ce365e08c4c1056d9388dcf9df78d7c92e0ed3e37f2d29a7af4f4`,
  `082eec344c6b1d66e57e117059a57aee16566fbf3223aaeb7b8bf5480d3491b8`,
  and
  `20e2b3bd822e12d7b0f7f134a200c66f77a51a9bedacb9ea5b6badcc2795c8fe`.
  The only anomalies remain historical shadow records: two same-label
  duplicate manual Parquet keys, three obsolete outcome-hash bindings, and
  146 orphan manual Parquets. Marker-bound records are internally consistent;
  duplicate manual keys, extra decisions, extra Parquets, and missing Parquets
  are all zero.
- Audit jobs from
  `/rav/rav-datakit-6854-audit-fast-1754-v1377` through
  `/rav/rav-datakit-6854-audit-fast-1811-v1416` independently reread the
  advancing completion markers. Two formerly blocking giant checkpoints
  completed:

  - p1 decision-file 44 offset 0 contains 128 pairs and 122,322,523 combined
    characters; its outcome SHA-256 is
    `516550f56d40a13c8411b3156fe8648f510850288adaa95f7615c029639635f6`;
  - p3 decision-file 109 offset 0 contains 128 pairs and 109,562,149 combined
    characters; its outcome SHA-256 is
    `e8f5f3a07b2b1385c15a6cc705a46a44bf403a95cfad5cb68e9720d943dc0012`.

  The audits rediscovered the 11 manually closed cases above and found no
  additional ambiguity. `/rav/rav-datakit-6854-audit-fast-1813-v1417`
  records the current four pending frontiers: p0 `(13, 0)`, p1 `(44, 128)`,
  p2 `(77, 0)`, and p3 `(109, 128)`. Each is an unfinished large-text
  checkpoint; combined character counts are 120,292,052, 64,263,142,
  139,043,746, and 48,729,275.
- The four semantic-review parents, four inference brokers, and four 2-H100
  workers remain running at batch priority. The durable code/logbook commit
  before this entry is
  `9e1b676311a903f1f297629bf1b02a20e643bd11`.

## 2026-07-27 18:52 UTC — 304,060-pair reconciliation

- Audit jobs
  `/rav/rav-datakit-6854-audit-fast-1832-v1419` through
  `/rav/rav-datakit-6854-audit-fast-1851-v1426` independently reread 23 new
  completion markers containing 2,944 baseline pairs. They reproduce 1,984
  false positives and 960 true duplicates, with no unresolved outcomes and no
  invalid judgments. The audited total is 304,316 pairs. The next frontiers are
  p0 `(13, 0)`, p1 `(44, 2176)`, p2 `(77, 0)`, and p3 `(109, 1024)`.
- `/rav/rav-datakit-6854-reconcile-manual-1849-v1425` scanned the stable prefix
  available at 18:50 UTC: 2,395 checkpoints and 304,060 pairs. All 362
  unresolved model outcomes have exact manual coverage. Applying 76
  false-positive and 286 true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 242,457 | 154,144 | 88,313 | 63.5758% |
  | treatment | 61,603 | 31,950 | 29,653 | 51.8644% |
  | combined | 304,060 | 186,094 | 117,966 | 61.2031% |

  The baseline-minus-treatment gap is 11.7115 percentage points. This snapshot
  covers 40.2579% of the 755,281 semantic candidates. Its immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1848.json`
  with SHA-256
  `9f1c89984ba0433f50e6780d3e22bf0f2018e2193fa8d9414a8d903fbe5f1243`.
  The semantic, manual-marker, and manual-Parquet path-manifest SHA-256 values
  are
  `450360157136893a0d24cc53f0928d2784f083e4030185b9f7d319f3f5036c6e`,
  `082eec344c6b1d66e57e117059a57aee16566fbf3223aaeb7b8bf5480d3491b8`,
  and
  `20e2b3bd822e12d7b0f7f134a200c66f77a51a9bedacb9ea5b6badcc2795c8fe`.
  There are no missing manual outcomes. Historical shadow-record counts remain
  unchanged: two same-label duplicate manual Parquet keys, three obsolete
  outcome-hash bindings, and 146 orphan manual Parquets; marker-bound records
  remain internally consistent.
- The first reconciliation attempt,
  `/rav/rav-datakit-6854-reconcile-manual-1848-v1424`, exhausted the 1 GB
  default memory during the manifest scan before writing output. The successful
  retry used four CPUs and 16 GB. The four semantic-review parents, brokers, and
  2-H100 workers remain healthy at batch priority.

## 2026-07-27 19:16 UTC — two new manual adjudications

- `/rav/rav-datakit-6854-audit-fast-1906-v1431` found two unresolved baseline
  outcomes among 512 pairs. Both source documents, semantic judgments, and
  hashes were independently reread before labeling:

  - decision-file 44, semantic offset 3,200, pair row 5,687 is a true
    duplicate. The 1,239-character member and 827-character canonical have
    0.5712 character and 0.5033 word-sequence similarity. Both are the same
    incoherent college-advice SEO template; the member's institution, location,
    and program phrases are slot substitutions rather than coherent distinct
    facts. The inspection, manual Parquet, and marker SHA-256 values are
    `e1e38b977608c4d462e00ae4636151eea0a133eed4b19cb43d24a3c65582b884`,
    `17afe007f60772c0d399ea8abdcc0641d8f4bef93cd60fa7a24a10a3d6fd2dca`,
    and
    `efdf2254efdfe774abf1683ed01cbfd0159f7abf07963ce74da98168de9bc807`.
  - decision-file 109, semantic offset 2,048, pair row 3,356 is a false
    positive. The 3,089-character member and 3,917-character canonical share
    the same Cambridge cocaine-addiction article, but the member uniquely adds
    a coherent claim that a licorice ingredient may counter cocaine toxicity
    and overdose, attributed to researchers in Korea and Pennsylvania. Its
    citation is truncated, but the factual claim is absent from the canonical.
    Character and word-sequence similarities are 0.8379 and 0.8311. The
    inspection, manual Parquet, and marker SHA-256 values are
    `ae358d38d56186b9a951840c89b5b4aae319b2d82879dd79918e33b1d474f6a2`,
    `d0c67e2e1e974403eec5d36af7d8dae6d539e60ff43e882a99bf7df7beeaaa26`,
    and
    `2cc0cafa1e6a7ab03aaaab99840e91b93c51217d7b5270389da1db9acc21ffac`.

  Publish jobs `/rav/rav-datakit-6854-publish-row5687-v1434` and
  `/rav/rav-datakit-6854-publish-row3356-v1435` wrote the hash-bound records.
  Verify-only jobs `/rav/rav-datakit-6854-verify-row5687-v1436` and
  `/rav/rav-datakit-6854-verify-row3356-v1437` reproduced their complete source,
  inspection, Parquet, marker, and evidence bytes.
- `/rav/rav-datakit-6854-reconcile-manual-1912-v1438` then verified 2,423
  checkpoints and complete manual coverage for all 364 unresolved outcomes.
  Applying 77 false-positive and 287 true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 246,041 | 156,170 | 89,871 | 63.4732% |
  | treatment | 61,603 | 31,950 | 29,653 | 51.8644% |
  | combined | 307,644 | 188,120 | 119,524 | 61.1486% |

  The baseline-minus-treatment gap is 11.6088 percentage points. The snapshot
  covers 40.7324% of the 755,281 semantic candidates. Its immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1912.json`
  with SHA-256
  `4e2eee51e76c43e584f8a55a5a17e36d18d10d1c137a5727a933f90ecfa8b65a`.
  The semantic, manual-marker, and manual-Parquet path-manifest SHA-256 values
  are
  `e6b3d814bf96199327234e0b32f81309d74dcaa0c6da2a8d954003fe9b6af7b7`,
  `11b86decfa9231291f7b722cdde1ed3b1cdc0d150b99b2341d84791b4eaec928`,
  and
  `d984206d211b578d5cb6729c2dea19a8f93e011c3454d236de68d8980dc020f5`.
  No manual outcome is missing. Historical shadow-record anomaly counts remain
  unchanged and marker-bound records remain internally consistent.
- `/rav/rav-datakit-6854-audit-fast-1915-v1439` independently verified another
  10 checkpoints containing 1,280 pairs: 744 false positives and 536 true
  duplicates, with no unresolved outcomes or invalid judgments. The audited
  total is 308,028 pairs. The next frontiers are p0 `(13, 0)`, p1 `(44, 3968)`,
  p2 `(77, 0)`, and p3 `(109, 2944)`. All four semantic-review parents,
  brokers, and 2-H100 workers remain healthy at batch priority.

## 2026-07-27 19:34 UTC — three formatting-only duplicates

- Audits `/rav/rav-datakit-6854-audit-fast-1918-v1440`,
  `/rav/rav-datakit-6854-audit-fast-1927-v1448`, and
  `/rav/rav-datakit-6854-audit-fast-1933-v1453` independently reread 22 new
  completion markers containing 2,816 pairs. After the three manual
  adjudications below, they contain 1,822 false positives and 994 true
  duplicates. The audits reproduced every persisted outcome and accepted
  8,748 valid model attempts; 19 invalid JSON attempts were excluded before
  persistence, and no resolved outcome depends on them.
- The three unresolved baseline outcomes were true duplicates. Full-text
  inspection found identical questions, options, reasoning traces,
  conclusions, and answers; only the final LaTeX answer wrapper differed:

  - decision-file 44, semantic offset 4,096, pair row 7,322 has 160 lines and
    differs only by `\boxed{C}` versus `\boxed{\text{C}}`. Its inspection,
    manual Parquet, and marker SHA-256 values are
    `3a02062662f02e530b900ce6db3bbaffe3e259809693eb1badfdc124f77fb177`,
    `9c10b040e1e8c108b2d71d852a9e3494fb21170086bbbbc0c80c49a39ae6762a`,
    and
    `adcddd6c6474432af33c0e6469ae5753104ced5fecbf581cd1858e4cf26084e1`.
  - decision-file 44, semantic offset 4,096, pair row 7,355 has 318 lines and
    differs only by `\boxed{H}` versus `\boxed{\text{H}}`. Its corresponding
    SHA-256 values are
    `1b8fbeff2ab35c980fe64775cab471a357d4cafbb2b1b13b16ff0af234d60997`,
    `598a6f6f2fec6b97407e4fb49f3a5bf3f0b3a939986342eb24e5bb28e2a2dfd4`,
    and
    `d77560b89227e3c09713fac3800550004c40f45dae031be20c908f7f7e6e508a`.
  - decision-file 44, semantic offset 4,352, pair row 7,598 has 258 lines and
    differs only by `\boxed{G}` versus `\boxed{\text{G}}`. Its corresponding
    SHA-256 values are
    `eef6da2d296ddc7ab8f4932e118007418b4464594e1bc8f1713ed7b0bb192d7a`,
    `49e1b14610d8b4cab158ca1da9bfb8ce66c3415132f6ddead10ed216c87a738f`,
    and
    `30aa220990434c92443b4268b4aef1fb39b781d70e3a2feb64d7089fcc620bec`.

  Separate publish and verify-only jobs wrote and then reproduced each complete
  source, inspection, manual Parquet, marker, and evidence payload.
- `/rav/rav-datakit-6854-reconcile-manual-1931-v1452` verified 2,444
  checkpoints and complete manual coverage for all 367 unresolved outcomes.
  Applying 77 false-positive and 290 true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 248,198 | 157,607 | 90,591 | 63.5005% |
  | treatment | 62,134 | 32,305 | 29,829 | 51.9925% |
  | combined | 310,332 | 189,912 | 120,420 | 61.1964% |

  The baseline-minus-treatment gap is 11.5080 percentage points. This snapshot
  covers 41.0883% of the 755,281 semantic candidates. Its immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-1931.json`
  with SHA-256
  `558de428a30f4c09bbbebf7ff81beedfc6e3932b071a3e98969928a193db8e6e`.
  The semantic, manual-marker, and manual-Parquet path-manifest SHA-256 values
  are
  `f8796c38d74285e26105317b63ae3f7fe08b7d3cd0fbffe6a3569ca958e3b8ba`,
  `bf3e1be04da0556e2d1f4d7b6e7d41424ad697268b2e5bfd4cfa650c4a459607`,
  and
  `4179bb0abd9ce3effeffc57d338c6913cdf3a0c4e883baf083216821e02ae92f`.
  No manual outcome is missing. Historical shadow-record anomaly counts remain
  unchanged and marker-bound records remain internally consistent.
- The independently audited total is 310,844 pairs. The next frontiers are p0
  `(13, 128)`, p1 `(44, 5376)`, p2 `(77, 0)`, and p3 `(109, 4224)`. All four
  semantic-review parents, brokers, and 2-H100 workers remain healthy at batch
  priority.

## 2026-07-27 20:17 UTC — two metadata-only duplicates

- Audits `/rav/rav-datakit-6854-audit-fast-1957-v1458`,
  `/rav/rav-datakit-6854-audit-fast-2009-v1463`, and
  `/rav/rav-datakit-6854-audit-fast-2015-v1468` independently reread 22 new
  checkpoints containing 2,810 pairs. They reproduced 10,509 valid model
  attempts; one invalid attempt had already been retried and excluded. Two
  treatment outcomes required full-text adjudication:

  - decision-file 109, semantic offset 4,736, pair row 7,985 is a true
    duplicate. Both documents contain the same Inspire2Live cancer article and
    all substantive claims. The differences are title and section-heading
    formatting, publication metadata, and a byline. Its inspection, manual
    Parquet, and marker SHA-256 values are
    `233da253fe899dae05a8a103244af4e7d47c21c804f11f1e779b8162ea331ef6`,
    `d02900f710cd49428de33bfb05fb79002ba6a4ead5f0aae731a4fb26249f0d75`,
    and
    `99116decd4064a5f9afd268e074e7345c14e22c1a5ac2be5153844fd18d51020`.
  - decision-file 109, semantic offset 5,632, pair row 8,861 is a true
    duplicate. Both documents contain the same nursing-center description. The
    member title restates the facility already named in the body, while its
    pricing footer supplies no price or distinct claim. Its corresponding
    SHA-256 values are
    `436882cc9e9130b6a8e984dcd441c06cecc9b7dd910e5ad80f1d499dbb299504`,
    `172994dbbf14a5462bda580d90bd20f76c1b9faefb5388a849f0c69faf3a0a5a`,
    and
    `e7941491f2dbe6a125a1030d16890ce70b580c403e384b9078504d471af06125`.

  Separate publish and verify-only jobs wrote and then reproduced each complete
  source, inspection, manual Parquet, marker, and evidence payload.
- `/rav/rav-datakit-6854-reconcile-manual-2014-v1467` verified 2,477
  checkpoints and complete manual coverage for all 369 unresolved outcomes.
  Applying 77 false-positive and 292 true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 250,458 | 159,345 | 91,113 | 63.6214% |
  | treatment | 64,077 | 33,260 | 30,817 | 51.9063% |
  | combined | 314,535 | 192,605 | 121,930 | 61.2348% |

  The baseline-minus-treatment gap is 11.7151 percentage points. This snapshot
  covers 41.6448% of the 755,281 semantic candidates. Its immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-2014.json`
  with SHA-256
  `693c460035dfa4319088392d5673e7a9691704e3f3b383b8cbf80bf3fb8914cb`.
  The semantic, manual-marker, and manual-Parquet path-manifest SHA-256 values
  are
  `5097302644b71f384bf9cef5f637490d5845bbd31556e5d8bbe92e93e8adb5a4`,
  `b538c8414d2b0bf61de92b3d773a065387c96793be1ad393d91cd35484e6747d`,
  and
  `f659b01c678bda00d96dcc1cc92b75721cf552a7a823c1140a1097c0675dd5ed`.
  No manual outcome is missing. Historical shadow-record anomaly counts remain
  unchanged and marker-bound records remain internally consistent.
- The independently audited total is 314,663 pairs. The next frontiers are p0
  `(13, 1664)`, p1 `(45, 0)`, p2 `(77, 128)`, and p3 `(110, 0)`. All four
  semantic-review parents, brokers, and 2-H100 workers remain healthy at batch
  priority.

## 2026-07-27 20:35 UTC — hosting-template adjudication

- Five checkpoint audits independently reread another 1,408 pairs and 3,719
  valid model judgments without an invalid attempt. One baseline outcome
  required full-text adjudication: decision-file 13, semantic offset 2,176,
  pair row 3,684 is a true duplicate. The 3,781-character member and
  3,737-character canonical are the same spun web-hosting template and
  preserve the same meaning. Their differences are paraphrases and broken
  offer/provider slots such as zero TLDs, a zero price, a nameserver domain,
  and a blank vendor, rather than distinct coherent content. The inspection,
  manual Parquet, and marker SHA-256 values are
  `3093b0eb6bc1bcb13aeade217b6647d49f4608672419213d03486cf45728cf14`,
  `1afb7ae8b688334d821af3685cc6c21e5f072fc58c7c1883f8b7c0dcf99a1c05`,
  and
  `e8702d3a8b7e74610d091b184c22ed0166afe0ffd528456caf9676a4c57b3cb4`.
  Publish job `/rav/rav-datakit-6854-publish-row3684-v1473` wrote the
  hash-bound decision, and verify-only job
  `/rav/rav-datakit-6854-verify-row3684-v1474` reproduced every input and
  output hash.
- `/rav/rav-datakit-6854-reconcile-manual-2029-v1475` verified 2,487
  checkpoints and complete manual coverage for all 370 unresolved outcomes.
  Applying 77 false-positive and 293 true-duplicate manual decisions gives:

  | Arm | Pairs | False positives | True duplicates | False-positive rate |
  | --- | ---: | ---: | ---: | ---: |
  | baseline | 251,738 | 159,955 | 91,783 | 63.5403% |
  | treatment | 64,077 | 33,260 | 30,817 | 51.9063% |
  | combined | 315,815 | 193,215 | 122,600 | 61.1798% |

  The baseline-minus-treatment gap is 11.6340 percentage points. This snapshot
  covers 41.8142% of the 755,281 semantic candidates. Its immutable report is
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-semantic-reconciliation-100b-20260727-v1/snapshots/20260727-2029.json`
  with SHA-256
  `b4bddfaead3d21f44c4f2cff438e78ea5af2472d90b32a719e1a9035eb79dd5a`.
  The semantic, manual-marker, and manual-Parquet path-manifest SHA-256 values
  are
  `a50def16a20ad74c62fa6e1d7b63b84af93c84279f7dbd9170aa2dc7813342b2`,
  `ce4010fd1c9349609838c87aa29ed6b5b318cb5120e057cf66b5a30b5d57799a`,
  and
  `380a01cae4f6dfc2271f33a7286594141b32e3ac531d16da19c62464a3957b8d`.
  No manual outcome is missing. Historical shadow-record anomaly counts remain
  unchanged and marker-bound records remain internally consistent.
- The independently audited total is 316,071 pairs. The next frontiers are p0
  `(13, 2944)`, p1 `(45, 0)`, p2 `(77, 256)`, and p3 `(110, 0)`. All four
  semantic-review parents, brokers, and 2-H100 workers remain healthy at batch
  priority.
