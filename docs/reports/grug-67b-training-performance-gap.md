# Grug 67B training performance gap

The frozen comparison remains unchanged. On one fixed logical update and a
four-host, 32-H100 topology, Levanter took 255.859 seconds. MarinSkyRL's fastest
valid path took 6,991.670 seconds. The gap is 6,735.810 seconds, or 27.33x.

A later MarinSkyRL measurement located 5,583.786 seconds of a 6,854.120-second
matched-CE step inside routed-MoE blocks. This is 81.466% of that present-day
wall. It does not assign the same time to the frozen cross-stack gap.

We tried to close that attribution gap with a same-revision eager-versus-
grouped experiment. The required eight-H100 semantic gate failed before the
32-H100 timing run. Exact route loads and representative gradients disagreed.
No causal grouped-expert recovery is claimed, and no new number is debited from
the frozen gap.

## Frozen comparison

Both stacks consumed the same content-addressed replay:

| Property | Value |
| --- | --- |
| Logical sequences | 4,096 |
| Allocated positions | 32,817,152 |
| Non-padding positions | 25,095,420 |
| Loss positions | 24,494,588 |
| Sequence shape | `[4096, 8012]` |
| Rank shards | 32 shards of 128 sequences |
| Logical batch SHA-256 | `e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d` |
| Manifest SHA-256 | `5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74` |

The manifest is
`s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json`.
The June step-630 MarinSkyRL export is pinned at model revision
`a822321c2c21af099189e7116104b3cf5142c119`.

| Stack and path | Wall seconds | Samples | Result |
| --- | ---: | ---: | --- |
| Levanter, native FA4 and ring EP8 | 255.859 | 3 | Fastest valid frozen path |
| MarinSkyRL, FlashAttention and FSDP2 EP1 | 6,991.670 | 3 | Fastest valid frozen path |
| MarinSkyRL, eager attention and FSDP2 EP1 | 7,214.213 | 1 | Direct attention control |

The direct MarinSkyRL FlashAttention intervention saved 222.543 seconds, or
3.085%, relative to eager attention. The operational comparison is useful for
capacity planning, but it is not a common-implementation benchmark. Levanter
uses FP32 master parameters, BF16 compute, stacked layers, ring EP8, and AdamH.
MarinSkyRL starts from a BF16 export, runs 128 serial microbatches per rank with
FSDP2 EP1, and uses AdamW.

The matched-CE control exposed the remaining semantic problem. Levanter's
median global CE was 1.994753. MarinSkyRL reported 2.013692 with FlashAttention
and 2.013704 with eager attention. The differences, 0.018939 and 0.018952,
failed the predeclared absolute 0.01 gate. The two stacks do not use identical
represented parameter values, so the cross-stack matched-CE timing is not a
causal component of the frozen gap.

## Present-day routed-block attribution

A separate present-day MarinSkyRL run used the same replay, model export,
matched global token-weighted CE objective, 32 H100s, and EP1. Hooks on the 26
routed-MoE blocks formed a non-overlapping partition on the critical rank:

| Partition | Seconds | Share of wall |
| --- | ---: | ---: |
| Routed blocks, initial forward | 246.012 | 3.589% |
| Routed blocks, backward including checkpoint recompute | 5,337.774 | 77.877% |
| Routed blocks, total | 5,583.786 | 81.466% |
| Everything else | 1,270.334 | 18.534% |
| Synchronized wall | 6,854.120 | 100% |

The source was MarinSkyRL commit
[`08f814440579854313a258a8dd658176557f907d`](https://github.com/marin-community/MarinSkyRL/commit/08f814440579854313a258a8dd658176557f907d),
and the image was
`ghcr.io/marin-community/marinskyrl@sha256:5f35056daee57d25f134aa2171126645be6750944c92bec27962cfae412041d3`.
An independent readback verified the result hash, finite gradients, expected
hook counts, loss invariance, and exact partition closure.

For scale only, subtracting Levanter's complete 253.176-second matched-CE wall
from the present routed-block span gives 5,330.611 seconds. This is not a lower
bound or frozen-gap closure. It crosses represented values and revisions, and
the 111.351-second total-wall bridge between frozen and present MarinSkyRL
cannot locate expert versus nonexpert drift.

## Eager-versus-grouped closeout gate

The final discriminator was built from the exact green head of
[MarinSkyRL #276](https://github.com/marin-community/MarinSkyRL/pull/276),
[`b37cd1cb4027c0a11d705734554c739e5f9f67f7`](https://github.com/marin-community/MarinSkyRL/commit/b37cd1cb4027c0a11d705734554c739e5f9f67f7).
[MarinSkyRL #249](https://github.com/marin-community/MarinSkyRL/pull/249), at
`7276dee1f7d9c94d4925bf91a1eff07d0d86295f`, was held common as a dependency.
It was not treated as an intervention.

One measurement-only commit added the fixed-replay driver, verifier, and common
routed-block timing seam:

| Identity | Value |
| --- | --- |
| Measurement source | [`d81f6364ab66947cdf520a3a42a274b586e830da`](https://github.com/marin-community/MarinSkyRL/commit/d81f6364ab66947cdf520a3a42a274b586e830da) |
| Source parent | `b37cd1cb4027c0a11d705734554c739e5f9f67f7` |
| Runtime image | `ghcr.io/marin-community/marinskyrl@sha256:188eb430485f12182f483a7ee1c2c50191898b5a91e0fa6fea9ef183c4b947a6` |
| Model revision | `a822321c2c21af099189e7116104b3cf5142c119` |
| Replay manifest | `5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74` |
| Logical batch | `e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d` |
| Topology | One complete host, 8 H100-80GB GPUs, EP1 |
| Objective | Matched global token-weighted CE through backward; no optimizer |

Before the run, the validity rule required exact source, image, model, replay,
configuration, and rank-to-GPU topology; both requested implementations; the
same non-overlapping timing boundary; exact per-layer route loads; output,
loss, and representative-gradient agreement at fixed tolerances; finite full
parameter gradients; and an unchanged eager instrumentation oracle. A
semantic, numeric, route, oracle, or identity failure could not be retried. Any
failure stopped the 32-H100 pair.

The gate ran three one-microbatch-per-rank arms sequentially in one reserved
eight-H100 pod. Each result records at least 476 gradient tensors and
8,384,693,120 gradient elements per rank. Both instrumented arms recorded 26
forward and 26 backward routed-block calls per rank, plus one route-load sample
per layer. The grouped logs proved that all 26 blocks entered the native
grouped path.

The raw gate timings below are diagnostics. They are not full-replay samples
and cannot be subtracted to claim recovery.

| Gate arm | Wall seconds | Routed-block seconds | Other seconds | Global CE |
| --- | ---: | ---: | ---: | ---: |
| Eager, no hooks | 18.174897 | — | — | 11.350234 |
| Eager, hooks | 18.098923 | 13.710684 | 4.388239 | 11.350234 |
| Grouped, hooks | 2.250760 | 0.633520 | 1.617240 | 11.351937 |

The frozen verifier stopped first because ranks 2 and 3 exchanged physical GPU
UUIDs between arms. A separate CPU readback showed that the same physical host
and eight-GPU set was reused, but also found two additional semantic gate
failures:

| Gate | Result | Evidence |
| --- | --- | --- |
| Common source, image, model, replay, configuration, staging, and warmup | Pass | Rehashed identities matched |
| Requested eager and native grouped paths | Pass | Path markers on every rank |
| Common timing seam and expected counts | Pass | 26 forward and 26 backward calls per rank |
| Exact rank-to-GPU topology | **Fail** | Ranks 2 and 3 exchanged GPU UUIDs |
| Exact per-layer route loads | **Fail** | All 8 ranks differed; 1,656 aggregated layer/expert cells differed; largest load delta 969 |
| Representative outputs | Pass | Maximum absolute difference 0.232764; `rtol=0.04`, `atol=0.004` |
| Global CE | Pass | Absolute difference 0.001702; `rtol=0.002`, `atol=0.002` |
| Representative gradients | **Fail** | 46 of 1,296 checks failed; worst was 5.746x the allowed difference; `rtol=0.08`, `atol=0.0001` |
| Full parameter gradients finite | Pass | No non-finite gradient tensors in either arm |
| Eager instrumentation oracle | Pass | Output and loss differences were zero; all gradient checks passed |

The two arms routed the same total 6,665,984 expert allocations, but equal
totals do not make the layer and expert workloads equivalent. The passing
output and loss tolerances also do not replace the failed route and gradient
contracts. The grouped timing observation is therefore ineligible for causal
interpretation.

The development allocation was released immediately after the failure. The
planned four-node, 32-H100 job was never submitted.

## Conclusion and next experiment

The strongest supported conclusions are:

- the frozen Levanter-to-MarinSkyRL operational gap is 6,735.810 seconds;
- routed blocks occupy 5,583.786 seconds of one present-day MarinSkyRL
  matched-CE wall;
- the attempted eager-to-grouped bridge failed before production timing; and
- no part of the frozen gap can yet be assigned to grouped experts.

The cheapest next experiment is not a 32-H100 rerun. Use one eight-H100 FSDP
initialization and one replay shard per rank. Snapshot the model and RNG state,
then compare eager and grouped one routed block at a time. Record the input
hidden state, router logits and margins, selected experts, per-expert loads,
block output, and selected input and parameter gradients. This locates the
first divergence without paying for a full replay or changing product code
before the cause is known.

Production timing can resume only after a newly pinned revision passes the
three-arm eight-H100 gate: exact rank topology, the predeclared route/load
contract on all 26 layers and eight ranks, all 1,296 representative-gradient
checks, output and loss tolerances, full-gradient finiteness, expected timing
counts, and the instrumentation oracle. If native grouped arithmetic cannot
preserve exact downstream routes, a replacement per-layer load contract must
be justified and frozen before collecting another sample. This failed result
must not be reinterpreted under a weaker post-hoc contract.

## Artifacts

The executed launchers and independent readback are preserved on the
[`grug-eager-grouped-gate-evidence-20260802`](https://github.com/marin-community/MarinSkyRL/tree/a779e8690ec84e8102a1d00ab5e3baaced4ccb1f/evidence/grug_paired_20260802)
evidence branch. The exact measurement branch remains pinned at
[`d81f636`](https://github.com/marin-community/MarinSkyRL/tree/d81f6364ab66947cdf520a3a42a274b586e830da).
The prior full report machinery is preserved on Marin branch
[`grug-training-perf-gap-evidence-98766a-20260802`](https://github.com/marin-community/marin/tree/98766a743f9751a8894618381812ba1893c01aff).

Gate and readback jobs:

- [Eight-H100 gate](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fdev-gpu-romain-grug-paired-d81f636)
- [Independent failure readback](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-preflight-failure-readback-d81f636-r2-20260802)
- [Exact image build](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-paired-gpu-rl-amd64-d81f636-r2-20260802)

Content-addressed gate results:

| Arm | Object | `result_sha256` |
| --- | --- | --- |
| Eager oracle | `preflight/eager-oracle-s1.json` | `af9a94639ce03fae1449ab0d625632790d27e24b8644182f78a411383a17e9e2` |
| Eager instrumented | `preflight/eager-instrumented-s1.json` | `20545a39a1fe87910c1e1d789d67e17cf9b9784e2c871a897d4aa0c4abf0950e` |
| Grouped instrumented | `preflight/grouped-instrumented-s1.json` | `5bde7e38c3a2132153feb4320795f473172becde909e2911061baa8a24d508d5` |

These objects share the prefix
`s3://marin-us-east-02a/iris/grug-training-perf-gap/20260802/paired-d81f636/`.
The independent readback report SHA-256 is
`06a6a954b1bf9d48ff5c134024e647fc965683949d36c4e947c9c49d1b519816`.

The frozen accepted objects and their verification machinery remain on the
preserved Marin evidence branch. Every displayed timing is tied to its recorded
source, image, topology, replay, and result hash; no failed preflight is
relabeled as a headline.
