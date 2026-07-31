# GrugMoE inference preflight findings

Status: **GO for the exact serving architecture and for a later, separately
authorized architecture experiment.**

The frozen reference now starts and generates on the target GB200 topology.
The exact tensor fixture, prefix behavior, active KV accounting, P0
configuration families, unattended two-node path, and unattended four-node
acceptance all passed. The final two warmed arms differed by 0.3324%, below
the 2% gate.

This is not trained-model throughput evidence. The one permitted Snowball
attempt failed on path-style S3 listing before model load. The accepted run
used deterministic dummy weights, whose routing is reproducible but strongly
skewed. No architecture comparison, profile, lower-precision run, H100 run,
full RL cycle, 131K run, capacity-factor sweep, or fault-injection run was
performed.

## Recommendation

Use the exact reference as the baseline for the later architecture experiment.
Keep the Marin and vLLM evidence commits pinned together, retain the
counter-based throughput measurement, and preserve the same unattended Iris
path for every candidate.

Do not use the accepted throughput as a claim about the trained Snowball
checkpoint. Repair and review its object-store access before a trained-model
experiment. Also report routing histograms beside future throughput: the dummy
router left 121 of 128 experts unused, so its absolute load balance is not
representative.

## Frozen provenance

Every live job below was submitted from a clean, pushed Marin commit that
pinned the exact pushed vLLM evidence SHA.

| Item | Frozen value |
|---|---|
| Marin branch | [`grugmoe-inference-preflight`](https://github.com/marin-community/marin/tree/grugmoe-inference-preflight) |
| Requested frozen Marin reference | `75bf2437035cf731d1a4bd71266229dfcdda9478` |
| Resumed preflight commit | `db0fd88acad7fc61022c074470c310f0d62c045a` |
| Current branch merge base | `2f49a34509fa01352d6b8137c52632e55c913854` |
| Final live acceptance commit | `a3320a3043018ee923bc98bf2e6e6eef3f03a6fe` |
| vLLM branch | [`grugmoe-inference-preflight`](https://github.com/marin-community/vllm/tree/grugmoe-inference-preflight) |
| vLLM base | `afb26719464d5957e695bde478ae93a160b11d14` |
| vLLM evidence commit | `2c2bef33dfbd7aef3c9d4433a7e4110f77d56a4a` |
| vLLM reviewed branch head | `cdfde7e24d8aa3339b4f22444db7b45d43e018fa` |
| Training semantic oracle | `fd3e9bc5b428633027f944be7fdf1136567db028` |
| Cluster / priority | `cw-us-east-08a` / `interactive` |
| Task image | `ghcr.io/marin-community/iris-task@sha256:5e2a69af91a000cb999e6ff0d92933874bd3142eb45469fc64fc7a3f5db64fbb` |
| Image amd64 child | `sha256:6d9946f50a81a0bfbe31516274e2318cc70a9d0b683d898b33b42fe0a84e5d9b` |
| Image arm64 child | `sha256:9a68d25676c45cddee848c552bf57ebe46dc74ff1dcc648d8fec05b72e3bc900` |
| Acceptance workload SHA-256 | `e298b1f4925421a1d759ab7d92fb8310cfb8c9f074d0ece2277c713d2c5b7c62` |
| Acceptance config SHA-256 | `88844012a67ca8fec2da879e4b8ff3245d12d3276fb5146ed89ca9cc38bc2169` |
| Correctness workload SHA-256 | `882b51d0959d8dd6e875d04be49d197ceef3e4c4e1d3c9cd141c6166b5fe5dc1` |
| Dependency lock SHA-256 | `600b2abe4b5e8027c3783adc8cc45924c71be1a357c1e23eac6ab9049d5f6a14` |

The reviewed vLLM head is one commit after the live-evidence SHA. It only
rejects malformed unstacked expert tensors and adds a regression test. Every
in-scope fixture/export tensor is stacked 3D, so the measured serving path and
the evidence pin remain unchanged.

The multi-architecture image is a generic Iris task image. The unattended
worker synchronizes the pushed source named in the manifest at job start.

## Exact implementation

The accepted model is the frozen `d6144`, 48-layer reference:

- 48 query heads with head dimension 128;
- 12 stored/local KV heads and 6 global KV heads;
- local attention window 512 and global attention every 6 layers;
- top-4 of 128 routed experts with intermediate width 3072;
- two separate, nonzero shared experts of width 3072 each;
- fused half RoPE on local layers, gated normalization, attention gating,
  XSA, and QB routing;
- SConv kernel 4 at K, V, attention output, and MLP output;
- MTP depth remains in the checkpoint/configuration, while ordinary serving
  returns trunk logits and does not execute the dense training head;
- BF16 weights and BF16 KV cache, prefix caching, and chunked prefill;
- PP1, TP1, DP16, and EP16 for acceptance.

The vLLM branch adds only the reference-specific model/configuration behavior,
four SConv paths, heterogeneous hybrid-cache grouping, active-block telemetry,
and tests. The two shared experts remain two modules and their outputs are
summed; they are not silently replaced by one learned full-width expert.

The Marin branch adds the frozen fixture and oracle, the small case catalogue,
the request/correctness/KV harness, and one zero-retry unattended Iris gang
entrypoint. The top-level result is the literal conjunction of placement,
all-rank health, correctness, duration, token count, repeatability, and
artifact readback.

## Assumption ledger

No preflight assumption remains uncertain because of missing model, cache, or
launcher machinery.

| Assumption | Status | Decision-ready evidence |
|---|---|---|
| The exact frozen reference starts | **confirmed** | Exact EP8 and EP16 jobs loaded all custom paths and generated successfully. |
| Ordinary serving omits the dense MTP head | **confirmed** | The fixture pins `mtp_depth=1`; vLLM returns trunk logits and the oracle manifest records the training head as excluded. |
| Every selected top-K contribution is dispatched | **confirmed** | The pinned `FusedMoE` route has no capacity clipping or token drop; fixture and live route captures include every selected assignment. |
| Levanter and vLLM agree on the same tensors | **confirmed** | Selected experts match, normalized routing weights differ by at most `2.3841858e-07`, and next-token probabilities satisfy the repository `NextTokenParity` tolerance. |
| Two half-width shared experts preserve training semantics | **confirmed** | Both fixture experts are nonzero and pairwise distinct. On frozen oracle inputs, separate-sum and fused-concat representations agree within `2.3283064e-10`; vLLM's two-module sum is unit-tested and covered by end-to-end logprob parity. |
| Prefix reuse preserves token, logprob, and route results | **confirmed** | Cold and reused requests agree across a physical KV-block boundary and the 512-token window boundary in fixture and live P0 jobs. |
| One-token prefix mutation causes a miss | **confirmed** | Mutated requests reported zero reused tokens and zero new hit-counter tokens at both boundaries. |
| Append-to-conversation request shape works | **confirmed for the exact dummy path** | Appends produce real hits, sampled-token logprobs, routed IDs, fixed response lengths, and synchronized counters. |
| Seeded dummy routing is reproducible | **confirmed** | Repeated correctness checks and the two acceptance arms produced stable route evidence. |
| Seeded dummy routing is balanced | **rejected** | 121 experts were unused; the busiest EP rank had 26,592 assignments versus a 6,648 mean. The cyclic balanced case is an instrumentation control only. |
| The pinned Snowball export is directly loadable by this request path | **rejected** | The only allowed attempt failed before load because path-style `ListObjectsV2` was rejected. It was not retried or copied. |
| Local active KV plateaus while global active KV grows | **confirmed** | With one active request, local blocks stayed at 33 while global blocks grew from 180 at 6,144 tokens to 2,039 at 65,536 tokens. |
| Semantic, reserved, and active KV can be separated | **confirmed** | Active physical bytes matched group payload exactly; reserved capacity stayed at 61,899,276,288 bytes while active use grew from 299,630,592 to 1,761,607,680 bytes. |
| The same unattended path works at EP8 and EP16 | **confirmed** | Two and four whole GB200 nodes rendezvoused through the checked-in Iris entrypoint with hard `nvlink.domain` coscheduling. |
| Two warmed acceptance arms differ by no more than 2% | **confirmed** | Stable live-counter means were 1,578.2497 and 1,583.5041 generated tokens/s, a 0.3324% difference. |
| Claimed S3 bundles survive independent readback | **confirmed** | Separate authorized reader jobs read every claimed object and verified byte identity and recorded hashes. |

## Frozen tensor and prefix parity

The downscaled exact fixture has seven layers, hidden size 64, four query
heads, two local KV heads, one global KV head, two-of-four routed experts, and
two separate shared experts of width 16. It enables every custom path used by
the full reference.

```text
model.safetensors SHA-256:
  6f96fee7651e44e1dd610d7e73ab7df668b3e81411a32713a60c9b9e31b8137d
observations.npz SHA-256:
  3458fb4c89d101a030724f1cddbac7698168894cdb56b92f4ad7fdcc5b6240ce
GPU job:
  /romain/grugmoe-tiny-fixture13-ee7e05dc3-20260731
S3:
  s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/tiny/fixture13-ee7e05dc3-20260731/
independent reader:
  /romain/grugmoe-fixture13-inspect-20260731
result SHA-256:
  3beb9f7c361f773296dffb167317e5448fa47f352acf219b2def09a9ebbc9827
```

The tensor oracle checked 33 tokens in every layer. Selected experts matched;
the maximum normalized-weight error was `2.384185791015625e-07`. The two
shared experts were nonzero and distinct. On their frozen oracle inputs, the
separate-sum and fused-concat representations differed by at most
`2.3283064365386963e-10`. vLLM's two-module sum has direct unit coverage, and
the live next-token comparison covers the summed path end to end.

At 33 semantic tokens, the live server and frozen oracle chose token 63 and
the maximum probability error was `8.575512504406177e-05`. At the 514-token
boundary, cold and 512-token-reused results were byte-identical in returned
scores and routes. The live token 24 and frozen token 22 were separated by
only `8.536872245598626e-06` in the frozen distribution; the measured maximum
probability error was `4.70345119299477e-05`, well inside the repository
`0.075` parity bound.

## P0 readiness and live run index

All distinct implementation families are smokeable. `config-only` means the
new exact path needs no additional code for that value; `custom-code` means
the branch supplies the path that was missing at the vLLM base.

| P0 family | Implementation | Readiness | Live case |
|---|---|---|---|
| uniform KV / every 4 / SConv off | `config-only` | `ready` | `legacy-control-ep4` |
| heterogeneous KV / every 6 / SConv on | `custom-code` | `ready` | `one-node-ep4`, then exact `reference-ep8` |
| global KV 2 / window 2,048 | `config-only` on the heterogeneous path | `ready` | `kv2-window2048-ep4` |
| top-8 / 256 experts / EP16 | `config-only` | `ready` | `granular-ep16` |
| exact reference / EP16 | `custom-code` plus unattended launcher | `ready` | `exact-reference-ep16` acceptance |

Every row below ended with all seven aggregate checks true and an independent
reader in `succeeded`.

| Case | GPU job and S3 prefix | Independent hashes |
|---|---|---|
| One-node exact path, EP4 | `/romain/grugmoe-one-node-ep4-p0-one-node-ep4-hybrid-align-224c78d21-20260731`<br>`s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/one-node-ep4/p0-one-node-ep4-hybrid-align-224c78d21-20260731/` | result `26234fd38273af6a541e9f7e815a7249e02f31829db13e9011b1eaa828032683`<br>manifest `d9ae9d1f344ba1447d2ed2712c2418adfb22265c56b6e9057d3afab9a0161932` |
| Legacy control, EP4 | `/romain/grugmoe-legacy-control-ep4-p0-legacy-control-ep4-224c78d21-20260731`<br>`s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/legacy-control-ep4/p0-legacy-control-ep4-224c78d21-20260731/` | result `afb928778ce9c24cca8ecd7a6dc0bffc5b0430de5e1869afc50aed147dfbd9d9`<br>manifest `70234b178fbe223f2c4c3449775cbcd4f5c5d661c83b9c725cbb9b52881939b5` |
| KV2/window-2048, EP4 | `/romain/grugmoe-kv2-window2048-ep4-p0-kv2-window2048-ep4-224c78d21-20260731`<br>`s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/kv2-window2048-ep4/p0-kv2-window2048-ep4-224c78d21-20260731/` | result `e8551deedc3ea7490d4a81d98e0216b9af24d0ec6a06ecae5f1b444c1d052aef`<br>manifest `6976962c1bc83fe55b29c612696818cde37981f8fc6db4d1647a86203ef6d4a9` |
| Exact reference, EP8 | `/romain/grugmoe-reference-ep8-p0-reference-ep8-node-proof-03e3767e4-20260731`<br>`s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/p0-reference-ep8-node-proof-03e3767e4-20260731/` | result `7415c33b2051b688beca9c6b956fbbcf7643c0f1af9cbb1f36772c3c9dfb8d20`<br>manifest `4f4d2fb991b12c39491f062739e628c92f5834bf7b311395687c66d0104c29e5` |
| Granular top-8/256, EP16 | `/romain/grugmoe-granular-ep16-p0-granular-ep16-03e3767e4-20260731`<br>`s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/granular-ep16/p0-granular-ep16-03e3767e4-20260731/` | result `607dd5d0039b5a75db5d506a5dc485bfbf1b4587efe8cd73554a84276e843802`<br>manifest `adaff6ae99c3c57a55c844fc72f781822a5c4d7a1d56edecca849da19211e2a1` |

The exact EP8 job used two distinct whole nodes in
`DH1-392-US-EAST-08A`. The granular EP16 job used four whole nodes in
`DH1-136-US-EAST-08A`. Both used the same zero-retry submit, rendezvous,
health, correctness, aggregation, upload, and readback path later used by the
exact acceptance.

## Exact live KV result

```text
GPU job:
  /romain/grugmoe-reference-ep8-kv-reference-ep8-live-03e3767e4-20260731
S3:
  s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/kv-reference-ep8-live-03e3767e4-20260731/
independent reader:
  /romain/grugmoe-kv-reference-ep8-inspect-20260731
bundle digest:
  d419d4691fa26f6ba4960653d35cbc2ca2e7462a8213a223c2defc0d83a6eaa6
result SHA-256:
  6440a3d4f140c0bc29d47f346a8b646e3420ed59c6fe016b004ae3c38287ef02
manifest SHA-256:
  d8c7a99b620a4c5733c7d7bb10f7b5b5205eed2a4c9c8adbdd51fd4f9f3fdce0
KV summary SHA-256:
  3d887ad3c21f729c2271424b38300e26c7abad848490587daccf3d8adda226e5
```

Prefix caching remained enabled. Both observations held exactly one active
request on DP rank 0.

| Final tokens | Local / global / SConv active blocks | Attention active bytes | SConv active bytes | Physical active bytes | Reserved physical bytes | Predicted attention bytes | Prediction gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,144 | 33 / 180 / 2 | 271,319,040 | 28,311,552 | 299,630,592 | 61,899,276,288 | 276,824,064 | 1.9886% |
| 65,536 | 33 / 2,039 / 2 | 1,733,296,128 | 28,311,552 | 1,761,607,680 | 61,899,276,288 | 1,736,441,856 | 0.1812% |

Local active blocks plateau at the 512-token window. Global blocks grow with
context. SConv state also stays bounded. Physical active bytes equal the
reported group payload, so there is no unexplained active-use gap, let alone
one above 10%.

The large reserved/active gap is real but has a different meaning: vLLM
reserves a 61.9 GB page pool up front and then assigns only the active pages
shown above. Reserved capacity must not be presented as per-request occupancy.
There were 30 live groups: 6 attention and 24 SConv; 29 were sliding-window
groups and one was full attention.

## Four-node acceptance

### First attempt: terminal failure with a diagnosed measurement defect

The first exact job exercised the full model, placement, correctness, warm
phase, workload, two arms, and S3 readback. It failed duration and
repeatability because the harness credited all 2,048 generated tokens in the
minute when a request returned. Requests took 57--95 seconds, so continuously
busy minutes could appear as zero.

```text
job:
  /romain/grugmoe-exact-reference-ep16-acceptance-03e3767e4-20260731
S3:
  s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/exact-reference-ep16/acceptance-03e3767e4-20260731/
independent reader:
  /romain/grugmoe-exact-acceptance-inspect-20260731
bundle / result / manifest:
  fb1e3ea3b7a4a3cc90842061a998cfc356f6e2c19910fc612c8a7e6333cb3cc0
  b5e2847017b68d8edb2a01dd307252b569f2a3b0c6209f34a41cd46bfc291969
  f5bc773f9ade1f3e266bab0ff8126175f60d2f110101cc124fe64eed9ff56839
```

The vLLM generation counter exactly matched each arm's completed-token total.
The fix therefore sampled `vllm:generation_tokens` at fixed wall-clock
boundaries while requests remained active and used adjacent counter deltas.
A compressed unit test makes requests slower than the sample interval and
proves that busy intervals do not become false zeroes. No model, workload,
topology, acceptance threshold, or result predicate changed.

### Single diagnosed-fix rerun: passed

```text
job:
  /romain/grugmoe-exact-reference-ep16-acceptance-counter-a3320a304-20260731
S3:
  s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/exact-reference-ep16/acceptance-counter-a3320a304-20260731/
independent reader:
  /romain/grugmoe-exact-acceptance-counter-inspect-20260731
bundle digest:
  8b10500844d3df77a4b5af73e5ad34336cb08c0b7d83f4bcc6d95004e9767841
result SHA-256:
  cba763ac6b5fda8593fe52cfb6d1d1b1a5a8ce2d7e4d2330a27ca855b88b0b35
manifest SHA-256:
  6b4abccb705aabd60b8c0a8125148819fc259615f87c223aaa20889b7737b631
objects read: 51
```

The job used four distinct whole GB200 nodes in one NVLink domain:

```text
rank 0: s9lrxs64  10.186.210.237
rank 1: scspxs64  10.186.210.213
rank 2: shmsxs64  10.186.210.227
rank 3: sjjvxs64  10.186.210.207
domain: DH1-129-US-EAST-08A
rack:   dh1-r129-us-east-08a
```

The workload had 18 roots and 8 branches per root. Six roots used each cached
history length 10,240, 30,720, and 62,464. Every branch appended 1,024 tokens
and generated 2,048, producing 48 requests at each final length 13,312,
33,792, and 65,536. The warm phase covered every prefix cohort with 162
requests and 331,776 generated tokens.

| Measure | Arm 1 | Arm 2 |
|---|---:|---:|
| Wall time | 603.7255 s | 600.1098 s |
| Requests | 464 | 464 |
| Generated tokens | 950,272 | 950,272 |
| Branches covered | 144 / 144 | 144 / 144 |
| Concurrency | 64 | 64 |
| Full-arm mean | 1,574.0133 tok/s | 1,583.4970 tok/s |
| Ten-interval stable mean | 1,578.2497 tok/s | 1,583.5041 tok/s |
| Stable counter window | 602.1050 s | 600.1071 s |
| Prefix 10,240 generated tokens | 360,448 | 360,448 |
| Prefix 30,720 generated tokens | 294,912 | 294,912 |
| Prefix 62,464 generated tokens | 294,912 | 294,912 |
| Preemptions | 0 | 0 |

Stable live-counter deltas were:

```text
arm 1:
  131072,65285,79598,72034,78555,130512,119618,79617,63685,130296
arm 2:
  131072,73724,74520,43721,103205,123854,96242,71422,101506,131006
```

Every interval was positive. The stable means differed by
`0.33237259800135577%`. All seven required checks were true:

```text
placement=true
all_rank_health=true
correctness=true
duration=true
token_count=true
repeatability=true
artifact_readback=true
```

All 33 manifest objects were byte-identical on readback, all four rank
receipts passed, the aggregate result was byte-identical, and the independent
reader itself reached `succeeded`. No third acceptance was run.

## Snowball bounded result

The one allowed trained-checkpoint request-path attempt used the pinned export:

```text
s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/
```

It failed before model load when path-style `ListObjectsV2` was rejected. The
branch now writes virtual-hosted S3 configuration and rejects zombie launcher
parents, but the goal forbade a retry or an object copy. Therefore:

- the exact dummy request shape is confirmed;
- the pinned Snowball storage/request path is rejected at this evidence
  revision;
- no Snowball throughput or model behavior is claimed.

## Ranked remaining risks

1. **Trained-checkpoint access.** The exact serving implementation is proven
   with frozen and deterministic dummy tensors, but the pinned Snowball export
   did not reach load. Review and validate object-store access before calling
   any later run trained-model evidence.
2. **Representative routing.** Dummy routing is highly skewed. Future
   architecture results need model routing histograms and EP max/mean beside
   throughput. The cyclic balanced case validates instrumentation only.
3. **Fork maintenance.** Exact support touches the vLLM model, hybrid cache,
   scheduler telemetry, and SConv code. It needs normal owner review before
   long-term use.
4. **Capacity interpretation.** The runtime reserves much more KV memory than
   one request actively occupies. Capacity conclusions must use active pages
   and resident-request behavior, not divide the whole reserved pool by a
   single sequence.
5. **Experiment scope.** This preflight deliberately did not compare
   candidates or test 131K context. Those remain work for the later
   architecture protocol, not missing preflight evidence.

## Local validation

Marin:

```text
uv run pytest -q \
  experiments/grug/moe/test_inference_preflight.py \
  tests/cluster/vllm/backend_parity.py \
  tests/cluster/vllm/test_grug_exact_reference_check.py \
  scripts/iris/tests/test_grugmoe_inference_preflight.py \
  tests/inference/test_serve.py

95 passed
```

The repository-specific `./infra/pre-commit.py` checks and
`git diff --check` pass for the changed Marin files.

vLLM:

```text
PYTHONPATH=$PWD \
  /home/romain/dev/marin-wt/grug-pp2-multinode-vllm-20260729/.venv/bin/python \
  -m pytest -q tests/models/test_grugmoe.py \
  tests/v1/core/test_prefix_caching.py

100 passed, 2 skipped
```

All applicable vLLM pre-commit hooks and `git diff --check` pass for the ten
changed files.

The first four local Max-effort goal reviews produced three passes and one
request for two narrow corrections: reject malformed unstacked expert tensors,
and describe the shared-expert equivalence measurement precisely. Both are
resolved in the reviewed vLLM head and this findings revision. Final review is
run after these corrections and clean validation.
