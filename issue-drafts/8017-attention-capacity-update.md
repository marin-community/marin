🤖

## Attention-capacity follow-up

**On throughput and capacity under this frozen BF16 dummy-weight GB200 rollout benchmark, keep global attention every six layers.** Both variants reached the largest tested load, 1,536 concurrent rollout requests, without a KV-cache preemption. The full A→B and B→A confirmation makes `>=1536` the maximum-safe lower bound for both variants. It does not show a larger capacity ceiling for global attention every four layers. The throughput knee is 144 for both variants: 144 is the lowest stable point within 5% of that variant's best stable throughput. This is inference-cost evidence, not model-quality evidence; the final architecture choice remains with modeling.

The short provisional scouts favored every four at 768–1,152 concurrency, but those single-order results were not confirmed. At 1,536, the full confirmation found every four 5.642% slower in A→B order and 5.223% slower in B→A order, for an order-balanced 5.435% slowdown. The intermediate scout advantage remains unconfirmed and does not set the decision.

| Concurrent requests | Evidence | Global every 6, generated tok/s/GPU | Global every 4, generated tok/s/GPU | Every-4 change |
| ---: | --- | ---: | ---: | ---: |
| 144 | stable A→B / B→A | 300.498 / 298.899 | 280.845 / 283.769 | -6.540% / -5.062% |
| 384 | provisional, separate allocations | 605.524 | 579.866 | not paired |
| 768 | provisional A→B scout | 501.522 | 555.294 | +10.722% |
| 1,152 | provisional A→B scout | 268.915 | 310.358 | +15.411% |
| 1,536 | stable A→B / B→A | 262.630 / 256.754 | 247.813 / 243.343 | -5.642% / -5.223% |

| Variant | Best stable throughput | Throughput knee | Maximum safe concurrency |
| --- | ---: | ---: | ---: |
| Global every 6 | 304.694 generated tok/s/GPU at 144 | 144 | `>=1536` |
| Global every 4 | 282.307 generated tok/s/GPU at 144 | 144 | `>=1536` |

“Stable” means the full confirmation passed the applicable [#7912](https://github.com/marin-community/marin/issues/7912) load, duration, token, manifest, health, and provenance gates with zero vLLM KV-cache preemptions. Scout rows deliberately used lower floors and are provisional. Higher is better for generated-token throughput.

The two best-stable cells are per-variant maxima, not a matched cross-variant comparison. The every-six value is the standalone result from #8017's accepted `max_num_batched_tokens=8192` reference calibration; the every-four value is the mean of its paired C144 A→B and B→A rates. Use the order-matched rows in the first table for relative cost. The knee is literal under the pinned definition and the confirmed stable grid; it is not a claim that throughput saturated or that a capacity cliff occurred at 144.

<details>
<summary><strong>Observed load, queues, request counts, and latency</strong></summary>

The accepted lower-concurrency reference curve was reused without another GPU run:

| Concurrent requests | Global every 6, generated tok/s/GPU | Evidence |
| ---: | ---: | --- |
| 24 | 63.524 | stable reference calibration |
| 48 | 111.566 | stable reference calibration |
| 72 | 153.382 | stable reference calibration |
| 96 | 196.250 | stable reference calibration |
| 144 | 304.694 | stable reference calibration |

The client held exactly 1,536 occupied slots throughout every accepted C1536 plateau: minimum, mean, maximum, and close were all 1,536. Each request completion retained its slot until the controller consumed it and submitted the frozen same-cohort successor.

| Order | Variant | Plateau / whole / drain successes | Running requests/rank, plateau mean range | Waiting requests/rank, plateau mean range | Peak resident requests/rank range | TPOT p50 / p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A→B | Global every 6 | 1,667 / 3,203 / 1,536 | 71.1–79.3 | 16.2–24.4 | 93–96 | 0.350s / 0.399s |
| A→B | Global every 4 | 1,666 / 3,202 / 1,536 | 50.3–60.3 | 34.9–44.9 | 72–82 | 0.250s / 0.299s |
| B→A | Global every 6 | 1,702 / 3,238 / 1,536 | 72.0–80.5 | 15.0–23.5 | 94–96 | 0.350s / 0.399s |
| B→A | Global every 4 | 1,649 / 3,185 / 1,536 | 49.1–59.9 | 35.4–46.1 | 69–86 | 0.250s / 0.299s |

Peak resident sequences by DP rank 0–15 were:

- A→B every six: `[96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 93, 96]`.
- A→B every four: `[82, 74, 76, 77, 74, 74, 79, 72, 76, 76, 78, 76, 79, 82, 79, 77]`.
- B→A every six: `[95, 96, 96, 96, 96, 96, 96, 96, 95, 94, 96, 96, 96, 96, 96, 96]`.
- B→A every four: `[79, 80, 73, 76, 84, 82, 77, 75, 78, 80, 78, 76, 76, 86, 69, 76]`.

Time per output token (TPOT) is the server's decode-only interval [`(last token timestamp - first token timestamp) / (generated tokens - 1)`](https://github.com/marin-community/vllm/blob/06af5cff3b97723356ec590b9ecf635b7690bd40/vllm/v1/metrics/stats.py#L440-L459). It excludes queue time and prefill; lower is better. A lower TPOT can coexist with lower aggregate throughput when queueing, prefill, or the number of resident sequences differs.

</details>

<details>
<summary><strong>KV cache and resident sequences</strong></summary>

| Order | Variant | Peak active sequences | Physical / semantic KV | KV per active sequence | Attention / SConv KV | Reserved KV | Preemptions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B | Global every 6 | 1,391 | 1,385.317 / 1,385.317 GiB | 0.996 GiB | 1,183.828 / 201.489 GiB | 1,444.031 GiB | 0 |
| A→B | Global every 4 | 960 | 1,386.537 / 1,386.537 GiB | 1.444 GiB | 1,198.003 / 188.534 GiB | 1,444.025 GiB | 0 |
| B→A | Global every 6 | 1,362 | 1,391.859 / 1,391.859 GiB | 1.022 GiB | 1,201.285 / 190.573 GiB | 1,444.031 GiB | 0 |
| B→A | Global every 4 | 933 | 1,384.996 / 1,384.996 GiB | 1.484 GiB | 1,199.529 / 185.467 GiB | 1,444.025 GiB | 0 |

Semantic KV is the cache implied by the model layout. Physical KV is the allocator-padded byte count reported by the GrugMoE instrumentation. The peak is a sampled instant, so its active-request count need not equal the 1,536 client slots. Reserved bytes are the cache capacity made available to the 16 engines.

</details>

<details>
<summary><strong>Protocol, provenance, and independent readback</strong></summary>

The accepted 144-request pair is reused unchanged from #8017. The upper ladder added deterministic prompt roots in balanced blocks while keeping BF16 dummy weights, EP16, 16 collocated GB200 GPUs, the rolling 13K/34K/65K mixed workload, eight candidates per prompt, R3 off, seed 1234, `max_num_batched_tokens=8192`, prefix-cache reset behavior, and all serving flags fixed. Client concurrency and `max_num_seqs` were the only load-limit changes. Each confirmed order ran both variants sequentially in one four-node allocation and one NVLink domain with zero retries.

Before live submission, the focused CPU suite returned 76 passing tests with targeted Ruff and whitespace checks clean. It covers exact C144 manifest compatibility, unique request IDs and full prompts, balanced roots across DP16, paired input and flag identity, capacity result math, and independent readback.

The C1536 confirmations use [Marin `8516be8422b1b1155cdf59fb51763f62fa695ed8`](https://github.com/marin-community/marin/commit/8516be8422b1b1155cdf59fb51763f62fa695ed8). The reused C144 pair uses [Marin `59f6a8e0e0be64ca9ced3146ef41e07cdf6b9a8e`](https://github.com/marin-community/marin/commit/59f6a8e0e0be64ca9ced3146ef41e07cdf6b9a8e). All runs pin [vLLM `06af5cff3b97723356ec590b9ecf635b7690bd40`](https://github.com/marin-community/vllm/commit/06af5cff3b97723356ec590b9ecf635b7690bd40) and this immutable image:

```text
ghcr.io/marin-community/iris-task@sha256:9af9a3d38f57c2ed8dfe1d6f6657a9f4a00c582ec06a5ac2af8fcddbe51da03c
```

- C1536 A→B: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-confirm-c1536-matched-ab-prod-20260807t1707z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-readback-capacity-confirm-c1536-matched-ab-prod-20260807t1707z).
- C1536 B→A: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-confirm-c1536-matched-ba-prod-20260807t1938z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-readback-capacity-confirm-c1536-matched-ba-prod-20260807t1938z).
- C144 A→B and B→A: [producer A→B](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-global-every4-ep16-ab-20260806t0545z), [reader A→B](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-readback-global-every4-ep16-ab-20260806t0545z), [producer B→A](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-global-every4-ep16-ba-20260806t0629z), [reader B→A](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-readback-global-every4-ep16-ba-20260806t0629z).
- Provisional ladder: [C384 every-four producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c384-matched-ab-20260807t0202z), [C384 every-six producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c384-reference-single-20260807t0301z), [C768 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c768-matched-ab-20260807t0328z), [C1152 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c1152-matched-ab-prod-20260807t1326z), [C1536 scout producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c1536-matched-ab-prod-20260807t1503z).
- Lower reference curve: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-instrument-v1-instrument-v1-0057bb4500-20260806t0144z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-instrument-v1-readback-instrument-v1-0057bb4500-20260806t0144z).

Producer results and independently generated receipts:

| Evidence | Producer result | Independent receipt |
| --- | --- | --- |
| Lower reference curve | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-0/instrument-v1/instrument-v1-0057bb4500-20260806t0144z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/instrument-v1/instrument-v1-0057bb4500-20260806t0144z/independent-readback.json) |
| C144 A→B | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-pair-v1/global-every4-ep16-ab-20260806t0545z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-pair-v1/global-every4-ep16-ab-20260806t0545z/independent-readback.json) |
| C144 B→A | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-pair-v1/global-every4-ep16-ba-20260806t0629z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-pair-v1/global-every4-ep16-ba-20260806t0629z/independent-readback.json) |
| C384 every four | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c384-matched-ab-20260807t0202z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c384-matched-ab-20260807t0202z/independent-readback.json) |
| C384 every six | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c384-reference-single-20260807t0301z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c384-reference-single-20260807t0301z/independent-readback.json) |
| C768 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c768-matched-ab-20260807t0328z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c768-matched-ab-20260807t0328z/independent-readback.json) |
| C1152 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c1152-matched-ab-prod-20260807t1326z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c1152-matched-ab-prod-20260807t1326z/independent-readback.json) |
| C1536 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c1536-matched-ab-prod-20260807t1503z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c1536-matched-ab-prod-20260807t1503z/independent-readback.json) |
| C1536 A→B confirmation | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-confirm-c1536-matched-ab-prod-20260807t1707z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-confirm-c1536-matched-ab-prod-20260807t1707z/independent-readback.json) |
| C1536 B→A confirmation | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-confirm-c1536-matched-ba-prod-20260807t1938z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-confirm-c1536-matched-ba-prod-20260807t1938z/independent-readback.json) |

```text
C144 A→B: manifest.json 910a709c98ef71312dfbd6be96b5fcf628407646924894881f820c46f8a6856a; result.json e9b5ae7aa954ec9294c2e05199425dbd4ebf97e222bbd35cb6272590969ad561; independent-readback.json ef80d873ab451b183737dc56a4e3e9f95c8b58b1ebcdf0221d3eaf96480f585b
C144 B→A: manifest.json 006766b9ed8befd0130263061e4ca83a76e39512dda6ab4ac166e7269f1c2304; result.json 59825ec0c8b93fba9a7b9426a35f8e04fa66f69504d4bfcdf2ef11f17ea11afc; independent-readback.json b856a5276f6992be869dbbe624d5c6d89e4463237ca69cfffc315d31cf7c6a7e
C1536 A→B: manifest.json 3cefae1dbf341a4dbae3762e42b33c496bfd504b15be2e76d8b034a471be7e00; result.json 04623d13f63eaffa3bc509c5e454b35c6f3ca671fe1a0031ced7e02bf55c0466; independent-readback.json 83226268b52125700ce9e913b5d659c146da5a5a79d1f8e34a487fc16134c057
C1536 B→A: manifest.json 4713138c6eb9a8059e291e13c539fdf622fc85aa8f3a98dfdc9c8932f36a4d32; result.json 896d905aa83f3ca417585dc6239f00c94aaaf257b90a8c4cbd317c6602f31f03; independent-readback.json 278b43b7c190143dddd88ec7b1adb2dd5c247fdb28048b88bb270e268629d0df
```

</details>

<details>
<summary><strong>Excluded and provisional evidence</strong></summary>

- The first C384 attempt is excluded. Its provisional reference plateau had no long-cohort completion, so latency summarization failed and no arm result was accepted. The harness was narrowed to require one completion from every frozen cohort before a scout plateau may close.
- The repaired matched C384 attempt lost one rank while fetching the pinned vLLM commit from GitHub. Its every-six phase produced no arm. The later every-four phase and a separate reference-only retry each passed independent readback, but they used separate allocations; their rates are retained only as one-sided scout evidence and no matched percentage is reported.
- C768, C1152, and C1536 scout arms passed their applicable gates with zero KV preemptions, but they did not meet the full duration, generated-token, and manifest floors. They selected C1536 for confirmation; they do not set the knee, the maximum-safe bound, or the relative-throughput conclusion.
- An interactive C1152 submission remained pre-admission because no NVLink domain could fit four whole nodes. It consumed no GPU benchmark time and contributes no curve point. The later authorized production scout is the provisional C1152 row above.
- No failed, unmatched, pre-admission, stopped, or unread artifact contributes to a stable result.

</details>
