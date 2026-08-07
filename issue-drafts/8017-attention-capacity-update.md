## Attention-capacity follow-up

> [!IMPORTANT]
> **Keep global attention every six layers for inference cost.** At 1,536 concurrent rollout requests, global-every-4 was 5.435% slower after balancing both run orders. Both variants served the workload with zero vLLM KV-cache preemptions.

Larry asked how throughput changes as batch size grows. In continuous vLLM serving, the closest control is **concurrency**: the number of requests the client keeps in flight.

| Question | Answer |
| --- | --- |
| Which attention schedule is faster? | Global-every-6. Global-every-4 was 5.06–6.54% slower at 144 requests and 5.22–5.64% slower at 1,536. |
| Does the larger every-4 KV cache create pressure? | Yes. Every-4 kept roughly one-third fewer sequences resident and queued more work. It still completed the 1,536-request workload without cache preemption. |
| Where does throughput saturate? | The provisional curve peaks near 384 requests, but the exact point is not confirmed. The 384–1,152 scouts used shorter checks and were not repeated in both run orders. |

For the decision-driving points, each variant ran back-to-back on the same 16 collocated GB200 GPUs, then we reversed the order. This controls for run-order drift.

| Concurrent requests | Run order | Global-every-6, generated tok/s/GPU | Global-every-4, generated tok/s/GPU | Every-4 change |
| ---: | --- | ---: | ---: | ---: |
| 144 | Every-6 then every-4 | 300.498 | 280.845 | -6.540% |
| 144 | Every-4 then every-6 | 298.899 | 283.769 | -5.062% |
| 1,536 | Every-6 then every-4 | 262.630 | 247.813 | -5.642% |
| 1,536 | Every-4 then every-6 | 256.754 | 243.343 | -5.223% |

This is BF16 dummy-weight inference-cost evidence for the frozen 13K/34K/65K rolling workload. It does not measure model quality, and the final architecture choice remains with modeling.

<details>
<summary><strong>Full capacity curve and server health</strong></summary>

The accepted lower-concurrency global-every-6 curve was reused from #8017:

| Concurrent requests | Generated tok/s/GPU |
| ---: | ---: |
| 24 | 63.524 |
| 48 | 111.566 |
| 72 | 153.382 |
| 96 | 196.250 |
| 144 | 304.694 |

The upper scouts suggest a peak near 384, followed by lower throughput as concurrency grows. These points are useful directionally, but they did not meet the full confirmation standard:

| Concurrent requests | Global-every-6, generated tok/s/GPU | Global-every-4, generated tok/s/GPU | Evidence |
| ---: | ---: | ---: | --- |
| 384 | 605.524 | 579.866 | Separate allocations; not paired |
| 768 | 501.522 | 555.294 | Single-order scout |
| 1,152 | 268.915 | 310.358 | Single-order scout |

At 1,536 requests, the client kept all 1,536 slots occupied throughout every accepted measurement plateau. Each completed request was immediately replaced with another request from the same frozen cohort.

In the table below, A→B means global-every-6 followed by global-every-4; B→A means the reverse.

| Order | Variant | Plateau / whole / drain successes | Running requests/rank, plateau mean range | Waiting requests/rank, plateau mean range | Peak resident requests/rank range | TPOT p50 / p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A→B | Global every 6 | 1,667 / 3,203 / 1,536 | 71.1–79.3 | 16.2–24.4 | 93–96 | 0.350s / 0.399s |
| A→B | Global every 4 | 1,666 / 3,202 / 1,536 | 50.3–60.3 | 34.9–44.9 | 72–82 | 0.250s / 0.299s |
| B→A | Global every 6 | 1,702 / 3,238 / 1,536 | 72.0–80.5 | 15.0–23.5 | 94–96 | 0.350s / 0.399s |
| B→A | Global every 4 | 1,649 / 3,185 / 1,536 | 49.1–59.9 | 35.4–46.1 | 69–86 | 0.250s / 0.299s |

Time per output token (TPOT) is the server's decode-only interval [`(last token timestamp - first token timestamp) / (generated tokens - 1)`](https://github.com/marin-community/vllm/blob/06af5cff3b97723356ec590b9ecf635b7690bd40/vllm/v1/metrics/stats.py#L440-L459). It excludes queue time and prefill. A lower TPOT can coexist with lower aggregate throughput when queueing, prefill, or the number of resident sequences differs.

</details>

<details>
<summary><strong>KV-cache pressure</strong></summary>

The following values are totals across the 16 engines at the sampled KV peak:

| Order | Variant | Peak active sequences | Physical / semantic KV | KV per active sequence | Attention / SConv KV | Reserved KV | Preemptions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B | Global every 6 | 1,391 | 1,385.317 / 1,385.317 GiB | 0.996 GiB | 1,183.828 / 201.489 GiB | 1,444.031 GiB | 0 |
| A→B | Global every 4 | 960 | 1,386.537 / 1,386.537 GiB | 1.444 GiB | 1,198.003 / 188.534 GiB | 1,444.025 GiB | 0 |
| B→A | Global every 6 | 1,362 | 1,391.859 / 1,391.859 GiB | 1.022 GiB | 1,201.285 / 190.573 GiB | 1,444.031 GiB | 0 |
| B→A | Global every 4 | 933 | 1,384.996 / 1,384.996 GiB | 1.484 GiB | 1,199.529 / 185.467 GiB | 1,444.025 GiB | 0 |

Every-4 used more KV memory per active sequence, so fewer sequences remained resident and more requests waited. The remaining requests queued rather than causing KV-cache preemption.

Semantic KV is the cache implied by the model layout. Physical KV is the allocator-padded byte count reported by the GrugMoE instrumentation. The peak is a sampled instant, so its active-request count need not equal the 1,536 client slots. Reserved KV is the total cache capacity available to the 16 engines.

</details>

<details>
<summary><strong>Protocol, provenance, and independent readback</strong></summary>

The accepted 144-request pair is reused unchanged from #8017. The upper ladder added deterministic prompt roots in balanced blocks while keeping BF16 dummy weights, EP16, 16 collocated GB200 GPUs, the rolling 13K/34K/65K mixed workload, eight candidates per prompt, R3 off, seed 1234, `max_num_batched_tokens=8192`, prefix-cache reset behavior, and all serving flags fixed. Client concurrency and `max_num_seqs` were the only load-limit changes. Each confirmed order ran both variants sequentially in one four-node allocation and one NVLink domain with zero retries.

Before live submission, the focused CPU suite returned 76 passing tests with targeted Ruff and whitespace checks clean. It covers exact C144 manifest compatibility, unique request IDs and full prompts, balanced roots across DP16, paired input and flag identity, capacity-result math, and independent readback.

The 1,536-request confirmations use [Marin `8516be8422b1b1155cdf59fb51763f62fa695ed8`](https://github.com/marin-community/marin/commit/8516be8422b1b1155cdf59fb51763f62fa695ed8). The reused 144-request pair uses [Marin `59f6a8e0e0be64ca9ced3146ef41e07cdf6b9a8e`](https://github.com/marin-community/marin/commit/59f6a8e0e0be64ca9ced3146ef41e07cdf6b9a8e). All runs pin [vLLM `06af5cff3b97723356ec590b9ecf635b7690bd40`](https://github.com/marin-community/vllm/commit/06af5cff3b97723356ec590b9ecf635b7690bd40) and this immutable image:

```text
ghcr.io/marin-community/iris-task@sha256:9af9a3d38f57c2ed8dfe1d6f6657a9f4a00c582ec06a5ac2af8fcddbe51da03c
```

- 1,536 A→B: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-confirm-c1536-matched-ab-prod-20260807t1707z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-readback-capacity-confirm-c1536-matched-ab-prod-20260807t1707z).
- 1,536 B→A: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-confirm-c1536-matched-ba-prod-20260807t1938z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-readback-capacity-confirm-c1536-matched-ba-prod-20260807t1938z).
- 144 A→B and B→A: [producer A→B](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-global-every4-ep16-ab-20260806t0545z), [reader A→B](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-readback-global-every4-ep16-ab-20260806t0545z), [producer B→A](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-global-every4-ep16-ba-20260806t0629z), [reader B→A](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-pair-v1-readback-global-every4-ep16-ba-20260806t0629z).
- Provisional ladder: [384 every-4 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c384-matched-ab-20260807t0202z), [384 every-6 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c384-reference-single-20260807t0301z), [768 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c768-matched-ab-20260807t0328z), [1,152 producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c1152-matched-ab-prod-20260807t1326z), [1,536 scout producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-attention-capacity-v1-capacity-scout-c1536-matched-ab-prod-20260807t1503z).
- Lower reference curve: [producer](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-instrument-v1-instrument-v1-0057bb4500-20260806t0144z), [independent reader](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fromain%2Fgrugmoe-instrument-v1-readback-instrument-v1-0057bb4500-20260806t0144z).

Producer results and independently generated receipts:

| Evidence | Producer result | Independent receipt |
| --- | --- | --- |
| Lower reference curve | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-0/instrument-v1/instrument-v1-0057bb4500-20260806t0144z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/instrument-v1/instrument-v1-0057bb4500-20260806t0144z/independent-readback.json) |
| 144 A→B | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-pair-v1/global-every4-ep16-ab-20260806t0545z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-pair-v1/global-every4-ep16-ab-20260806t0545z/independent-readback.json) |
| 144 B→A | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-pair-v1/global-every4-ep16-ba-20260806t0629z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-pair-v1/global-every4-ep16-ba-20260806t0629z/independent-readback.json) |
| 384 every 4 | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c384-matched-ab-20260807t0202z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c384-matched-ab-20260807t0202z/independent-readback.json) |
| 384 every 6 | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c384-reference-single-20260807t0301z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c384-reference-single-20260807t0301z/independent-readback.json) |
| 768 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c768-matched-ab-20260807t0328z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c768-matched-ab-20260807t0328z/independent-readback.json) |
| 1,152 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c1152-matched-ab-prod-20260807t1326z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c1152-matched-ab-prod-20260807t1326z/independent-readback.json) |
| 1,536 scout | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-scout-c1536-matched-ab-prod-20260807t1503z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-scout-c1536-matched-ab-prod-20260807t1503z/independent-readback.json) |
| 1,536 A→B confirmation | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-confirm-c1536-matched-ab-prod-20260807t1707z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-confirm-c1536-matched-ab-prod-20260807t1707z/independent-readback.json) |
| 1,536 B→A confirmation | [result.json](s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3/attention-capacity-v1/capacity-confirm-c1536-matched-ba-prod-20260807t1938z/result.json) | [independent-readback.json](s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/matrix-control/attention-capacity-v1/capacity-confirm-c1536-matched-ba-prod-20260807t1938z/independent-readback.json) |

```text
144 A→B: manifest.json 910a709c98ef71312dfbd6be96b5fcf628407646924894881f820c46f8a6856a; result.json e9b5ae7aa954ec9294c2e05199425dbd4ebf97e222bbd35cb6272590969ad561; independent-readback.json ef80d873ab451b183737dc56a4e3e9f95c8b58b1ebcdf0221d3eaf96480f585b
144 B→A: manifest.json 006766b9ed8befd0130263061e4ca83a76e39512dda6ab4ac166e7269f1c2304; result.json 59825ec0c8b93fba9a7b9426a35f8e04fa66f69504d4bfcdf2ef11f17ea11afc; independent-readback.json b856a5276f6992be869dbbe624d5c6d89e4463237ca69cfffc315d31cf7c6a7e
1,536 A→B: manifest.json 3cefae1dbf341a4dbae3762e42b33c496bfd504b15be2e76d8b034a471be7e00; result.json 04623d13f63eaffa3bc509c5e454b35c6f3ca671fe1a0031ced7e02bf55c0466; independent-readback.json 83226268b52125700ce9e913b5d659c146da5a5a79d1f8e34a487fc16134c057
1,536 B→A: manifest.json 4713138c6eb9a8059e291e13c539fdf622fc85aa8f3a98dfdc9c8932f36a4d32; result.json 896d905aa83f3ca417585dc6239f00c94aaaf257b90a8c4cbd317c6602f31f03; independent-readback.json 278b43b7c190143dddd88ec7b1adb2dd5c247fdb28048b88bb270e268629d0df
```

</details>

<details>
<summary><strong>Excluded and provisional attempts</strong></summary>

- The first 384-request attempt is excluded. Its reference plateau had no long-cohort completion, so latency summarization failed and no arm result was accepted. The harness now requires one completion from every frozen cohort before a scout plateau may close.
- The repaired matched 384-request attempt lost one rank while fetching the pinned vLLM commit from GitHub. Its every-6 phase produced no arm. The later every-4 phase and a separate every-6 retry each passed independent readback, but they used separate allocations; no matched percentage is reported.
- The 768-, 1,152-, and 1,536-request scouts passed their applicable gates with zero KV preemptions, but they did not meet the full duration, generated-token, and manifest floors. They selected the 1,536-request point for confirmation; they do not determine the exact saturation point or the relative-throughput conclusion.
- An interactive 1,152-request submission remained pre-admission because no NVLink domain could fit four whole nodes. It consumed no GPU benchmark time and contributes no curve point. The later authorized production scout is the provisional 1,152-request row above.
- No failed, unmatched, pre-admission, stopped, or unread artifact contributes to a confirmed result.

</details>
