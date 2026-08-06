# Visual-token budget sweep: max throughput and CPU:GPU ratio per budget

Grid sweep of Infinity-Parser2-Flash doc2md serving on GB200, over visual-token
budget (page size) × in-flight sizing (client concurrency, engine slots,
API-server count, pod CPU, pod RAM). One GPU per vLLM instance, FlashInfer
attention + FlashInfer GDN (prebuilt cubin/jit-cache artifacts), max_model_len
24576, max_tokens 4096, temperature 0. Payloads are focus-crawl PDF pages from
`part-00000` of the 2026-22 10% sample, rendered at the budget with a 300-DPI
upscale cap. Data: `ocr-budget-*.jsonl` result files, produced by the sweep
harness (arms ran solo and sequentially; each wrote durable results after
every point). Raw records carry full config, latency percentiles, DPI
distributions, and failure counts. The harness and raw results live at commit
`39b4095fa` under `experiments/b200_ocr/`.

Arms (pod = CPU cores / RAM / vLLM API servers, per GPU):

| arm | pod | engine | concurrency |
|---|---|---|---|
| base | 64 / 220g / 4 | seqs 1024, btok 131072 | 256, 512, 768 |
| lean | 32 / 160g / 2 | seqs 1024, btok 131072 | 256, 512, 768 |
| rich | 96 / 280g / 8 | seqs 1024, btok 131072 | 256, 512, 768 |
| highpar | 64 / 220g / 4 | seqs 2048, btok 262144 | 512, 1024, 1536 (budgets ≤2048) |
| richpar | 96 / 280g / 8 | seqs 2048, btok 262144 | 768, 1024, 1536 (budgets ≤2048) |

## Max throughput and CPU:GPU ratio per budget

The best point at every budget is the rich pod at concurrency 512 with
seqs 1024. Render rate is single-core PyMuPDF render+PNG+base64 of the same
pages (`ocr-budget-cpubench`); the render CPU:GPU ratio is what a feeder fleet
must supply per GPU to sustain the max.

| budget | MP/page | median DPI | pages <100 DPI | max p/s/GPU | p50 latency | GPU-h per M pages | render p/core/s | render cores : GPU |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.50 | 72 | 99% | 19.5 | 19s | 14.2 | 29.3 | 0.67 |
| 1024 | 1.02 | 102 | 2% | 19.3 | 20s | 14.4 | 20.3 | 0.95 |
| 2048 | 2.07 | 146 | 0% | 18.9 | 21s | 14.7 | 13.2 | 1.43 |
| 4096 | 4.14 | 207 | 0% | 14.9 | 26s | 18.6 | 8.4 | 1.79 |
| 8192 | 8.07 | 294 | 0% | 12.3 | 31s | 22.6 | 5.2 | 2.37 |

Budget 16384 was omitted: the 300-DPI upscale cap already binds at 8192
(mean 8.07 MP against the 8.39 MP budget), so a 16384 arm renders near-identical
payloads. Raising the budget past 8192 requires raising the DPI cap, which buys
no glyph detail.

Throughput is strikingly budget-insensitive below 2048: 16× smaller pages
(8192 → 512) buy only ~1.6× throughput. Output length is ~740 completion tokens
regardless of page size, so decode dominates once prefill shrinks; page-size
savings mostly stop mattering below ~2 MP.

The DPI column decides the low end: 512 renders 99% of pages below the ~100 DPI
legibility floor and 1024 sits right at it, while both serve within ~3% of
2048's throughput. 2048 (~146 DPI median, everything legible) is therefore the
floor worth operating at; below it, quality is surrendered for nothing. Above
it the trade is real: −21% throughput for 207 DPI at 4096, −35% for ~294 DPI at
8192.

## Which resources actually matter

**API-side CPU and server count set throughput at every budget.** rich
(96c/api8) beats base (64c/api4) by 10–25% across the board; lean (32c/api2)
matches base at small budgets and trails ~4–7% at 4096–8192. The API servers do
the multimodal preprocessing, and they are the bottleneck ahead of the GPU in
the lean/base pods.

**More engine slots hurt.** Opening seqs 1024→2048 (+btok 262144) lowered
throughput at matched pod and equal-or-higher concurrency (richpar 18.6 max vs
rich 19.5; at conc 768 richpar hit 14.5 where rich@512 gives 19.5). Concurrency
512 against seqs 1024 is the operating point; past it, added in-flight only adds
latency (highpar reached rich-level throughput only at conc 1536 with ~60s p50,
3× rich's p50).

**The collapse boundary is API-count × page size, not RAM.** At budget 8192 and
conc 768, base (api4/220g) and rich (api8/280g) both collapsed (76%+ request
failures, pod OOM), while lean (api2/160g) survived and even peaked there —
fewer API servers throttle concurrent preprocessing, capping the transient-RAM
peak. Big-page serving must either cap in-flight at ≤512 per instance or run
few API servers.

## Per-node packing changes the answer

GB200 nodes have 4 GPUs and ~144 usable cores. The rich pod (96c/GPU) packs
1/node; base (64c) packs 2/node; lean (32c) packs 4/node. Per-GPU winners are
not per-node winners:

| config | packs | est. p/s per node (budget 1024 / 8192) |
|---|---|---|
| rich × 1 | 1/node | 19.3 / 12.3 (3 GPUs idle) |
| base × 2 | 2/node | 32.0 / 22.8 |
| lean × 4 | 4/node | 68.7 / 44.1 |

Lean × 4 dominates per node at every budget despite the lower per-GPU number.
The measured 4-GPU validation of this estimate is in the next section.

## Full-node serving at budget 2048 (4 GPUs, brokered)

`results/ocr-node2048-*.jsonl`: 4 one-GPU instances behind the marin broker
(`--instances 4`, `--max-in-flight 512`), budget 2048, total concurrency
1024/2048/3072, three pod shapes. All arms clean:

| arm | pod per GPU | p/s per 4 GPUs (conc 1024 / 2048 / 3072) |
|---|---|---|
| lean-api2 | 32c / 160g / api2 | 67.9 / **71.4** / 67.8 |
| lean-api4 | 32c / 160g / api4 | 55.9 / 69.9 / 69.0 |
| base | 64c / 220g / api4 | 61.7 / 72.1 / 71.4 |

Three results:

- **~71 pages/s per 4 GPUs (15.6 GPU-h per M pages) at total concurrency
  2048**, i.e. 512 per instance — the same per-instance operating point as the
  solo sweep. 1024 under-fills; 3072 only adds latency (p50 21s → 33s).
- **Pod shape stops mattering under the broker** (all arms within ~3%): the
  broker holds every engine at its full 512 in-flight continuously, so the
  API-side burst capacity that separated lean/base/rich under direct
  closed-loop load is no longer exercised at this page size. The 32-core pod is
  the fleet config: it packs 4/node and loses ~1% to the 64-core pod.
- **Brokered beats direct per GPU** (17.8–18.0 vs 15.1–15.6 solo at the same
  pods): a client-side pool dips while requests drain and refill; the broker
  queue never does. The solo-sweep per-GPU numbers are therefore conservative
  for production, and the lean×4 node estimate above (61) undershoots the
  measured 71.

Placement caveat: iris bin-packs and 4×32c fits one node, but the hub DB clears
placement after job end, so co-location was inferred, not verified.

The first run of this topology found two more default-concurrency caps in the
brokered path (both fixed on this branch, validated by the numbers above):
anyio's 40-thread `to_thread` limiter in the inference proxy capped any
brokered fleet at ~40 in-flight requests total, and the inference worker's
httpx client capped forwarding at 100 connections against a 512-thread pool.
With the dashboard-proxy pool fix, that is three separate library-default
concurrency ceilings this campaign hit in the serving path.

## Planning numbers (per budget, packable lean config, solo-measured)

| budget | p/s/GPU | p/s/node (×4) | GPU-h per M pages | render cores per GPU | render cores per node |
|---|---|---|---|---|---|
| 512 | 16.8 | 67 | 16.6 | 0.57 | 2.3 |
| 1024 | 17.2 | 69 | 16.2 | 0.85 | 3.4 |
| 2048 | 15.1 | 61 | 18.3 | 1.14 | 4.6 |
| 4096 | 12.2 | 49 | 22.8 | 1.46 | 5.8 |
| 8192 | 11.0 | 44 | 25.2 | 2.12 | 8.5 |

Render feeding is cheap relative to serving at every budget: ≤9 render cores
per 4-GPU node, plus the in-pod API-side cores already counted in the pod
shape. The DPI-quality side of the budget choice (what 512–2048 tokens does to
legibility, `frac_below_floor` in the cpu records) is a separate decision
tracked with the quality work in #7619; this sweep prices the throughput axis.
