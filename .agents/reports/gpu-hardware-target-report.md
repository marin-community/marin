# H100 and Blackwell Hardware Assumptions for Training Cost Modeling

Date: 2026-07-07

Coordinating issue: https://github.com/marin-community/marin/issues/7010

## TL;DR

- H100 per-GPU compute and HBM values are experimentally grounded for our JAX stack: 700 TFLOP/s bf16, 1.25-1.45 PFLOP/s fp8, and 2.85 TB/s measured HBM stream bandwidth. The H100 SXM capacity and peak reference values come from NVIDIA's product specs.
- Blackwell is only partially measured. We measured an 8-GPU Blackwell proxy node, not a GB200 NVL72 rack. The proxy reached 1.88 PFLOP/s bf16, 3.56 PFLOP/s fp8, and 6.42 TB/s HBM stream bandwidth on the same JAX-style benchmark. GB200/NVL72 capacity and topology values are source-backed, not locally measured.
- H100 intra-node collectives are measured through JAX/XLA. All-reduce and reduce-scatter look usable for modeling. The one-node all-gather `busbw` is implausibly high after NCCL-style accounting and should not be treated as a hardware lower bound without native `nccl-tests`.
- H100 two-node collectives are measured for all-reduce through 1 GiB and reduce-scatter through 256 MB. Two-node all-gather is not measured: both normal and tiled JAX `all_gather` initialized all 16 ranks and then hung before the first 1 MB timing record.
- We do not have experimental evidence for GB200 intra-NVL72 all-gather/reduce-scatter or inter-rack all-gather/reduce-scatter. Public provider examples mainly report all-reduce.
- Treat 400 Gb/s scale-out networking as effectively per GPU for DGX/HGX-style H100/B200 systems unless the provider SKU says otherwise. On the measured H100 cluster, the 1 GiB two-node all-reduce `busbw` of 365 GB/s is not compatible with a single 400 Gb/s NIC per 8-GPU node. Fabric oversubscription is still unmeasured.

## Evidence Labels

This report uses four labels:

- `Measured`: produced by a local benchmark run in this investigation.
- `Source-backed`: taken from public vendor/provider documentation or public benchmark logs.
- `Inferred`: derived from measured/source-backed numbers, usually by per-GPU division or NCCL busbw formulas.
- `Unknown`: not measured and not confirmed by a sufficiently specific source.

## Recommended Modeling Inputs

Dense tensor throughput, per GPU. These are compute-bound GEMM/HBM inputs, not end-to-end training MFU.

| Target | bf16 TFLOP/s | fp8 TFLOP/s | HBM capacity | HBM bandwidth | Evidence | Notes |
|---|---:|---:|---:|---:|---|---|
| H100 SXM | 700 | 1,250 | 80 GB | 2.85 TB/s measured, 3.0-3.1 TB/s planning | Measured + source-backed | 700 bf16 is the measured 16k GEMM result. fp8 measured 1,453 TFLOP/s at 16k, but 1,250 is a conservative planning value. |
| Blackwell proxy node | 1,550-1,880 | 3,500-3,700 | 275 GB measured on proxy GPU | 6.42 TB/s measured | Measured | This is not GB200. Use as a proxy for current Blackwell access only. |
| B200/HGX | 1,550 | 3,700 | 180 GB | 7.4 TB/s planning | Source-backed + inferred | DGX B200 lists 1,440 GB and 64 TB/s HBM3e across 8 GPUs, or 180 GB and 8 TB/s peak per GPU. |
| GB200 NVL72 | 1,750 | 4,100 | 186 GB | 7.35-7.5 TB/s planning | Source-backed + inferred | NVIDIA lists 13.4 TB HBM and 576 TB/s HBM bandwidth per NVL72 rack. Azure reports 7.35 TB/s achieved HBM bandwidth on ND GB200 v6. We did not measure this. |

The H100 public spec lists 80 GB HBM3, 3.35 TB/s memory bandwidth, sparse bf16 tensor throughput of 1,979 TFLOP/s, sparse fp8 tensor throughput of 3,958 TFLOP/s, 900 GB/s NVLink, and up to 700 W TDP for SXM. Dense peak is half the sparse tensor number. Our H100 JAX GEMM number is about 71% of dense bf16 peak.

NVIDIA's DGX B200 page lists 8 Blackwell GPUs, 1,440 GB total HBM, 64 TB/s HBM3e bandwidth, FP8 Tensor Core performance of 72 PFLOP/s sparse, and says dense performance is half sparse. Per GPU, that is 180 GB, 8 TB/s peak HBM, and 4.5 PFLOP/s dense fp8 peak before framework losses.

GB200 NVL72 is not a single GPU SKU. NVIDIA describes it as 36 Grace CPUs and 72 Blackwell GPUs in one rack-scale 72-GPU NVLink domain with 13.4 TB HBM and 576 TB/s HBM bandwidth. Per GPU, simple division gives about 186 GB and 8 TB/s peak HBM. We should keep GB200 separate from the measured Blackwell proxy.

## Experimental Setup

The local benchmark is `scripts/bench/gpu_microbench.py`. It uses JAX 0.10.x, bf16/fp8 GEMMs, a simple JAX HBM stream kernel, and JAX/XLA collectives under `pmap`. Collective `busbw` follows the NCCL `nccl-tests` convention:

- all-reduce: `busbw = algbw * 2 * (n - 1) / n`
- all-gather and reduce-scatter: `busbw = algbw * (n - 1) / n`

For all-gather and reduce-scatter, the benchmark reports `logical_mb = per_rank_chunk_mb * ranks`, matching NCCL's definition of `S` for these collectives.

The two-node H100 runs used 2 Iris tasks, each with `H100x8`, and `--processes-per-task 8`, giving 16 JAX processes with one local GPU per process. A one-process-per-node attempt initialized two JAX processes with eight local GPUs each and hung in XLA/NCCL clique setup. Use one process per GPU for this path.

## Compute and HBM Measurements

| Platform | Shape / kernel | bf16 TFLOP/s | fp8_e4m3 TFLOP/s | HBM stream TB/s | Evidence |
|---|---:|---:|---:|---:|---|
| H100 | 8192 GEMM | 753.3 | 1300.3 | - | Measured |
| H100 | 16384 GEMM | 700.6 | 1452.7 | 2.85 | Measured |
| Blackwell proxy | 8192 GEMM | 1578.0 | 2771.7 | - | Measured |
| Blackwell proxy | 16384 GEMM | 1880.7 | 3556.4 | 6.42 | Measured |

Interpretation:

- H100: use 700 bf16 TFLOP/s as the conservative framework-achievable GEMM input. Use 1,250 fp8 TFLOP/s if the objective needs a conservative fp8 number; the local 16k GEMM measured 1,453 TFLOP/s.
- Blackwell proxy: use 1,550 bf16 and 3,700 fp8 for a conservative Blackwell proxy, or 1,880 bf16 and 3,556 fp8 if the model wants the exact measured 16k GEMM result. Do not label this as GB200 evidence.
- HBM: use measured values for framework-facing models. Use source-backed peak/near-peak values only for hardware sensitivity studies.

## Collective Bandwidth: What Is Measured

All bandwidths in this section are NCCL-style `busbw` in GB/s. The JAX/XLA measurements are useful for framework-facing cost estimates, but they are not a substitute for native `nccl-tests`.

### H100, One Node, 8 GPUs, JAX/XLA

| Message | all-reduce | all-gather | reduce-scatter | Evidence |
|---:|---:|---:|---:|---|
| 1 MB | 3.3 | 16.0 | 15.2 | Measured, AG/RS corrected |
| 8 MB | 26.9 | 118.4 | 84.0 | Measured, AG/RS corrected |
| 64 MB | 148.3 | 445.1 | 218.0 | Measured, AG/RS corrected |
| 256 MB | 271.1 | 1084.4 | 254.7 | Measured, AG/RS corrected |
| 512 MB | 316.2 | 1265.2 | 276.7 | Measured, AG/RS corrected |
| 1024 MB | 368.9 | 1352.9 | 284.6 | Measured, AG/RS corrected |

Caveat: the all-gather numbers are too high to treat as a hardware lower bound. They are a JAX/XLA microbenchmark result and may reflect the specific lowering or an accounting mismatch that native `all_gather_perf` would expose. The reduce-scatter numbers are plausible after the accounting correction.

### Blackwell Proxy, One Node, 8 GPUs, JAX/XLA

| Message | all-reduce | all-gather | reduce-scatter | Evidence |
|---:|---:|---:|---:|---|
| 1 MB | 2.1 | 11.2 | 12.0 | Measured, AG/RS corrected |
| 8 MB | 23.1 | 93.6 | 75.2 | Measured, AG/RS corrected |
| 64 MB | 131.7 | 614.8 | 313.7 | Measured, AG/RS corrected |
| 256 MB | 333.5 | 2036.8 | 509.8 | Measured, AG/RS corrected |
| 512 MB | 476.7 | 3308.9 | 555.4 | Measured, AG/RS corrected |
| 1024 MB | 576.5 | 4405.4 | 583.5 | Measured, AG/RS corrected |

Caveat: this is an 8-GPU Blackwell proxy node, not GB200 NVL72. The one-node all-gather numbers are even less suitable as hardware bounds than the H100 all-gather numbers.

### H100, Two Nodes, 16 GPUs, JAX/XLA

All-reduce completed through 1 GiB:

| Message | seconds | busbw GB/s | Evidence |
|---:|---:|---:|---|
| 1 MB | 0.000401 | 4.9 | Measured |
| 8 MB | 0.001865 | 8.4 | Measured |
| 64 MB | 0.000701 | 179.5 | Measured |
| 256 MB | 0.001681 | 299.3 | Measured |
| 512 MB | 0.002959 | 340.2 | Measured |
| 1024 MB | 0.005518 | 364.9 | Measured |

Reduce-scatter completed through 256 MB per output chunk:

| Output chunk | logical array | seconds | busbw GB/s | Evidence |
|---:|---:|---:|---:|---|
| 1 MB | 16 MB | 0.000627 | 25.1 | Measured |
| 8 MB | 128 MB | 0.000810 | 155.4 | Measured |
| 64 MB | 1024 MB | 0.003793 | 265.4 | Measured |
| 256 MB | 4096 MB | 0.014493 | 277.8 | Measured |

All-gather did not complete:

- Normal `lax.all_gather(v, "rank", axis=0)` initialized all 16 ranks and then hung before the first 1 MB timing record.
- Tiled `lax.all_gather(v, "rank", axis=0, tiled=True)` on a `(1, chunk)` input had the same behavior.

Large reduce-scatter did not complete:

- The 512 MB / 1 GiB reduce-scatter job initialized all 16 ranks and then produced no 512 MB timing record after about five minutes. It was stopped.

Marginal fits:

- Two-node H100 all-reduce over 64, 256, 512, and 1024 MB:
  `time ~= 0.000390 + 0.00513 * GiB`, where GiB is per-rank payload. `R2=0.99999`.
- Two-node H100 reduce-scatter over 64 and 256 MB chunks:
  `time ~= 0.000226 + 0.0571 * GiB_per_output_chunk`.
  This is only a two-point fit. Treat it as exploratory.

## Collective Bandwidth: What Is Source-Backed or Inferred

Public sources help fill some all-reduce anchors, but they do not close the all-gather/reduce-scatter gaps.

### H100 Inter-Node / Scale-Out IB

CoreWeave's public `nccl-tests` README includes a 64-GPU H100 all-reduce example around 356 GB/s at 512 MB and 366 GB/s at 1 GiB. Our two-node / 16-rank H100 all-reduce measured 340 GB/s at 512 MB and 365 GB/s at 1 GiB, so the local result is consistent with that public anchor.

We do not have native H100 `all_gather_perf` or `reduce_scatter_perf` logs for the target cluster. The two-node JAX reduce-scatter result is useful but not native NCCL evidence. The two-node JAX all-gather result is a failure, not a bandwidth measurement.

### GB200 / NVL72

NVIDIA confirms the architectural shape: 72 Blackwell GPUs in one NVLink domain and 130 TB/s rack NVLink fabric. CoreWeave's public `nccl-tests` README includes GB200 all-reduce examples:

- Single rack: about 586 GB/s at 512 MB and 681-724 GB/s at 1 GiB.
- Two racks: about 494-509 GB/s at 512 MB and 596-598 GB/s at 1 GiB.
- Larger multi-rack examples fall much lower at the same sizes.

We do not have GB200/NVL72 native all-gather or reduce-scatter numbers. It is reasonable to expect large-message `busbw` for all-gather and reduce-scatter to live in the same broad fabric family as all-reduce because NCCL's busbw normalization is designed to make collectives comparable on the same bottleneck. That is an inference, not a measurement.

## Topology and Network Assumptions

### H100

CoreWeave's H100 InfiniBand product page says the instance has 8 H100 GPUs, 80 GB HBM3 per GPU, NVLink within the node, and a 400G NDR InfiniBand fabric between nodes. That wording alone is ambiguous about whether 400G is per node or per GPU.

The effective measured behavior points away from a single 400G link per 8-GPU node. A single 400G link is about 50 GB/s line rate before overhead. The two-node H100 all-reduce measured 365 GB/s `busbw` at 1 GiB. That requires multiple network rails or equivalent aggregate fabric bandwidth. DGX/HGX-style reference designs also use multiple 400G adapters for an 8-GPU system. For PP/DP modeling, treat scale-out H100 as effectively one 400G rail per GPU unless the exact SKU contradicts that.

Oversubscription remains unknown. We measured two coscheduled nodes. That does not prove the fabric is non-blocking at larger allocation sizes, across racks, or under concurrent tenant load.

### GB200 / NVL72

The 72-GPU NVLink domain is source-backed. If the target is actually GB200 NVL72, PP/DP cost should model traffic inside the NVL72 rack as NVLink/NVSwitch traffic, not ordinary inter-node IB traffic.

The scale-out fabric between NVL72 racks is SKU-dependent. Public CoreWeave material currently distinguishes GB200 and GB300:

- GB200 NVL72 material references fifth-generation NVLink and Quantum-2 InfiniBand networking.
- GB300/Blackwell Ultra material explicitly mentions Quantum-X800 and ConnectX-8 SuperNICs with 800 Gb/s connectivity per GPU.

Do not assume ConnectX-8 800 Gb/s for plain GB200 unless the provider confirms it for the target allocation.

## Marginal Communication Costs for the Planner

Use the measured wall-clock fits for framework-facing cost modeling, and keep the evidence limits attached.

| Fabric / placement | Collective | Usable fit | Evidence |
|---|---|---:|---|
| H100 one node, 8 GPUs | all-reduce | `0.000576 + 0.00457 * GiB` | Measured |
| H100 one node, 8 GPUs | reduce-scatter | `0.000721 + 0.00322 * GiB_logical` | Measured after accounting correction |
| H100 two nodes, 16 GPUs | all-reduce | `0.000390 + 0.00513 * GiB` | Measured |
| H100 two nodes, 16 GPUs | reduce-scatter | `0.000226 + 0.0571 * GiB_output_chunk` | Exploratory, two points |
| Blackwell proxy one node, 8 GPUs | all-reduce | `0.000747 + 0.00251 * GiB` | Measured |
| Blackwell proxy one node, 8 GPUs | reduce-scatter | `0.000691 + 0.00152 * GiB_logical` | Measured after accounting correction |

All-gather should not use the JAX microbenchmark fit in a planner. The one-node values are suspiciously high, and the two-node path hangs. Use native NCCL all-gather numbers once available, or use all-reduce/all-gather public anchors only as a sensitivity band with an explicit caveat.

## What Is Still Unknown

- Native `nccl-tests` for the target H100 cluster: all-reduce, all-gather, and reduce-scatter across the full 1 MB to 1 GiB range, with exact placement metadata.
- Native `nccl-tests` for Blackwell proxy nodes. We only have JAX/XLA collectives.
- Any measured GB200 NVL72 collectives from our environment. Public GB200 data found in this pass is mostly all-reduce.
- GB200/NVL72 all-gather and reduce-scatter at 1 MB to 1 GiB, both intra-rack and inter-rack.
- Fabric oversubscription for large H100 or GB200 allocations. Two-node success does not prove non-blocking behavior at larger scale.
- Exact Target B SKU. B200/HGX, GB200 NVL72, and GB300/Blackwell Ultra have different memory capacities and scale-out assumptions.
- Energy/cost objective. H100 SXM TDP is source-backed at up to 700 W, and the Blackwell proxy reported a 1100 W GPU power limit, but this report does not compute energy-normalized cost.

## Decision for the Cost Model

Use the following default inputs now:

| Input | Value | Evidence boundary |
|---|---:|---|
| H100 bf16 | 700 TFLOP/s per GPU | Measured |
| H100 fp8 | 1,250 TFLOP/s per GPU | Conservative from measured 1,453 |
| H100 HBM | 80 GB, 2.85 TB/s measured | Capacity source-backed, bandwidth measured |
| H100 one-node all-reduce | measured table above | JAX/XLA, not native NCCL |
| H100 two-node all-reduce | measured table above | JAX/XLA, consistent with public native all-reduce anchor |
| H100 two-node reduce-scatter | use measured through 256 MB only | JAX/XLA, no 512/1024 MB result |
| H100 all-gather | do not use local JAX result | Needs native NCCL |
| Target B compute | use Blackwell proxy measured values for proxy studies; use GB200 source-backed values only for GB200 sensitivity | No direct GB200 measurement |
| Target B intra-NVL72 | use public GB200 all-reduce anchors for sensitivity | No local measurement, no AG/RS |
| Target B inter-rack | use public GB200 all-reduce anchors for sensitivity | No local measurement, no AG/RS |

For PP/DP cost, separate these cases:

1. H100 intra-node NVLink/NVSwitch.
2. H100 inter-node IB, likely multi-rail 400G-class scale-out.
3. Blackwell proxy intra-node NVLink.
4. GB200 intra-NVL72 NVLink domain.
5. GB200 inter-rack IB.

Do not collapse GB200 intra-NVL72 and GB200 inter-rack into one bandwidth. The public all-reduce anchors differ enough that placement will change the objective.

## Appendix: Run Ledger

- H100 one-node GEMM/HBM/collectives: `/dlwh/gpu-bench-h100-1node-r3`, `/dlwh/gpu-bench-h100-coll-r4`.
- Blackwell proxy one-node GEMM/HBM/collectives: Slurm jobs `64971`, `64973`.
- H100 two-node all-reduce: `/dlwh/gpu-bench-h100-2node-ar-r9`.
- H100 two-node reduce-scatter through 256 MB: `/dlwh/gpu-bench-h100-2node-rs-r13-retry`.
- H100 two-node all-gather normal lowering: `/dlwh/gpu-bench-h100-2node-ag-r12-retry`, stopped after no timing record.
- H100 two-node all-gather tiled lowering: `/dlwh/gpu-bench-h100-2node-ag-tiled-r15`, stopped after no timing record.
- H100 two-node reduce-scatter 512/1024 MB: `/dlwh/gpu-bench-h100-2node-rs-r14-large`, stopped after no timing record.

## Sources

- NVIDIA H100 product specs: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA DGX B200 product specs: https://www.nvidia.com/en-us/data-center/dgx-b200/
- NVIDIA GB200 NVL72 product page: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA `nccl-tests` performance definitions: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- CoreWeave H100 InfiniBand product page: https://docs.coreweave.com/platform/instances/gpu/gd-8xh100ib-i128
- CoreWeave public `nccl-tests` examples: https://github.com/coreweave/nccl-tests
- CoreWeave Blackwell product page: https://www.coreweave.com/products/nvidia-blackwell
- Azure ND GB200 v6 performance note: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/unpacking-nvidia-gb200-gpu-performance-on-azure-virtual-machines/4390442
