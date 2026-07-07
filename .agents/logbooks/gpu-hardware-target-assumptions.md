---
topic: gpu-hardware-target-assumptions
issue: https://github.com/marin-community/marin/issues/7010
description: H100 and Blackwell hardware/network assumptions for training cost modeling
author: dlwh
---

# GPU Hardware Target Assumptions: Task Logbook

## Scope
- Goal: Establish cost-model inputs for Target A = H100 and Target B = Blackwell/GB200-class hardware.
- Primary metrics: Achievable GEMM TFLOP/s, HBM capacity and bandwidth, NCCL bus bandwidth by topology, and topology assumptions for PP/DP communication costs.
- Constraints: Use public/durable sources only in this brief.
- Coordinating issue: https://github.com/marin-community/marin/issues/7010

## Current TL;DR
- Use H100 SXM for Target A unless the SKU is explicitly PCIe: 700 bf16 TFLOP/s, 1,250 fp8 TFLOP/s, 80 GB HBM, 3.05 TB/s achievable HBM bandwidth.
- For the currently available Blackwell proxy, use B200/HGX values: 1,550 bf16 TFLOP/s, 3,700 fp8 TFLOP/s, 180 GB HBM, 7.4 TB/s achievable HBM bandwidth.
- Add a true GB200 NVL72 sensitivity only when the deployment is confirmed: roughly 1,750 bf16 TFLOP/s, 4,100 fp8 TFLOP/s, 186 GB HBM, 7.5 TB/s HBM bandwidth.
- For DGX/HGX-style H100 and GB200 designs, model scale-out IB as one 400 Gb/s rail per GPU, not one 400 Gb/s link per node. Confirm vendor SKU details because some cloud pages summarize this ambiguously.
- GB200 NVL72 has a confirmed 72-GPU NVLink domain and 130 TB/s rack NVLink fabric. Public GB200 reference material points to 400G ConnectX-7 scale-out; do not assume ConnectX-8 800G unless the exact SKU says GB300/Blackwell Ultra or otherwise confirms it.
- H100 two-node / 16-rank JAX evidence now exists for all-reduce through 1 GiB and reduce-scatter through 256 MB. All-gather is not experimentally backed by this JAX multinode harness: both stacked and tiled `lax.all_gather` forms initialized 16 ranks but hung before the first 1 MB timing record.

## Recommended Modeling Values

Dense tensor throughput, per GPU, no structured sparsity. These are for compute-bound training GEMMs, not end-to-end model MFU.

| Target | Use case | Achievable bf16 | Achievable fp8 | HBM capacity | Achievable HBM bandwidth | Confidence |
|---|---:|---:|---:|---:|---:|---|
| H100 SXM | Target A default | 700 TFLOP/s | 1,250 TFLOP/s | 80 GB | 3.0-3.1 TB/s | High |
| H100 PCIe | Sensitivity only | 520 TFLOP/s | 950 TFLOP/s | 80 GB | 1.8 TB/s | Medium |
| B200/HGX | Target B proxy | 1,550 TFLOP/s | 3,700 TFLOP/s | 180 GB | 7.4 TB/s | Medium |
| GB200 NVL72 per GPU | Target B if confirmed GB200 | 1,750 TFLOP/s | 4,100 TFLOP/s | 186 GB | 7.5 TB/s | Medium-low |

Notes:
- NVIDIA's H100 specs list 80 GB HBM3, 3.35 TB/s HBM bandwidth, 900 GB/s NVLink, and sparse tensor numbers; dense H100 SXM is about 989 bf16 TFLOP/s and 1,979 fp8 TFLOP/s after halving the sparse figures ([NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)).
- NVIDIA's GB200 NVL72 page lists 72 Blackwell GPUs, 13.4 TB HBM3e, 576 TB/s HBM bandwidth, 130 TB/s NVLink bandwidth, 720 PFLOP/s FP8/FP6 tensor and 360 PFLOP/s FP16/BF16 tensor for the rack; interpreted as sparse tensor figures, dense per-GPU is approximately half of rack/72 ([NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)).
- NVIDIA DGX B200 lists 8 Blackwell GPUs, 1,440 GB total HBM, and 64 TB/s HBM bandwidth, implying 180 GB and 8 TB/s per GPU ([NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)).
- Azure reported sustained 2,744 TFLOP/s FP8 and 7.35 TB/s HBM bandwidth for ND GB200 v6; use this as an external sanity check for the GB200 sensitivity ([Azure GB200 performance note](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/unpacking-the-performance-of-microsoft-azure-nd-gb200-v6-virtual-machines/4390442)).
- Public B200/Blackwell GEMM reports vary heavily by matrix shape. The B200 proxy above is intentionally below ideal square-GEMM/peak values to account for framework and shape variance.

## NCCL Bus Bandwidth Assumptions

NCCL `busbw` is a normalized hardware-bottleneck estimate, not a literal wire-rate counter. NCCL computes bus bandwidth from algorithm bandwidth using operation-specific correction factors: all-reduce uses `2*(n-1)/n`; all-gather and reduce-scatter use `(n-1)/n`. Therefore, for cost modeling, all-reduce, all-gather, and reduce-scatter should converge to similar `busbw` at large messages on the same fabric, while their algorithm bandwidth and latency differ ([NVIDIA nccl-tests performance doc](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)).

Use these as planning ranges for all-reduce / all-gather / reduce-scatter `busbw` unless measured values for the exact cluster are available:

| Fabric / placement | 1 MB | 8 MB | 64 MB | 256 MB | 512 MB | 1 GB |
|---|---:|---:|---:|---:|---:|---:|
| H100 intra-node NVLink/NVSwitch, 8 GPUs | 25-60 GB/s | 100-140 GB/s | 260-320 GB/s | 390-420 GB/s | 425-435 GB/s | 450-480 GB/s |
| H100 inter-node NDR IB, well-tuned 8-rail nodes | 25-60 GB/s | 80-140 GB/s | 250-320 GB/s | 330-400 GB/s | 350-430 GB/s | 350-470 GB/s |
| GB200 intra-NVL72 rack NVLink domain | 50-150 GB/s | 150-350 GB/s | 350-600 GB/s | 500-750 GB/s | 580-700 GB/s | 680-730 GB/s |
| GB200 inter-rack IB, 2 racks / 144 GPUs | 30-100 GB/s | 100-250 GB/s | 250-450 GB/s | 400-600 GB/s | 490-510 GB/s | 595-600 GB/s |
| GB200 large multi-rack IB example | 10-60 GB/s | 50-150 GB/s | 100-250 GB/s | 150-300 GB/s | about 150 GB/s | about 220 GB/s |

Published anchors:
- CoreWeave's H100 NCCL example shows 64-GPU all-reduce busbw around 356 GB/s at 512 MB, 366 GB/s at 1 GB, and 386 GB/s at 8 GB ([CoreWeave nccl-tests README](https://github.com/coreweave/nccl-tests)).
- CoreWeave's GB200 NCCL examples show single-rack all-reduce busbw around 586 GB/s at 512 MB and 681-724 GB/s at 1 GB; 2-rack examples show about 494-509 GB/s at 512 MB and about 596-598 GB/s at 1 GB; 20-rack examples show about 150 GB/s at 512 MB and 219 GB/s at 1 GB ([CoreWeave nccl-tests README](https://github.com/coreweave/nccl-tests)).
- NVIDIA's NCCL tuning note warns that message-size curves can be non-monotonic or plateau when NCCL selects poor algorithms/protocols; benchmark exact software versions before treating any busbw row as stable ([NVIDIA NCCL tuning blog](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)).

## Topology Conclusions

H100:
- NVIDIA DGX H100 has 8 H100 GPUs, 900 GB/s per-GPU NVLink, 4 NVSwitches, and 10 ConnectX-7 400 Gb/s NICs, of which 8 are usually the GPU scale-out fabric and 2 are storage/management in DGX reference designs ([NVIDIA DGX H100](https://www.nvidia.com/en-eu/data-center/dgx-h100/)).
- CoreWeave's H100 InfiniBand SKU says 8 H100 GPUs with NVLink, and nodes linked by 400G NDR InfiniBand fabric. The same product family advertises 3,200 Gb/s InfiniBand for 8-GPU H200 nodes, which is consistent with one 400G rail per GPU, not one 400G link per node ([CoreWeave H100 IB](https://docs.coreweave.com/platform/instances/gpu/gd-8xh100ib-i128), [CoreWeave GPU compute](https://www.coreweave.com/products/gpu-compute)).

GB200 / NVL72:
- NVIDIA confirms GB200 NVL72 as 36 Grace CPUs plus 72 Blackwell GPUs in one rack-scale 72-GPU NVLink domain, with 130 TB/s low-latency GPU communication and 1.8 TB/s GPU-to-GPU interconnect per GPU ([NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)).
- Provider writeups align with the public architecture: in-rack communication is NVLink/NVSwitch; cross-rack communication falls back to InfiniBand and must be topology-aware because the bandwidth gap is large ([Nebius GB200 NVL72 interconnect note](https://nebius.com/blog/posts/leveraging-nvidia-gb200-nvl72-gpu-interconnect)).
- CoreWeave says GB200 NVL72 used NVIDIA Quantum-2 InfiniBand at 400 Gb/s per GPU, and notes future 800 Gb/s InfiniBand separately. CoreWeave's newer GB300/Blackwell Ultra page confirms ConnectX-8 800 Gb/s per GPU for GB300, not plain GB200 ([CoreWeave GB200 NVL72 note](https://www.coreweave.com/blog/coreweave-unleashes-the-power-of-the-nvidia-gb200-nvl72-a-glimpse-into-the-future-of-ai), [CoreWeave Blackwell](https://www.coreweave.com/products/nvidia-blackwell)).

Oversubscription:
- Assume no oversubscription only if the provider confirms a non-blocking or full-fat-tree RDMA fabric and placement within the same relevant topology domain.
- Marin's Iris topology labels already reflect the right distinction: H100 InfiniBand placement has fabric/superpod/leafgroup levels, while GB200 capacity has a hard `nvlink.domain` level for single-domain colocation ([Iris CoreWeave topology labels](../../lib/iris/src/iris/cluster/platforms/k8s/coreweave_topology.py)).
- PP/DP cost should distinguish: intra-node H100 NVLink, intra-NVL72 GB200 NVLink, and inter-rack IB. Treat crossing the NVL72 boundary as a much larger communication penalty than crossing H100 nodes within a well-tuned IB leaf group.

## Source Ledger

| Source | Type | Claim used |
|---|---|---|
| [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) | Official specs | H100 memory, tensor, NVLink, TDP figures |
| [NVIDIA Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | Official technical blog | H100 dense/sparse tensor and HBM details |
| [NVIDIA DGX H100](https://www.nvidia.com/en-eu/data-center/dgx-h100/) | Official product page | 8-GPU node, NVLink, ConnectX-7 400G networking |
| [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | Official product page | 8x B200, 1,440 GB HBM, 64 TB/s HBM bandwidth |
| [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) | Official product page | 72-GPU NVLink domain, 130 TB/s NVLink, rack performance |
| [NVIDIA nccl-tests performance doc](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) | Official benchmark docs | `busbw` formulas by collective |
| [NVIDIA NCCL tuning blog](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) | Official technical blog | NCCL message-size behavior and tuning caveats |
| [CoreWeave nccl-tests README](https://github.com/coreweave/nccl-tests) | Provider benchmark repo | H100 and GB200 all-reduce busbw anchors |
| [CoreWeave H100 IB](https://docs.coreweave.com/platform/instances/gpu/gd-8xh100ib-i128) | Provider docs | H100 IB SKU shape and 400G NDR fabric |
| [CoreWeave GPU compute](https://www.coreweave.com/products/gpu-compute) | Provider product page | 3,200 Gb/s H200 node networking; GB200 rack power |
| [CoreWeave GB200 NVL72 note](https://www.coreweave.com/blog/coreweave-unleashes-the-power-of-the-nvidia-gb200-nvl72-a-glimpse-into-the-future-of-ai) | Provider blog | GB200 400 Gb/s per GPU; future 800G note |
| [CoreWeave Blackwell](https://www.coreweave.com/products/nvidia-blackwell) | Provider product page | GB300 ConnectX-8 800 Gb/s per GPU |
| [Nebius GB200 NVL72 interconnect note](https://nebius.com/blog/posts/leveraging-nvidia-gb200-nvl72-gpu-interconnect) | Provider technical blog | GB200 NVL72 topology and scheduling implications |
| [Azure GB200 performance note](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/unpacking-the-performance-of-microsoft-azure-nd-gb200-v6-virtual-machines/4390442) | Provider benchmark note | GB200 sustained FP8 and HBM bandwidth sanity check |

## Open Questions
- Get exact NCCL all-gather and reduce-scatter logs for the target clusters; public sources mostly report all-reduce.
- Confirm exact Target B SKU: B200 HGX/DGX proxy, GB200 NVL72, or GB300/Blackwell Ultra. The networking assumption changes materially.
- Confirm whether the specific provider fabric is non-blocking at the requested allocation size and whether jobs can request same leaf group / same NVLink domain placement.

## Entry Log

### 2026-07-06 01:15 - Framework Microbenchmarks On H100 And B300
- Hypothesis: The planning defaults are close to framework-achievable JAX GEMM/HBM behavior on H100 and Blackwell-class GPUs; one-node collectives can be measured through JAX/XLA collectives as a framework-facing NCCL proxy.
- Commit Hash: uncommitted workspace changes.
- Command:
  - H100 one-node: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --timeout 1800 --job-name gpu-bench-h100-1node-r3 --cpu 32 --memory 256GB --disk 128GB --gpu H100x8 --enable-extra-resources --extra gpu --sync-package marin-iris --sync-package marin-levanter -- python scripts/bench/gpu_microbench.py --iterations 5 --warmup 2 --gemm-sizes 8192,16384 --hbm-size-mb 2048 --collective-message-mb 1,8,64,256,512`
  - H100 one-node collectives: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --timeout 1800 --job-name gpu-bench-h100-coll-r4 --cpu 32 --memory 256GB --disk 128GB --gpu H100x8 --enable-extra-resources --extra gpu --sync-package marin-iris --sync-package marin-levanter -- python scripts/bench/gpu_microbench.py --skip-gemm --skip-hbm --iterations 5 --warmup 2 --collective-message-mb 1,8,64,256,512,1024`
  - B300 one-node: Slurm job `64971`, `b300`, 1 node, 8 GPUs, `scripts/bench/gpu_microbench.py --iterations 5 --warmup 2 --gemm-sizes 8192,16384 --hbm-size-mb 4096 --collective-message-mb 1,8,64,256,512,1024`.
  - B300 one-node collectives: Slurm job `64973`, `b300`, 1 node, 8 GPUs, `scripts/bench/gpu_microbench.py --skip-gemm --skip-hbm --iterations 5 --warmup 2 --collective-message-mb 1,8,64,256,512,1024`.
- Config:
  - Benchmark script: `scripts/bench/gpu_microbench.py`.
  - H100: 8x H100, JAX 0.10.1, one Iris task.
  - B300: 8x NVIDIA B300 SXM6 AC, driver 595.71.05, 275040 MiB HBM per GPU, 1100 W power limit, JAX 0.10.0.
- Result:
  - GEMM/HBM:

    | Platform | Shape / kernel | bf16 TFLOP/s | fp8_e4m3 TFLOP/s | HBM stream TB/s |
    |---|---:|---:|---:|---:|
    | H100 | 8192 GEMM | 753.3 | 1300.3 | - |
    | H100 | 16384 GEMM | 700.6 | 1452.7 | 2.85 |
    | B300 | 8192 GEMM | 1578.0 | 2771.7 | - |
    | B300 | 16384 GEMM | 1880.7 | 3556.4 | 6.42 |

  - H100 one-node JAX/XLA collective `busbw` in GB/s. Correction: all-gather and reduce-scatter in this table used per-rank shard bytes as `S`; see the 2026-07-07 correction entry for NCCL-style logical-array `S`.

    | Message | all-reduce | all-gather | reduce-scatter |
    |---:|---:|---:|---:|
    | 1 MB | 3.3 | 2.0 | 1.9 |
    | 8 MB | 26.9 | 14.8 | 10.5 |
    | 64 MB | 148.3 | 55.6 | 27.3 |
    | 256 MB | 271.1 | 135.6 | 31.8 |
    | 512 MB | 316.2 | 158.1 | 34.6 |
    | 1024 MB | 368.9 | 169.1 | 35.6 |

  - B300 one-node JAX/XLA collective `busbw` in GB/s. Correction: all-gather and reduce-scatter in this table used per-rank shard bytes as `S`; see the 2026-07-07 correction entry for NCCL-style logical-array `S`.

    | Message | all-reduce | all-gather | reduce-scatter |
    |---:|---:|---:|---:|
    | 1 MB | 2.1 | 1.4 | 1.5 |
    | 8 MB | 23.1 | 11.7 | 9.4 |
    | 64 MB | 131.7 | 76.8 | 39.2 |
    | 256 MB | 333.5 | 254.6 | 63.7 |
    | 512 MB | 476.7 | 413.6 | 69.4 |
    | 1024 MB | 576.5 | 550.7 | 72.9 |

  - H100 two-node JAX/XLA collective attempt was blocked: the job joined two Iris tasks but collective compilation failed with `Loaded runtime CuDNN library: 9.10.2 but source was compiled with: 9.12.0`, followed by `dnn_support != nullptr`. The job was stopped after it stuck running with no usable records.
- Interpretation:
  - H100 GEMM confirms the earlier default: 700 bf16 TFLOP/s is a good conservative one-GPU JAX value; fp8_e4m3 on this stack reached 1.45 PFLOP/s at 16384, above the prior 1.25 PFLOP/s default but still below peak.
  - B300 is materially stronger than the B200 proxy defaults: 1.88 PFLOP/s bf16 and 3.56 PFLOP/s fp8_e4m3 at 16384. HBM stream reached 6.42 TB/s with this simple JAX kernel, below the public 7.4 TB/s STREAM-like B200/GB200 reports but still far above H100.
  - The collective numbers are framework-facing JAX/XLA collectives, not native `nccl-tests`. Treat all-reduce as the most reliable of the three. The reduce-scatter path is suspiciously low and may reflect the benchmark's `psum_scatter` shape/lowering rather than hardware.
  - Native `nccl-tests` was not preinstalled on the B300 Slurm environment; no NCCL module or `all_reduce_perf` binary was found.
- Next action:
  - For production NCCL numbers, build or bring a known `nccl-tests` binary/container and run `all_reduce_perf`, `all_gather_perf`, and `reduce_scatter_perf`.
  - For H100 inter-node JAX, fix the task-image CUDA/cuDNN mismatch or pin JAX/JAXLIB to the image's runtime libraries before reattempting.

### 2026-07-07 00:00 - Marginal Collective Cost Fit
- Hypothesis: The large-message region of the one-node JAX/XLA collective measurements can be approximated as `time = alpha + beta * message_GiB`, giving a useful marginal communication-cost model for planner sensitivity studies.
- Commit Hash: uncommitted workspace changes.
- Command: Fit a least-squares line over the 64 MB, 256 MB, 512 MB, and 1024 MB points from the 2026-07-06 benchmark entry.
- Config:
  - `message_GiB` is the per-rank payload passed to the collective.
  - Slopes are wall-clock marginal cost for the JAX collective call, not raw link transfer time.
  - For `busbw` conversion on 8 ranks: all-reduce multiplies algorithm bandwidth by `1.75`; all-gather/reduce-scatter multiply by `0.875`.
- Result:

  | Platform | Collective | alpha ms | marginal ms/GiB | marginal us/MiB | effective alg GB/s | fit R2 |
  |---|---|---:|---:|---:|---:|---:|
  | H100 | all-reduce | 0.576 | 4.57 | 4.46 | 235 | 0.998 |
  | H100 | all-gather | 0.620 | 4.87 | 4.76 | 220 | 0.997 |
  | H100 | reduce-scatter | 0.721 | 25.73 | 25.13 | 41.7 | 1.000 |
  | B300 | all-reduce | 0.747 | 2.51 | 2.45 | 429 | 0.999 |
  | B300 | all-gather | 0.674 | 1.01 | 0.99 | 1062 | 0.994 |
  | B300 | reduce-scatter | 0.691 | 12.18 | 11.89 | 88.2 | 1.000 |

- Interpretation:
  - A planner can use these direct wall-clock models for one-node JAX/XLA collectives:
    - H100 all-reduce: `0.000576 + 0.00457 * GiB`
    - H100 all-gather: `0.000620 + 0.00487 * GiB`
    - H100 reduce-scatter: `0.000721 + 0.02573 * GiB`
    - B300 all-reduce: `0.000747 + 0.00251 * GiB`
    - B300 all-gather: `0.000674 + 0.00101 * GiB`
    - B300 reduce-scatter: `0.000691 + 0.01218 * GiB`
  - Treat reduce-scatter as a measured framework penalty, not a hardware lower bound. It is far slower than expected relative to all-reduce/all-gather and likely reflects the `lax.psum_scatter` benchmark/lowering.
  - For a hardware-optimistic model, derive reduce-scatter from all-gather/all-reduce bandwidth instead: on the same fabric, large-message reduce-scatter should be in the same `busbw` family as all-gather.
- Next action:
  - Replace the JAX reduce-scatter estimate with native `nccl-tests reduce_scatter_perf` before using it as a firm PP/DP objective input.

### 2026-07-07 00:20 - Corrected AG/RS Bandwidth Semantics And XLA Flags
- Hypothesis: The apparent reduce-scatter slowness is partly a benchmark accounting error rather than a pure hardware or XLA pathology; public XLA flags exist for collective combining, pipelining, and overlap.
- Commit Hash: uncommitted workspace changes.
- Command:
  - Public-source review: NVIDIA `nccl-tests` performance definitions, OpenXLA GPU flag guidance, JAX GPU performance tips, JAX `psum_scatter` docs, and MaxText GPU launch flags.
  - Local validation: `python -m py_compile scripts/bench/gpu_microbench.py`.
- Config:
  - `scripts/bench/gpu_microbench.py` now reports `logical_mb` and computes AG/RS algorithm bandwidth with NCCL-style logical size `S = per_rank_chunk * ranks`.
  - For 8 ranks, AG/RS `busbw` from the earlier table was undercounted by 8x. All-reduce was already using the correct `S`.
- Result:
  - Corrected one-node JAX/XLA collective `busbw` in GB/s:

    | Platform | Message | all-reduce | all-gather | reduce-scatter |
    |---|---:|---:|---:|---:|
    | H100 | 64 MB | 148.3 | 445.1 | 218.0 |
    | H100 | 256 MB | 271.1 | 1084.4 | 254.7 |
    | H100 | 512 MB | 316.2 | 1265.2 | 276.7 |
    | H100 | 1024 MB | 368.9 | 1352.9 | 284.6 |
    | B300 | 64 MB | 131.7 | 614.8 | 313.7 |
    | B300 | 256 MB | 333.5 | 2036.8 | 509.8 |
    | B300 | 512 MB | 476.7 | 3308.9 | 555.4 |
    | B300 | 1024 MB | 576.5 | 4405.4 | 583.5 |

  - Corrected large-message marginal fits over 64 MB, 256 MB, 512 MB, and 1024 MB:

    | Platform | Collective | alpha ms | marginal ms/GiB logical | marginal us/MiB logical | effective alg GB/s | fit R2 |
    |---|---|---:|---:|---:|---:|---:|
    | H100 | all-reduce | 0.576 | 4.57 | 4.46 | 235 | 0.998 |
    | H100 | all-gather | 0.620 | 0.609 | 0.595 | 1762 | 0.997 |
    | H100 | reduce-scatter | 0.721 | 3.22 | 3.14 | 334 | 1.000 |
    | B300 | all-reduce | 0.747 | 2.51 | 2.45 | 429 | 0.999 |
    | B300 | all-gather | 0.674 | 0.126 | 0.123 | 8499 | 0.994 |
    | B300 | reduce-scatter | 0.691 | 1.52 | 1.49 | 706 | 1.000 |

  - Direct wall-clock fit against per-rank input/output chunk size is unchanged from the previous entry:
    - H100 reduce-scatter: `0.000721 + 0.02573 * GiB_per_output_chunk`
    - B300 reduce-scatter: `0.000691 + 0.01218 * GiB_per_output_chunk`
- Interpretation:
  - NCCL docs define reduce-scatter and all-gather `S` as the total logical array size, not the per-rank chunk. This correction makes reduce-scatter much less anomalous: B300 1 GiB chunk RS is essentially at the all-reduce busbw level, and H100 1 GiB chunk RS is slower but no longer off by an order of magnitude.
  - The very high all-gather `busbw` values are probably a framework benchmark/lowering artifact or an unusually favorable single-node path; do not use them as a hardware lower bound without native `nccl-tests`.
  - Public docs do not show a clear current "JAX reduce-scatter is inherently slow on GPU" issue. The known knobs are XLA collective combining, pipelining, latency-hiding scheduling, PGLE, and NCCL algorithm/protocol selection.
  - Candidate XLA flag bundle to test on training workloads:

    ```bash
    export XLA_FLAGS="$XLA_FLAGS \
      --xla_gpu_enable_latency_hiding_scheduler=true \
      --xla_gpu_enable_highest_priority_async_stream=true \
      --xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
      --xla_gpu_all_gather_combine_threshold_bytes=134217728 \
      --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 \
      --xla_gpu_enable_pipelined_all_gather=true \
      --xla_gpu_enable_pipelined_reduce_scatter=true \
      --xla_gpu_enable_pipelined_all_reduce=true \
      --xla_gpu_enable_all_gather_combine_by_dim=false \
      --xla_gpu_enable_reduce_scatter_combine_by_dim=false"
    ```

  - Optional diagnostics/sweeps:
    - Force NCCL lowering when available: `--xla_gpu_collectives_implementation=nccl`.
    - Sweep NCCL behavior outside XLA: `NCCL_ALGO=Ring,Tree`, `NCCL_PROTO=Simple,LL,LL128`, `NCCL_DEBUG=INFO`.
    - Dump HLO to verify whether `psum_scatter` lowers to a reduce-scatter NCCL custom call: `--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text`.
    - Use JAX PGLE for real training steps; isolated single collective microbenchmarks have no compute to overlap, so pipelining and latency-hiding flags may not move them much.
- Next action:
  - Re-run the collective microbenchmarks with the corrected script output for clean JSON records.
  - Build or stage native `nccl-tests` binaries and compare `all_gather_perf` and `reduce_scatter_perf` against the framework-facing JAX/XLA path.

### 2026-07-07 12:05 - Two-Node H100 JAX Collective Evidence
- Hypothesis: Multinode H100 collectives can be measured through the same JAX/XLA harness if each 8-GPU node runs one JAX process per GPU; this should provide evidence for inter-node planning costs, at least for all-reduce and reduce-scatter.
- Commit Hash: `ffa3014af4` plus uncommitted benchmark/logbook files.
- Command:
  - Fast-forward before retrying after the earlier cuDNN mismatch: `git fetch origin main && git merge origin/main`.
  - Working shape for multinode H100: `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --timeout <timeout> --max-retries <0 or 5> --cpu 32 --memory 256GB --disk 128GB --gpu H100x8 --replicas 2 --processes-per-task 8 --enable-extra-resources --extra gpu --sync-package marin-iris --sync-package marin-levanter -- bash -lc '<set LD_LIBRARY_PATH from venv NVIDIA wheels>; .venv/bin/python scripts/bench/gpu_microbench.py --skip-gemm --skip-hbm ...'`.
  - Successful all-reduce job: `/dlwh/gpu-bench-h100-2node-ar-r9`, `--collectives all_reduce --collective-message-mb 1,8,64,256,512,1024 --iterations 3 --warmup 1`.
  - Successful reduce-scatter job: `/dlwh/gpu-bench-h100-2node-rs-r13-retry`, `--collectives reduce_scatter --collective-message-mb 1,8,64,256 --iterations 3 --warmup 1`.
  - All-gather negative controls: `/dlwh/gpu-bench-h100-2node-ag-r12-retry` used the script's normal `lax.all_gather(v, "rank", axis=0)`; `/dlwh/gpu-bench-h100-2node-ag-tiled-r15` used `lax.all_gather(v, "rank", axis=0, tiled=True)` on a `(1, chunk)` input.
- Config:
  - 2 Iris tasks, each `H100x8`, coscheduled by leaf group; `--processes-per-task 8` gives 16 JAX processes and one local GPU per process.
  - JAX 0.10.1, 16 total devices, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS="... --xla_gpu_enable_command_buffer="`.
  - Each task prepended Python wheel CUDA library paths for cuDNN/NCCL/CUDA runtime/cuBLAS/cuSPARSE/cuSOLVER to `LD_LIBRARY_PATH`.
  - Iris `origin/main` setup restores CUDA 13 cuDNN precedence. Some H100 nodes lack the `nvidia-cudnn-cu13==9.19.0.56` wheel in the offline uv cache; jobs with `--max-retries 5` can work around this when a retry lands on cached nodes.
- Result:
  - All-reduce completed across 16 ranks. Median over 16 rank-local JSON records:

    | Message | seconds | busbw GB/s |
    |---:|---:|---:|
    | 1 MB | 0.000401 | 4.9 |
    | 8 MB | 0.001865 | 8.4 |
    | 64 MB | 0.000701 | 179.5 |
    | 256 MB | 0.001681 | 299.3 |
    | 512 MB | 0.002959 | 340.2 |
    | 1024 MB | 0.005518 | 364.9 |

  - Reduce-scatter completed through 256 MB per output chunk. Median over 16 rank-local JSON records:

    | Chunk message | logical array | seconds | busbw GB/s |
    |---:|---:|---:|---:|
    | 1 MB | 16 MB | 0.000627 | 25.1 |
    | 8 MB | 128 MB | 0.000810 | 155.4 |
    | 64 MB | 1024 MB | 0.003793 | 265.4 |
    | 256 MB | 4096 MB | 0.014493 | 277.8 |

  - Large reduce-scatter `/dlwh/gpu-bench-h100-2node-rs-r14-large` retried past an offline cache miss, initialized all 16 JAX processes, then produced no 512 MB timing record after about five minutes and was stopped.
  - All-gather did not produce any busbw records. Normal stacked-axis all-gather and tiled all-gather both initialized all 16 JAX processes, printed summaries, then hung before the first 1 MB warmup/timing record; both jobs were stopped.
  - Failed/blocked setup attempts:
    - One-process-per-node multinode JAX initialized only two processes with eight local devices each and hung in XLA/NCCL clique initialization. Use one process per GPU for this path.
    - Before the fast-forward, multinode GPU collectives hit a cuDNN runtime/source mismatch (`9.10.2` runtime vs `9.12.0` compiled expectation).
    - After the fast-forward, some attempts failed before user code because the node-local offline uv cache lacked the CUDA 13 cuDNN wheel.
- Marginal fit:
  - All-reduce over 64, 256, 512, and 1024 MB: `time ~= 0.000390 + 0.00513 * GiB`, where GiB is the per-rank all-reduce payload. Fit `R2=0.99999`.
  - Reduce-scatter over 64 and 256 MB chunks: `time ~= 0.000226 + 0.0571 * GiB_per_output_chunk`. Equivalently, `0.00357 s/GiB` of logical reduce-scatter array at 16 ranks. This is only a two-point fit and should be treated as exploratory.
- Interpretation:
  - Experimental evidence supports H100 inter-node all-reduce busbw at roughly 365 GB/s for 16 ranks / 2 nodes / 1 GiB payload under this JAX/XLA stack.
  - Experimental evidence supports H100 inter-node reduce-scatter through 256 MB per output chunk, reaching roughly 278 GB/s busbw.
  - The requested multinode H100 all-gather row is not experimentally backed by this harness. Public/native NCCL numbers should still be used for planning until `all_gather_perf` or a non-hanging JAX lowering is available.
  - The 512 MB and 1 GiB reduce-scatter points are also not backed by completed experimental records from this run.
- Next action:
  - Build or stage native `nccl-tests` on the H100 Iris image and run `all_gather_perf` / `reduce_scatter_perf` to separate hardware/NCCL behavior from JAX/XLA lowering hangs.
  - Fix the Iris GPU setup cache issue so CUDA 13 cuDNN wheel availability is deterministic across H100 nodes.

### 2026-07-07 12:30 - Full Evidence Report Drafted
- Hypothesis: The hardware-planning brief should separate measured values from source-backed, inferred, and unknown values so the cost model does not accidentally treat proxy or failed measurements as facts.
- Commit Hash: `ffa3014af4` plus uncommitted benchmark/logbook/report files.
- Command:
  - Drafted `.agents/reports/gpu-hardware-target-report.md` from the logbook and current public source checks.
  - Ran a private-name hygiene scan over `.agents/reports`, `.agents/logbooks`, and `scripts/bench`.
- Config:
  - Report scope: H100 target, Blackwell proxy measurements, GB200/NVL72 source-backed sensitivity, one-node and two-node collective evidence, topology assumptions, and open gaps.
  - Durable report avoids private cluster names and labels evidence as measured, source-backed, inferred, or unknown.
- Result:
  - Report path: `.agents/reports/gpu-hardware-target-report.md`.
  - Coordinating issue: https://github.com/marin-community/marin/issues/7010.
  - Private-name hygiene check returned no matches.
- Interpretation:
  - The report answers the original request where the evidence exists, and explicitly marks the missing pieces: GB200/NVL72 local measurements, native NCCL all-gather/reduce-scatter, large H100 two-node reduce-scatter, and fabric oversubscription.
- Next action:
  - Use the report as the current cost-model input document until native `nccl-tests` measurements are available.
