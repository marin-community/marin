---
topic: cute-nvshmem-transport
issue: https://github.com/marin-community/marin/issues/7114
description: Evaluate CuTe DSL and NVSHMEM push, pull, and direct peer transport for MoE payloads.
author: dlwh
---

# CuTe NVSHMEM Transport: Research Logbook

## Current TL;DR

- The CoreWeave node is a fully connected 8×H100 80GB SXM topology (`NV18` for every GPU pair), suitable for same-node peer-tensor experiments.
- Stage 1 passes on 2 and 8 H100s: collective symmetric allocation, host peer aliases, CuTe compilation linked with NVSHMEM device bitcode, and direct `get_peer_tensor` ring stores returned the expected rank-coded values.
- The reproducible transport extra pins NVSHMEM 3.7.0, NVSHMEM4Py 0.3.1, and CUTLASS DSL 4.4.2. NVSHMEM4Py's CuTe integration is incompatible with Marin's normal CUTLASS DSL 4.5.2 environment.
- Device-only push and pull correctness passes on 8 H100s for RMA and direct peer variants, 1/2/4/8 slots, and one million aggregate single-slot transfers per principal variant. Ordinary CuTe payload loads after `signal_wait` can observe stale data; a system fence plus cache-volatile loads is required by the validated implementation.
- JAX interoperability is zero-copy when ownership is explicit: NVSHMEM buffers import into JAX at the identical pointer, and JAX-owned sources work as local RMA operands through a non-owning `cuda.core.Buffer` plus DLPack stream handoff. A two-PE JAX→NVSHMEM→JAX push passed with pointer identity on the receiver.
- The corrected 8-PE transport sweep contains 77 rows across 16 B–384 KiB payloads. At the 6144-byte production anchor, warp `put_signal` reaches 5.30 µs, 1.16 GB/s per PE, and 9.27 GB/s aggregate; warp blocking pull reaches 14.68 µs and 0.419 GB/s per PE. Scalar direct-peer loops are noncompetitive and are not a valid proxy for cooperative peer copies.
- All accelerator experiments will use CoreWeave H100 clusters. Host-only inspection may run locally.

## Scope

- Goal: Determine whether NVSHMEM `put_signal`, NVSHMEM `get_nbi` plus completion, direct peer-tensor stores, or direct peer-tensor loads should replace or complement the current Pallas/Mosaic source-push path.
- Primary metrics: correctness under device-only synchronization and repeated slot reuse; zero-copy JAX interoperability; payload latency; effective and aggregate bandwidth; readiness-to-consumption latency; communication resource use; concurrent GEMM degradation.
- Constraints: Remote payload and control endpoints must be symmetric objects; distinguish RMA operations from peer aliases; no steady-state host synchronization; do not port MoE GEMMs or the full persistent kernel; use CoreWeave H100s for GPU experiments.
- Coordinating issue: https://github.com/marin-community/marin/issues/7114
- Experiment prefix: `NVTP`
- Shared tags: `NVTP`, `issue-7114`, `cw-h100`, `cute-nvshmem`

## Current Baseline

- Date: 2026-07-10
- Code ref: `b09baa125ab4`
- Baseline path: current Pallas/Mosaic source-push implementation; exact code path and benchmark numbers pending internal forage.

## Hypothesis Queue

### Active

- `NVTP-002`: NVSHMEM put-with-signal is device-side correct with monotonic epochs and safe reuse. Next test: measure latency/bandwidth and cooperative scopes against explicit put+signal.
- `NVTP-003`: Blocking get and `get_nbi` plus low-level device `quiet` are device-side correct. Next test: compare per-get quiet with multiple gets per quiet and measure completion cost.
- `NVTP-004`: Direct peer stores and loads are device-side correct with explicit system memory operations. Next test: compare performance against RMA on production payload sizes.
- `NVTP-005`: JAX buffers participate without payload copies when wrapped with explicit owner/lifetime and DLPack stream handoff. Next test: integrate the handoff into device-kernel launch chaining without test-only host stream synchronization.
- `NVTP-006`: A transport can overlap with a production-relevant GEMM without unacceptable compute-throughput degradation. Next test: establish standalone communication and compute baselines before concurrent execution.
- `NVTP-007`: Warp-cooperative put-with-signal is the leading measured transport. Next test: add block/cooperative direct-peer variants, Pallas comparison, and GEMM overlap before making a production recommendation.

### Blocked

- Scalar direct-peer copies at production sizes: one-thread word loops plateau near 0.0025 GB/s for stores and 0.0046 GB/s for loads. Evidence: `NVTP-005` entry and artifact `cute_nvshmem_transport/artifacts/nvtp-benchmark-cw-h100-8pe.json`.

### Falsified / Dead End

- None.

### Promoted

- `NVTP-001`: A coherent isolated stack supports collective symmetric allocation, deterministic addressing, host peer aliases, and CuTe `get_peer_tensor` on a fully connected CoreWeave 8×H100 node. Evidence: `NVTP-002` entry below.
- `NVTP-002-CORRECTNESS`: Push and pull RMA/direct-peer protocols pass deterministic device-only ring validation through repeated slot reuse. Evidence: `NVTP-003` entry below and commit `c58969f590`.
- `NVTP-005-LOCAL`: JAX source and destination aliases interoperate zero-copy with symmetric transport when wrapped at the boundary. Evidence: `NVTP-004` entry below and commit `abebeb98dd`.

## Decision Log

- 2026-07-10: Use CoreWeave H100 clusters for all accelerator correctness and performance results, per user direction.
- 2026-07-10: Keep NVSHMEM RMA and direct peer-tensor access as separate benchmark families.
- 2026-07-10: Isolate NVSHMEM transport from the normal GPU/Torch extras and pin CUTLASS DSL 4.4.2 for compatibility with NVSHMEM4Py 0.3.1.
- 2026-07-10: Use explicit system fencing plus cache-volatile payload loads after signal waits; ordinary CuTe indexing produced reproducible stale-epoch failures.
- 2026-07-10: Use `nvshmem.bindings.device.cute.quiet` for device `_nbi` completion because the high-level `nvshmem.core.device.cute` module does not export quiet.
- 2026-07-10: Wrap JAX device pointers in non-owning `cuda.core.Buffer` objects with the JAX array as owner; raw JAX objects are not accepted by NVSHMEM host RMA.
- 2026-07-10: Benchmark repetitions use all-PE barriers before signal reset and after each kernel. Per-rank local completion is insufficient because late remote signals can cross a repetition boundary.

## Negative Results Index

- None.

## Stop Criteria

- Stop the CuTe/NVSHMEM path if symmetric-memory integration requires payload copies, JAX stream integration requires host synchronization, device-side pull completion is prohibitively coarse, direct peer access cannot compete with the staged baseline, or runtime and memory-lifecycle complexity outweigh the transport benefit.

## Initial Experiment Matrix

1. `NVTP-001`: symmetric arena, deterministic offsets, local view, and peer view.
2. `NVTP-002`: blocking put, put-with-signal, nonblocking put-with-signal, and safe reuse.
3. `NVTP-003`: publication, blocking get, nonblocking get plus completion, batched completion, and safe reuse.
4. `NVTP-004`: direct peer stores and loads with explicit ordering and signaling.
5. `NVTP-005`: JAX zero-copy endpoint and stream interoperability matrix.
6. `NVTP-006`: transport sweep, all-PE scaling, compute overlap, and comparison with Pallas.

## Entry Log

### 2026-07-10 - NVTP-000 research prologue

- Hypothesis: A fixed symmetric arena can support a controlled four-way comparison of NVSHMEM push, NVSHMEM pull, peer-tensor stores, and peer-tensor loads without requiring arbitrary remote CUDA pointers.
- Commit Hash: `b09baa125ab4` (starting revision)
- Command: Repository and issue searches plus local code inventory; no GPU command run yet.
- Config: Dedicated branch `research/cute-nvshmem-transport`; accelerator target CoreWeave 8×H100; issue #7114; experiment prefix `NVTP`.
- Result: No matching open issue was found. Created https://github.com/marin-community/marin/issues/7114 and initialized this logbook and hypothesis queue.
- Interpretation: Begin high-effort internal and external prior-work foraging before reserving an expensive H100 node. Use the smallest environment probe to decide whether Stage 1 is immediately implementable with the installed APIs.
- Next action: Locate the production baseline and payload shapes; verify current official API and memory-ordering semantics; inspect CoreWeave image/package availability.

### 2026-07-10 - NVTP-001 environment and API gate

- Hypothesis: The checked-in GPU dependencies expose the CuTe NVSHMEM device API on a same-node CoreWeave 8×H100 pod.
- Commit Hash: `b09baa125ab4` (starting revision)
- Command: Reserve `cw-us-east-02a` with `scripts/iris/dev_gpu.py`; inspect `nvidia-smi -L`, `nvidia-smi topo -m`, installed distributions, and imports; then install `nvshmem4py-cu13==0.3.1` and `nvidia-nvshmem-cu13==3.7.0` inside the disposable pod.
- Config: One CoreWeave node, 8×H100 80GB HBM3, all GPU pairs `NV18`; JAX 0.10.1; CUTLASS DSL 4.5.2.
- Result: The repo environment installed NVSHMEM runtime 3.4.5 but no NVSHMEM4Py package; `import nvshmem` failed. Installing NVSHMEM4Py 0.3.1 alone allowed host imports but the CuTe device module failed. Upgrading the runtime to 3.7.0 made `nvshmem.core.device.cute` import and exposed `get_peer_tensor`, blocking/nonblocking put/get, thread/warp/block variants, put-with-signal, signal operations, and waits.
- Interpretation: `NVTP-001` is partially supported but the checked-in version matrix is falsified. A dependency upgrade is required before a reproducible Stage 1 harness. Import success does not prove compilation, initialization, ordering, or JAX interoperability.
- Next action: Pin a coherent NVSHMEM 3.7.0/NVSHMEM4Py 0.3.1 environment on the research branch, implement UID bootstrap for one PE per GPU, and compile the smallest `my_pe` plus symmetric peer read/write probe.

## Background Research Brief

- Effort: high
- Stop rule: stopped when additional official queries repeated the same API pages without resolving documented contradictions.
- Date: 2026-07-10

### Current Marin Context

- The most complete Pallas/Mosaic semantic source-push baseline is pinned at https://github.com/marin-community/marin/commit/c53bbcdfba, with a fused W13 follow-up at https://github.com/marin-community/marin/commit/3b334d7510. It is not present at the starting revision.
- Existing CuTe/JAX integration in `lib/levanter/src/levanter/grug/attention/_fa4_thd.py` provides reusable `cutlass_call`, dtype, descriptor, and SM90 patterns but no transport or NVSHMEM lifecycle support.
- DeepEP transport under `lib/levanter/src/levanter/kernels/deepep/` is the closest peer-NVLink baseline, but its device-scalar read performs D2H plus stream synchronization and is unsuitable as the steady-state acceptance model.
- Production payload anchor: Grug MoE uses hidden size 3072, making one bf16 routed row 6144 bytes. Initial chunk sweep: 1, 2, 4, 8, 16, 32, and 64 rows across 2, 4, and 8 GPUs.

### Evidence Map

#### Claim: Symmetric identity, not arbitrary remote CUDA pointers, is the correct addressing model

- Support: NVIDIA documents collective symmetric allocation and `nvshmem_ptr`/peer views derived from a local symmetric address plus PE.
- Contradiction: Peer aliases may be unavailable for non-P2P peers, and passing a translated alias where an NVSHMEM API requires the original symmetric address is undefined.
- Confidence: stable for the documented model; runtime availability still requires a topology probe.
- Action: Keep RMA endpoints as original symmetric tensors and use peer aliases only for direct CuTe loads/stores.

#### Claim: Ordering and completion must be tested separately

- Support: NVIDIA documents fence as ordering, quiet as completion, and put-with-signal as an associated payload/notification primitive.
- Contradiction: CuTe `_nbi` prose loosely suggests fence or synchronization for completion, conflicting with the core fence semantics.
- Confidence: stable that fence alone is not a safe completion assumption.
- Action: Use quiet/barrier for `_nbi` completion and include a fence-only negative probe.

#### Claim: Official JAX-to-symmetric-memory interoperability is unproven

- Support: NVSHMEM buffers are DLPack-compatible and CuTe accepts DLPack producers.
- Negative result: No official end-to-end example shows JAX owning an NVSHMEM symmetric allocation or registering an ordinary JAX allocation as a remotely addressable symmetric endpoint.
- Confidence: exploratory.
- Action: Treat ownership, lifetime, aliasing, and stream handoff as a separate gate before performance work.

### Recommended Next Experiments

1. Compile a two-PE symmetric allocation and `my_pe`/peer read-write probe on the same H100 node.
2. Validate blocking put/get and `_nbi` plus quiet, then put-with-signal across one million epoch-reused transfers.
3. Prove local JAX DLPack aliasing and stream ordering independently of remote transport.
4. Sweep 6 KiB row multiples and compare effective bandwidth/latency against local copy, DeepEP, and the pinned semantic source-push harness.

### Source Ledger

- NVIDIA NVSHMEM memory and registration: https://docs.nvidia.com/nvshmem/api/gen/api/memory.html
- NVIDIA NVSHMEM pointer contract: https://docs.nvidia.com/nvshmem/api/gen/api/setup.html
- NVSHMEM4Py compatibility matrix: https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/overview.html
- CuTe device RMA API: https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/rma.html
- CuTe device API inventory: https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/index.html
- NVSHMEM ordering model: https://docs.nvidia.com/nvshmem/api/using.html
- NVSHMEM topology FAQ: https://docs.nvidia.com/nvshmem/api/faq.html
- CUTLASS CuTe runtime/DLPack API: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_runtime.html

### 2026-07-10 - NVTP-002 Stage 1 symmetric arena and peer tensor

- Hypothesis: A coherent NVSHMEM4Py stack can collectively allocate symmetric memory and construct working host and CuTe peer aliases across all eight H100s.
- Commit Hash: `225960f42f`
- Command: `NVTP_NUM_PES={2,8} uv run --package marin-levanter --extra nvshmem-transport python -m cute_nvshmem_transport.launch`
- Config: CoreWeave `cw-us-east-02a`, one 8×H100 80GB node, all GPU pairs `NV18`; Python 3.12.13; CUDA Python 13.2.0; cuda-core 1.1.0; CUTLASS DSL 4.4.2; NVSHMEM 3.7.0; NVSHMEM4Py 0.3.1.
- Result: Both 2-PE and 8-PE runs passed. Each process initialized one PE per GPU from a shared UID, collectively allocated corresponding buffers, read its successor through `get_peer_buffer`, compiled a CuTe kernel linked with NVSHMEM device bitcode, and stored its rank-coded value into its successor through `get_peer_tensor`. Every PE observed the expected successor value through the host alias and predecessor value through the device ring store.
- Interpretation: `NVTP-001` is promoted. Same-node symmetric addressing and direct peer tensor construction are feasible on the target CoreWeave H100 topology. This does not yet establish RMA ordering, slot reuse, bandwidth, or JAX ownership semantics.
- Negative results: Runtime 3.4.5 had no NVSHMEM4Py binding. NVSHMEM4Py 0.3.1 plus runtime 3.4.5 failed the CuTe import. Runtime 3.7.0 plus CUTLASS DSL 4.5.2 failed inside CuTe interop because `Constexpr` was removed. The eight-rank ring also exposed a two-rank-only oracle error: predecessor and successor coincide at two ranks but differ at eight; correcting the oracle made all observed values pass.
- Next action: Implement blocking put and put-with-signal correctness with monotonic epochs and safe slot reuse.

### 2026-07-10 - NVTP-003 push/pull correctness matrix

- Hypothesis: Push, pull, and direct peer transport can run device-only with monotonic epochs, payload-before-ready ordering, completion-before-consumption, and safe slot reuse.
- Commit Hash: `c58969f590`
- Commands: `NVTP_NUM_PES=8 NVTP_NUM_EPOCHS=125000 NVTP_NUM_SLOTS=1 NVTP_{PUSH,PULL}_OPERATION=<variant> uv run --package marin-levanter --extra nvshmem-transport python -m cute_nvshmem_transport.correctness_{push,pull}`; repeat with `NVTP_NUM_EPOCHS=1000` and `NVTP_NUM_SLOTS={2,4,8}`.
- Config: Same CoreWeave 8×H100 `NV18` node and dependency matrix as NVTP-002. Payloads encode producer rank, monotonic epoch, slot, and rank-XOR-epoch checksum. Each PE runs producer and consumer warps in one kernel; no host synchronization occurs inside the transfer loop.
- Result: Zero validation errors for blocking `put_signal`, `put_signal_nbi+quiet`, `put_nbi+quiet+signal`, peer stores+signal, blocking get, `get_nbi+quiet`, and peer loads. Every variant passed 8 PEs with 8 slots and 1,000 epochs. Each principal variant also passed 8 PEs × 125,000 one-slot epochs, or one million aggregate transfers. Final ready and consumed epochs matched every slot's expected last epoch.
- Timing note: End-to-end process times for the million-transfer runs were about 11.7–13.5 seconds, including eight Python process startups, per-rank compilation, NVSHMEM initialization/finalization, and output. These are correctness-run wall times and must not be interpreted as transport performance.
- Interpretation: Device-initiated push, pull, direct peer stores, and direct peer loads are all feasible on the target topology. Completion is available for `_nbi` through the generated low-level `quiet` binding even though the high-level CuTe module omits it. Performance, batching, cooperative scopes, large payload volume, and JAX interoperability remain open.
- Negative results: Ordinary `inbox[i]` loads after `signal_wait` observed a complete but stale previous-epoch payload (for example epoch 9 saw epoch 8). A system-scope fence followed by PTX cache-volatile loads eliminated the failure through millions of transfers. Combining acquire ordering and `.cv` on one PTX load is rejected by the NVVM verifier, so the working sequence separates the fence from weak cache-volatile loads. Initial multi-slot reuse also deadlocked when a negative `epoch-slots` threshold cast to `uint64`; initial fills now skip the consumed wait and wrapped epochs wait for the previous matching slot epoch.
- Next action: Validate JAX DLPack ownership/lifetime and stream handoff, then build payload-size and completion-batching benchmarks.

### 2026-07-10 - NVTP-004 JAX zero-copy interoperability

- Hypothesis: JAX-owned local operands and JAX views of symmetric memory can participate without payload copies when ownership and stream handoff are explicit.
- Commit Hash: `abebeb98dd`
- Commands: `uv run --package marin-levanter --extra nvshmem-transport python -m cute_nvshmem_transport.jax_interop`; repeat with `NVTP_JAX_REMOTE=1` for the two-PE path.
- Config: CoreWeave 8×H100 node; one-PE local probe and two-PE remote push; 16-byte `uint8` payload; JAX 0.10.1 plus the validated NVSHMEM transport stack.
- Result: `jax.dlpack.from_dlpack` imported an NVSHMEM symmetric buffer at the identical device pointer. JAX GPU computation observed external writes, while a previously materialized NumPy host array remained stale. Pure JAX operations produced a distinct output allocation. NVSHMEM host `put` rejected a raw JAX array but accepted a non-owning `cuda.core.Buffer.from_handle` with the JAX array as owner. JAX accepted DLPack handoff to the NVSHMEM CUDA stream, and the wrapped put reproduced all source values. In the two-PE test, a JAX-owned rank-0 source was put-with-signal into rank-1 symmetric memory; a pre-existing rank-1 JAX view had identical pointer ownership and computed the expected sum 120 without a payload copy.
- Interpretation: Questions 1–3 and the push-consume row of the interoperability matrix are feasible with a thin ownership wrapper. NVSHMEM owns symmetric allocations; JAX may own local sources or view symmetric destinations. Direct JAX objects are not an accepted NVSHMEM host operand type. Existing host materializations must not be reused as coherence evidence.
- Limitations: The correctness probe synchronizes the CUDA stream before final validation. DLPack establishes JAX→external-stream producer handoff, but a fused steady-state launch chain still needs an explicit external-stream/event integration rather than test-only host synchronization.
- Next action: Build machine-readable transport benchmarks for production payload sizes, cooperative scopes, completion batching, and overlap.

### 2026-07-10 - NVTP-005 corrected 8-PE transport sweep

- Hypothesis: Cooperative NVSHMEM operations outperform thread-scoped RMA and scalar direct-peer loops for routed-row payloads on the fully connected CoreWeave H100 node.
- Commit Hash: `74b485ca7b`
- Command: `uv run --package marin-levanter --extra nvshmem-transport python -m cute_nvshmem_transport.benchmark_transport --num-pes 8 --num-slots 8 --repetitions 3 --payload-bytes 16 256 4096 6144 24576 98304 393216 --output /tmp/nvtp-benchmark-v2.json`
- Config: CoreWeave `cw-us-east-02a`, 8×H100 80GB, fully connected `NV18`; 8 PEs; 8 reusable slots; three repetitions; payload-dependent epochs targeting 64 MiB for payloads above 256 bytes. Timings cover kernel launch through an all-PE post-kernel barrier and use the maximum rank median.
- Result: All 77 rows passed payload and epoch validation. At 6144 bytes, warp `put_signal` measured 5.304 µs, 1.158 GB/s per PE, and 9.267 GB/s aggregate. Warp `put_signal_nbi+quiet` measured 7.156 µs and 0.859 GB/s per PE. Warp blocking pull measured 14.677 µs and 0.419 GB/s per PE; warp `get_nbi+quiet` measured 16.385 µs and 0.375 GB/s per PE. Thread-scoped put measured 43.929 µs and 0.140 GB/s per PE. Scalar peer store/load loops measured 0.00246/0.00461 GB/s per PE. Warp put rose to 2.564 GB/s per PE at 384 KiB; warp pull plateaued near 0.548 GB/s per PE.
- Negative result: The first sweep failed on repetition two because ranks reset their local signals before slower peers globally completed repetition one; rank 7 consumed epoch 7 from an epoch-15 overwrite. A pre-reset barrier removed the stale payload, and a post-kernel barrier was additionally required before final host validation so the last remote consumed signal was visible. The corrected failing-case retest (8 PEs, 256 bytes, 10,000 epochs, 3 repetitions, direct peer load) passed before the full relaunch.
- Interpretation: Warp `put_signal` is the leading implemented path across the measured payload range. Blocking warp put consistently beats nonblocking-plus-quiet for this one-transfer-per-epoch protocol. Pull is substantially slower. Scalar peer results falsify only the scalar implementation, not cooperative/vectorized peer copies. Results are exploratory because the sweep is one node and lacks the Pallas and overlap baselines.
- Artifact: `cute_nvshmem_transport/artifacts/nvtp-benchmark-cw-h100-8pe.json` (structured rows) and `.jsonl` (streamed run record).
- Next action: Implement block/cooperative direct-peer variants and batched pull completion, then run the Pallas comparison and GEMM-overlap matrix.
