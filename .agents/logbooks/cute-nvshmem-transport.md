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

- `NVTP-002`: NVSHMEM put-with-signal provides the simplest correct device-side push primitive and competitive transport performance. Next test: two-PE blocking and nonblocking correctness with deterministic payloads and monotonic epochs.
- `NVTP-003`: Batched nonblocking gets amortize device-side completion enough for destination-directed pull to be viable. Next test: compare blocking get, per-get completion, and four-get batched completion.
- `NVTP-004`: Direct peer tensor stores or loads reduce transport overhead relative to NVSHMEM RMA while retaining correct symmetric addressing. Next test: compare minimal cooperative peer store/load probes against RMA on the same arena.
- `NVTP-005`: JAX buffers can participate in the local endpoint or symmetric allocation views without payload copies or host synchronization. Next test: build an ownership, DLPack lifetime, and stream-handoff interoperability matrix.
- `NVTP-006`: A transport can overlap with a production-relevant GEMM without unacceptable compute-throughput degradation. Next test: establish standalone communication and compute baselines before concurrent execution.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- `NVTP-001`: A coherent isolated stack supports collective symmetric allocation, deterministic addressing, host peer aliases, and CuTe `get_peer_tensor` on a fully connected CoreWeave 8×H100 node. Evidence: `NVTP-002` entry below.

## Decision Log

- 2026-07-10: Use CoreWeave H100 clusters for all accelerator correctness and performance results, per user direction.
- 2026-07-10: Keep NVSHMEM RMA and direct peer-tensor access as separate benchmark families.
- 2026-07-10: Isolate NVSHMEM transport from the normal GPU/Torch extras and pin CUTLASS DSL 4.4.2 for compatibility with NVSHMEM4Py 0.3.1.

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
