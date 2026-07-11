# CuTe DSL and NVSHMEM transport on eight H100s

## TL;DR

Do not replace the current Mosaic source-push path with this CuTe/NVSHMEM4Py
prototype.

Warp-cooperative `put_signal` was the fastest implemented transport. It moved a
6144-byte routed row in 5.30 µs at 1.16 GB/s per PE and 9.27 GB/s aggregate on
one fully connected 8×H100 CoreWeave node. The same transport reduced a
4096×4096 bf16 GEMM from 731 to 327 TFLOP/s per PE when run concurrently. Its
own bandwidth fell from 0.859 to 0.423 GB/s per PE.

Pull, direct peer access, and device-side completion batching were slower. A
stream-correct JAX→CuTe→NVSHMEM custom call did not compile against CUTLASS DSL
4.4.2 and NVSHMEM4Py 0.3.1. The working JAX routes require a host-synchronized
external CUDA stream. This violates the persistent-kernel acceptance criterion.

The results support keeping Mosaic while preserving this branch as a transport
and ordering reference. A future NVSHMEM binding with a working JAX custom-call
ABI could justify a new test.

## Setup

All accelerator results used CoreWeave `cw-us-east-02a` H100 nodes. The measured
node had eight H100 80GB SXM GPUs, with `NV18` reported between every GPU pair.
The transport environment used JAX 0.10.1, CUTLASS DSL 4.4.2, NVSHMEM 3.7.0,
NVSHMEM4Py 0.3.1, and CUDA Python 13.2.

Each PE owns the same symmetric layout. Remote locations are addressed by a
symmetric allocation, a deterministic byte offset, and a PE. RMA calls retain
the original symmetric address. Direct peer variants use `get_peer_tensor`
aliases only for ordinary CuTe loads and stores.

The principal benchmark is an eight-PE ring with eight reusable slots. Timings
start after compilation and an all-PE reset barrier, end after an all-PE
completion barrier, and report the slowest rank's median of three repetitions.
The raw 77-row sweep is in
[`artifacts/nvtp-benchmark-cw-h100-8pe.json`](artifacts/nvtp-benchmark-cw-h100-8pe.json).

## Correctness and ordering

Push and pull passed pair, ring, and all-to-all patterns with all eight GPUs
active. The all-to-all matrix covered push and pull, 2/4/8 slots, and 20-byte,
256-byte, and 6144-byte payloads. The ring tests covered one million aggregate
small transfers per principal variant. A large-volume gate moved 39.3 GB per PE
and 314.6 GB aggregate for each of warp push and warp pull while reusing eight
slots.

The validated payload sequence is:

1. The producer waits for the previous consumed epoch before reusing a slot.
2. The producer writes deterministic payload fields and applies a system fence.
3. Push uses put-with-signal. Pull publishes a ready signal, then the consumer
   issues a get and waits for completion.
4. After observing readiness or completion, the consumer applies a system fence
   and uses cache-volatile (`.cv`) loads for validation.
5. The consumer signals the consumed epoch to the producer.

Ordinary CuTe payload loads after `signal_wait` repeatedly observed the previous
epoch. A system fence followed by `.cv` loads removed the failure across millions
of transfers. NVVM rejects an acquire load combined with `.cv`, so the fence and
load are separate operations.

Benchmark repetitions also require two global boundaries. Without a pre-reset
barrier, a late consumed signal from repetition one can arrive after repetition
two resets its local epochs. Without a post-kernel barrier, the final remote
consumed signal may not be visible before host validation.

## Transport performance

### One 6144-byte routed row

| Protocol | Operation | Latency (µs) | GB/s per PE | Aggregate GB/s |
|---|---|---:|---:|---:|
| Push | warp `put_signal` | 5.304 | 1.158 | 9.267 |
| Push | warp `put_signal_nbi` + `quiet` | 7.156 | 0.859 | 6.869 |
| Push | thread `put_signal` | 43.929 | 0.140 | 1.119 |
| Push | thread `put_nbi` + `quiet` + signal | 44.107 | 0.139 | 1.114 |
| Pull | warp blocking get | 14.677 | 0.419 | 3.349 |
| Pull | warp `get_nbi` + `quiet` | 16.385 | 0.375 | 3.000 |
| Pull | four `get_nbi` slices + one `quiet` | 183.838 | 0.033 | 0.267 |
| Peer store | warp-cooperative stores + signal | 170.393 | 0.036 | 0.288 |
| Peer load | warp-cooperative `.cv` loads | 49.604 | 0.124 | 0.991 |

Blocking warp operations beat their nonblocking-plus-quiet counterparts for one
transfer per epoch. Four nonblocking gets followed by one quiet did not amortize
completion. It performed like the thread-scoped get path.

Warp push reached 2.564 GB/s per PE at 384 KiB in the size sweep and 2.605 GB/s
per PE in the 314.6 GB aggregate volume run. Warp pull plateaued near 0.549
GB/s per PE. Cooperative peer stores plateaued near 0.038 GB/s per PE and peer
loads near 0.143 GB/s per PE. NVSHMEM contributes a substantially better copy
implementation here, not only symmetric addressing and signaling.

CTA-wide NVSHMEM block operations did not compile. Linking
`nvshmemx_*_block` symbols produced a `ptxas` parse error near `.nvvm` in the
installed CUDA 13 stack. No block-scope performance claim is made. No applicable
TMA-backed NVSHMEM4Py API was present.

## Compute overlap

The overlap harness compiles eight transport ranks and eight per-GPU JAX GEMM
workers before releasing one filesystem start gate. The GEMM is 4096×4096 bf16.
`concurrent wall` is the maximum gated communication or compute duration; Python
worker teardown is excluded.

| Transport | Standalone comm. | Concurrent comm. | Comm. loss | Standalone GEMM | Concurrent GEMM | Compute loss | Concurrent wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Push: warp `put_signal_nbi` + `quiet` | 0.859 GB/s | 0.423 GB/s | 50.8% | 731 TFLOP/s | 327 TFLOP/s | 55.2% | 1.452 s |
| Pull: warp `get_nbi` + `quiet` | 0.357 GB/s | 0.267 GB/s | 25.2% | 834 TFLOP/s | 387 TFLOP/s | 53.6% | 2.302 s |

Dedicated communication concurrency overlaps in wall-clock time, but both
protocols consume enough GPU resources to cut GEMM throughput by more than half.
The `_nbi` path does not provide a production-relevant overlap win in this
organization.

## JAX interoperability

| Local endpoint | Remote symmetric endpoint | Operation | Copy required | Stream-correct result |
|---|---|---|---:|---|
| JAX source | NVSHMEM inbox | host put through non-owning `cuda.core.Buffer` | no | Works with DLPack handoff; final validation used host stream sync |
| CuTe symmetric source | NVSHMEM inbox | device put | no | Works in the standalone CuTe kernel |
| NVSHMEM source | ordinary JAX destination | get | no payload copy in principle | Unsupported: mutating an ordinary live JAX array is not a legal JAX output contract |
| NVSHMEM source | NVSHMEM destination | device get | no | Works in pull correctness kernels |
| NVSHMEM source | CuTe staging destination | device get | no | Symmetric CuTe destination works; a non-symmetric staging allocation was not accepted as a JAX-visible result |
| JAX writes symmetric view | NVSHMEM source | publish/pull | no | Unsupported: ordinary JAX arithmetic allocates a distinct output pointer |
| NVSHMEM inbox | JAX reads symmetric view | consume push | no | Pointer identity and values pass after host stream sync; XLA-stream custom call fails to compile |

NVSHMEM owns symmetric allocations. JAX may retain a DLPack view while the
NVSHMEM allocation and its owner remain alive. A JAX-owned local source can be
wrapped by `cuda.core.Buffer.from_handle` with the JAX array as owner. Raw JAX
objects are not accepted by NVSHMEM host RMA.

The attempted XLA-stream custom call used input/output aliasing for both the JAX
source and symmetric inbox view. CUTLASS JAX 4.4 did not preserve the pointer
alignment and scalar types required by NVSHMEM4Py's generated CuTe FFI
prototypes. Captured runtime tensors, typed DLPack arguments, byte views with an
in-kernel signal reinterpretation, and the low-level bindings all failed before
kernel launch. The prototype therefore has no steady-state host-free JAX launch
chain.

## Mosaic comparison

The closest source-push baseline is commit `c53bbcdfba`, variant
`copy_release_only`, with `semaphore_only` as its control-cost decomposition.
It uses 32,768 tokens per rank, hidden size 2560, top-k 4, EP=8, 288 entries per
destination, and 327,680-byte live entries. This is an all-to-all queue workload,
not the one-peer ring microbenchmark, so a raw latency ratio would be invalid.

Both historical variants failed on the current CoreWeave image and their pinned
JAX 0.10.1 environment. Mosaic lowering reports that `semaphore_wait` is
unimplemented for `LoweringSemantics.Lane` with Warpgroup user semantics. The
failed run is preserved in
[`artifacts/pallas-source-push-c53bbcdfba-cw-h100.jsonl`](artifacts/pallas-source-push-c53bbcdfba-cw-h100.jsonl).
No Mosaic bandwidth number or NVSHMEM speedup is reported.

## Conclusions

### NVSHMEM push

Warp `put_signal` is the best implemented transport. Put-with-signal also has
the simplest payload/readiness completion contract. It is not suitable for the
current persistent kernel because the JAX stream boundary fails and concurrent
GEMM throughput drops 55%.

### NVSHMEM pull

Pull is 2.8× slower than push at 6144 bytes and requires explicit local
completion. Four gets per quiet did not improve throughput. Destination-directed
scheduling does not compensate for the measured copy and completion costs.

### Direct peer stores

Warp-cooperative direct stores are 32× slower than warp `put_signal` at 6144
bytes. The implementation uses lane-striped 32-bit stores and one system fence
before signaling. Direct stores are not a competitive transport on this stack.

### Direct peer loads

Warp-cooperative peer loads are 3.4× slower than blocking warp get at 6144
bytes. They require explicit system ordering and `.cv` loads for correctness.
Direct loads are not a competitive replacement for NVSHMEM get.

### JAX interoperability

Allocation aliasing is zero-copy, but steady-state stream integration is not.
The working routes rely on explicit host-side CUDA stream synchronization. The
XLA custom-call route fails at the CUTLASS/NVSHMEM FFI type boundary.

### Persistent-kernel suitability

Return to Mosaic for production work. The NVSHMEM prototype fails two explicit
stop criteria: JAX integration still requires host synchronization, and
communication reduces concurrent GEMM throughput by more than half. Pull adds
completion overhead, direct peer access is slower, and the isolated dependency
stack conflicts with Marin's normal GPU/Torch environment.

## Reproduction

Install and inspect the isolated environment:

```bash
uv sync --package marin-levanter --extra nvshmem-transport
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.environment
```

Run the corrected eight-PE transport sweep:

```bash
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.benchmark_transport \
  --num-pes 8 --num-slots 8 --repetitions 3 \
  --payload-bytes 16 256 4096 6144 24576 98304 393216
```

Run all-to-all correctness:

```bash
NVTP_NUM_PES=8 NVTP_NUM_EPOCHS=100 NVTP_NUM_SLOTS=8 \
NVTP_PAYLOAD_BYTES=6144 NVTP_DIRECTION=push NVTP_PATTERN=all_to_all \
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.correctness_patterns
```

Run the start-gated overlap benchmark:

```bash
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.benchmark_overlap \
  --protocol push --operation put_signal_nbi_warp_quiet \
  --num-pes 8 --payload-bytes 6144 --num-epochs 100000
```

Detailed experiment commands, failures, and snapshot hashes are in the
[research logbook](../.agents/logbooks/cute-nvshmem-transport.md) and issue
[#7114](https://github.com/marin-community/marin/issues/7114).
