# Natural routed sparse attention: matched H100 checkpoint

Date: 2026-08-07

This checkpoint measures Shuttle's generated query-major sparse-attention path
against pinned MIT Block-Sparse-Attention at the same natural program boundary.
It supersedes the unmatched 2.388 ms Seer denominator for this workload.

Both timed paths include:

1. the FP32 metadata contraction used by the router;
2. the causal block-domain restriction;
3. sorted top-k selection and index-plane construction; and
4. BF16 causal exact attention over the selected blocks, with native GQA and a
   BF16 output.

QKV and output projections are excluded from both paths. The generated path
uses compact `RelationPlan` block lists; the oracle constructs the equivalent
dense block mask required by its public interface. This is a physical
representation choice, not a semantic-boundary difference. See
`raw/boundary-manifest.json`.

## Configuration

- GPU: one NVIDIA H100 80GB HBM3.
- Driver: 595.71.05.
- Power limit: 700 W.
- Observed active clocks: 1830 MHz SM and 2619 MHz memory.
- Sequence: 16,384.
- Query/KV block: 128/128.
- Selected blocks: at most eight per query block.
- Heads: 32 query, eight KV.
- Head dimension: 128.
- Router dimension: 64.
- Dtype: BF16 payload, FP32 routing and online state.
- Warmups/repeats: 10/30 per implementation.

## Matched results

| Implementation | Median | Range | Mean |
| --- | ---: | ---: | ---: |
| Shuttle generated query-major | 0.580496 ms | 0.577472–0.599584 ms | 0.582261 ms |
| Block-Sparse-Attention oracle | 1.426256 ms | 1.400896–1.463680 ms | 1.428406 ms |

The generated/oracle ratio is **0.407007x**. This clears the 1.20x completion
target on the matched boundary. A prior independent 30-sample distribution is
also retained in `raw/s16384-b128-k8-matched-pre-oracle-hash.json`; it measured
0.586288 ms generated and 1.422160 ms oracle.

Those first captures always launched generated before oracle, so they do not
satisfy the subsequently frozen counterbalancing rule. Two new independent
captures alternate every pair between generated→oracle and oracle→generated.
Their pooled results are:

| Implementation | Pooled median | Range | Mean |
| --- | ---: | ---: | ---: |
| Shuttle generated query-major | 0.617584 ms | 0.585536–0.645536 ms | 0.610611 ms |
| Block-Sparse-Attention oracle | 1.423632 ms | 1.396576–1.478144 ms | 1.428772 ms |

The counterbalanced ratio is **0.433809x**. The frozen oracle target remains
1.424720 ms; this confirmation does not move it. Every warmup and sample launch
order is recorded in `raw/counterbalanced/`.

The result should not be interpreted as a universal sparse-attention algorithm
comparison. The pinned oracle is an SM80-style implementation compiled for
SM90, whereas Shuttle's physical skeleton uses Hopper TMA/WGMMA scheduling.
It is the strongest matching, buildable oracle currently wired to this natural
boundary.

## Correctness and determinism

- Generated versus sampled independent semantic reference:
  - maximum absolute error: 0.00790286;
  - mean absolute error: 0.000179032.
- Generated versus oracle:
  - maximum absolute difference: 0.00390625;
  - mean absolute difference: 0.0000651722.
- Generated repeated-output SHA-256:
  `e9399766068941b3b60329760c04b576c79cec38fd036c8cbdfe43cdf8da3a83`.
- Oracle repeated-output SHA-256:
  `86d62d4d69acf008eb073c029da0810c524d4e6ce1abdfc5d5735d326e70c1b3`.
- Relation SHA-256:
  `0a2a06781755f5f577237a2e48c810cb160fd88db3295835458350f47ad61cbb`.

Both implementations reproduced their output hashes bit-for-bit.

## Synthesis and lineage

The accepted generated path begins with ordinary JAX serialized as StableHLO.
Recovery validates name erasure into `Contract`, `Map`, `Fold`, `Relation`,
`RelationPlan`, and `DomainRestriction`; the recorded validation error list is
empty. Physical execution instantiates Shuttle's generated SM90
QK/online-Fold/PV skeleton. It does not call Block-Sparse-Attention, Seer, FSA,
FlashMoBA, or an official FlashAttention forward function.

The skeleton retains generic CuTe/FlashAttention helper machinery for tensor
layouts, online-softmax operations, and block-list traversal. Exact helper
hashes are in `raw/external-source.sha256`; Shuttle-owned generator and skeleton
hashes are in `raw/generated-source.sha256`. The oracle is used only for the
comparison path.

The compiler now executes both orientations on H100. The bounded KV-major path
serializes selected-slot waves to preserve source order. Within each wave it
groups relation edges by right-side KV block, splits large incident groups into
fixed-capacity tasks, stages one KV-head block into dynamic shared memory, and
reuses it for the task's query consumers. Each query has one writer per wave,
so the inverse route writes directly into the global online state without
atomics or an edge-partial buffer.

At the primary shape, capacity two produces 671 tasks for 996 relation edges,
stages 65,536 bytes of K/V per CTA, materializes 272,629,760 bytes of global
online state, and materializes zero per-edge partial state. It measures
107.879105 ms versus 0.574656 ms for query-major in the same process. Maximum
and mean output differences are 0.015625 and 0.00006397, and repeated output is
bitwise identical.

This is a structural proof, not a competitive KV-major kernel. The emitted
body uses CUDA-core QK/PV work, one global state pass per slot wave, and no TMA,
WGMMA, cluster multicast, or cross-head K/V sharing. A capacity-one mutation
creates 996 tasks and measures 103.355042 ms while producing bitwise-identical
output through the same generated source. The result says that reuse is legal
and physically real, but the current task grouping loses more parallelism than
it saves in K/V traffic. Query-major remains the selected schedule.

The KV-major implementation is generated from the same `RelationPlan`, score
Map, `DomainRestriction`, and normalized-exponential Fold used by query-major.
It does not copy or call an oracle sparse-attention body. Detailed lineage and
known physical gaps are in `raw/kv-major/lineage.md`.

## Pins and build notes

- Shuttle checkout: `4fba36752bdbfd28ad9a0ea8dee121bb382b21c9` plus the source hashes recorded in
  this artifact.
- Block-Sparse-Attention: `49d6c39e4dc0303442cda3bb758b3925d4399c49`.
- Oracle CUTLASS submodule: `a75b4ac483166189a45290783cb0a18af5ff0ea5`.
- Torch: 2.11.0+cu130.
- CUDA compiler: 13.0.88.
- CUTLASS DSL: 4.5.2.
- FlashAttention CuTe helpers: `flash-attn-4==4.0.0b16`.
- QuACK helpers: 0.5.0.

The holder image initially lacked `nvcc`. A minimal CUDA compiler was installed
ephemerally. CUDA 12.8 then failed against Debian 13's glibc math declarations;
CUDA 13.0 with GCC 13 built the unmodified pinned oracle successfully. The
successful build log and complete environment are preserved under `raw/`.
Repository linting removed trailing spaces from `raw/nvidia-smi-q.txt` after
capture; the telemetry fields and line order are unchanged, and `RAW.sha256`
records the normalized bytes.
The source build used:

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
CC=gcc-13 CXX=g++-13 \
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE \
BLOCK_SPARSE_ATTN_CUDA_ARCHS=90 \
MAX_JOBS=8 NVCC_THREADS=2 \
uv pip install --python /tmp/shuttle-sparse-env/bin/python \
  --no-build-isolation /tmp/block-sparse-attention
```

The counterbalanced confirmation and KV-major structural run used the same
driver, GPU, Torch, CuTe, QuACK, and FlashAttention helper revisions with CUDA
compiler 13.1.115. The unchanged oracle built successfully from the same pinned
source and CUTLASS revisions; its build log is
`raw/counterbalanced/block-sparse-build-cu131.log`.

## Reproduction

After installing the pinned dependencies and oracle extension:

```bash
PYTHONPATH=lib/tile_lifetime:lib/tile_lifetime/src \
python lib/tile_lifetime/benchmarks/h100_natural_routed_streaming_attention.py \
  --sequence 16384 \
  --block 128 \
  --slots 8 \
  --router-dimension 64 \
  --warmups 10 \
  --repeats 30 \
  --include-block-sparse-oracle \
  --json-output result.json
```

Add the following flags to reproduce the bounded KV-major structural run and
its capacity mutation:

```bash
  --include-kv-major \
  --kv-query-capacity 2 \
  --include-kv-capacity-mutation
```

`RAW.sha256` covers every raw evidence file with portable relative paths.
