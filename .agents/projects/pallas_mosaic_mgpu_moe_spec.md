# Spec: Pallas Mosaic MGPU Fused MoE Dispatch + W13/SiLU for Single-Node EP

Issue: https://github.com/marin-community/marin/issues/6597
Research logbook: `.agents/logbooks/pallas-mosaic-mgpu-moe.md`

## Context

We want a DeepEP-free expert-parallel MoE path for Grug MoE on CoreWeave-style
8xH100 nodes and future NVL72 GB200 systems. DeepEP can sometimes reach useful
throughput, but it has repeatedly failed with internode recv-counter timeouts,
and its FFI boundary makes profiling and fusion hard. Ring and ragged all-to-all
are easier to reason about, but current profiles show MoE transport and
permutation/combine overhead remain large enough to justify a lower-level
single-node path.

The first target is **single host, 8xH100, EP=8, NVLink only**. Do not target
NIC, InfiniBand, NCCL EP, NVSHMEM, or DeepEP in this path. The medium-term
reason to build this is NVL72/GB200, but the first implementation should prove
the idea on our existing 8xH100 nodes.

Current in-repo baselines and integration points:

- `ring`: `lib/levanter/src/levanter/grug/_moe/ep_ring.py`
- `ragged_all_to_all`: `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`
- `padded_all_to_all`: `lib/levanter/src/levanter/grug/_moe/ep_padded_all_to_all.py`
- `sonic`: `lib/levanter/src/levanter/grug/_moe/sonic.py` for local expert
  kernel/layout ideas. SonicMoE does not implement expert-parallel transport,
  so it is a compute-kernel reference, not an EP communication reference.
- dispatch entrypoint: `lib/levanter/src/levanter/grug/grug_moe.py`
- default May D2560 config: 256 experts, top-k 4, hidden dim 2560,
  intermediate dim about hidden/2 rounded to 128, L26 for May full-shape tests.

Useful recent measurements:

- EP16/N2 DeepEP B64 `save_moe` May387: about 17.0 MFU, but DeepEP is fragile.
- EP8/B16 readable profiles May363/May378: MoE is about 37-40% of step time.
- Ring/ragged A2A comparisons showed `ragged_all_to_all` was slower than ring
  for the tested shape, so a new path must beat both transport overhead and
  profiling opacity.

## Objective

Build a Hopper-targeted Pallas Mosaic MGPU MoE backend whose product target is a
fused dispatch + `W_gate/W_up` + `SiLU(gate) * up` forward kernel for
single-node expert parallelism.

The product target is:

```text
route metadata
  -> pack/write routed token blocks directly into destination expert-major tiles
  -> consume those tiles with local expert W_gate/W_up WGMMA
  -> produce SiLU(gate) * up without materializing [tokens, 2 * d_ff]
```

The full MoE backend also needs `W_down` and combine/scatter to become useful in
training:

```text
fused dispatch + W13/SiLU output
  -> W_down
  -> combine/scatter returns token outputs
```

The training target also requires a credible backward path. The forward kernel
should be designed so a custom VJP can reuse the routing metadata and avoid
reintroducing the same bad collective/permutation path in reverse.

The implementation plan deliberately separates the first validation slice from
the product target. First prove that Mosaic remote refs can put routed tokens in
the exact expert-major layout the W13 kernel will consume; then fuse the
dispatch and W13/SiLU execution boundary. That validation slice is not the
destination.

## Non-goals

- No DeepEP, FFI transport, NCCL EP, NIC, RDMA, or NVSHMEM.
- No tensor parallelism. Always `model_axis=1`.
- No multi-host path in the first version.
- No FP8 in v0.
- No dropless overflow path in v0.
- No custom W_down/combine rewrite until fused dispatch + W13/SiLU is correct
  and measurable.
- No replacement for all MoE implementations. Add a new experimental backend,
  likely `mosaic_mgpu`, next to ring/ragged A2A.

## Shape Targets

### Correctness smoke targets

```text
EP=2, experts_per_rank=1, top_k=1
EP=8, experts_per_rank=1, top_k=1
EP=8, experts_per_rank=4, top_k=1
EP=8, experts_per_rank=4, top_k=4
small d_model/d_ff for CPU/reference parity
```

### First realistic target

```text
device: 8xH100 SXM, single host
EP: 8
global experts: 256
experts_per_rank: 32
top_k: 4
dtype: bf16 activations and weights
batch/sequence: match current single-node profile shape first, then B16/B64
d_model: 2560
d_ff: current Grug May intermediate dim
capacity: static padded capacity, explicit overflow policy outside kernel
```

Top-k 1 is allowed only for proving the primitive. Top-k 4 is the first
claim-relevant target because current Grug uses `num_experts_per_token=4`.

## Public API

Add a backend under the existing Grug MoE dispatch surface, not a separate model
variant. A plausible low-level API:

```python
def moe_mosaic_mgpu_forward(
    x_local: jax.Array,                 # [tokens_per_rank, hidden], bf16
    expert_ids: jax.Array,              # [tokens_per_rank, top_k], int32 global expert ids
    expert_weights: jax.Array,          # [tokens_per_rank, top_k], bf16/fp32 router weights
    w_gate_up_local: jax.Array,         # [experts_per_rank, hidden, 2 * d_ff], bf16
    w_down_local: jax.Array | None,     # [experts_per_rank, d_ff, hidden], bf16
    *,
    ep_axis_name: str,
    config: MosaicMgpuMoEConfig,
) -> tuple[jax.Array, MosaicMgpuMoEMetadata]:
    ...
```

For staged development, expose internal primitives, but keep the intended fast
path explicit:

```python
prepare_mosaic_mgpu_routing(...)
mosaic_mgpu_dispatch(...)
mosaic_mgpu_fused_dispatch_w13_silu(...)
local_or_mgpu_w2(...)
mosaic_mgpu_combine(...)
mosaic_mgpu_fused_dispatch_w13_silu_bwd(...)
mosaic_mgpu_combine_bwd(...)
```

The first two primitives may exist temporarily for testing. The backend should
not stop at `mosaic_mgpu_dispatch(...)`; its first performance-relevant forward
kernel is `mosaic_mgpu_fused_dispatch_w13_silu(...)`.

The public training backend should be selected as a new `MoeImplementation`
literal, for example `mosaic_mgpu`, and wired through
`lib/levanter/src/levanter/grug/grug_moe.py`.

## Data Layout Contract

The dispatch primitive must produce a destination-owned expert-major buffer:

```text
recv_x: [max_recv_tokens, hidden] bf16
rows_per_expert: [experts_per_rank] int32
expert_base: [experts_per_rank] int32
```

Rows are grouped by local expert. Within each expert, rows are grouped by source
rank and then by source-local position:

```text
row = expert_base[local_expert]
    + src_base[src_rank, local_expert]
    + local_pos_within_src_expert
```

For top-k 4, treat each `(token, choice)` assignment as a routed row. The row
metadata must preserve enough information to combine later:

```text
recv_src_rank: [max_recv_tokens] int32
recv_src_token_idx: [max_recv_tokens] int32
recv_topk_slot: [max_recv_tokens] int32
recv_router_weight: [max_recv_tokens] bf16/fp32
```

If metadata storage is expensive, v0 can return a compact packed metadata buffer
with the same row ordering. Do not rely on reconstructing source token identity
from sorting unless there is a checked reference proving it.

## Backward Contract

The implementation must plan for a custom VJP from the beginning. A forward-only
Mosaic kernel is useful for microbenchmarks but does not answer the training
question.

Forward values to save or recompute:

```text
expert_ids / top-k slots
router weights
recv row metadata
rows_per_expert
expert_base / src_base
optionally recv_x, gate, up, or h depending on remat policy
```

Backward for the fused W13/SiLU region receives:

```text
dh_local: [max_recv_tokens, d_ff]
recv_x or recomputed recv_x tiles
w_gate_up_local
metadata needed to return dx_local to source ranks
```

It must produce:

```text
dw_gate_up_local: [experts_per_rank, hidden, 2 * d_ff]
dx_local: [tokens_per_rank, hidden]
drouter_weight: [tokens_per_rank, top_k] if router weights require gradients
```

The W13/SiLU derivatives are:

```text
h = silu(gate) * up
dup = dh * silu(gate)
dgate = dh * up * d_silu(gate)
```

Then:

```text
dW_gate += X^T @ dgate
dW_up   += X^T @ dup
dX      += dgate @ W_gate^T + dup @ W_up^T
```

The backward transport should mirror the forward row metadata:

```text
destination-rank expert-major dX rows
  -> remote scatter/add back to source-rank token-gradient rows
```

For top-k 4, multiple expert choices for the same source token accumulate into
the same `dx_local[token]`. This accumulation must be deterministic enough for
bf16 training tolerances. If remote atomic add is unavailable or too slow, use a
two-stage design: remote-write per-assignment `dx` rows back to a source-owned
buffer, then locally reduce the top-k rows per token.

The custom VJP should expose remat choices explicitly:

```text
save_recv_x: bool
save_gate_up: bool
save_h: bool
```

Default for the first training profile should favor correctness and readable
profiles over memory minimality. Later variants can recompute W13 forward tiles
inside backward to reduce saved activations.

## Recommended Milestones

### Milestone 0: routing and layout reference

Implement a pure JAX reference that converts local tokens and top-k expert ids
into the exact destination expert-major layout above.

Required outputs:

```text
recv_x_ref
rows_per_expert_ref
expert_base_ref
recv_src_rank_ref
recv_src_token_idx_ref
recv_topk_slot_ref
recv_router_weight_ref
```

Use small CPU-compatible tests and a local 8-device fake mesh. This is the
oracle for every later kernel.

### Milestone 1: Mosaic remote dispatch validation slice

Implement a Pallas Mosaic MGPU primitive that moves **prepacked** source-rank
send buffers into remote destination `recv_x`.

Prepacked input:

```text
send_x_by_dst: [EP, max_send_rows_per_dst, hidden] bf16
send_meta_by_dst: [EP, max_send_rows_per_dst, meta_fields]
send_count_by_dst: [EP] int32
```

Output on each destination rank:

```text
recv_x
recv_meta
rows_per_expert
```

This first version may be non-overlapped:

```text
send all blocks
wait for all source ranks
return recv_x + metadata
```

Success criterion: exact layout parity with the JAX reference on EP=2 and EP=8,
including zero-token experts, skewed experts, and ragged tails.

This milestone exists to validate remote writes, semaphores, and row metadata.
It is not a performance endpoint and should not be presented as the planned MoE
backend.

### Milestone 2: fused dispatch + W13/SiLU

Build the first product-target kernel: remote dispatch directly feeds the
expert-local `W_gate/W_up` matmul and produces the activated `h` output.

Target fused region:

```text
routed source token blocks
  -> remote expert-major staging / tiles
  -> W_gate/W_up WGMMA
  -> SiLU(gate) * up
  -> h_local [max_recv_tokens, d_ff]
```

The first implementation may have an explicit `recv_x` staging buffer between
the remote-write phase and the W13 phase if Mosaic requires it. The optimization
target, however, is to eliminate any unnecessary global-memory materialization
and to make source/expert ranges consumable by W13 as soon as they are ready.

Activation fusion is part of this milestone. A temporary debug mode may
materialize `tmp_gate_up: [tokens, 2 * d_ff]`, but the milestone is not complete
until the main path avoids storing the full gate/up intermediate to global
memory.

Success criterion: W13/SiLU parity against the JAX reference for top-k 1 and
top-k 4, plus a phase table that separates:

```text
routing/prepack
remote dispatch
W13 WGMMA
SiLU epilogue
staging/materialization bytes, if any
```

### Milestone 3: existing W_down consumes fused W13/SiLU output

Feed `h_local`, `rows_per_expert`, and local `W_down` into the existing or
lightly adapted grouped expert matmul path.

This is allowed to be a separate matmul. Do not fuse W_down into the first
Mosaic kernel until dispatch + W13/SiLU is already correct and profiled.

Success criterion: expert-major down output parity against reference and phase
timings for:

```text
fused dispatch+W13/SiLU
W_down
```

### Milestone 4: Mosaic combine/scatter

Implement the reverse primitive:

```text
expert-major down output + recv_meta + router weights
  -> source-rank token output accumulation
```

For top-k 4, combine must add four weighted expert outputs back to the source
token. Use bf16 or fp32 accumulation according to current model behavior.

This is at least as important as dispatch. Ring’s backward/profile pain is
partly combine/scatter/psum shaped, so a dispatch-only primitive is not enough
for an end-to-end backend.

Success criterion: forward MoE output parity against the JAX reference and
current backend, including top-k 4.

### Milestone 5: overlap dispatch with expert compute

Add source/expert readiness:

```text
ready[src_rank, local_expert]
```

The matmul scheduler may consume source/expert row ranges as soon as the
corresponding remote writes are visible. This is the first version that tests
the ring-like "send phase while computing ready phase" hypothesis.

This is the final form of the fused dispatch + W13/SiLU target: incoming
source/expert ranges should be scheduled into W13 as they become ready, rather
than waiting for the whole `recv_x` buffer to fill.

### Milestone 6: reference custom VJP

Before writing backward Mosaic kernels, implement a custom VJP wrapper whose
forward may call the Mosaic forward path but whose backward is pure JAX/reference
math using the saved metadata.

This milestone should answer:

```text
is the metadata sufficient?
are top-k 4 source-token accumulations well-defined?
which tensors must be saved versus recomputed?
does the API fit JAX autodiff and Grug training?
```

Success criterion: gradient parity against the existing backend on small shapes
for `x`, `w_gate_up`, `w_down`, and router weights if applicable.

### Milestone 7: Mosaic backward for W13/SiLU

Implement the backward of the fused W13/SiLU region:

```text
dh_local
  -> dgate / dup epilogue
  -> dW_gate / dW_up grouped expert GEMMs
  -> expert-major dX rows
```

The first implementation may materialize `dgate` and `dup` for clarity. The
optimized target should fuse derivative computation into the backward GEMM
schedule where practical.

Success criterion: value/gradient parity and a phase table:

```text
dgate/dup epilogue
dW WGMMA
dX WGMMA
temporary bytes
```

### Milestone 8: Mosaic backward transport to source tokens

Use the forward metadata to return expert-major `dX` rows to source-rank
`dx_local` rows.

Start with a conservative two-stage design if needed:

```text
remote write per-assignment dX rows to source-owned [tokens, top_k, hidden]
local reduce over top_k into [tokens, hidden]
```

Then optimize toward direct remote scatter/add if Pallas Mosaic supports it
efficiently and deterministically.

Success criterion: `dx_local` parity for top-k 4 and measured transport time.

### Milestone 9: end-to-end training integration

Integrate forward and backward into the experimental Grug backend and run
fwd+bwd+SGD profiles. Do not use Muon in the first integrated benchmark; SGD
keeps optimizer overhead out of the MoE backend decision.

Success criterion: one-node EP8 fwd+bwd+SGD profile with named scopes for:

```text
mosaic_mgpu/prepack
mosaic_mgpu/fused_dispatch_w13_silu
mosaic_mgpu/w2
mosaic_mgpu/combine
mosaic_mgpu/bwd_w2
mosaic_mgpu/bwd_w13_silu
mosaic_mgpu/bwd_transport
```

## Pallas/Mosaic Implementation Notes

Use the JAX examples as references:

```text
jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py
jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py
jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py
lib/levanter/src/levanter/grug/_moe/sonic.py
```

Follow the communication pattern in `collective_matmul_mgpu.py`:

```text
remote_ref
semaphore_signal
semaphore_wait
copy_smem_to_gmem / wait_smem_to_gmem as needed
```

Follow the ragged grouped matmul structure in `ragged_dot_mgpu.py` only after
the transport layout is proven.

Use SonicMoE as an in-repo reference for local expert compute decomposition,
activation placement, and layout choices. Do not copy its assumptions about
communication: SonicMoE is not an EP backend and does not solve cross-rank
dispatch/combine.

Start with prepacked sends. Do not start with direct irregular gathers from
`x_local[token_idx]`; that couples histogramming, routing, transport, and memory
coalescing too early. Once the primitive works, profile whether prepack is a
real cost.

## Config Dataclass

```python
@dataclass(frozen=True, slots=True)
class MosaicMgpuMoEConfig:
    ep_size: int = 8
    top_k: int = 4
    block_m: int = 64
    block_n: int = 128
    block_k: int = 64
    num_stages: int = 2
    prepacked_send: bool = True
    overlap_dispatch_compute: bool = False
    fuse_w13_silu: bool = False
    backward_impl: str = "reference"
    backward_dx_transport: str = "two_stage"
    save_recv_x: bool = True
    save_gate_up: bool = False
    save_h: bool = True
    static_capacity_factor: float = 1.25
    fail_on_overflow: bool = True
```

Tune only after correctness. Initial tuning candidates:

```text
block_m: 64, 128
block_n: 64, 128, 256
block_k: 64
num_stages: 2, 3
```

Record tile choices, device type, CUDA/JAX versions, and exact shape in every
benchmark result.

## Correctness Requirements

Compare against a simple JAX reference and at least one existing backend. Tests:

```text
EP=2, experts_per_rank=1, top_k=1
EP=8, experts_per_rank=1, top_k=1
EP=8, experts_per_rank=4, top_k=1
EP=8, experts_per_rank=4, top_k=4
zero tokens for some experts
zero tokens from some source ranks
one hot/skewed routing
ragged expert sizes not divisible by block_m
capacity overflow fails outside kernel
```

Value tolerance for bf16:

```text
rtol=2e-2
atol=2e-2
```

Gradient parity is required before training integration. It can start with
small shapes and a custom VJP if the forward primitive is not naturally
differentiable through remote refs. Do not rely on end-to-end training as the
first gradient test.

Backward tests must include:

```text
top_k=1 and top_k=4
multiple choices from one token landing on different destination ranks
multiple choices accumulating back into the same source token
zero-token experts
skewed routing
capacity tails not divisible by block_m
```

## Performance Harness

Create a dedicated harness before integrating into full training. It should run
on one 8xH100 node and report:

```text
prepack time
dispatch time
w13 time
activation time
w2 time
combine time
bwd_w2 time
bwd_w13_silu time
bwd_transport time
end-to-end MoE forward time
end-to-end MoE forward+backward time
NVLink effective bandwidth for dispatch/combine
achieved W13/W2 TFLOP/s
scratch bytes
compile time
steady-state median/p50/p90
```

Baselines:

```text
A: current ring backend, same shape
B: current ragged_all_to_all backend, same shape
C: current local grouped matmul on already expert-major input
D: Mosaic dispatch validation slice plus existing grouped matmul
E: fused Mosaic dispatch + W13/SiLU plus existing W_down/combine
F: fused Mosaic dispatch + W13/SiLU + Mosaic combine
G: full custom VJP with reference backward
H: full Mosaic forward+backward
```

Use W&B for scalar series when running on CoreWeave. Use a dedicated project or
at least a stable run prefix, for example `MOSAIC-MGPU-MOE-*`.

## Integration Plan

1. Add kernel code under a Pallas/Mosaic-specific module, not inside DeepEP:

```text
lib/levanter/src/levanter/kernels/pallas/mosaic_moe/
```

2. Add a Grug MoE backend wrapper:

```text
lib/levanter/src/levanter/grug/_moe/ep_mosaic_mgpu.py
```

3. Register the backend in:

```text
lib/levanter/src/levanter/grug/_moe/common.py
lib/levanter/src/levanter/grug/grug_moe.py
```

4. Add tests under:

```text
lib/levanter/tests/kernels/
lib/levanter/tests/grug/
```

5. Add a standalone benchmark script before full training:

```text
lib/levanter/scripts/bench/bench_mosaic_mgpu_moe.py
```

6. Only then add a launch script for the May D2560 profile shape.

7. Add custom-VJP and backward tests before any fwd+bwd training claim.

## Risks and Open Questions

- Pallas MGPU remote refs may have constraints around process topology. Verify
  whether we need one process controlling 8 local devices or one process per GPU.
  This is a milestone-zero runtime check.
- If remote refs are not compatible with the current Fray/JAX process topology,
  the first deliverable should be a minimal repro and a required topology note.
- Top-k 4 metadata volume may make combine the bigger problem than dispatch.
  Measure it explicitly.
- Backward may become the real bottleneck even if forward is good. Keep forward
  and backward phase tables separate.
- Direct remote scatter/add for `dx_local` may not be available or deterministic
  enough. The fallback is per-assignment remote write plus local top-k reduce.
- Prepacking may become the bottleneck. Do not optimize it until the transport
  primitive works, but keep it separately timed.
- Existing grouped matmul may not accept the exact expert-major layout without a
  copy. That copy must be measured in the validation slice; it should not exist
  in the fused W13/SiLU product path.
- If the first end-to-end forward is slower, it can still be a success if the
  phase table shows remote dispatch + W13/SiLU is controllable and the remaining
  time is in W_down or combine.

## Definition of Done for the First Useful Prototype

- EP8 single-node 8xH100 fused dispatch + W13/SiLU works for top-k 4.
- A custom VJP exists and passes small-shape gradient parity, even if the first
  backward implementation is reference/JAX rather than Mosaic.
- Values match the JAX reference and one existing backend on small and realistic
  shapes.
- Phase table reports prepack, remote dispatch, W13 WGMMA, SiLU epilogue,
  staging bytes, W2, and combine separately.
- The path uses no DeepEP, no NCCL EP, and no opaque FFI transport.
- The code can be enabled as an experimental Grug MoE backend without changing
  model code outside backend selection.
- A short report compares it to ring and ragged A2A on the same shape.

## Definition of Done for Training Integration

- EP8 single-node 8xH100 fwd+bwd+SGD profile runs with `model_axis=1`.
- Forward uses fused Mosaic dispatch + W13/SiLU.
- Backward uses either reference custom VJP or Mosaic kernels, with the
  implementation clearly labeled in metrics.
- Gradients match the reference backend on small test shapes.
- Profile separates forward transport, forward W13/SiLU, W2, combine,
  backward W13/SiLU, backward transport, and W2 gradients.
- Throughput is compared against ring and ragged A2A on the same shape.

## Suggested Experiment Issue Framing

Title:

```text
Experiment: Pallas Mosaic MGPU fused MoE dispatch + W13/SiLU for single-node EP8
```

Goal:

```text
Build a DeepEP-free single-node EP8 fused MoE forward path using Pallas Mosaic
remote refs: routed token dispatch should feed W_gate/W_up directly and produce
SiLU(gate) * up without materializing the full gate/up intermediate.
```

Primary metrics:

```text
MoE forward time
fused dispatch + W13/SiLU time
dispatch/combine time
MoE backward time
backward transport time
end-to-end fwd+bwd+SGD MFU after integration
```

Stop criteria:

```text
Stop or redesign if remote refs cannot express EP8 dispatch/combine under our
JAX process topology, or if dispatch/combine remain slower than ring/ragged A2A
after removing avoidable copies.
```
