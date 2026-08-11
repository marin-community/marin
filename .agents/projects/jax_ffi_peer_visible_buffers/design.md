# Peer-visible JAX FFI buffers without memory-space copies

`mok_like` now removes global GPU barriers, but each training step still moves
about 108.09 GiB per GPU between XLA arrays and runtime-owned peer-visible
staging buffers. We want XLA to own those buffers while allowing a typed GPU
FFI to read and write them from peer GPUs for the duration of the call. The
barrier-free staged backend remains supported until an explicit JAX/OpenXLA
contract removes the copies in optimized HLO and a production profile.

## Challenges

Remote GPU access is invisible to XLA's ordinary liveness analysis. A local
custom call may have returned on the host while another GPU still reads its
operand, or XLA may schedule a consumer while another GPU still writes its
result. The FFI must close those accesses through work ordered on every local
XLA stream.

OpenXLA already colors custom-call buffers as collective memory, and the
four-GB200 probe proves those allocations are peer-visible. Coloring is not
enough for this use case: optimized HLO copies a default-space input into color
1 and copies both results back. Applying that mechanism directly to the full
MoK ABI would relocate the staging traffic instead of removing it. See the
[research record](research.md) for the exact HLO and hardware results.

## Costs / Risks

- This crosses JAX lowering, OpenXLA buffer assignment, and PJRT GPU allocation.
  Marin cannot make the lifetime and allocation guarantee in Levanter alone.
- A peer-visible default allocation may reserve VMM mappings on every local GPU
  and reduce allocator flexibility. Unsupported topologies must fail rather
  than silently stage.
- Incorrect stream completion is data corruption, not merely a performance
  bug. Concurrency and buffer-reuse gates are part of the API contract.
- The change has no immediate benefit on hosts without a full local P2P clique;
  the staged backend remains necessary.

## Design

Add `peer_visible` as a PJRT GPU memory kind and as a semantic capability on
typed GPU FFI operands/results. The capability is orthogonal to HLO
memory-space color: an array remains in ordinary device memory, but its
allocation is read/write mapped on every device in an explicit same-process
participant group. This avoids introducing a color boundary around ordinary
JAX producers and consumers.

External parameters, donated arrays, and pre-existing PJRT buffers must already
use the `peer_visible` memory kind; a custom call cannot retrofit them. JAX
users select that kind through the existing sharding memory-kind surface when
placing inputs or constraining producer outputs. At executable load or
execution, PJRT validates each external buffer against the participant group.
Failure is explicit and is never repaired with a copy.

JAX exposes one peer-access descriptor on `jax.ffi.ffi_call` and
`jax.ffi.ffi_lowering`. It names flattened operand/result indices, a static
instance ID, and disjoint participant groups in PJRT client device-ID space.
The lowering emits checked custom-call frontend attributes. OpenXLA propagates
the capability through complete alias sets: if any value sharing a
`BufferAllocation` requires peer visibility, the allocation requires it for its
whole lifetime. Alias conflicts are unioned only when every external buffer is
already capable; otherwise they are rejected. Donation never upgrades an
ordinary allocation.

The typed FFI lifetime remains stream-scoped. “Call active” means the interval
between producer readiness and the closing dependency on the owning local XLA
stream, not the host handler's stack lifetime. Each remotely read operand has
this dependency graph:

```text
owner producer -> owner ready -> accessor wait -> remote read
              -> accessor done -> owner wait -> owner reuse
```

For each remotely written result, pointer publication precedes the remote write
and the owner stream waits for every writer before local consumption. The
handler may return after all closing waits are enqueued; it need not block until
they execute. If a participant fails after remote work begins, no rank may
return normally until the work is quiescent. If quiescence cannot be proven,
PJRT poisons the executable or client so XLA cannot reuse affected buffers.

`ffi::RunId` identifies concurrent executions; the static instance ID separates
multiple collectives within one executable. Every device belongs to exactly one
participant group and every group rank invokes the site once per dynamic
instance. Coordinators key state by client epoch, RunId, instance ID, and group.
A pointer published for one owner allocation must be dereferenceable unchanged
from every group device; separate rank-owned allocations need not use equal
addresses. No remote access may occur after the closing stream dependency.

Keep the existing collective-memory color API as a separate public option for
callers that need a distinct collective or symmetric allocator. JAX should
surface both concepts rather than treating them as synonyms:

- `memory_space="collective"` selects color 1 and may introduce explicit
  boundary copies;
- `peer_access=...` retains ordinary device memory and guarantees P2P access
  without boundary copies.

After the upstream contract lands, Marin adds an experimental `mok_like`
storage backend. It passes only remote-accessed activation/router operands and
results as peer-visible, reuses the proven readiness/completion protocol, and
deletes the corresponding native staging copies. We do not change kernel math,
schedule construction, routing, parameter leaves, or numerical tolerances.
The runtime-owned staged backend remains selectable until the zero-copy path
passes the full correctness, concurrency, training, and profile gates.

## Testing

JAX/OpenXLA needs an end-to-end four-GPU typed-FFI test. One call remotely reads
the next rank's operand and writes the next rank's result, then validates exact
vectors. Optimized HLO and buffer assignment must show peer-visible default
allocations and zero copies around the call. An invalid capability/topology
must fail at compilation or client initialization.

The same compiled executable then runs twice concurrently with distinct data;
both results must remain exact. Delayed owner readiness, a delayed remote
kernel, and a rapid buffer-reuse loop catch missing edges. Tests cover external
parameters, donation/alias sets, multiple participant groups, missing ranks,
and partial-failure poisoning. Unrelated values may be physically P2P-mapped by
an allocator, but must not acquire the semantic capability or constrain buffer
assignment.

Marin adopts the API only after the isolated gate passes. The production gate
reuses balanced, zero-token, skewed, all-to-one, one/two/eight-macrobuffer,
saved-context, corrupt-context, concurrent-VJP, 25-step, and 100-step tests.
Nsight must show none of the five identified staging copies, and steady
drop-adjusted throughput must not regress against the barrier-free staged
backend.

## Open Questions

- Does OpenXLA need a distinct buffer-allocation color that is layout-compatible
  with default device memory, or can buffer assignment carry an orthogonal
  allocator capability without perturbing copy insertion?
- Should a peer-visible alias mismatch always reject, or may XLA union the alias
  set when every external member is already backed by the PJRT memory kind?
- Can the first version standardize cross-device event waits, or should the FFI
  contract expose only generation buffers and local-stream wait kernels?
