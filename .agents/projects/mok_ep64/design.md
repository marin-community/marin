# Mixture-of-Kittens EP64 on GB200 NVLink

Marin should run the upstream Mixture-of-Kittens kernel across all 64 GB200s in one NVL72 rack,
rather than using EP4 islands with data parallelism. This gives the production EP hero two routed
experts per GPU and lets us compare MoK directly with the existing EP64 fixed-all-to-all baseline.
Today the kernel itself supports EP64, but Marin's JAX adapter and workspace runtime are fixed to
four GPUs in one process. [Research](research.md) contains the source and experiment history.

## Challenges

The fused kernel dereferences one pointer per remote expert rank. XLA allocations are not mapped
across processes, and Marin's current runtime allocates all peer buffers with process-local
`cudaMalloc`. EP64 also magnifies failure semantics: one rank that returns early can strand 63
peers inside a device wait and later leave JAX collectives mismatched. Finally, factor 4 is only a
strict capacity bound for EP4; EP64 needs a newly explicit capacity contract.

## Costs / Risks

- The first transport uses PyTorch's private symmetric-memory API and a second NCCL process group.
  We pin its version and isolate it behind a workspace interface, but it remains a dependency risk.
- Runtime staging adds local copies. Cross-process XLA zero-copy is explicitly deferred.
- One process per GPU means 64 JAX processes. Compile and host-memory behavior must be measured,
  not inferred from the sealed four-GPU-process runs.
- Correct failure closure across 64 ranks adds a small full-expert status agreement on the healthy
  path and requires a rack negative gate.

## Design

Iris will continue to allocate 16 tasks with four GB200s each in one hard NVLink domain, but will
set `processes_per_task=4`. Its existing supervisor gives each child one device and a stable global
rank. JAX forms mesh `(replica_dcn=1, data=1, expert=64, model=1)`; Torch forms a separate 64-rank
NCCL process group with the same ordering and a separately discovered TCP endpoint.

A new symmetric workspace owner allocates one flat, identically sized byte arena per rank and
workspace slot through `torch.distributed._symmetric_memory`. The arena packs staged forward input,
combine output, backward input, routed input gradient, router weights, router-weight gradient,
generation/status cells, and debug counters at validated aligned offsets. The owner retains the
Torch tensor, rendezvous handle, and process group for the complete native-runtime lifetime. It
passes the local pointer plus all 64 remote aliases to the native initializer. A standalone probe
must pass before this owner is connected to MoK. If the backend cannot expose all 64 ranks, the
same interface will be implemented with native CUDA fabric VMM; the FFI and model contracts do not
change.

The EP64 native target is distinct from EP4. It selects the upstream BF16 EP64 instantiation,
uses a 64-rank pointer table, and has one local FFI handler per process. Host arrival masks and
process-local four-GPU rendezvous disappear. A deterministic operation stamp derived from runtime
epoch, static collective ID, ordinal, and phase guards the one workspace slot. Device readiness,
completion, and cancellation cells require exact stamp equality. Any mismatch cancels the
operation instead of accepting a later generation.

Initial forward and backward storage are both runtime-staged. JAX retains route `all_gather`,
schedule construction, shared-weight reductions, and a full-expert-axis `pmax` of the native
failure status. Shared gradient reductions execute only in the uniform success branch. Native
cancellation must first make all 64 FFI handlers return; the JAX agreement cannot recover a rank
still spinning inside native code.

The launcher is a dedicated EP64 MoK arm derived from the existing EP hero, not a reinterpretation
of the E8/EP4 weak-scale launcher. It uses E128, top-4, d6144/i3072, 48 layers, batch 1024, EP64,
DP1, two local experts, one slot, runtime staging, host-memory cap 176 GiB, and zero retry/failure
tolerance. Run identity names EP64, DP1, 64 JAX processes, symmetric staging, capacity policy, and
the workspace schema. Strict all-to-one dropless means factor 64. Smaller factors are allowed only
when tagged capacity-limited and audited for exact drops.

Teardown is collective: stop new calls, wait for native quiescence, synchronize the local device,
barrier the Torch group, close the native runtime, barrier again, then release symmetric-memory
owners and destroy the Torch group. Sticky/asynchronous CUDA faults remain process-terminal.

## Testing

Local tests cover arena layout, rank ordering, EP64 schedule routes through ranks 31/32/63,
factor-64 all-to-one capacity, limited-capacity drop accounting, dynamic audit counts, and abstract
EP64 lowering. Hardware gates are deliberately staged:

1. two-node EP8 JAX+Torch symmetric-memory read/write and repeated-generation probe;
2. one-rack EP64 probe, including ranks 0/31/32/63 and clean teardown;
3. small EP64 forward/VJP parity for balanced, zero-token, skewed, and all-to-one routes;
4. rank-0 and rank-63 failure injection at every forward/backward boundary;
5. one-layer production-shape compile/update and memory gate;
6. 25-step stability and a 100-step matched performance seal against fixed all-to-all.

No training arm is promoted unless all processes exit cleanly, losses are finite, drops match the
declared capacity policy, handler/staging counts are exact, and generation/reuse/protocol counters
are zero.

## Open Questions

- Does PyTorch 2.11 symmetric memory expose all 64 pointers in this container/rack, or must the
  implementation switch immediately to native fabric VMM?
- Is factor 64 affordable at the production shape, or should the promoted contract be a measured
  capacity percentile with explicit nonzero drops?
- What healthy-path overhead does full-expert failure agreement add at EP64, and can it be fused
  with an existing schedule collective after correctness is sealed?
