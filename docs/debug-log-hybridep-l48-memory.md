# Debugging log for HybridEP L48 memory

Run the locked L48/d5120/E256/top8/EP64 configuration with receiver-pooled
token dropping at no more than 3% physical overflow.

## Initial status

The same HybridEP path completed a two-step L4 smoke on 64 GB200 GPUs with
0.11% mean physical overflow. The first L48 attempt that completed compilation
failed in the first metadata preprocessing call:

- 184.30 GiB total HBM, 48.25 MiB free
- 6.98 GiB allocated by PyTorch
- a 256 MiB `torch::empty` request failed

## Hypothesis 1

DeepEP allocates fixed-shape metadata after XLA has planned and allocated the
training executable. At EP64 with 65,536 local tokens, the first
`sparse_to_dense_map` is `[65536, 64]` int32, exactly 256 MiB. The same call
also allocates a 64 MiB routing map and a 256 MiB dense-to-expert map.

## Changes to make

Reserve three sets of caching-allocator blocks matching DeepEP's three large metadata tensors
while the HybridEP runtime initializes, before JAX initializes its allocator.
The tensors are released immediately, leaving 1,728 MiB of PyTorch allocator
headroom that DeepEP can reuse at dispatch.

## Results

- The bridge configures one NVLink domain with 64 ranks and one node.
- DeepEP allocates about 576 MiB of metadata per dispatch handle.
- The Iris task-exec read-only probe failed because the Kubernetes backend
  targeted a nonexistent container named `task`; allocator environment and live
  process memory still need confirmation through another read-only path.
- Commit `2eac7cc45660ae5608814231444cc5a0e8958e20` preserves the state before
  allocator headroom was reserved.
- A 4-GPU gradient smoke showed that XLA may start a backward dispatch while one
  forward handle is still live. The first one-slot build rejected this valid
  schedule before the artifact upload. The complete custom-VJP smoke peaked at
  three live handles, so the reservation now has three slots.
- The three-slot artifact passed the 4-GPU round-trip, repeat, gradient, and
  separated custom-VJP smoke. It reserved 64 MiB per rank for the smoke geometry
  and was uploaded with SHA-256
  `6c995c2d46dd15e89fb3b1f810989df3fbbe53c1d2b10f8f2aa275e4dbeba8cd`.
- The locked L48 EP64 v8 run initialized HybridEP before XLA with
  1,059,061,760 bytes in the PyTorch cache, compiled, and entered the first
  dispatch on all ranks without the previous 256 MiB allocation failure. It did
  not complete step 0: after five gang attempts, the rank-0 JAX coordinator
  disappeared and Iris terminated the other 15 tasks as coscheduled failures.
  This is tracked as issue #7650.
- The fresh locked v9 retry reproduced issue #7650 across five gang attempts.
  Its longest-lived attempt reached the HybridEP dispatch and reported five
  simultaneous dispatch handles on multiple ranks. No PyTorch allocation
  failure was reported. About one minute later, JAX aborted the live gang when
  a replacement process 9 connected to the coordinator with a different
  incarnation. The run therefore establishes that the reserved allocator
  headroom clears the earlier first-dispatch OOM through a five-handle schedule,
  but it still has no completed training step or MFU measurement.

## Hypothesis 2

The L48 allocation failure after 21 dispatches is a custom-VJP ownership leak,
not ordinary activation liveness. Under `jax.checkpoint`, the reverse scan
replays the forward dispatch to reconstruct the expert input needed by the
expert-MLP gradient. It does not replay the corresponding forward combine.
DeepEP therefore inserts the replay dispatch's roughly 576 MiB handle into its
process-global map, but no FFI call consumes it.

## Changes to make

Retain the replay dispatch handle as a custom-VJP residual. Pass it alongside
the real backward handle to combine-with-probabilities, which consumes the
backward handle and releases the replay handle in one stream synchronization.

## Results

- A reduced JAXPR reproduces the reverse-scan sequence exactly: one replay
  dispatch, one backward dispatch, and one combine consuming only the backward
  handle.
- The existing L48 logs show the corresponding runtime behavior: active
  handles rise monotonically to 21 with no release before the next 256 MiB
  metadata allocation fails.
- A 48-layer, 65,536-token, d5120 residual-scan gradient smoke passed on four
  GB200 GPUs. All four ranks repeated the bounded live-handle sequence
  `1 → 2 → 3 → 2 → 1` through the full reverse scan and passed the hidden and
  routing-probability gradient checks.
- The validated bridge bundle was uploaded with SHA-256
  `b8b7da360f29866b697905efde827be3b44046feab766d45556bb7cd8c39a3c5`.

## Future work

- [ ] Confirm the PyTorch reservation and JAX allocator split on GB200.
- [ ] Identify metadata unused by fused permute-dispatch/unpermute-combine.
- [ ] Validate the fix in the locked EP64 training executable, then measure MFU
      and overflow.
