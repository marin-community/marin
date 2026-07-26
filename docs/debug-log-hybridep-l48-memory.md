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

Determine whether XLA's allocator reservation can leave enough headroom for
DeepEP without increasing actual live memory. If not, move the metadata into
caller-owned buffers so XLA accounts for its liveness, or remove metadata that
the fused permute path does not consume.

## Results

- The bridge configures one NVLink domain with 64 ranks and one node.
- DeepEP allocates about 576 MiB of metadata per dispatch handle.
- The Iris task-exec read-only probe failed because the Kubernetes backend
  targeted a nonexistent container named `task`; allocator environment and live
  process memory still need confirmation through another read-only path.

## Future work

- [ ] Confirm the JAX allocator mode in the live task.
- [ ] Identify metadata unused by fused permute-dispatch/unpermute-combine.
- [ ] Validate the smallest memory fix on L48, then measure MFU and overflow.
