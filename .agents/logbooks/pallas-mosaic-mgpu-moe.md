# Pallas Mosaic MGPU MoE: Research Logbook

## Scope

- Goal: Build a DeepEP-free single-node EP8 Grug MoE backend using Pallas
  Mosaic MGPU remote refs, with fused dispatch + W13/SiLU as the first product
  target and a custom-VJP/backward path before training integration.
- Primary metrics: fused dispatch + W13/SiLU time, MoE forward time,
  MoE forward+backward time, end-to-end fwd+bwd+SGD MFU.
- Constraints: single-node 8xH100 first, `model_axis=1`, no DeepEP/NCCL
  EP/FFI transport/NIC/RDMA/NVSHMEM in this path.
- Experiment issue: https://github.com/marin-community/marin/issues/6597
- Spec: `.agents/projects/pallas_mosaic_mgpu_moe_spec.md`

## Baseline

- Date: 2026-06-23
- Code refs:
  - `lib/levanter/src/levanter/grug/_moe/ep_ring.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_padded_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/sonic.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
- Baseline numbers:
  - May387 DeepEP EP16/N2 B64 `save_moe`: about 17 MFU, but DeepEP remains
    fragile.
  - May363/May378 EP8/B16 readable profiles: MoE around 37-40% of step time.
  - Ring/ragged A2A comparison: `ragged_all_to_all` was slower than ring in the
    tested shape.

## Experiment Log

### 2026-06-23 - Kickoff spec handoff

- Hypothesis: A Pallas Mosaic MGPU backend can route token blocks directly into
  destination expert-major tiles and consume them with W13/SiLU, avoiding
  DeepEP's opaque FFI/runtime fragility and avoiding high-level collective
  permutation overhead.
- Command: No benchmark command yet; this entry creates the research branch,
  spec, and issue.
- Config: Target is single-node EP8 8xH100, top-k 4, bf16 activations/weights,
  May D2560 Grug MoE shape.
- Result: Created issue #6597 and checked in the initial implementation spec.
- Interpretation: The spec deliberately distinguishes a remote-dispatch
  validation slice from the product target. The first product target is fused
  dispatch + W13/SiLU; training integration also requires backward/custom VJP.
- Next action: Assign an implementation agent to start with the routing/layout
  JAX reference and Mosaic remote-dispatch validation slice.
