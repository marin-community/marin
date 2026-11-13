# Hero EP MuonH CPU oracle

`golden.npz` freezes three updates from Marin revision `4af966070e5d0252e87d404196400c35bd04883b`, using `experiments/grug/moe_hero_ep/{optimizer,grugmuon_hero,adamh}.py` and Levanter's Newton–Schulz coefficient table. SHA-256: `bd5ec624264e5ff28e97e83ff526fb34ec5f647d1de3088c90f12acb0da3b5e9`.

The fixture uses two layers and three routed experts with unequal norms. It covers Hero EP's default whole-bank HyperBall sphere, separate shared-expert spheres, fused Q/K/V and gate/up matrices, GatedNorm, embeddings, routers, norm gains, ShortConv and AdamH readout. Final matrix axes are transposed into Megatron layout. Router bias is included as a numerical Adam leaf; the runtime's frozen router-bias buffer is not an optimizer parameter.

Parameters, momentum and Adam moments are FP32. Newton–Schulz runs five BF16 quintic iterations with Nesterov momentum 0.95. MuonH/AdamH LR is 0.03; Adam LR is 0.004; betas are (0.9, 0.95); epsilons are 1e-8. The schedule is constant, with no clipping, warmup or decay. The named CPU mesh has one expert device and `use_syrk=False`. Like the earlier Grug golden generator, the outer update is not jitted; JAX executes its individual compiled primitives. No Torch optimizer produces these values.

Generate from the Marin root in its locked CPU environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run --frozen --no-sync python \
  vendor/skyrl/skyrl-train/tests/fixtures/grug_muonh_hero_ep/generate.py
```

The CPU regression preserves the earlier Grug golden tolerances: MuonH weights use `rtol=3e-3, atol=1.5e-3`; FP32 state and Adam/AdamH weights use `rtol=2e-6, atol=5e-7`. Every route also compares the actual parameter update with the reference update at relative L2 error below `1e-3`, and integer counters must match exactly. The measured maximum relative update error on Torch 2.11 CPU and JAX 0.11.1 was `9.47e-6`; maximum parameter absolute error was below `1e-6`. These measurements choose a stricter update check without changing the existing weight tolerances.

A serialized optimizer round trip occurs after update 1. A second test divides one expert bank unevenly across two Gloo processes and compares update 3 with the same JAX oracle. CUDA BF16 wrappers, pinned-memory transfers and full-model memory fit require the Mochi GB200 run.
