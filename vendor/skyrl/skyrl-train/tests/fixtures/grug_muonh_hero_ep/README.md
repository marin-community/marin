# Hero EP MuonH CPU oracle

These fixtures contain three updates from Hero EP at Marin revision
`4af966070e5d0252e87d404196400c35bd04883b`. The generator calls its JAX optimizer;
Torch does not produce the expected values. Parameters and state are FP32,
Newton–Schulz uses BF16, and the outer update is not jitted. Matrix axes are
transposed into Megatron layout. Inputs include two layers and three routed
experts with unequal norms.

- `golden.npz`: whole-bank normalization; SHA-256 `bd5ec624264e5ff28e97e83ff526fb34ec5f647d1de3088c90f12acb0da3b5e9`.
- `golden_per_expert.npz`: same inputs with per-expert normalization; SHA-256 `37aeb3d57fdd016dd6b9ca548de265357a2a436c460344b62b0a3c3b85002701`.

Run from the Marin repository root containing this generator, in its locked
CPU environment. The generator imports this checkout's reference code. Verify
`experiments/grug/moe_hero_ep/{optimizer,grugmuon_hero,adamh}.py` and Levanter's
`optim/util.py` coefficient table against the pinned revision. These files are
unchanged at extraction base `2af3f828aa08cae6359fef4ba1c6ac9d81ccde61`.

```bash
for mode in all_experts per_expert; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    uv run --frozen --no-sync python \
    vendor/skyrl/skyrl-train/tests/fixtures/grug_muonh_hero_ep/generate.py --expert-normalization "$mode"
done
```
