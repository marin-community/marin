# Mixed E4M3/E5M2 FP8 wgmma — forked-jaxlib approach

The FP8 ragged-dot hybrid (`haliax._src.fp8_ragged`) wants the Transformer-Engine recipe:
**E4M3** activations/weights on the forward, **E5M2** output-gradient on the backward. That
makes the two backward grouped GEMMs *mixed*-FP8:

- dgrad (`dlhs`): `e5m2`-grad × `e4m3`-rhs  → `jax.experimental.pallas.ops.gpu.ragged_dot_mgpu`
- wgrad (`drhs`): `e4m3`-act × `e5m2`-grad  → `haliax._src.transposed_ragged_dot_mgpu`

Hopper's `wgmma.mma_async` instruction natively supports this — it takes **independent**
`.atype`/`.btype` operands, each `.e4m3` or `.e5m2`. But stock Mosaic-GPU (jax/jaxlib 0.10.0)
rejects a mixed pair in three places, all of which assume `a` and `b` share one element type:

| layer | location | role |
|-------|----------|------|
| jaxlib C++ | `mosaic_gpu.cc` `WGMMAOp::verify` | dialect verifier (gate) |
| jax python | `mosaic/gpu/wgmma.py` | PTX emitter — emits `.{el_ty}.{el_ty}` (functional) |
| jax python | `_src/pallas/mosaic_gpu/primitives.py` `wgmma` | pallas primitive dtype gate |
| jax python | `pallas/ops/gpu/ragged_dot_mgpu.py` | grouped-GEMM `dlhs` dtype guard |

The emitter is the only *functional* change (it must emit `.atype.btype` independently); the
other three are pure gates. With same-dtype operands the only f8 backward that lowers is
all-E4M3 — which loses the E5M2 dynamic range on the gradient.

## The fork

[`mcwitt/jax@mixed-fp8-wgmma-0.10.0`](https://github.com/mcwitt/jax/tree/mixed-fp8-wgmma-0.10.0)
relaxes all four layers (verifier + emitter + two python guards) to accept the e4m3/e5m2 pair,
keeping the same-dtype contract for every other type. The changes are upstreamable — they
match the hardware's actual capability and are covered by added tests in
`tests/mosaic/gpu_test.py`, `tests/mosaic/gpu_dialect_test.py`, `tests/pallas/mosaic_gpu_test.py`.

This branch carries **only** the haliax-side FP8 ragged subsystem (which already allows the
mixed pair in `transposed_ragged_dot_mgpu`). All jax/jaxlib changes live in the fork; there is
no runtime monkeypatching. The alternative — running the identical path on **stock** jaxlib via
an import-time python overlay plus a scoped verifier-disable — is the `grug-fp8-shim` branch.

## Install (cw-us-east-02a H100 container, repo synced at `/app`)

```bash
# Builds the forked jaxlib wheel (~11 min, CPU; the dialect is MLIR C++, no nvcc), installs it
# + the forked jax python package, and wires the Mosaic toolchain (ptxas/libdevice/cuDNN 9.12).
bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh

# To skip the build, point at a prebuilt wheel:
JAXLIB_WHEEL=/path/to/jaxlib-mixfp8-0.10.0-cp312-cp312-manylinux_2_27_x86_64.whl \
  bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh
```

The base jaxlib is independent of the CUDA plugin version, so the stock `jax-cuda13-plugin` /
`jax-cuda13-pjrt` 0.10.0 are kept as-is — only the base `jaxlib` wheel and the `jax` python
package are replaced.

## Run the hybrid benchmark

```bash
uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra gpu \
  -- python lib/levanter/scripts/bench/bench_ragged_mosaic_hybrid_e2e.py \
       --path mosaic --grad-dtype e5m2 --mosaic-wgrad fp8
```

Expected (real Grug MoE-MLP, T8192/D2048/F5632/E8): the all-f8 hybrid runs ~1.33× the bf16
baseline end-to-end, both backward GEMMs lowering as genuine mixed E4M3/E5M2 wgmma.

## Verify the mix lowers

```python
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu
m, k, n, g = 512, 256, 256, 4
lhs = jnp.ones((m, k), jnp.float8_e5m2)            # e5m2 (grad)
rhs = jnp.ones((g, k, n), jnp.float8_e4m3fn)       # e4m3 (rhs)
gs = jnp.full((g,), m // g, jnp.int32)
print(jax.jit(lambda a, b, s: ragged_dot_mgpu.ragged_dot(
    a, b, group_sizes=s, block_m=64, block_n=64, block_k=64,
    max_concurrent_steps=2, grid_block_n=1))(lhs, rhs, gs).shape)   # (512, 256), no verifier error
```
