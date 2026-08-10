# Fixed-capacity distributed expert JAX module CPU checkpoint

This artifact is a static and CPU correctness gate. It is not a GPU result and
contains no latency claim.

The natural boundary starts from ordinary JAX router, top-k, normalized route
weights, selected expert Contracts, and pairwise Map algebra. JAX produces the
whole-program VJP. Shuttle then instantiates five generic typed-FFI families at
fixed rank-local shapes: relation-edge Map/Fold, input-adjoint segmented
Contract/Map/Contract, W13 weight Contract, W2 weight Contract,
and the post-return source Fold. CUDA handlers are lowered to StableHLO custom
calls but are not compiled or executed on CPU. The concatenated W13 weight
cotangent is split back into natural gate/up optimizer storage outside the
generic Contract handler.

Four forced host devices execute a payload-only `jax.lax.all_to_all` round trip.
The stored collective StableHLO contains exactly two all-to-all operations and
the returned integer payload is exact. A second, single shard-mapped StableHLO
graph contains each of the five generated handlers once, three payload-only
all-to-all operations, the post-return source Fold, one JAX router-gradient
all-reduce, and ordinary JAX router-pullback algebra. Forward-layout W2 and W13
operands are explicitly transposed within each expert before the input-adjoint
Contract ABI.

This revision rejects an earlier dense expert-axis specialization. At the
primary per-rank shape `E=96, C=256, H=7168, I=3072`, that specialization would
have allocated 106,803,757,056 bytes of intermediates, evaluated
14,495,514,624 Map items, and repeated Contract work 96 times. The current
fixed-capacity segmented family uses 805,306,368 bytes across projection scratch
and its two outputs, evaluates 150,994,944 Map items with 64-bit indexing, and
performs each expert Contract once. This accounting is non-allocating.

The integrated graph is lowered but not compiled on CPU because its typed-FFI
handlers are CUDA implementations. The exact CPU numerical comparison executes
the same fixed-capacity relation and decomposed generic stage semantics without
those custom calls. A nonsquare `H=48, I=32` case and a segmented fixture with
nonuniform occupancy and an empty expert cover the transposed weight ABI. The
artifact records maximum and mean absolute and relative errors plus BF16 ULP
distance; relative and ULP maxima are diagnostic rather than acceptance bounds
when values cross zero. A four-device collective test also sends unique
logical-edge IDs through both inverse payload paths and recovers every original
source-item/route-slot coordinate exactly.

Reproduce with:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run --frozen --package marin-tile-lifetime --group test python \
  lib/tile_lifetime/benchmarks/cpu_distributed_expert_jax_module.py \
  --output-directory \
  lib/tile_lifetime/benchmarks/artifacts/distributed_expert_jax_module_cpu_v0
```

Remaining gates are CUDA source compilation, numerical execution of the
integrated graph, and a matched four-rank GB200 replay. No GPU result is implied
by this checkpoint.
