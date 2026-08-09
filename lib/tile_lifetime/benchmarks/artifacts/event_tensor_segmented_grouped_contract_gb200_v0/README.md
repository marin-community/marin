# Event Tensor relation-to-grouped-Contract linkage on GB200

This artifact records one bounded physical-linkage replay on one NVIDIA GB200.
It connects runtime `RelationPlan` tables to generic grouping/padding and then
to the generic SM100 grouped-Contract primitive through two JAX typed-FFI
handlers on the same device stream.

The two relations have counts `[64, 80, 48, 0]` and `[72, 56, 64, 0]`.
Both include one empty segment. The relation mutation preserves the program and
inner-Contract fingerprints while changing the runtime fingerprint. Both cases
match an independent segmented-Contract reference, are bitwise deterministic,
and execute exactly one grouping handler and one grouped-Contract handler per
invocation. The recorded handler count is 28 for each target: two determinism
checks, two warmups, and ten measured calls for each relation.

This is linkage and correctness evidence, not an overlap or tuning result. The
outer Event Tensor is legally erased by verified same-JAX-stream order. Shuttle
owns the generated wrapper synchronization ABI, but the external grouped-
Contract primitive still owns its internal `mbarrier` arrive/wait sites, TMA
issue, phase advancement, and accumulator release instructions.

The Python runtime is Torch-free: Torch never enters `sys.modules`, and `ldd`
shows no Torch or C10 dependency. The build uses Torch headers only because the
pinned MoK/ThunderKittens source includes them transitively. JAX owns the entry
point and AD; this forward-only replay does not register a custom adjoint.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/marin/lib/tile_lifetime/src \
/app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/gb200_jax_segmented_grouped_contract_event.py \
  --mok-root /tmp/mixture-of-kittens \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --torch-root /app/.venv/lib/python3.12/site-packages/torch \
  --build-directory /tmp/shuttle-event-gmm-build \
  --output /tmp/shuttle-event-gmm-result.json \
  --shuttle-revision 1f846b5baa7a606574b4a2082f8d9b7f088ea944 \
  --warmups 2 \
  --samples 10 \
  --requested-cpu 2 \
  --requested-priority batch
```

The first build attempt exposed a mixed compiler environment: CUDA 13.0
`nvcc`/`ptxas` had CUDA 13.3 NVVM/CRT installed, producing PTX 9.3 that the
9.0 assembler rejected. The successful replay pinned `nvidia-nvvm`,
`nvidia-cuda-crt`, and `nvidia-cuda-nvcc` to 13.0.88. The one-line
`EventRealizationAudit.entries` reporting fix in the final harness happened
after physical compilation; it did not change the generated library or
execution path.

## Results

| Relation | Median | Maximum absolute error | Deterministic |
| --- | ---: | ---: | --- |
| Primary `[64,80,48,0]` | 0.218432 ms | 0.0140762 | yes |
| Mutation `[72,56,64,0]` | 0.225696 ms | 0.0140762 | yes |

`result.json` contains every timing sample, HLO target occurrence counts,
program/runtime/inner fingerprints, output hashes, exact dynamic dependencies,
and source pins. `run.log` contains the successful compiler and benchmark
output. `environment.txt` records the physical device and toolchain.
