# XLA pair-Map replacement smoke

This artifact is a deliberately disposable proof of the XLA insertion point.
An ordinary JAX function computes two matrix products, applies a scalar
`tanh(left) * right` map, and consumes the result with a third matrix product.
At `PRE_SCHEDULER`, Shuttle's opcode/shape/dependency matcher recovers that
structure, generates a fixed-shape C++ implementation, and replaces the entry
region with one CPU custom call.

The transformed result matches the unmodified JAX executable with maximum
absolute error `1.9073486328125e-06` and mean absolute error
`4.333754475283058e-07`. The generated handler's execution counter is `1`, and
the transformed HLO contains the target exactly once. Neither recovery nor
generation uses a model or activation name.

This is not the production bridge. It uses an HLO text round trip and XLA's
removed legacy CPU custom-call ABI, which JAX 0.11 warns about at runtime. Those
choices are acceptable only for this isolated smoke. The frozen Grug module
must use typed C++ HLO mutation plus a supported FFI ABI, with sharding,
aliasing, side effects, and multi-output region boundaries preserved.

Reproduce from the repository root:

```shell
uv run \
  --config-file lib/tile_lifetime/benchmarks/jax011_probe_uv.toml \
  --isolated \
  --package marin-tile-lifetime \
  --group test \
  --with 'jax==0.11.0' \
  --with 'jaxlib==0.11.0' \
  python lib/tile_lifetime/benchmarks/xla_pair_map_custom_call_smoke.py \
  --artifact-directory \
  lib/tile_lifetime/benchmarks/artifacts/xla_pair_map_custom_call_smoke_jax011_v0
```

Files:

- `summary.json`: numerical evidence, handler execution count, and blockers.
- `generated_pair_map_handler.cc`: exact generated fixed-shape body.
- `original-pre-scheduler-hlo.txt.gz`: ordinary JAX HLO before recovery.
- `recovery-report.json`: generic structural match over the original HLO.
- `transformed-pre-scheduler-hlo.txt.gz`: the returned module containing the call.
