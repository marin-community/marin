# Grug backward multi-output region replacement

This artifact proves the multi-output boundary needed by the second generic
pair-Map region recovered from an ordinary one-layer Grug train step.

Starting only from the two recovered projection Contracts, Shuttle grows a
maximal entry-local region through pointwise and wrapper operations. Contracts
and reductions stop growth. Additional operands become explicit inputs, and
every value with a user outside the region becomes an explicit output.

For this module, the algorithm derives:

```text
inputs
    activation          f32[8,32]
    projection weight   f32[32,32]
    projection weight   f32[32,32]
    saved cotangent     bf16[8,32]

internal
    9 entry instructions

outputs
    scalar-Map VJP 0    f32[8,32] -> 2 downstream Contracts
    scalar-Map VJP 1    f32[8,32] -> 2 downstream Contracts
    saved forward Map   f32[8,32] -> 1 downstream Contract
```

The callback generates all three scalar expressions from the inlined HLO,
including every BF16 conversion boundary. It inserts one tuple-result custom
call and rewires the three live outputs through `get-tuple-element`. The
original internal instructions remain for the later XLA dead-code pass.

The complete 58-leaf train-step result is bitwise identical between the
unmodified and transformed executables. The two NaNs already present in the
baseline have identical payloads; all finite values have zero maximum and mean
absolute error. The tuple handler executed once, and the transformed HLO
contains one call and three tuple projections.

This remains a disposable CPU proof because it uses a structurally checked HLO
text edit. The three-result handler now uses XLA's supported typed FFI, so the
multi-result ABI is no longer an open question. Production needs typed C++
connected-region replacement, generic sharding/alias/effect transfer, and a GPU
skeleton consuming the same generated multi-output AST.

Reproduce from the repository root:

```shell
uv run \
  --config-file lib/tile_lifetime/benchmarks/jax011_probe_uv.toml \
  --isolated \
  --package marin-core \
  --extra cpu \
  --with 'jax==0.11.0' \
  --with 'jaxlib==0.11.0' \
  python \
  lib/tile_lifetime/benchmarks/xla_grug_backward_multi_output_custom_call_smoke.py \
  --artifact-directory \
  lib/tile_lifetime/benchmarks/artifacts/xla_grug_backward_multi_output_custom_call_smoke_jax011_v0
```

Files:

- `summary.json`: exact boundary, users, numerical evidence, and blockers.
- `generated_pair_map_handler.cc`: generated tuple-output physical body.
- `original-pre-scheduler-hlo.txt.gz`: unmodified natural train-step HLO.
- `recovery-report.json`: generic Contract-pair/Map recovery.
- `transformed-pre-scheduler-hlo.txt.gz`: tuple call plus three output rewires.
