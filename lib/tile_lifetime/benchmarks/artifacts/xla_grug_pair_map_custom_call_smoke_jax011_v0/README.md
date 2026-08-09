# Grug region-local XLA replacement smoke

This artifact extends the isolated whole-entry CPU smoke to a region-local
replacement inside the ordinary one-layer Grug training step. The JAX 0.11
`PRE_SCHEDULER` callback recovers two generic pair-Map regions. A structural
criterion selects the unique region whose consumer Contract preserves the row
domain. No model, module, activation, or source-location name participates.

The recovered physical boundary is:

```text
f32[8,32] activation
f32[32,32] projection weight
f32[32,32] projection weight
f32[32,32] consumer weight
    -> f32[8,32]
```

The callback replaces only the consumer entry instruction. All upstream
`convert`, `bitcast`, and wrapper fusions remain in the module. The generated
scalar expression retains all 16 BF16/F32 cast boundaries inside the recovered
Map, using a generated round-to-BF16 operation. The target instruction has no
explicit HLO sharding attribute, so the text smoke refuses rather than guesses
about sharding transfer.

The complete 58-leaf train-step result is bitwise identical between the
unmodified and transformed executables. This includes identical payloads for
the two NaNs already present in the semantic baseline; all finite values have
zero maximum and mean absolute error. The generated handler executed once and
the transformed HLO contains one call target.

The other recovered region is backward-facing and carries additional saved and
adjoint values. The sibling
`xla_grug_backward_multi_output_custom_call_smoke_jax011_v0` artifact grows that
region generically, emits a three-result body, and executes its tuple-result
replacement. The remaining blocker is a supported production multi-result FFI
and GPU lowering, not semantic region formation.

This remains disposable insertion-point evidence. It uses a line-level,
structurally checked HLO text edit and XLA's removed legacy CPU custom-call ABI.
Production needs typed C++ instruction replacement, supported XLA FFI, explicit
alias/sharding/effect transfer, and GPU lowering from the same generated AST.

Reproduce from the repository root:

```shell
uv run \
  --config-file lib/tile_lifetime/benchmarks/jax011_probe_uv.toml \
  --isolated \
  --package marin-core \
  --extra cpu \
  --with 'jax==0.11.0' \
  --with 'jaxlib==0.11.0' \
  python lib/tile_lifetime/benchmarks/xla_grug_pair_map_custom_call_smoke.py \
  --artifact-directory \
  lib/tile_lifetime/benchmarks/artifacts/xla_grug_pair_map_custom_call_smoke_jax011_v0
```

Files:

- `summary.json`: numerical evidence, exact boundary, and production blockers.
- `generated_pair_map_handler.cc`: generated fixed-shape physical body.
- `original-pre-scheduler-hlo.txt.gz`: ordinary natural train-step HLO.
- `recovery-report.json`: both generic recovered pair-Map regions.
- `transformed-pre-scheduler-hlo.txt.gz`: one region-local custom call.
