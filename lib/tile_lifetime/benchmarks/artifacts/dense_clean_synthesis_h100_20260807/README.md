# Dense clean-synthesis H100 acceptance rerun

This checkpoint exercises the accepted dense path:

```text
ordinary JAX
  -> frozen StableHLO
  -> named semantic recovery
  -> 36 generic Map / Contract / Fold / DomainRestriction operations
  -> generic eight-skeleton plan
  -> generated Contract tile programs and generated streaming attention
```

The measured path does not select a named GEMM callback or the official FA3
forward kernel. The JSON manifests record `named_gemm_callback_selected=false`,
`official_fa3_oracle_selected=false`, and `attention_backend=generated_sm90`.

## Region results

Measurements used 10 warmups followed by 30 interleaved samples of 10 region
iterations. Ratios use the frozen manual CODA/FA3 oracle for the same shape.

| Sequence | RMS placement | Median (ms) | Minimum (ms) | Oracle (ms) | Ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | source-ordered consumer prologue | 1.6872 | 1.5564 | 1.4561 | 1.159x |
| 2048 | delayed consumer epilogue | 1.6339 | 1.5188 | 1.4561 | 1.122x |
| 4096 | source-ordered consumer prologue | 3.4148 | 3.2044 | 3.0080 | 1.135x |
| 4096 | delayed consumer epilogue | 3.3848 | 3.2749 | 3.0080 | 1.125x |

All four measured candidates satisfy the clean-synthesis region target of at
most 1.20x the natural expert comparison. The raw distributions are preserved
in `region-s2048.json` and `region-s4096.json`.

## Correctness evidence

The package tests compare semantic recovery and rewrites against the natural
JAX/CPU definitions. On H100, generated Contract components were compared with
the matching named QuACK implementations used only as oracles. Generated
residual, RMS-partial, row-scale, and rotary outputs were bitwise equal to the
matched oracle. The direct scalar-AST SiLU expression differs from QuACK's
named fast-math helper after BF16 rounding:

| Sequence | Candidate | Maximum absolute error | Mean absolute error |
| ---: | --- | ---: | ---: |
| 2048 | consumer prologue gate/up | 0.0625 | 1.34e-8 |
| 2048 | delayed epilogue gate/up | 0.03125 | 7.34e-9 |
| 4096 | consumer prologue gate/up | 0.1250 | 1.25e-8 |
| 4096 | delayed epilogue gate/up | 0.03125 | 8.66e-9 |

This comparison is `real_algebra_equivalent`; the generated expression is the
natural scalar AST and the named fast-math helper is not the semantic source of
truth. Each JSON file preserves deterministic output hashes. Prologue versus
delayed placement differs by at most 0.03125 in `x2` and next-QKV for both
shapes, as expected from their different declared floating-point orderings.

The mutation test changes pairwise SwiGLU from
`SiLU(left) * right` to `left * right`. The same scalar-AST generator emits the
new arithmetic without a workload-name switch or handwritten kernel edit.

## Generated-code audit

The generated GEMM sources contain direct scalar-AST arithmetic. In
particular, pairwise rotary code is emitted from coefficient TileLoads and the
pair expression, and SwiGLU is emitted directly with `cute.exp`. The sources do
not reference `swiglu`, `rotary_cos_sin_load`, the official FlashAttention
forward, or named Transformer scheduling helpers.

External implementation dependencies on the accepted path are classified as:

| Dependency | Classification | Use |
| --- | --- | --- |
| CUDA, WGMMA, TMA | hardware/runtime primitive | matrix and asynchronous-memory instructions |
| QuACK/CuTe mainloop and layout machinery | generic compute primitive | Contract mainloop, tile copies, and epilogue framework |
| generated scalar/tile AST | generated Shuttle kernel | preparation, finalization, auxiliary Fold emission, RoPE, and SwiGLU |
| generated SM90 streaming skeleton | generated Shuttle kernel | QK/PV Contracts, online Fold state, and producer/consumer pipeline |
| named QuACK callbacks | expert/oracle-only | component comparison only; not selected by the measured region |
| official FA3 | expert/oracle-only | frozen historical comparison only; not selected by the measured region |

The generic Contract primitive derives from QuACK revision
`84ef91df9bec87c7e4938517234fafb07ef844dd`. It retains the WGMMA mainloop,
CuTe tile/layout machinery, and programmable epilogue framework. Transformer
operation selection and named epilogue visitors were removed from the accepted
interface; Shuttle supplies preparation, finalization, coefficient loads, and
auxiliary emissions as generic tile programs.

The generated streaming skeleton is FA3-inspired and retains low-level CuTe
matrix/pipeline machinery. Its body is supplied by generic QK/PV Contracts,
DomainRestriction, and normalized-exponential Fold lowering; the official FA3
forward entry point is not called.

## Environment and source identity

The run used one NVIDIA H100 80 GB HBM3 at a 700 W power limit, driver
595.71.05, CUDA 13.0, PyTorch 2.13.0+cu130, CUTLASS DSL 4.6.1, and
flash-attn-4 4.0.0b16. The generated sources and implementation files are
content-addressed in each region JSON. `shuttle_revision` records a dirty
prototype checkout; the implementation-file and generated-source SHA-256
values are therefore authoritative for this checkpoint. The QuACK patch hash
is also recorded.

## Reproduction

With the pinned QuACK source and its recorded patch installed:

```bash
export PYTHONPATH=/app/lib/tile_lifetime:/app/lib/tile_lifetime/src
python lib/tile_lifetime/benchmarks/h100_plan_runtime.py \
  --sequence 2048 --warmups 10 --repeats 30 --iterations 10 \
  --attention-backend generated_sm90 \
  --shuttle-revision dirty-dense-name-erasure \
  --quack-revision 84ef91df9bec87c7e4938517234fafb07ef844dd \
  --json-output region-s2048.json
python lib/tile_lifetime/benchmarks/h100_generated_gemm_programs.py \
  --sequence 2048 --warmups 10 --repeats 30 --iterations 10 \
  --shuttle-revision dirty-dense-name-erasure \
  --quack-revision 84ef91df9bec87c7e4938517234fafb07ef844dd \
  --json-output components-s2048.json
```

Repeat with `--sequence 4096` for the larger primary shape.
