# QuACK partitioned-mainloop source lineage

## Scope

This audit covers the physical backend needed by a generic
`PartitionedGemmProgram` with one unpartitioned left operand, independently
allocated right-hand-side N segments, coordinate-aligned accumulator groups,
and generated scalar/direct-store finalizations. It does not audit or depend on
model names.

The natural proof case has logical `(M, N, K) = (8, 68, 32)`, RHS segment
widths `[32, 32, 4]`, one generated scalar Map over the two 32-wide results,
and one direct 4-wide result. These dimensions are a compiler input, not a
backend dispatch key.

## Inspected sources

The original adapter plan pins Dao-AILab/QuACK revision
`84ef91df9bec87c7e4938517234fafb07ef844dd` (2026-07-29). The relevant source
hashes at that revision are:

| File | SHA-256 |
| --- | --- |
| `quack/gemm_sm90.py` | `93f676aa0439a3fe7e3bb389fcf7fd55f0ff9083e64f5e6b6fb4ad6c023a37d5` |
| `quack/gemm_base.py` | `9106f322aff597a2c6a02a69e4843531266103997e24cee0e049ddea458c19e8` |
| `quack/epilogue/frontend.py` | `48888becde4c21e7dc3d4e92e083d5d637db05cebd40e9a61de326915b2ce5e5` |
| `quack/epilogue/visit.py` | `9132378e13b00fee98c749029c874e42a293ac98a499746f968c850157e40466` |
| `quack/epilogue/ops.py` | `381416072a4d345eb2583ab1b68e98f5a7be3495d9c9c6f39f9675713efd6336` |

The package lock also pins `quack-kernels==0.5.0`. The inspected wheel has
SHA-256 `08821ebfb8e638cc20308d5c59410c6dbb3b637ccc7b07bd57c7a9261a06af74`.
Its relevant hashes are:

| File | SHA-256 |
| --- | --- |
| `quack/gemm_sm90.py` | `321085b836268deb96180ca5fa7d90bf56f58f40815430ec9959a2a955eb6cf9` |
| `quack/gemm_base.py` | `93d3e9c197d1b432d15e669641052cf436601c9f2561536d495638ba79b158d6` |
| `quack/epi_ops.py` | `eb25b7943627287ca928e899101e4fdbc414af8eebc590d5594484698c27d043` |
| `quack/gemm_default_epi.py` | `8cda07636fda14a081ab0536177f8091c49d6dc16ca6f7e523a279549c607d34` |

## Reusable machinery

The backend may retain the following generic QuACK/CuTe machinery:

- SM90 TMA and WGMMA setup;
- the staged A/B pipeline and its transaction barrier;
- tile scheduling and persistent CTA mechanics;
- register/shared/TMA output conversion and store plumbing;
- BF16 round-to-nearest-even conversion;
- the scalar epilogue frontend after it is presented with congruent generic
  accumulator fragments.

None of these interfaces needs to know which frontend program requested the
Contract.

## Missing stock interface

Both inspected implementations expose one `mB` tensor and one `tma_atom_b` in
`GemmSm90.__call__` and `GemmSm90.kernel`. The load warp constructs one global
B tile, one copy function, and one B fragment. The `concat_layout` option
relabels/interleaves a single physical tensor; it cannot join independently
allocated RHS tensors.

The epilogue frontends likewise receive one accumulator fragment. The pinned
revision's `acc_pair` mode groups adjacent N coordinates and halves every
output. It cannot combine corresponding coordinates from logical ranges
`[0,32)` and `[32,64)` while retaining `[64,68)` as an independent output.
QuACK 0.5.0 reorganizes the frontend but does not add a segmented RHS or
partitioned-accumulator interface.

Passing a concatenated temporary, launching three GEMMs, or treating the tail
as another pair would make the proof case executable, but each changes the
requested physical program. Those options are therefore excluded from this
candidate.

## Bounded physical candidate

`plan_quack_partitioned_mainloop` lowers the generic semantic program into one
CTA program with:

1. one A stage shared by all RHS groups;
2. one independently addressed B stage per static RHS segment;
3. one ordered K loop that issues each group's WGMMA operation before
   advancing the common A/B pipeline;
4. one FP32 accumulator group per logical N interval;
5. congruent register coordinates for equal-width groups;
6. BF16 round-to-nearest-even at every declared partition boundary;
7. generated scalar finalization over congruent groups;
8. a direct predicated store for the valid lanes of a padded tail group.

For widths `[32, 32, 4]`, the physical MMA-N widths are `[32, 32, 8]`; only the
first four lanes of the last group are valid. This padding is a generic SM90
instruction-shape legalization, not a semantic materialization.

The plan deliberately records one kernel, one K loop, and shared A staging. A
scalar-AST mutation changes the semantic/physical digest and scalar body while
preserving the tiled structure.

## Exact extension boundary

An executable implementation requires a reusable QuACK extension at these
sites:

1. `quack/gemm_sm90.py::__call__` and `kernel` accept a static tuple of RHS
   tensors rather than one `mB`;
2. the load warp stages one A tile and each RHS group under one common AB
   readiness barrier whose transaction count includes every active copy;
3. the math warpgroup owns one accumulator per RHS group and issues all group
   WGMMA operations in the same ordered K iteration;
4. `quack/epilogue/visit.py` presents congruent accumulator-group coordinates
   to a generated scalar function after the declared BF16 boundaries;
5. `quack/gemm_base.py` reuses its multi-output store driver for scalar and
   passthrough outputs, including valid-lane predication for padded groups.

The first isolated extension patch is
`lib/tile_lifetime/backends/h100/quack_partitioned_sm90.patch` (SHA-256
`0bbb2354cff80b2fdf475fce12cef277f961591623b0c078160c27f09e5658db`).
It adds reusable tuple validation, congruent accumulator partitioning,
ordered group-WGMMA issue, BF16-boundary helpers, and a bounded
`PartitionedGemmSm90` executor. The executor is intentionally nonpersistent:
one load warp stages one A tile and every static RHS tile, one math warpgroup
issues all accumulator groups in the same K loop, and a generated finalizer
writes scalar and passthrough outputs from BF16 shared-memory boundaries.
This avoids changing QuACK's persistent scheduler merely to preserve its
existing class shape.

The stock QuACK packages cannot execute this plan without the patch. The patch
and generated authoring source pass host-side syntax and mutation/ABI tests.
The single authorized H100 invocation installed the pinned QuACK, Torch, and
CUTLASS DSL environment, but stopped before the patched module import because
the narrow preflight environment omitted JAX while Shuttle's package
initializer eagerly imports a JAX transport module. CuTe device compilation
and execution therefore remain untested. This is not GPU correctness or
performance evidence.

`benchmarks/h100_quack_partitioned_mainloop_preflight.py` checks the source
revision, patch digest, patched-module import, and helper symbols on an H100
host without launching the device. It imports both the patched executor and a
generated module recovered from the supplied HLO, and requires the generic
entry point before any correctness or timing run. This prevents a stock
single-RHS or split-GEMM fallback from being reported as the requested
candidate. The exact failed H100 gate and released-allocation proof are under
`benchmarks/artifacts/quack_partitioned_sm90_h100_compile_gate_v0/`.
