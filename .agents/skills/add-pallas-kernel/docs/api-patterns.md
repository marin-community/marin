# API Patterns

Read this file before adding or changing a public kernel wrapper, backend
selection behavior, block-size config, tuned table lookup, or input
normalization contract.

## Recommended Module Layout

Tokamax-style decomposition is preferred for maintainability:

- `__init__.py`: public API, imports from `api.py`.
- `reference.py`: readable vanilla JAX oracle.
- `api.py`: stable user-facing entrypoint with `implementation=` override and
  fallback order.
- `config.py`: block-size dataclasses and implementation-neutral config types
  when needed.

Then add implementation variants as needed:

- `xla.py`: default implementation, if different from reference.
- `pallas_tpu.py`: TPU Pallas implementation.
- `pallas_mosaic_gpu.py`: GPU Pallas Mosaic implementation.
- `tuned_block_sizes.py`: checked-in tuned table for runtime selection.

Reference template:
`lib/levanter/src/levanter/kernels/pallas/template_kernel.py`

## Top-Level API

Use an entrypoint shaped like:

```python
Implementation: TypeAlias = typing.Literal["reference", "xla", "pallas_tpu", ...]


class BlockSizes(Protocol):
    b_block_size: int
    h_block_size: int
    v_block_size: int


def template(
    x: Float[Array, "B H V"],
    *,
    implementation: Implementation | Sequence[Implementation | Callable[..., jax.Array]] | None = None,
    block_sizes: BlockSizes | dict[Implementation, BlockSizes] | None = None,
) -> jax.Array:
```

## Block Size Config

Use a protocol or union of dataclasses to expose implementation-specific
block-size choices. Each implementation should have its own block-size class
unless the choices are genuinely identical.

Expose tile choices via a dataclass with explicit defaults:

```python
@dataclass(frozen=True, slots=True)
class BlockSizes:
    b_block_size: int = 1024
    h_block_size: int = 512
    v_block_size: int = 2048

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()
```

Rules:

- Validate backend-alignment constraints in the backend-specific implementation.
- Keep reference/XLA paths usable even when TPU or GPU constraints are not met.
- If a legacy `block_size` arg exists, map it clearly to the new config and
  raise on conflicting inputs.
- Keep tuned table keys stable and reviewable; do not key on every exact shape.

## Fallback Semantics

- If a single implementation is explicitly requested, such as
  `implementation="pallas_tpu"`, fail fast on unsupported backend/shape.
- If a sequence of implementations is requested, try each in order, warn on each
  fallback, and raise if none work.
- Treat a default implementation order the same as a sequence.
- Keep backend selection explicit and predictable in `api.py`.

## Input Normalization

Prefer one canonical kernel input shape and make callers normalize to it:

- Implement the core kernel for batched inputs.
- Define one canonical shape contract, such as rank-2/1/2 forms.
- Expect callers to flatten or reshape batch axes before kernel invocation.
- If wrapper reshaping helpers are useful, keep them thin and explicit at API
  boundaries.

For TPU kernels, also read the input-shape notes in [TPU tips](tpu-tips.md).

## Sharded Kernel Boundaries

Every accelerator kernel boundary should be wrapped in an explicit
`jax.shard_map`. This applies to `pl.pallas_call`, Mosaic GPU calls, and custom
FFI calls. Normalize and `reshard` inputs to the local kernel `PartitionSpec`
before entering the `shard_map`, and make the output spec match the public
wrapper contract. Only skip `shard_map` for wrappers whose inputs are explicitly
documented and tested as fully local or replicated.

Do not call a Pallas or FFI kernel directly on globally sharded tensors. Opaque
kernel boundaries give XLA too little information, and it can insert large
all-gathers or other global reshards around the call. When a kernel cannot
support a sharded dimension, assert that dimension is unsharded before the
`shard_map` rather than silently relying on a global reshard.

## Bad Patterns

- Exact-shape tuned tables instead of stable shape buckets.
- Silent fallback when the user explicitly requested one backend.
- Benchmark scripts imported by production code.
- Backend-specific shape hacks spread through callers instead of normalized at
  the API boundary.
- Hidden env-var defaults for critical behavior.
- Compatibility shims for old kernel arguments unless the user explicitly asks
  for them.
- Direct `pl.pallas_call`, Mosaic, or FFI calls on sharded arrays without a
  localizing `shard_map`.
