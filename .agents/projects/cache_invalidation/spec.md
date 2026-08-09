# Compile-cache invalidation contract

## Rigging content identities

File: `lib/rigging/src/rigging/cache.py`

```python
def file_content_hash(path: pathlib.Path) -> str: ...


def directory_content_hash(directory: pathlib.Path) -> str: ...


def combined_content_hash(components: Sequence[str]) -> str: ...


def workspace_lock_hash(start: pathlib.Path) -> str: ...
```

File hashes cover bytes. Directory hashes recursively cover sorted logical paths
and bytes while excluding absolute roots, mtimes, `.pyc`, and `__pycache__`.
Combiners consume ordered labeled strings with length framing. Missing inputs,
non-regular entries, and symlinks raise `ValueError`.

`workspace_lock_hash` finds the nearest Marin UV workspace and hashes its
`uv.lock`. An absent workspace or lock raises `ValueError`; callers must disable
shared persistence. The external dependency contract is a frozen installation
of that lock. Manual same-version patches outside the lock are unsupported.

## CuTeDSL object cache

File: `lib/levanter/src/levanter/cutlass_kernel_cache.py`

The artifact key is SHA-256 over:

1. cache schema;
2. factory module, qualified name, and sorted keyword configuration;
3. `combined_content_hash([schema, directory_content_hash(levanter/grug),
   workspace_lock_hash(cutlass_kernel_cache.py)])`;
4. stable `repr(spec)`;
5. observed JAX device platform and compute capability/device kind.

Any external dependency change in `uv.lock`, including Cutlass DSL/libraries,
FlashAttention, CUDA Python, TVM FFI, Quack, or nvdisasm, changes the key.

A launcher without an identity, a launcher defined outside `levanter.grug`, or
a spec containing a process address compiles without persistent storage. Failure
to construct the source/toolchain identity disables the persistent layer for the
installed wrapper; compilation still proceeds.

The positional launcher-factory argument remains the singleton CuTe module
bundle and does not enter launcher identity. New factories with positional
configuration are invalid and must change the contract before opting in.

## Fused cross-entropy autotune cache

File: `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`

The existing key gains schema and revision fields. The revision is
`combined_content_hash([schema, implementation, directory_content_hash(
fused_cross_entropy_loss), file_content_hash(pallas/autotune_utils.py),
workspace_lock_hash(api.py), JAX version, jaxlib version])`. Shapes, dtypes,
options, backend, observed device kind, and jaxpr remain.

If jaxpr tracing or revision construction fails, `_autotune_cache_key` returns
`None`. Autotuning may continue, but the cache is neither read nor written; this
applies to winners and negative entries.

## Unchanged contracts

- JAX persistent compilation keeps its native HLO/compiler/device key.
- XLA per-fusion autotuning keeps the launch-tree mirror until observed GPU
  identity, bounded remote generations, and local cleanup are designed together.
- Triton Sonic keeps its native cache at `/tmp/marin-triton-cache`.
- DeepEP layout and transport keep their local build caches pending a complete
  source/toolchain key plus atomic/private build publication.

Dataset, checkpoint, model-download, vLLM, and runtime attention KV caches remain
outside scope.
