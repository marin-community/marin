# Compile-cache invalidation contract

## Rigging content identities

File: `lib/rigging/src/rigging/cache.py`

```python
def compile_cache_key(
    source_paths: Sequence[pathlib.Path],
    *,
    environment: Sequence[str],
) -> str: ...


def installed_distribution_fingerprint(
    distribution_names: Sequence[str],
) -> str: ...
```

`compile_cache_key` consumes roots and environment strings in caller order. A
directory is recursive and sorted by logical relative path. SHA-256 inputs are
length framed and include root indices, paths, and bytes. Absolute roots, mtimes,
`.pyc`, and `__pycache__` are excluded. Missing roots, non-regular entries, and
symlinks raise `ValueError`.

`installed_distribution_fingerprint` consumes names in caller order and includes
installed distribution name, version, metadata-listed logical paths, and actual
file bytes. For PEP 610 editable installs it also hashes the source roots of all
import packages attributed to the distribution. Missing distributions, absent
file inventories, unresolved editable sources, missing files, non-regular files,
and symlinks raise `ValueError`.

## CuTeDSL object cache

File: `lib/levanter/src/levanter/cutlass_kernel_cache.py`

The artifact key is SHA-256 over:

1. cache schema;
2. factory module, qualified name, and sorted keyword configuration;
3. `compile_cache_key([levanter/grug], environment=[schema, toolchain digest])`;
4. stable `repr(spec)`;
5. observed JAX device platform and compute capability/device kind.

The toolchain digest covers actual installed bytes for the declared TVM FFI,
CUDA Python, nvdisasm, Cutlass DSL/libraries, FlashAttention 4, and Quack
distributions.

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
`compile_cache_key([fused_cross_entropy_loss package, pallas/autotune_utils.py],
environment=[schema, JAX version, jaxlib version, optional libtpu digest])`. TPU
uses actual installed `libtpu` bytes. Shapes, dtypes, options, backend, observed
device kind, and jaxpr remain.

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
