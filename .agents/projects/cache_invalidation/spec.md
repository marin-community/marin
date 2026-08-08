# Compile-cache invalidation contract

## Rigging API

File: `lib/rigging/src/rigging/cache.py`

```python
def compile_cache_key(
    source_paths: Sequence[pathlib.Path],
    *,
    environment: Sequence[str],
) -> str:
    """Return a SHA-256 content address for source trees and build inputs.

    Paths are consumed in caller order. Directories are traversed recursively in
    sorted logical-path order. The hash includes each logical path, file bytes,
    and each environment string. Absolute root paths, mtimes, Python bytecode,
    and `__pycache__` entries are excluded. Missing or non-file/non-directory
    paths raise `ValueError`.
    """


def sync_kv_cache(prefix: str, namespace: str, local: str) -> SyncedDirectory:
    """Mirror a compiler-owned local tree to a namespaced temp object directory.

    The remote root is `marin_temp_bucket(30, prefix)/namespace`. The caller owns
    invalidation and must supply a path-safe, non-empty namespace. Invalid
    namespaces raise `ValueError`. Transfer failures retain existing best-effort
    warning behavior.
    """
```

`sync_kv_cache` no longer reads `launch_provenance` and no longer returns `None`.
Callers decide whether the external state required to mirror a cache exists.

## Iris resource and XLA contracts

File: `lib/iris/src/iris/env_resources.py`

```python
@dataclass(frozen=True)
class TaskResources:
    memory_bytes: int
    cpu_cores: float
    gpu_count: int
    tpu_count: int
    gpu_variant: str | None = None
```

`TaskResources.from_environment()` copies `device.gpu.variant` from
`IRIS_TASK_RESOURCES`; absent/empty values become `None`.

File: `lib/iris/src/iris/runtime/jax_init.py`

```python
def _xla_autotune_namespace(*, xla_version: str, gpu_variant: str) -> str:
    """Return a path-safe namespace that changes with XLA and GPU type."""
```

For remote JAX compilation-cache configurations, GPU tasks use
`/cache/xla/per-fusion-autotune/<namespace>` and mirror it to
`xla-per-fusion-autotune/<namespace>`. Tasks without a GPU variant still use a
versioned node-local directory but do not mirror it to object storage.

## CuTeDSL contract

File: `lib/levanter/src/levanter/cutlass_kernel_cache.py`

`_kernel_key(fn, spec)` returns SHA-256 over:

1. cache schema;
2. factory module, qualified name, and sorted keyword configuration;
3. `compile_cache_key([levanter/grug], environment=<installed distributions>)`;
4. stable `repr(spec)`;
5. JAX device platform and compute capability/device kind.

A launcher without an identity, a launcher defined outside `levanter/grug`, or
a specification containing a process address is compiled without persistent
storage. Source-key construction errors propagate during launcher creation so an
uncovered source boundary cannot silently become cacheable.

## Fused cross-entropy contract

File: `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`

The existing autotune key gains one `revision=<sha256>` field. The revision is
`compile_cache_key([fused_cross_entropy_loss package], environment=[JAX version,
jaxlib version, cache schema])`. Existing shape, dtype, option, backend, device,
and jaxpr fields remain.

## DeepEP contract

Files:

- `lib/levanter/src/levanter/kernels/deepep/layout_ffi.py`
- `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`

Both artifact directories use the first 16 hexadecimal characters of a
`compile_cache_key`. Sources include the Levanter DeepEP package, external
`DEEPEP_SRC_ROOT/csrc`, and `jaxlib/include`. Environment values include cache
schema, effective CUDA architecture/compile flags, and `nvcc --version` output.
Transport additionally includes the patched intranode source bytes, Torch/raw
and Python-module modes, dispatch-thread override, compatibility signature, and
Python ABI tag. No absolute source or include directory enters either key.

If `nvcc` identity cannot be read, the environment records `nvcc=unknown`; the
subsequent build retains its existing explicit failure if the compiler is
unavailable.

## Out of scope

- Changing JAX's persistent compilation-cache key or storage.
- Persisting or remotely mirroring Triton's Sonic cache.
- vLLM compilation-cache generations.
- Dataset, checkpoint, Hugging Face download, and runtime attention KV caches.
