# Content-address compile caches by their real inputs

Levanter's CuTeDSL cache will survive unrelated Marin worktree changes without
serving stale kernels. Rigging will expose explicit source/environment keying and
caller-owned directory namespaces, so each compiler cache invalidates on the
inputs that can change its artifact. The full inventory and prior-work evidence
are in [research.md](research.md).

## Challenges

The caches do not share one correct revision key. JAX hashes lowered HLO and its
compiler environment; XLA owns per-fusion files but delegates version and GPU
separation to the caller; CuTeDSL emits object code before JAX can hash the
module; Levanter's autotuner stores a performance decision, including negative
results; DeepEP invokes `nvcc` over local and external source trees.

A narrow source key must remain fail-closed. PR #8065 showed that hashing a
launcher's defining file was incomplete: `_fa4_cute_kernels.py` dynamically
loads `_fa4_cute_segmented_bwd.py`, so the old key silently reused code compiled
before an edit. Commit hashes have the opposite defect for development: they do
not see dirty source.

## Costs / Risks

- Hashing conservative source directories costs extra filesystem reads once per
  process. The current Grug and DeepEP source sets are small compared with a
  kernel compile.
- Cache keys produced by this change abandon warm entries under the old schemes.
  The 30-day object lifecycle reclaims them.
- The XLA namespace trusts Iris's requested GPU variant. A malformed or missing
  variant will keep the cache node-local instead of sharing an ambiguous entry.
- Triton remains task-local until measurements justify remote mirroring.

## Design

Add a Rigging `compile_cache_key` helper that hashes an ordered set of files or
directory trees plus explicit environment strings. It includes logical relative
paths and file bytes, excludes `__pycache__`/`.pyc` output, and raises on a
missing source. It never includes absolute checkout paths, git metadata, or
timestamps.

CuTeDSL will replace `launch_provenance().tree_hash` in
[`cutlass_kernel_cache.py`](https://github.com/marin-community/marin/blob/b7169e65ac1b219887f1268df97d02f75caafb73/lib/levanter/src/levanter/cutlass_kernel_cache.py#L141-L161)
with a cached digest of `levanter/grug` and the installed distribution-version
set. The key will continue to include launcher name/configuration, stable
argument-spec representation, and GPU architecture. The decorator will reject
launchers defined outside the declared Grug source root, preventing a future
caller from receiving a persistent identity with an uncovered source file.

Rigging's
[`sync_kv_cache`](https://github.com/marin-community/marin/blob/b7169e65ac1b219887f1268df97d02f75caafb73/lib/rigging/src/rigging/cache.py#L251-L262)
will take an explicit namespace. Iris will derive the XLA namespace from
`jaxlib` version and `TaskResources.gpu_variant`; the same namespace will appear
in the node-local and object-store paths. Missing launch provenance still means
node-local only, and a missing GPU variant disables the remote mirror. JAX's
main compilation cache remains unchanged because its native key already covers
HLO, `jaxlib`, flags, and topology.

The fused cross-entropy autotune key will add a digest of its package source and
the JAX/`jaxlib` versions. Shapes, dtypes, options, device kind, and jaxpr stay in
the key. This makes both positive and negative entries expire when candidate
policy or the compiler changes.

DeepEP layout and transport will use `compile_cache_key` over the Levanter
DeepEP package, the external DeepEP `csrc` tree, and JAX FFI headers. Their
environment fields will cover schema, generated/patched source choices,
architecture and compile flags, compiler identity, build mode, and Python ABI
where relevant. Absolute paths will leave the key.

Triton and JAX keep their owner-generated keys. The PR will document why they do
not receive a git namespace, but it will not introduce another storage layer.

## Testing

Rigging tests will prove that identical content at different absolute paths has
one key; edits, logical path changes, and environment changes miss; generated
bytecode does not. Directory-sync tests will prove caller namespaces select
distinct remote roots.

CuTe tests will compile once across unrelated provenance changes, then recompile
after a Grug source or installed-package version change. A launcher outside the
source boundary will be rejected. Pallas tests will show compiler/source revision
changes select different autotune entries.

Iris tests will assert the XLA path and mirror namespace include XLA version and
GPU variant, and that an unknown GPU variant is never uploaded. DeepEP tests will
exercise key changes with source/header/toolchain changes without invoking a GPU
compiler.

## Open Questions

- Is `levanter/grug` the right conservative CuTe source boundary, or should the
  first version hash all of `levanter` for simpler ownership at the cost of more
  misses?
- Should Sonic's Triton cache move to `/cache` in this PR despite the lack of a
  startup measurement, or remain a measured follow-up?
