# Compile-cache invalidation research

## Scope

This review covers caches in `lib/levanter`, the Rigging cache primitives they
use, and the Iris setup code that selects storage for XLA. Dataset, checkpoint,
Hugging Face download, and runtime KV caches are outside scope because they do
not store compilation or autotuning results.

The starting revision is
[`b7169e65`](https://github.com/marin-community/marin/tree/b7169e65ac1b219887f1268df97d02f75caafb73).
Prior work is recorded in
[#8045](https://github.com/marin-community/marin/pull/8045),
[#8061](https://github.com/marin-community/marin/pull/8061), and
[#8065](https://github.com/marin-community/marin/pull/8065). The last change is
the safety baseline: hashing only the Python file that defined a launcher missed
the imported 1,600-line segmented-backward implementation and allowed a stale
CuTeDSL object to run.

## Inventory and recommendation

| Cache | Stored artifact and location | Current invalidation inputs | Finding | Recommendation |
| --- | --- | --- | --- | --- |
| JAX persistent compilation cache | XLA executable in region-local object storage | Non-optimized HLO, `jaxlib` version, relevant XLA flags, device topology, compression, optional custom hook | The owner already hashes the computation and compilation environment. Adding a git identity would only create misses. Multi-node jobs need shared storage because only process 0 writes. | Keep the native JAX key and shared object-store directory. Do not add a git hash. |
| XLA per-fusion autotune cache | Compiler-owned directory under `/cache`, mirrored by `SyncedDirectory` | XLA owns per-fusion entries; Rigging adds the entire launch tree as the outer directory | The launch tree is unrelated to most autotune results, so every source edit starts cold. It is also the wrong safety boundary: OpenXLA requires callers to separate XLA versions and GPU types. A local directory reused across jobs can mix versions before mirroring. | Namespace both local and remote directories by `jaxlib`/XLA version and Iris GPU variant. Make `sync_kv_cache` accept the namespace instead of reading git provenance. |
| CuTeDSL object cache | One object per key in a 30-day region-local bucket | Launcher identity/config, full launch tree hash, argument spec, GPU architecture | Safe but too broad. Commit hashes miss dirty edits; a single module hash is unsafe; the full git tree invalidates on docs, experiments, Iris, and unrelated Levanter changes. | Hash the complete Levanter CuTe source set (`levanter/grug`) and the installed package-version set, then combine it with launcher config, spec, and device architecture. Work outside that declared source root should fail closed. |
| Fused cross-entropy autotune cache | Selected block sizes or a negative result in `PersistentKvCache` | Backend, device kind, shapes/dtypes/options, and a jaxpr digest; encoding changes rely on manual `block_sizes_vN` bumps | A jaxpr change invalidates kernel semantics, but an XLA/JAX upgrade or a candidate-policy change can leave the jaxpr unchanged. Negative entries are the highest-risk case because they suppress every future compile attempt. | Add a source hash for the full fused-CE package and JAX/`jaxlib` versions to the key. Retain shapes, device, options, and jaxpr. |
| DeepEP layout FFI | Local `.so` under `~/.cache/marin` or `MARIN_DEEPEP_CACHE_DIR` | Two CUDA files, absolute source/include paths, architecture, manual schema | Absolute paths cause false misses. Transitive DeepEP headers, JAX FFI header content, build logic, and `nvcc` identity can change without a miss. | Hash content, not absolute paths: local DeepEP shim sources, the DeepEP `csrc` tree, JAX FFI headers, compile flags, schema, and compiler identity. |
| DeepEP transport FFI | Local `.so`/Python extension beside raw build outputs | Selected CUDA/header contents, transport module bytes, paths, flags/options, manual schema | Better than layout, but the selected-file list can miss transitive headers and changes in shared availability/build helpers; paths still cause false misses; compiler identity is absent. | Use the same complete source-set/environment hash as layout, plus Python ABI and Torch build mode when selected. |
| Triton/JAX-Triton Sonic cache | Triton-owned directory at `/tmp/marin-triton-cache` | Triton's native kernel source/AST, signature/constants, backend/compiler target and versions | Invalidation belongs to Triton and should not be wrapped in a git namespace. The current location only survives within the task/container lifetime. | Keep the native key. Moving the directory to persistent node storage is useful, but remote mirroring should wait until we have measured Sonic compile cost and concurrent-writer behavior. |
| Python/JAX in-process memoization | Launcher instances, `cutlass_call` closures, traces and lowerings in memory | Python function/launcher identity and call arguments | Process-local only; stale cross-run hits are impossible. Recreating closures causes misses but not incorrect hits. | Keep the existing `lru_cache` wrappers. |

JAX documents the fields in its persistent key and the rank-0 write behavior in
its [persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
OpenXLA explicitly says the per-fusion directory can accumulate entries across
different models, while callers must separate XLA versions and use results on
the same GPU type in its
[persisted autotuning guide](https://openxla.org/xla/persisted_autotuning).

## Strategy comparison

`git commit hash` is neither a content key nor a safe development key. Rebases
and amended metadata miss despite identical inputs, while dirty and untracked
source changes can hit an old commit's cache.

`git worktree/tree hash` is content-addressed and includes dirty tracked and
untracked content in Marin's provenance implementation. It is a safe fallback
when the dependency closure is unknown, but it couples every artifact to every
file in the launch bundle. It should remain an image/build identity, not the
default compile-cache key.

`module hash` is safe only when “module” means the complete source closure. The
defining Python file is not enough: a launcher can instantiate kernel classes
from sibling Levanter modules and external distributions. Static import-graph
walking is also incomplete in this code because the FA4 adapters use dynamic
`importlib.import_module` calls.

`local source-set hash + environment` is the recommended application-owned key.
The caller declares a conservative directory boundary and Rigging hashes every
source file beneath it, excluding generated Python bytecode. The environment
part includes compiler/package versions, device identity, compile options, and
an explicit schema. This allows reuse across commits and unrelated worktree
edits while retaining fail-closed invalidation.

## Surprises and remaining uncertainty

The broad git-tree namespace is not universally safer. XLA's documented safety
boundary is XLA version plus GPU type; the current namespace changes on unrelated
source edits but does not directly identify the GPU model. The node-local XLA
directory also lacks any namespace, so old files can be uploaded into a new
remote namespace after a code rollout.

CuTeDSL object code is generated before JAX forms the HLO cache key, so JAX's
persistent cache cannot rescue this compile. The CuTe cache needs its own source
and environment identity even when a warm JAX executable exists.

Triton's native cache appears to own the right semantic inputs. We have not
measured whether persisting Sonic's cache beyond `/tmp` changes startup enough
to justify another mirror and object-store prefix.
