# Compile-cache invalidation research

## Scope

This review covers compilation and autotuning caches in `lib/levanter`, the
Rigging primitives they use, and Iris's XLA cache setup. Dataset, checkpoint,
model-download, and runtime attention KV caches are outside scope.

The starting revision is
[`b7169e65`](https://github.com/marin-community/marin/tree/b7169e65ac1b219887f1268df97d02f75caafb73).
Prior work is recorded in
[#8045](https://github.com/marin-community/marin/pull/8045),
[#8061](https://github.com/marin-community/marin/pull/8061), and
[#8065](https://github.com/marin-community/marin/pull/8065). The last change is
the safety baseline: hashing only the Python file that defined a launcher missed
the imported segmented-backward implementation and allowed a stale CuTeDSL
object to run.

## Inventory and recommendation

| Cache | Stored artifact and location | Current invalidation inputs | Finding | Recommendation |
| --- | --- | --- | --- | --- |
| JAX persistent compilation cache | XLA executable in region-local object storage | Non-optimized HLO, `jaxlib`, relevant XLA flags, device topology, compression, optional custom hook | The compiler already owns the semantic key. A git identity would only add false misses. Multi-node jobs need shared storage because only process 0 writes. | Keep the native JAX key and shared directory. |
| XLA per-fusion autotune cache | Compiler-owned directory under `/cache`, mirrored by `SyncedDirectory` | XLA owns per-fusion entries; Rigging adds the launch tree as the outer directory | The tree is broader than the artifact, but replacing it with requested Iris GPU resources is unsafe: requests can be `auto`, contain alternatives, and differ from placement. A fleet-wide version directory would also make `SyncedDirectory` fetch an unbounded tree. | Keep the bounded tree namespace in this PR. Follow up with observed-device identity, a bounded remote layout, and local generation cleanup as one change. |
| CuTeDSL object cache | One object per key in a 30-day region-local bucket | Launcher identity/config, full launch tree, argument spec, GPU architecture | Safe but far too broad. Commit hashes miss dirty edits; one defining module is unsafe; the whole tree invalidates on docs, experiments, Iris, and unrelated Levanter work. | Hash the complete `levanter/grug` tree and actual installed bytes of the declared CuTe toolchain distributions. Retain launcher config, spec, and observed device. Do not persist launchers defined outside that boundary. |
| Fused cross-entropy autotune cache | Selected block sizes or a negative result in `PersistentKvCache` | Backend, device kind, shapes/dtypes/options, jaxpr digest | Candidate-policy, shared autotune-helper, JAX, `jaxlib`, or `libtpu` changes can alter the decision without changing the jaxpr. Negative entries are highest risk because they suppress later attempts. | Add the full fused-CE source tree, shared Pallas autotune helpers, JAX, `jaxlib`, and actual `libtpu` bytes for TPU. Do not share a result when jaxpr or source identity is unavailable. |
| DeepEP layout FFI | Local `.so` under `~/.cache/marin` or `MARIN_DEEPEP_CACHE_DIR` | Two CUDA files, absolute paths, architecture, manual schema | Transitive headers, JAX FFI headers, build logic, host compiler, `nvcc`, and atomic publication are outside the key. Absolute paths create false misses. | Treat as a follow-up: content-address the full inputs and make concurrent builds private and atomic together. A partial key edit would imply safety it does not provide. |
| DeepEP transport FFI | Local `.so`/extension beside build outputs | Selected source/header bytes, module bytes, paths, flags/options, manual schema | Better than layout, but selected files can miss transitive headers and shared build helpers. Host compiler, Torch/Python ABI details, absolute paths, and concurrent publication remain. | Use the same complete build-input and atomic-publication follow-up as layout. |
| Triton/JAX-Triton Sonic cache | Triton-owned directory at `/tmp/marin-triton-cache` | Native kernel source/AST, signature/constants, backend/compiler target and versions | Invalidation belongs to Triton. The directory only survives the task/container. | Keep the native key. Measure compile cost and concurrent-writer behavior before adding persistence. |
| Python/JAX in-process memoization | Launchers, `cutlass_call` closures, traces and lowerings | Function/launcher identity and call arguments | Process-local only, so stale cross-run hits are impossible. | Keep the existing `lru_cache` wrappers. |

JAX documents its persistent key and rank-0 write behavior in the
[persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
OpenXLA says callers must separate XLA versions and use autotune results on the
same GPU type in its
[persisted autotuning guide](https://openxla.org/xla/persisted_autotuning).

## Strategy comparison

`git commit hash` is neither a content key nor a safe development key. Rebases
and metadata changes miss despite identical inputs, while dirty source can hit
the committed cache.

Marin's `git worktree/tree hash` uses `git stash create`, so it covers tracked
staged and unstaged changes but not untracked file contents. Untracked state is
reported separately as dirty. Even when complete, a repository-wide tree is a
safe fallback rather than a precise cache key: it couples every artifact to
every file in the launch bundle.

`module hash` is safe only when “module” means the complete source closure. A
defining Python file is not enough: launchers instantiate kernels from sibling
modules and external distributions, and FA4 uses dynamic imports that defeat a
simple static import walk.

`local source-set hash + environment` is the recommended application-owned key.
The caller declares a conservative directory boundary; Rigging hashes logical
paths and actual bytes, then adds compiler/runtime identities, observed device,
compile options, and an explicit schema. This enables reuse across commits and
unrelated worktree edits without accepting stale artifacts.

## Remaining uncertainty

The installed CuTe distributions total roughly 100 MB and `libtpu` roughly 200
MB. Their installed bytes, plus resolved import roots for editable installs, are
hashed once per process at first cache use. Version or `RECORD` metadata would be
cheaper but would miss locally patched installations; the byte scan is the cost
of fail-closed persistence.

CuTe's declared Levanter boundary is definition-site based: all current opted-in
factories live below `levanter.grug`. Future runtime patching from outside that
tree must either expand the boundary or disable persistence for that launcher.

The XLA directory and DeepEP builds need storage/publication changes as well as
new identities. They remain deliberately conservative rather than receiving
partial invalidation fixes.
