# Content-address Levanter compile caches by their real inputs

Levanter's CuTeDSL object cache will survive unrelated Marin worktree changes
without serving stale kernels. The fused cross-entropy autotuner will also
invalidate positive and negative decisions when source or compiler inputs
change. The full inventory is in [research.md](research.md), and Claude's
independent review is summarized in [peer_review.md](peer_review.md).

## Challenges

The caches do not share one correct revision key. JAX hashes lowered HLO and its
compiler environment; XLA owns per-fusion files but delegates version and GPU
separation to the caller; CuTeDSL emits object code before JAX can hash the
module; Levanter's autotuner stores a performance decision; DeepEP invokes local
compilers over multiple source trees.

A narrow key must remain fail-closed. PR #8065 showed that hashing a launcher's
defining file was incomplete because it dynamically loaded a sibling kernel
implementation. Package versions have the same weakness for editable, rebuilt,
or locally patched installs.

## Costs and risks

- The CuTe toolchain and `libtpu` scans read about 100 MB and 200 MB respectively
  once per process, when those caches are first used.
- New schemas leave old entries cold; the 30-day object lifecycle reclaims them.
- `levanter/grug` is a declared definition-site boundary, not a dynamic import
  tracer. Opted-in launchers outside it compile without persistent storage.
- XLA and DeepEP retain broad/local keys until their synchronization and build
  publication rules can be changed safely as a whole.

## Design

Rigging adds `compile_cache_key`, which hashes ordered file or directory roots
plus explicit environment strings. Length-framed components include root index,
logical relative path, and file bytes. Absolute checkout paths and mtimes are
excluded. Python bytecode is excluded. Missing paths and symlinks raise rather
than producing an ambiguous key.

Rigging also adds `installed_distribution_fingerprint`. It hashes distribution
name, version, installed logical paths, and the actual bytes listed by package
metadata. For editable installs it additionally resolves the distribution's
import packages and hashes their source roots. Actual bytes, rather than
`RECORD` checksums alone, cover local patches made without metadata updates.

CuTeDSL replaces `launch_provenance().tree_hash` with a process-cached digest of
`levanter/grug` and a declared Cutlass/CuTe/FA4/Quack toolchain distribution
set. The artifact key retains launcher configuration, stable argument spec, and
observed JAX device architecture. The launcher factory marks whether its
definition is inside `levanter.grug`; uncovered launchers compile normally but
are never written to shared storage. If source identity cannot be built, the
entire CuTe persistent layer degrades to compile-only for that process.

The fused cross-entropy key adds a digest of its package source, shared Pallas
autotune helpers, and JAX/`jaxlib` versions. TPU decisions additionally include
actual installed `libtpu` bytes. Shapes, dtypes, options, backend, observed
device kind, and jaxpr remain. If jaxpr tracing or source identity fails, the
sweep can run but neither a winner nor a negative result is shared.

JAX and Triton keep their compiler-native keys. XLA retains the existing
tree-scoped mirror in this PR: an observed GPU identity is not available at the
current setup boundary, and a fleet-wide version directory would make each task
download an unbounded cache tree. DeepEP remains a follow-up because correct
invalidation also requires atomic/private build publication.

## Testing

Rigging tests cover checkout-location independence; source content, logical path,
and environment invalidation; bytecode exclusion; missing/symlink rejection; and
actual installed-file changes without a version bump.

CuTe tests cover reuse across a simulated process restart, source-revision
invalidation, source-identity failure, and a launcher outside the declared
boundary. Fused-CE tests cover revision invalidation and refusal to share when a
jaxpr is unavailable. Existing tests retain launcher config, spec, cache
concurrency, and positive/negative decision coverage.

## Decisions

- Use source-set plus environment hashes for Levanter-owned artifacts.
- Use actual installed bytes where editable or patched packages can affect code
  generation.
- Keep compiler-native keys when the compiler already owns the semantic inputs.
- Keep a broader safe boundary when narrowing requires missing observed state or
  a storage/publication redesign.
