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
implementation. External packages have a simpler repository-level generation:
Marin installs them from the committed `uv.lock` with `uv sync --frozen`.

## Costs and risks

- Any `uv.lock` change starts Levanter-owned compile caches cold, including
  changes to unrelated dependencies. Dependency changes are infrequent enough
  that this broad boundary is preferable to runtime package inspection.
- A task without the Marin workspace lock disables shared Levanter-owned caches.
  An environment manually patched away from its lock violates the cache contract.
- New schemas leave old entries cold; the 30-day object lifecycle reclaims them.
- `levanter/grug` is a declared definition-site boundary, not a dynamic import
  tracer. Opted-in launchers outside it compile without persistent storage.
- XLA and DeepEP retain broad/local keys until their synchronization and build
  publication rules can be changed safely as a whole.

## Design

Rigging adds three small content primitives: `directory_content_hash` for an
internal source tree, `file_content_hash` for a file such as `uv.lock`, and
`combined_content_hash` for ordered labeled identities. Components are length
framed. Directory hashes include logical relative paths and bytes but exclude
absolute checkout paths, mtimes, and Python bytecode. Missing paths and symlinks
raise rather than producing an ambiguous key. `workspace_lock_hash` locates and
hashes the Marin lockfile, raising when no locked workspace is available.

CuTeDSL replaces `launch_provenance().tree_hash` with a process-cached digest of
the whole `levanter/grug` directory and `uv.lock`. The artifact key retains
launcher configuration, stable argument spec, and observed JAX device
architecture. The launcher factory marks whether its definition is inside
`levanter.grug`; uncovered launchers compile normally but are never written to
shared storage. If source or lock identity cannot be built, the entire CuTe
persistent layer degrades to compile-only for that process.

The fused cross-entropy key adds a digest of its package source, shared Pallas
autotune helpers, `uv.lock`, and observed JAX/`jaxlib` versions. Shapes, dtypes,
options, backend, observed device kind, and jaxpr remain. If jaxpr tracing,
source identity, or the lock is unavailable, the sweep can run but neither a
winner nor a negative result is shared.

JAX and Triton keep their compiler-native keys. XLA retains the existing
tree-scoped mirror in this PR: an observed GPU identity is not available at the
current setup boundary, and a fleet-wide version directory would make each task
download an unbounded cache tree. DeepEP remains a follow-up because correct
invalidation also requires atomic/private build publication.

## Testing

Rigging tests cover checkout-location independence; directory content and
logical-path invalidation; file and lockfile invalidation; unambiguous combining;
bytecode exclusion; and missing/symlink rejection.

CuTe tests cover reuse across a simulated process restart, source-revision
invalidation, source-identity failure, and a launcher outside the declared
boundary. Fused-CE tests cover revision invalidation and refusal to share when a
jaxpr is unavailable. Existing tests retain launcher config, spec, cache
concurrency, and positive/negative decision coverage.

## Decisions

- Use internal source-tree hashes plus `uv.lock` for Levanter-owned artifacts.
- Treat a frozen lock-derived environment as a cache invariant; fail closed when
  the lock is unavailable.
- Keep compiler-native keys when the compiler already owns the semantic inputs.
- Keep a broader safe boundary when narrowing requires missing observed state or
  a storage/publication redesign.
