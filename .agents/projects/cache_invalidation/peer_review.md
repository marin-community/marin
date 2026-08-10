# Claude peer review

Claude reviewed the initial inventory and design against the repository on
2026-08-08. The review rejected several proposed shortcuts before implementation.

## Findings incorporated

- Iris task resources describe requests, not observed placement. They can be
  `auto`, list alternatives, differ in case, or disagree with the installed GPU.
  They are not a safe XLA autotune identity.
- A fleet-wide XLA version/GPU directory would make `SyncedDirectory` fetch the
  entire accumulated tree, while the current launch-tree namespace bounds each
  download. Local generations would also need cleanup.
- CuTe depends on external kernel/compiler bodies, including Cutlass DSL,
  FlashAttention, and Quack. Version strings alone do not cover locally patched
  or rebuilt installs.
- Fused-CE TPU autotuning must include `libtpu`. A shared
  `jaxpr=unavailable` key can collide, and a negative entry under that key can
  suppress valid future work.
- Marin's launch tree uses `git stash create`; it covers tracked changes, not
  untracked file contents.
- A source helper needs unambiguous framing, root-qualified logical paths,
  explicit bytecode/symlink behavior, and errors for incomplete inputs.
- DeepEP keys omit host compiler, transitive headers, some runtime/ABI inputs,
  and atomic publication. Its low-value local cache should not receive a partial
  safety claim in this PR.

## Resulting boundary

The implementation is limited to a robust Rigging content-identity API, the
CuTeDSL object cache, and fused-CE autotuning. XLA and DeepEP remain unchanged
and have explicit follow-up requirements. The implementation strengthens the
review's package recommendation by hashing actual installed file bytes rather
than trusting `RECORD` data, which can remain unchanged after an in-place patch.
