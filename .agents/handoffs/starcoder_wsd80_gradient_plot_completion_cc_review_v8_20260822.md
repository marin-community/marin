PASS_AFTER_BLOCKERS_RESOLVED

# CC review: StarCoder WSD80 gradient plot completion v8

Review method: read-only `claude -p` using `claude-opus-5` at max effort,
`plambdafour@proton.me`, `stripe_subscription`, with `ANTHROPIC_API_KEY`
removed from the child environment.

- Initial review session: `58be8ce3-4cf4-45e6-859d-dd4b52536005`
- Follow-up review session: `16995bea-1942-4e0a-8fa6-e74088d7ab1c`

## Initial findings

The first v8 review agreed that unconditional rebinding closes the exact v7
output-root failure, but blocked freezing on three related gaps:

1. Two remaining module-level verification caches could cross the same
   `__main__` cloudpickle boundary and suppress worker-side source or package
   verification.
2. The original adapter preflight changed bindings only in-process and could
   not demonstrate the Iris process boundary.
3. Stage 1 would have been the first execution of the worker package, source,
   configuration, and root gates, across eight workers at once.

## Resolution reviewed

The follow-up review re-read the patched runtime harness and freezer and found
all three blockers resolved:

- All runtime and installed-distribution memoization was removed. Every worker
  independently verifies the release document, 1,035-file historical source
  manifest, recovery and parent implementation maps, recovery implementation
  identity, Python version, and required JAX/JAXLIB/libtpu versions before
  rebinding the immutable v10 kernel.
- Every process entry reapplies the five required bindings. The captured
  original functions remain immutable, so repeated application cannot wrap a
  wrapper or accumulate recursion.
- The local cloudpickle round trip is correctly scoped as a binding smoke test,
  not as process-boundary evidence.
- A one-worker v5p-8 remote canary uses the same central1 resources, requested
  task image, and `RemoteCallable` serialization path as Stage 1. It verifies
  packages, bundled sources, group configuration, restored-step semantics, and
  the v8 result root without restoring a checkpoint or reading endpoint
  metrics.
- The canary writes a stable, release-specific, create-only GCS marker. A
  repeated invocation validates and skips that marker. Authorization and every
  Stage launch fail closed until the marker's release, identity, device,
  package, source, and implementation evidence passes.
- The remote marker lives outside the numerical result glob and cannot alter
  the exact-output audit.
- V7 is hash-pinned as non-consumable and disjoint in both result root and
  artifact version. Its eight failed workers produced no row JSON or completion
  markers.

## Scientific and execution scope

The reviewer found no change to the numerical gradient kernel, checkpoint
identities, source or target batches, 288-row manifest, stage split
`8/16/32/232`, acceptance tolerances, or idempotent create-only result
semantics. The remote canary addresses the v7 failure class; it intentionally
does not claim to exercise checkpoint restoration or numerical probes, which
remain gated by Stage 1 and its exact audit.

Required order:

1. Freeze v8 from the detached historical runtime tree.
2. Run local runtime-adapter preflight.
3. Run the one-worker remote adapter canary.
4. Authorize the release.
5. Run Stage 1 with eight rows and audit it standalone.
6. Promote Stages 2-4 only after each exact prior-stage audit passes.

The follow-up verdict was: no launch-blocking defect remains.
